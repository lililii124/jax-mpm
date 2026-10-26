"""Independent RT0 exact-characteristic transport candidate around fixed fibres.

Experimental independent stage: no cure/full FSI. Only run after operator checks
and an authenticated full-turn forming checkpoint. No final-cylinder particles
are prefilled or faded in. Every source mass/energy and outflow is explicit.
"""
import argparse,json,time,shutil,hashlib
from pathlib import Path
import numpy as np


def run(checkpoint,plan,output,steps=64,resume_from=None,source_until=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.three_strand_resin_geometry import obstacle_from_checkpoint,continued_mac_face_inlet_schedule
    from mpm_jax.resin_apic import MACLayout
    from mpm_jax.resin_fill import FlowState,make_step
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise ValueError('Fresh candidate output must be empty; resume not implemented')
    if not 1<=steps<=512:raise ValueError('Bounded1..512 fill steps only')
    geometry=obstacle_from_checkpoint(checkpoint,plan)
    density=1100.;viscosity=.05;dt=.00004
    horizons=json.loads((resume_from/'plan.json').read_text()).get('source_horizons_s',[.32]) if resume_from else [.32]
    if source_until is not None and source_until>horizons[-1]:horizons=horizons+[source_until]
    source=continued_mac_face_inlet_schedule(horizons,density=density,dx=geometry['dx']);count=source['capacity'];layout=MACLayout(tuple(geometry['shape']),tuple(geometry['origin']),geometry['dx'])
    boundary=tuple(jnp.asarray(v) for v in geometry['fixed']);target=tuple(jnp.asarray(v) for v in geometry['prescribed'])
    solid_grid=jnp.asarray(geometry['collision_solid'])
    step=make_step(layout,boundary,target,jnp.asarray(geometry['open_cells']),density=density,viscosity=viscosity,dt=dt,solid=solid_grid,transport='rt0',capture_pressure=True,outlet_x=geometry['receipt']['cylinder_limits_m'][1])
    state=FlowState(jnp.asarray(source['x']),jnp.zeros((count,3)),jnp.zeros((count,3,3)),jnp.asarray(source['mass']),jnp.zeros(count,dtype=bool),jnp.asarray(0.),jnp.asarray(True))
    emitted=np.zeros(count,dtype=bool);source_mass=0.;outflow_mass=0.;source_energy=0.;outflow_energy=0.;rows=[];h=layout.dx;work_total=0.;loss_total=0.
    def inlet_accounting(time_s):
        counts_by_face=np.bincount(source['face_id'][emitted],minlength=source['face_count'])
        flux_volume=source['face_flux_m3_per_s']*time_s
        raw_remainder=flux_volume-counts_by_face*source['packet_volume_m3']
        tolerance=64*np.finfo(float).eps*np.maximum(flux_volume,source['packet_volume_m3'])
        valid=bool(np.all(raw_remainder>=-tolerance)&np.all(raw_remainder<source['packet_volume_m3']+tolerance))
        remainder=np.clip(raw_remainder,0.,np.nextafter(source['packet_volume_m3'],0.))
        return counts_by_face,flux_volume,remainder,valid
    start=0;source_time_s=0.;transport_logJ=np.zeros(count);transport_reference=None;pressure_components=None
    if resume_from:
        with np.load(resume_from/'checkpoint.npz',allow_pickle=False) as cp:
            old_count=len(cp['mass']);assert old_count<=count and np.array_equal(cp['mass'],source['mass'][:old_count])
            leaves={name:cp[name].copy() for name in FlowState._fields};start=int(cp['step']);emitted=cp['emitted'].copy()
            if int(cp['step'])!=start:raise ValueError('Checkpoint step index mismatch')
            source_time_s=start*dt;actual_clock=float(cp['time']);clock_bound=8*max(start,1)*abs(np.spacing(source_time_s))
            if abs(actual_clock-source_time_s)>clock_bound:raise ValueError('Checkpoint physical time is inconsistent with its step index')
            if 'inlet_source_time_s' in cp and float(cp['inlet_source_time_s'])!=source_time_s:raise ValueError('Checkpoint nominal inlet clock mismatch')
            stored_remainder_time=float(cp['inlet_source_time_s']) if 'inlet_source_time_s' in cp else actual_clock
            if old_count<count:
                assert np.all(source['birth_time'][old_count:]>float(cp['time']))
                before={name:hashlib.sha256(leaves[name].tobytes()).hexdigest() for name in ['x','v','C','mass','active']}
                for name in ['x','v','C','mass','active']:
                    tail=source[name][old_count:] if name in ['x','mass'] else np.zeros((count-old_count,)+leaves[name].shape[1:],dtype=leaves[name].dtype)
                    leaves[name]=np.concatenate((leaves[name],tail),axis=0)
                    assert hashlib.sha256(leaves[name][:old_count].tobytes()).hexdigest()==before[name]
                emitted=np.pad(emitted,(0,count-old_count));(folder/'capacity-bridge.json').write_text(json.dumps(dict(old_particles=old_count,new_capacity=count,physical_time_s=float(cp['time']),existing_state_prefix_sha256=before,all_new_births_strictly_future=True,physical_inlet_unchanged=True,existing_mass_rescaled=False),indent=2))
            state=FlowState(*(jnp.asarray(leaves[name]) for name in FlowState._fields))
            has_transport_history='transport_logJ' in cp
            if has_transport_history:transport_logJ=np.pad(cp['transport_logJ'],(0,count-old_count))
            stored_inlet_remainder=cp['inlet_face_remainder_m3'].copy() if 'inlet_face_remainder_m3' in cp else None
            source_mass=float(cp['source_mass_kg']);outflow_mass=float(cp['outflow_mass_kg']);source_energy=float(cp['source_energy_J']);outflow_energy=float(cp['outflow_energy_J'])
            if not bool(state.valid) or not np.array_equal(np.asarray(state.mass),source['mass']):raise ValueError('Invalid changed source mass state')
        old=json.loads((resume_from/'observables.json').read_text())[-1]
        if old['step']!=start:raise ValueError('Checkpoint and ledger endpoint mismatch')
        work_total=old['fixture_work_cumulative_J'];loss_total=old['loss_cumulative_J']
        previous=json.loads((resume_from/'plan.json').read_text());transport_reference=previous.get('transport_reference_checkpoint_sha256')
        if transport_reference is None and not has_transport_history:
            from examples.three_strand_resin_geometry import sha256
            transport_reference=sha256(resume_from/'checkpoint.npz')
        if previous['forming']!=geometry['receipt'] or previous['dt_s']!=dt:raise ValueError('Changed frozen geometry/fluid parameters')
        if stored_inlet_remainder is None:raise ValueError('Resume checkpoint lacks conservative per-face inlet remainder')
        _,_,parent_remainder,remainder_valid=inlet_accounting(stored_remainder_time)
        if not remainder_valid or not np.allclose(parent_remainder,stored_inlet_remainder,rtol=0.,atol=1e-24):raise ValueError('Per-face inlet remainder changed across resume')
    meta=dict(source_horizons_s=horizons,inlet_protocol_scope='Future-only conservative MAC face packet source. A parcel is born after one full Vp crosses its face at nominal time (layer+1)*Vp/Qface. Transverse location is MAC face-centred; axial location is left+Vp/(2*dx^2). Face remainder is retained and recorded. Birth is injected at the end of the first solver interval at or after nominal time, then enters P2G, pressure, and RT0 on the next interval. Time quantization is at most dt, with first-order time error and displacement bound umax*dt; no substep transport is claimed. No past state or mass changes',transport_reference_checkpoint_sha256=transport_reference,transport_scope='Old active particles use the unchanged full-step RT0 frozen-grid characteristic in the actual mass-supported pressure domain; no air extension or artificial mass. Newborn particles start with logJ=0 and are not advanced until the next complete step. Accumulated logJ never corrects density',particle_boundary='RT0 shared zero-flux voxel faces; exact outlet-plane TOI ends material tracking and explicit sink removes exited mass/energy; no finite position projection or momentum impulse',case='three-strand-resin-fill-rt0-candidate',experimental=True,forming=geometry['receipt'],density_kg_m3=density,viscosity_Pa_s=viscosity,dt_s=dt,source_particle_capacity=count,source_packet_volume_m3=source['packet_volume_m3'],inlet_mac_spacing_m=source['mac_spacing_m'],inlet_mac_face_count=source['face_count'],inlet_maximum_birth_time_quantization_s=dt,inlet_maximum_birth_displacement_m=source['maximum_inlet_speed_m_per_s']*dt,nominal_flow_m3_per_s=source['nominal_flow_m3_per_s'],explicit_viscous_number=12*viscosity*dt/(density*h*h),nominal_advective_CFL=.2*dt/h,nominal_inlet_Reynolds=density*.1*.005/viscosity,pressure='mass-metric mixed JAX LSQR; original threshold plus complete positive-face owned/reachable domain closure; true primal residual<=1e-7, stationarity<=1e-10; pressure reconstruction diagnosed separately',physical_scope='Fixed actual fibre obstacles; laminar Newtonian filling, no cure, no full FSI, no prescribed final resin geometry',production_eligible=False)
    meta['pressure_surface']='supported-empty-surface-v1: remove a threshold-only empty cell if a free face has zero particle mass; every particle-owned cell remains mandatory and all six pressure faces require positive mass or prescribed velocity. No artificial face inertia or unprojected particle transport.'
    (folder/'plan.json').write_text(json.dumps(meta,indent=2));np.savez(folder/'fixed-fibres.npz',x=geometry['fibre_x'],fibre_id=geometry['fibre_id'])
    def save(k,rejected=False):
        leaves={name:np.asarray(v) for name,v in state._asdict().items()}
        if pressure_components is not None:leaves.update(last_pressure_multiplier_base=np.asarray(pressure_components[0]),last_pressure_multiplier_correction=np.asarray(pressure_components[1]),last_pressure_dt_s=dt,last_pressure_scope='Pre-advection last-step physical impulse potentials; use D.T(base)+D.T(correction), no rounded-potential collapse')
        _,flux_volume,remainder,_=inlet_accounting(source_time_s)
        np.savez(folder/('rejected-state.npz' if rejected else 'checkpoint.tmp.npz'),**leaves,transport_logJ=transport_logJ,step=k,emitted=emitted,source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,source_energy_J=source_energy,outflow_energy_J=outflow_energy,inlet_face_flux_volume_m3=flux_volume,inlet_face_remainder_m3=remainder,inlet_source_time_s=source_time_s)
        if not rejected:(folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
        (folder/'observables.json').write_text(json.dumps(rows,indent=2))
    save(start)
    if start==0:shutil.copyfile(folder/'checkpoint.npz',folder/'initial-checkpoint.npz')
    np.savez_compressed(folder/f'frame-{start:06d}.npz',x=np.asarray(state.x)[np.asarray(state.active)],time=np.asarray(state.time));started=time.monotonic()
    for k in range(start+1,start+steps+1):
        interval_start=(k-1)*dt
        state,metrics,increment,pressure_input=step(state);pressure_components=pressure_input[4:6];transport_logJ+=np.asarray(increment);values={name:(bool(v) if name.endswith('valid') else float(v)) for name,v in metrics.items()}
        interval_end=k*dt
        source_time_s=interval_end
        xyz=np.asarray(state.x);active=np.asarray(state.active);v=np.asarray(state.v);C=np.asarray(state.C)
        exits=active&(np.asarray(pressure_input[6])|(xyz[:,0]>.025));outflow_mass+=float(source['mass'][exits].sum());outflow_energy+=float(.5*np.sum(source['mass'][exits]*(np.sum(v[exits]**2,axis=1)+h*h/4*np.sum(C[exits]**2))))
        active=active&~exits;state=state._replace(active=jnp.asarray(active))
        # A full MAC-face packet is born at the interval endpoint. It does not
        # enter this interval's P2G, pressure, or RT0 transport; its first full
        # physical step starts on the next iteration.
        born=(source['birth_time']<=interval_end)&~emitted
        born_array=jnp.asarray(born);source_v=jnp.asarray(source['v'])
        state=state._replace(x=jnp.where(born_array[:,None],jnp.asarray(source['x']),state.x),v=jnp.where(born_array[:,None],source_v,state.v),C=jnp.where(born_array[:,None,None],jnp.zeros_like(state.C),state.C),active=state.active|born_array)
        emitted|=born;source_mass+=float(source['mass'][born].sum());source_energy+=float(.5*np.sum(source['mass'][born,None]*source['v'][born]**2))
        active=np.asarray(state.active)
        xyz=np.asarray(state.x);v=np.asarray(state.v);C=np.asarray(state.C)
        coordinates=np.floor((xyz-geometry['origin'])/h).astype(int);in_grid=np.all((coordinates>=0)&(coordinates<geometry['shape']),axis=1);safe=np.clip(coordinates,0,geometry['shape']-1);solid_occ=geometry['solid'][safe[:,0],safe[:,1],safe[:,2]]
        intrusion=int(np.sum(active&solid_occ));escaped=int(np.sum(active&~in_grid));mass=float(source['mass'][active].sum());mass_error=mass-(source_mass-outflow_mass)
        work_total+=values['fixture_work_J']
        loss_total+=sum(values[name] for name in ['p2g_loss_J','boundary_loss_J','viscous_loss_J','pressure_loss_J','g2p_loss_J','particle_wall_loss_J'])
        active_energy=float(.5*np.sum(source['mass'][active]*(np.sum(v[active]**2,axis=1)+h*h/4*np.sum(C[active]**2,axis=(1,2)))))
        global_identity=source_energy+work_total-outflow_energy-active_energy-loss_total
        counts_by_face,flux_volume,remainder,remainder_valid=inlet_accounting(source_time_s)
        quantization_delay=np.maximum(interval_end-source['birth_time'][born],0.)
        inlet_volume_balance_error=float(flux_volume.sum()-counts_by_face.sum()*source['packet_volume_m3']-remainder.sum())
        row=dict(inlet_mac_cumulative_flux_volume_m3=float(flux_volume.sum()),inlet_packet_volume_m3=float(counts_by_face.sum()*source['packet_volume_m3']),inlet_volume_remainder_m3=float(remainder.sum()),inlet_volume_balance_error_m3=inlet_volume_balance_error,inlet_face_remainder_min_m3=float(remainder.min()),inlet_face_remainder_max_m3=float(remainder.max()),inlet_face_remainder_valid=remainder_valid,newborn_quantization_delay_max_s=float(quantization_delay.max()) if born.any() else 0.,transport_logJ_reference_active_mean=float(transport_logJ[active].mean()) if active.any() else 0.,transport_logJ_reference_active_max=float(transport_logJ[active].max()) if active.any() else 0.,transport_logJ_reference_active_min=float(transport_logJ[active].min()) if active.any() else 0.,step=k,time_s=float(state.time),actual_state_time_s=float(state.time),source_time_s=source_time_s,source_clock_error_s=float(state.time)-source_time_s,active_particles=int(active.sum()),source_interval_start_s=interval_start,source_interval_end_s=interval_end,newborn_count=int(born.sum()),source_particles=int(emitted.sum()),source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,active_mass_kg=mass,mass_identity_kg=mass_error,source_energy_J=source_energy,outflow_energy_J=outflow_energy,fixture_work_cumulative_J=work_total,loss_cumulative_J=loss_total,active_kinetic_J=active_energy,global_energy_identity_J=global_identity,solid_cell_intrusions=intrusion,escaped_particles=escaped,**values);rows.append(row)
        valid=bool(state.valid) and intrusion==0 and escaped==0 and abs(mass_error)<1e-12*max(source_mass,1e-30) and row['maximum_speed_m_per_s']*dt/h<.25 and remainder_valid and abs(inlet_volume_balance_error)<1e-20 and row['newborn_quantization_delay_max_s']<=dt+1e-15
        if not valid:
            
            velocity_input,mass_input,fixed_input,fluid_input=pressure_input[:4]
            np.savez(folder/'rejected-pressure-input.npz',**{f'{prefix}{i}':np.asarray(value) for prefix,fields in [('velocity',velocity_input),('mass',mass_input),('fixed',fixed_input)] for i,value in enumerate(fields)},fluid=np.asarray(fluid_input),h=h,dt=dt,step=k)
            save(k,True);raise FloatingPointError(f'Rejected filling candidate at {k}; all diagnostics retained')
        if k%16==0 or k==start+steps:
            save(k);np.savez_compressed(folder/f'frame-{k:06d}.npz',x=xyz[active],time=np.asarray(state.time));print(json.dumps(row),flush=True)
    (folder/'review.json').write_text(json.dumps(dict(accepted_steps=[start,start+steps],wall_seconds=time.monotonic()-started,latest=rows[-1],production_eligible=False),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--forming-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--resume-from',type=Path);p.add_argument('--source-until',type=float);a=p.parse_args();run(a.checkpoint,a.forming_plan,a.output,a.steps,a.resume_from,a.source_until)

if __name__=='__main__':main()
