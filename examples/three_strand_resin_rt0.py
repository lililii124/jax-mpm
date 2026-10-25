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
    from mpm_jax.resin_transport_rt0 import advect as advect_rt0
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
    start=0;transport_logJ=np.zeros(count);transport_reference=None;pressure_components=None
    if resume_from:
        with np.load(resume_from/'checkpoint.npz',allow_pickle=False) as cp:
            old_count=len(cp['mass']);assert old_count<=count and np.array_equal(cp['mass'],source['mass'][:old_count])
            leaves={name:cp[name].copy() for name in FlowState._fields};start=int(cp['step']);emitted=cp['emitted'].copy()
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
    meta=dict(source_horizons_s=horizons,inlet_protocol_scope='Future-only MAC inlet face flux parcels; each packet volume is apportioned by face-normal flux and transverse births lie on MAC face centres. Births are injected after the interval pressure/MAC solve and RT0-advected only for the exact remaining characteristic time; no past state or mass changes',transport_reference_checkpoint_sha256=transport_reference,transport_scope='RT0 exact frozen-grid characteristics wholly inside real mass-supported pressure domain; no air extension or artificial mass. APIC momentum unchanged; accumulated logJ never corrects old density',particle_boundary='RT0 shared zero-flux voxel faces; exact outlet-plane TOI ends material tracking and explicit sink removes exited mass/energy; no finite position projection or momentum impulse',case='three-strand-resin-fill-rt0-candidate',experimental=True,forming=geometry['receipt'],density_kg_m3=density,viscosity_Pa_s=viscosity,dt_s=dt,source_particle_capacity=count,source_packet_volume_m3=source['packet_volume_m3'],inlet_mac_spacing_m=source['mac_spacing_m'],inlet_mac_face_count=source['face_count'],nominal_flow_m3_per_s=source['nominal_flow_m3_per_s'],explicit_viscous_number=12*viscosity*dt/(density*h*h),nominal_advective_CFL=.2*dt/h,nominal_inlet_Reynolds=density*.1*.005/viscosity,pressure='mass-metric mixed JAX LSQR; original threshold plus complete positive-face owned/reachable domain closure; true primal residual<=1e-7, stationarity<=1e-10; pressure reconstruction diagnosed separately',physical_scope='Fixed actual fibre obstacles; laminar Newtonian filling, no cure, no full FSI, no prescribed final resin geometry',production_eligible=False)
    (folder/'plan.json').write_text(json.dumps(meta,indent=2));np.savez(folder/'fixed-fibres.npz',x=geometry['fibre_x'],fibre_id=geometry['fibre_id'])
    def save(k,rejected=False):
        leaves={name:np.asarray(v) for name,v in state._asdict().items()}
        if pressure_components is not None:leaves.update(last_pressure_multiplier_base=np.asarray(pressure_components[0]),last_pressure_multiplier_correction=np.asarray(pressure_components[1]),last_pressure_dt_s=dt,last_pressure_scope='Pre-advection last-step physical impulse potentials; use D.T(base)+D.T(correction), no rounded-potential collapse')
        np.savez(folder/('rejected-state.npz' if rejected else 'checkpoint.tmp.npz'),**leaves,transport_logJ=transport_logJ,step=k,emitted=emitted,source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,source_energy_J=source_energy,outflow_energy_J=outflow_energy)
        if not rejected:(folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
        (folder/'observables.json').write_text(json.dumps(rows,indent=2))
    save(start)
    if start==0:shutil.copyfile(folder/'checkpoint.npz',folder/'initial-checkpoint.npz')
    np.savez_compressed(folder/f'frame-{start:06d}.npz',x=np.asarray(state.x)[np.asarray(state.active)],time=np.asarray(state.time));started=time.monotonic()
    for k in range(start+1,start+steps+1):
        interval_start=float(state.time)
        state,metrics,increment,pressure_input=step(state);pressure_components=pressure_input[4:6];transport_logJ+=np.asarray(increment);values={name:(bool(v) if name.endswith('valid') else float(v)) for name,v in metrics.items()}
        xyz=np.asarray(state.x);active=np.asarray(state.active);v=np.asarray(state.v);C=np.asarray(state.C)
        exits=active&(np.asarray(pressure_input[6])|(xyz[:,0]>.025));outflow_mass+=float(source['mass'][exits].sum());outflow_energy+=float(.5*np.sum(source['mass'][exits]*(np.sum(v[exits]**2,axis=1)+h*h/4*np.sum(C[exits]**2))))
        active=active&~exits;state=state._replace(active=jnp.asarray(active))
        # Births are excluded from this interval's P2G, pressure and work solve.
        # After its MAC field is available, inject each parcel and integrate the
        # exact frozen RT0 characteristic for only its remaining interval time.
        interval_end=float(state.time)
        born=(source['birth_time']>interval_start+1e-15)&(source['birth_time']<=interval_end+1e-15)&~emitted
        remaining=np.maximum(interval_end-source['birth_time'],0.)
        newborn=advect_rt0(jnp.asarray(source['x']),pressure_input[7],jnp.asarray(born),solid_grid,pressure_input[8],layout,jnp.asarray(remaining),outlet_x=geometry['receipt']['cylinder_limits_m'][1])
        born_array=jnp.asarray(born);source_v=jnp.asarray(source['v'])
        state=state._replace(x=jnp.where(born_array[:,None],newborn.x,state.x),v=jnp.where(born_array[:,None],source_v,state.v),C=jnp.where(born_array[:,None,None],jnp.zeros_like(state.C),state.C),active=state.active|born_array,valid=state.valid&newborn.valid)
        transport_logJ[born]+=np.asarray(newborn.logJ)[born]
        emitted|=born;source_mass+=float(source['mass'][born].sum());source_energy+=float(.5*np.sum(source['mass'][born,None]*source['v'][born]**2))
        birth_exits=born&np.asarray(newborn.exited)
        if birth_exits.any():
            outflow_mass+=float(source['mass'][birth_exits].sum());outflow_energy+=float(.5*np.sum(source['mass'][birth_exits]*np.sum(source['v'][birth_exits]**2,axis=1)))
        active=np.asarray(state.active)&~birth_exits;state=state._replace(active=jnp.asarray(active))
        xyz=np.asarray(state.x);v=np.asarray(state.v);C=np.asarray(state.C)
        coordinates=np.floor((xyz-geometry['origin'])/h).astype(int);in_grid=np.all((coordinates>=0)&(coordinates<geometry['shape']),axis=1);safe=np.clip(coordinates,0,geometry['shape']-1);solid_occ=geometry['solid'][safe[:,0],safe[:,1],safe[:,2]]
        intrusion=int(np.sum(active&solid_occ));escaped=int(np.sum(active&~in_grid));mass=float(source['mass'][active].sum());mass_error=mass-(source_mass-outflow_mass)
        work_total+=values['fixture_work_J']
        loss_total+=sum(values[name] for name in ['p2g_loss_J','boundary_loss_J','viscous_loss_J','pressure_loss_J','g2p_loss_J','particle_wall_loss_J'])
        active_energy=float(.5*np.sum(source['mass'][active]*(np.sum(v[active]**2,axis=1)+h*h/4*np.sum(C[active]**2,axis=(1,2)))))
        global_identity=source_energy+work_total-outflow_energy-active_energy-loss_total
        row=dict(transport_logJ_reference_active_mean=float(transport_logJ[active].mean()) if active.any() else 0.,transport_logJ_reference_active_max=float(transport_logJ[active].max()) if active.any() else 0.,transport_logJ_reference_active_min=float(transport_logJ[active].min()) if active.any() else 0.,step=k,time_s=float(state.time),active_particles=int(active.sum()),source_interval_start_s=interval_start,source_interval_end_s=interval_end,newborn_count=int(born.sum()),newborn_remaining_time_min_s=float(remaining[born].min()) if born.any() else 0.,newborn_remaining_time_max_s=float(remaining[born].max()) if born.any() else 0.,source_particles=int(emitted.sum()),source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,active_mass_kg=mass,mass_identity_kg=mass_error,source_energy_J=source_energy,outflow_energy_J=outflow_energy,fixture_work_cumulative_J=work_total,loss_cumulative_J=loss_total,active_kinetic_J=active_energy,global_energy_identity_J=global_identity,solid_cell_intrusions=intrusion,escaped_particles=escaped,**values);rows.append(row)
        valid=bool(state.valid) and intrusion==0 and escaped==0 and abs(mass_error)<1e-12*max(source_mass,1e-30) and row['maximum_speed_m_per_s']*dt/h<.25
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
