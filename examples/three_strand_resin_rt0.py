"""Independent RT0 exact-characteristic transport candidate around fixed fibres.

Experimental independent stage: no cure/full FSI. Only run after operator checks
and an authenticated full-turn forming checkpoint. No final-cylinder particles
are prefilled or faded in. Every source mass/energy and outflow is explicit.
"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np


def run(checkpoint,plan,output,steps=64,resume_from=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.three_strand_resin_geometry import obstacle_from_checkpoint,inlet_schedule
    from mpm_jax.resin_apic import MACLayout
    from mpm_jax.resin_fill import FlowState,make_step
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):raise ValueError('Fresh candidate output must be empty; resume not implemented')
    if not 1<=steps<=512:raise ValueError('Bounded1..512 fill steps only')
    geometry=obstacle_from_checkpoint(checkpoint,plan)
    density=1100.;viscosity=.05;dt=.00004
    source=inlet_schedule(density=density);count=source['capacity'];layout=MACLayout(tuple(geometry['shape']),tuple(geometry['origin']),geometry['dx'])
    boundary=tuple(jnp.asarray(v) for v in geometry['fixed']);target=tuple(jnp.asarray(v) for v in geometry['prescribed'])
    step=make_step(layout,boundary,target,jnp.asarray(geometry['open_cells']),density=density,viscosity=viscosity,dt=dt,solid=jnp.asarray(geometry['collision_solid']),transport='rt0',capture_pressure=True)
    state=FlowState(jnp.asarray(source['x']),jnp.zeros((count,3)),jnp.zeros((count,3,3)),jnp.asarray(source['mass']),jnp.zeros(count,dtype=bool),jnp.asarray(0.),jnp.asarray(True))
    emitted=np.zeros(count,dtype=bool);source_mass=0.;outflow_mass=0.;source_energy=0.;outflow_energy=0.;rows=[];h=layout.dx;work_total=0.;loss_total=0.
    start=0;transport_logJ=np.zeros(count);transport_reference=None
    if resume_from:
        with np.load(resume_from/'checkpoint.npz',allow_pickle=False) as cp:
            state=FlowState(*(jnp.asarray(cp[name]) for name in FlowState._fields));start=int(cp['step']);emitted=cp['emitted'].copy()
            if 'transport_logJ' in cp:transport_logJ=cp['transport_logJ'].copy()
            source_mass=float(cp['source_mass_kg']);outflow_mass=float(cp['outflow_mass_kg']);source_energy=float(cp['source_energy_J']);outflow_energy=float(cp['outflow_energy_J'])
            if not bool(state.valid) or not np.array_equal(cp['mass'],source['mass']):raise ValueError('Invalid changed source mass state')
        old=json.loads((resume_from/'observables.json').read_text())[-1]
        if old['step']!=start:raise ValueError('Checkpoint and ledger endpoint mismatch')
        work_total=old['fixture_work_cumulative_J'];loss_total=old['loss_cumulative_J']
        previous=json.loads((resume_from/'plan.json').read_text());transport_reference=previous.get('transport_reference_checkpoint_sha256')
        if transport_reference is None:
            from examples.three_strand_resin_geometry import sha256
            transport_reference=sha256(resume_from/'checkpoint.npz')
        if previous['forming']!=geometry['receipt'] or previous['dt_s']!=dt:raise ValueError('Changed frozen geometry/fluid parameters')
    meta=dict(transport_reference_checkpoint_sha256=transport_reference,transport_scope='RT0 exact frozen-grid characteristics wholly inside real mass-supported pressure domain; no air extension or artificial mass. APIC momentum unchanged; accumulated logJ never corrects old density',particle_boundary='RT0 shared zero-flux voxel faces; no finite particle position projection or momentum impulse',case='three-strand-resin-fill-rt0-candidate',experimental=True,forming=geometry['receipt'],density_kg_m3=density,viscosity_Pa_s=viscosity,dt_s=dt,source_particle_capacity=count,nominal_flow_m3_per_s=source['nominal_flow_m3_per_s'],explicit_viscous_number=12*viscosity*dt/(density*h*h),nominal_advective_CFL=.2*dt/h,nominal_inlet_Reynolds=density*.1*.005/viscosity,pressure='mass-metric mixed JAX LSQR; original threshold plus complete positive-face owned/reachable domain closure; true primal residual<=1e-7, stationarity<=1e-10; pressure reconstruction diagnosed separately',physical_scope='Fixed actual fibre obstacles; laminar Newtonian filling, no cure, no full FSI, no prescribed final resin geometry',production_eligible=False)
    (folder/'plan.json').write_text(json.dumps(meta,indent=2));np.savez(folder/'fixed-fibres.npz',x=geometry['fibre_x'],fibre_id=geometry['fibre_id'])
    def save(k,rejected=False):
        leaves={name:np.asarray(v) for name,v in state._asdict().items()};np.savez(folder/('rejected-state.npz' if rejected else 'checkpoint.tmp.npz'),**leaves,transport_logJ=transport_logJ,step=k,emitted=emitted,source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,source_energy_J=source_energy,outflow_energy_J=outflow_energy)
        if not rejected:(folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
        (folder/'observables.json').write_text(json.dumps(rows,indent=2))
    save(start)
    if start==0:shutil.copyfile(folder/'checkpoint.npz',folder/'initial-checkpoint.npz')
    np.savez_compressed(folder/f'frame-{start:06d}.npz',x=np.asarray(state.x)[np.asarray(state.active)],time=np.asarray(state.time));started=time.monotonic()
    for k in range(start+1,start+steps+1):
        # Activate exclusively at the physical inlet. Quantization error is <=dt;
        # source kinetic energy is recorded before transfer, not fixture work.
        born=(source['birth_time']<=k*dt)&~emitted;emitted|=born
        source_mass+=float(source['mass'][born].sum());source_energy+=float(.5*np.sum(source['mass'][born,None]*source['v'][born]**2))
        state=state._replace(active=state.active|jnp.asarray(born),v=jnp.where(jnp.asarray(born)[:,None],jnp.asarray(source['v']),state.v))
        state,metrics,increment,pressure_input=step(state);transport_logJ+=np.asarray(increment);values={name:(bool(v) if name.endswith('valid') else float(v)) for name,v in metrics.items()}
        xyz=np.asarray(state.x);active=np.asarray(state.active);v=np.asarray(state.v);C=np.asarray(state.C)
        exits=active&(xyz[:,0]>.025);outflow_mass+=float(source['mass'][exits].sum());outflow_energy+=float(.5*np.sum(source['mass'][exits]*(np.sum(v[exits]**2,axis=1)+h*h/4*np.sum(C[exits]**2,axis=(1,2)))))
        active=active&~exits;state=state._replace(active=jnp.asarray(active))
        coordinates=np.floor((xyz-geometry['origin'])/h).astype(int);in_grid=np.all((coordinates>=0)&(coordinates<geometry['shape']),axis=1);safe=np.clip(coordinates,0,geometry['shape']-1);solid=geometry['solid'][safe[:,0],safe[:,1],safe[:,2]]
        intrusion=int(np.sum(active&solid));escaped=int(np.sum(active&~in_grid));mass=float(source['mass'][active].sum());mass_error=mass-(source_mass-outflow_mass)
        work_total+=values['fixture_work_J']
        loss_total+=sum(values[name] for name in ['p2g_loss_J','boundary_loss_J','viscous_loss_J','pressure_loss_J','g2p_loss_J','particle_wall_loss_J'])
        active_energy=float(.5*np.sum(source['mass'][active]*(np.sum(v[active]**2,axis=1)+h*h/4*np.sum(C[active]**2,axis=(1,2)))))
        global_identity=source_energy+work_total-outflow_energy-active_energy-loss_total
        row=dict(transport_logJ_reference_active_mean=float(transport_logJ[active].mean()) if active.any() else 0.,transport_logJ_reference_active_max=float(transport_logJ[active].max()) if active.any() else 0.,transport_logJ_reference_active_min=float(transport_logJ[active].min()) if active.any() else 0.,step=k,time_s=float(state.time),active_particles=int(active.sum()),source_particles=int(emitted.sum()),source_mass_kg=source_mass,outflow_mass_kg=outflow_mass,active_mass_kg=mass,mass_identity_kg=mass_error,source_energy_J=source_energy,outflow_energy_J=outflow_energy,fixture_work_cumulative_J=work_total,loss_cumulative_J=loss_total,active_kinetic_J=active_energy,global_energy_identity_J=global_identity,solid_cell_intrusions=intrusion,escaped_particles=escaped,**values);rows.append(row)
        valid=bool(state.valid) and intrusion==0 and escaped==0 and abs(mass_error)<1e-12*max(source_mass,1e-30) and row['maximum_speed_m_per_s']*dt/h<.25
        if not valid:
            
            velocity_input,mass_input,fixed_input,fluid_input=pressure_input
            np.savez(folder/'rejected-pressure-input.npz',**{f'{prefix}{i}':np.asarray(value) for prefix,fields in [('velocity',velocity_input),('mass',mass_input),('fixed',fixed_input)] for i,value in enumerate(fields)},fluid=np.asarray(fluid_input),h=h,dt=dt,step=k)
            save(k,True);raise FloatingPointError(f'Rejected filling candidate at {k}; all diagnostics retained')
        if k%16==0 or k==start+steps:
            save(k);np.savez_compressed(folder/f'frame-{k:06d}.npz',x=xyz[active],time=np.asarray(state.time));print(json.dumps(row),flush=True)
    (folder/'review.json').write_text(json.dumps(dict(accepted_steps=[start,start+steps],wall_seconds=time.monotonic()-started,latest=rows[-1],production_eligible=False),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--forming-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--resume-from',type=Path);a=p.parse_args();run(a.checkpoint,a.forming_plan,a.output,a.steps,a.resume_from)

if __name__=='__main__':main()
