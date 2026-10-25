"""Finite 0/+60/-60 dry weaving station: endpoint fixtures + real APIC/Hertz.

Default execution prints a NumPy plan only. No server dispatch or media is
performed here. A full protocol is not qualified by a two-step admission probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .triaxial_weaving_geometry import make_process,fixture_position,swept_corridor_seeds


def json_value(value):
    if isinstance(value,dict):return {k:json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return value.item()
    return value


def plan_process(per_family=2,dx=.000075,pitch=.0027,diameter=.000225,particle_spacing=.00006):
    geometry=make_process(per_family=per_family,pitch=pitch,diameter=diameter,spacing=particle_spacing)
    young,poisson,density=10e6,.25,1500.
    lam=young*poisson/((1+poisson)*(1-2*poisson));mu=young/(2*(1+poisson))
    wavespeed=np.sqrt((lam+2*mu)/density);dt=.08*dx/wavespeed
    bounds=geometry['swept_reference_bounds'];shape=np.ceil(bounds[:,:,1]/dx).astype(int)-np.floor(bounds[:,:,0]/dx).astype(int)+13
    nodes=int(np.prod(shape,axis=1).sum());segments=[f['longitudinal_sections']-1 for f in geometry['metadata']['fibres']]
    candidates=sum(segments[a]*segments[b] for a,b in [c['fibre_ids'] for c in geometry['contact_candidates']])
    maximum=geometry['metadata']['maximum_fixture_displacement_m']
    record=dict(case='finite-dry-triaxial-weaving-process-v1',particles=len(geometry['x']),
        fibres=len(segments),per_family=per_family,dx=dx,dt=float(dt),nominal_elastic_cfl=.08,
        duration_s=geometry['metadata']['duration_s'],full_steps=int(np.ceil(geometry['metadata']['duration_s']/dt)),
        young_Pa=young,poisson=poisson,density_kg_m3=density,tensile_strength_Pa=.2*young,fracture_energy_J_m2=1000.,
        swept_dense_nodes=nodes,seven_grid_scalars_subtotal_bytes=nodes*7*8,
        coverage_pair_upper_bound=candidates,coverage_skin_m=2*maximum+2*geometry['metadata']['diameter_m'],
        contact_active_cell_capacity=1024,contact_patch_capacity=64,contact_iterations=128,
        boundary_particle_count=int(np.any(np.asarray(geometry['masks']),axis=0).sum()),
        body_particles_unprescribed=int((~np.any(np.asarray(geometry['masks']),axis=0)).sum()),
        protocol=geometry['metadata']['stages'],matrix_present=False,
        scope='Ideal flexible circular cords, finite machine-end commands; no friction, resin, industrial rate or strength calibration',
        admission='Actual resource pilot required; analytic storage is not peak memory',
        validation_required=['mass/IDs constant','no unsupported end or same-family contact','contact coverage valid',
            'small local Hertz footprint and connected extent','undegraded material','positive detF',
            'finite strain and radial deformation','guide errors','actual crossing order after closing',
            'boundary/contact work and momentum','dt and discretization sensitivity before physical performance claims'])
    record['within_initial_resource_limits']=bool(nodes<=30_000_000 and candidates<=2_000_000)
    return geometry,record


def run(output,*,per_family=2,steps=None,saved_frames=8,resume=False,dx=.000075,pitch=.0027,diameter=.000225,particle_spacing=.00006,grid="dense",upgrade_work_observer=False):
    geometry,plan=plan_process(per_family,dx,pitch,diameter,particle_spacing)
    if not plan['within_initial_resource_limits']:raise ValueError('Need a new resource plan and swept sparse contact/grid strategy for this larger station')
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import (FibreConfig,build_layout,make_state,reference_grips,
        step_with_diagnostics,validate_state,stored_energy,FibreState,build_tube_layout)
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle,BrittleHistory
    from mpm_jax.fibre_coupling import FibreCouplingHistory
    from mpm_jax.fibre_contact import FibreContactHistory
    from mpm_jax.fibre_coupling import build_contact_data,section_centres,closest_segments
    from mpm_jax.fibre_contact import fibre_contact_parameters
    from mpm_jax.contact_patches import patch_neighbours,contact_patches
    from mpm_jax.fibre_patch_contact import initial_patch_history,evaluate_hertz_patches
    output=Path(output)
    if resume:
        if not output.is_dir():raise ValueError('Resume requires existing evidence directory')
    else:output.mkdir(parents=True,exist_ok=False)
    count=plan['full_steps'] if steps is None else steps
    if not 1<=count<=plan['full_steps']:raise ValueError('Bounded positive steps within fixed protocol required')
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),root/'examples/triaxial_weaving_geometry.py',root/'examples/triaxial_fibre_geometry.py',*sorted((root/'mpm_jax').rglob('*.py'))]
    plan['grid_allocation']=grid
    if grid not in ('dense','tube'):raise ValueError('Unknown grid allocation')
    plan['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    runtime=dict(jax=jax.__version__,jaxlib=__import__('jaxlib').__version__,numpy=np.__version__,x64=bool(jax.config.jax_enable_x64))
    signature=hashlib.sha256(json.dumps(json_value(dict(plan=plan,geometry=geometry['metadata'],runtime=runtime)),sort_keys=True).encode()).hexdigest()
    plan['requested_steps']=count;plan['signature']=signature;plan['runtime']=runtime
    if resume:
        prior=json.loads((output/'plan.json').read_text())
        if prior['signature']!=signature:
            if not upgrade_work_observer:raise ValueError('Source signature differs; explicit documented observer upgrade required')
            omit={'source_sha256','requested_steps','signature','runtime'}
            assert {k:v for k,v in prior.items() if k not in omit}=={k:json_value(v) for k,v in plan.items() if k not in omit}
            old_sources=prior['source_sha256'];new_sources=plan['source_sha256']
            assert old_sources['examples/triaxial_weaving_process.py']=='d6af5976dae9b1429fdf5601c4230adb172ecefafcdde2b4def2550006d938f4','Only the audited pre-work-observer source may upgrade'
            assert set(old_sources)==set(new_sources)
            assert all(old_sources[k]==new_sources[k] for k in old_sources if k!='examples/triaxial_weaving_process.py')
            (output/'work-observer-upgrade.json').write_text(json.dumps(dict(old_signature=prior['signature'],new_signature=signature,
                only_changed_source='examples/triaxial_weaving_process.py',scope='Add physical fixture work and projection-loss observer; no solver/material/guide changes'),indent=2))
    (output/'plan.json').write_text(json.dumps(json_value(plan),indent=2))
    (output/'geometry.json').write_text(json.dumps(json_value(geometry['metadata']),indent=2))
    np.savez_compressed(output/'reference.npz',reference=geometry['x'],mass=geometry['mass'],volume=geometry['volume'],
        fibre_id=geometry['fibre_id'],family_id=geometry['family_id'],grip_masks=np.asarray(geometry['masks']))
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],
        tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx'])
    if grid=='tube':
        seeds,seed_ids=swept_corridor_seeds(geometry,plan['dx'])
        layout=build_tube_layout(seeds,seed_ids,dx=plan['dx'],padding_cells=6,
            reference_bounds=geometry['swept_reference_bounds'],max_nodes=30_000_000)
    else:
        layout=build_layout(geometry['x'],geometry['fibre_id'],dx=plan['dx'],padding_cells=6,
            reference_bounds=geometry['swept_reference_bounds'],max_nodes=30_000_000)
    (output/'layout.json').write_text(json.dumps(dict(allocation=grid,active_nodes=layout.node_count,
        dense_address_count=int(np.sum(np.prod(np.asarray(layout.shape),axis=1))))))
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout)
    config=FibreConfig(dt=plan['dt']);times=geometry['metadata']['fixture_times'];positions=geometry['metadata']['fixture_positions']
    def grip_speed(group):
        return lambda t:(fixture_position(t,times,positions,jnp)[group]-fixture_position(t-config.dt,times,positions,jnp)[group])/config.dt
    boundary=reference_grips(geometry['masks'],tuple(grip_speed(i) for i in range(len(geometry['masks']))))
    data=build_contact_data(geometry,search_window=8,coverage_skin=plan['coverage_skin_m'],max_coverage_pairs=2_000_000)
    neighbours=patch_neighbours(np.asarray(data.coverage_pairs),np.asarray(data.segment_fibre))
    law=fibre_contact_parameters(effective_modulus=plan['young_Pa']/(2*(1-plan['poisson']**2)),
        effective_radius=geometry['metadata']['diameter_m']/2,work_of_adhesion=0.,model='hertz')
    history=initial_patch_history(data,max_patches=64)
    @jax.jit
    def evaluate(s,h):
        force,updated,metrics=evaluate_hertz_patches(s,h,data,law,neighbours,capacity=1024,max_patches=64,max_iterations=128)
        centres,_=section_centres(s.x,s.mass,data);near=data.coverage_pairs;ends=data.segments[near]
        first,second,_,_,distance=closest_segments(centres[ends[:,0,0]],centres[ends[:,0,1]],centres[ends[:,1,0]],centres[ends[:,1,1]])
        gap=distance-jnp.sum(data.fibre_radius[data.segment_fibre[near]],axis=1)
        selection=contact_patches(gap,gap<0,neighbours,capacity=1024,max_patches=64,max_iterations=128)
        cells=selection.cells;live=cells<len(near);points=jnp.concatenate(((first+second)/2,jnp.zeros((1,3))))[cells]
        low=jnp.full((1025,3),jnp.inf).at[selection.labels].min(jnp.where(live[:,None],points,jnp.inf))
        high=jnp.full((1025,3),-jnp.inf).at[selection.labels].max(jnp.where(live[:,None],points,-jnp.inf))
        roots=jnp.nonzero(live&(selection.labels==jnp.arange(1024)),size=64,fill_value=1024)[0]
        extent=jnp.linalg.norm(jnp.where((roots<1024)[:,None],high[roots]-low[roots],0.),axis=1)
        spacing=jnp.max(jnp.linalg.norm(centres[data.segments[:,1]]-centres[data.segments[:,0]],axis=1))
        local=selection.valid&jnp.all(extent<=jnp.maximum(2*spacing,4*jnp.max(metrics.contact_radius,initial=0.)))
        return force,updated,metrics,jnp.max(extent,initial=0.),local
    @jax.jit
    def advance(s,h,accumulator,n):
        def body(_,carry):
            old,attachment,a=carry
            def attempt(_):
                force,new_h,metrics,extent,local=evaluate(old,attachment)
                new,diag=step_with_diagnostics(old,material,layout,config,force,boundary=boundary)
                stretches=jnp.linalg.svd(new.F,compute_uv=False)
                constitutive_ok=(jnp.all(jnp.abs(jnp.log(stretches))<=.12)&jnp.all(new.history.damage<=1e-10))
                new=new._replace(valid=new.valid&metrics.valid&local&constitutive_ok)
                contact_work=jnp.sum(force*(new.x-old.x))
                return new,new_h,a+jnp.array([contact_work,jnp.abs(diag.boundary_work),
                    jnp.linalg.norm(metrics.force_residual),jnp.linalg.norm(metrics.torque_residual),
                    jnp.sum(diag.group_impulses*jnp.stack([grip_speed(i)(old.time+config.dt) for i in range(len(geometry['masks']))])),
                    jnp.sum(diag.group_impulses*jnp.stack([grip_speed(i)(old.time+config.dt) for i in range(len(geometry['masks']))]))-diag.boundary_work])
            return jax.lax.cond(old.valid,attempt,lambda _:carry,None)
        return jax.lax.fori_loop(0,n,body,(s,h,accumulator))
    observations=[];accumulator=jnp.zeros(6);start=0
    observer_start=0;observer_initial_energy=0.;observer_initial_grid_work=0.;observer_initial_abs_work=0.
    if resume:
        with np.load(output/'checkpoint.npz',allow_pickle=False) as cp:
            assert (str(cp['signature'])==signature or (upgrade_work_observer and str(cp['signature'])==prior['signature'])) and json.loads(str(cp['runtime']))==runtime
            values={name:jnp.asarray(cp['state_'+name]) for name in state._fields if name!='history'}
            values['history']=BrittleHistory(*(jnp.asarray(cp['history_'+name]) for name in state.history._fields))
            state=FibreState(**values)
            history=FibreCouplingHistory(FibreContactHistory(jnp.asarray(cp['contact_law_active']),jnp.asarray(cp['contact_law_valid'])),jnp.asarray(cp['contact_radius']),jnp.asarray(cp['contact_segments']),jnp.asarray(cp['contact_valid']))
            accumulator=jnp.asarray(cp['accumulator']);start=int(cp['step'])
            if len(accumulator)==4:
                assert upgrade_work_observer
                accumulator=jnp.concatenate([accumulator,jnp.zeros(2)])
                observer_start=start;observer_initial_grid_work=float(state.boundary_work);observer_initial_abs_work=float(accumulator[1])
                observer_initial_energy=float(.5*jnp.sum(state.mass[:,None]*state.v**2)+stored_energy(state,material)+jnp.sum(evaluate(state,history)[2].potential))
            else:
                observer_start=int(cp['observer_start']);observer_initial_energy=float(cp['observer_initial_energy'])
                observer_initial_grid_work=float(cp['observer_initial_grid_work']);observer_initial_abs_work=float(cp['observer_initial_abs_work'])
            exact={name:bool(np.array_equal(cp['state_'+name],np.asarray(getattr(state,name)))) for name in state._fields if name!='history'}
            exact.update({'history_'+name:bool(np.array_equal(cp['history_'+name],np.asarray(getattr(state.history,name)))) for name in state.history._fields})
            assert all(exact.values()) and np.array_equal(cp['accumulator'][:4],np.asarray(accumulator)[:4])
            (output/'restore-exact.json').write_text(json.dumps(dict(step=start,physical_fields=exact,original_accumulator_first4_exact=True),indent=2))
        if not start<count:raise ValueError('Resume endpoint must advance the actual checkpoint')
        observations=json.loads((output/'observations.json').read_text());assert observations[-1]['step']==start
        validate_state(state,layout)
        with np.load(output/f'frame_{start:07d}.npz') as frame:assert np.array_equal(np.asarray(state.x),frame['x'])

    def save(step):
        validate_state(state,layout)
        force,_,metrics,extent,local=evaluate(state,history)
        if not bool(metrics.valid&local):raise RuntimeError('Contact applicability failed at saved state')
        targets=fixture_position(float(state.time),times,positions);error=max(float(np.max(np.linalg.norm(np.asarray(state.x)[mask]-geometry['x'][mask]-(target-start),axis=1))) for mask,target,start in zip(geometry['masks'],targets,positions[0]))
        if error>plan['dx']*.02:raise RuntimeError('Actual endcap motion differs from prescribed machine command')
        K=float(.5*jnp.sum(state.mass[:,None]*state.v**2));U=float(stored_energy(state,material));H=float(jnp.sum(metrics.potential))
        row=dict(step=step,time_s=float(state.time),K=K,U=U,Hertz_potential=H,boundary_work=float(state.boundary_work),
            contact_particle_work=float(accumulator[0]),absolute_boundary_step_work=float(accumulator[1]),
            work_observer_start_step=observer_start,window_machine_work=float(accumulator[4]),window_projection_loss=float(accumulator[5]),
            window_grid_kinetic_projection_work=float(state.boundary_work)-observer_initial_grid_work,
            window_energy_change=K+U+H-observer_initial_energy,
            window_energy_minus_machine_work=K+U+H-observer_initial_energy-float(accumulator[4]),
            window_absolute_grid_work=float(accumulator[1])-observer_initial_abs_work,
            energy_minus_boundary_work=K+U+H-float(state.boundary_work),maximum_grip_error_m=error,
            minimum_gap_m=float(metrics.minimum_coverage_gap),contact_patches=int(metrics.patch_count),maximum_patch_extent_m=float(extent),
            force_residual=float(jnp.linalg.norm(metrics.force_residual)),torque_residual=float(jnp.linalg.norm(metrics.torque_residual)),
            actual_mpm_updates=step,contact_valid=True,mpm_valid=bool(state.valid))
        observations.append(row);(output/'observations.json').write_text(json.dumps(observations,indent=2))
        np.savez_compressed(output/f'frame_{step:07d}.npz',x=np.asarray(state.x),v=np.asarray(state.v),F=np.asarray(state.F),
            C=np.asarray(state.C),damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed),
            fibre_id=geometry['fibre_id'],family_id=geometry['family_id'],mass=np.asarray(state.mass),time=float(state.time),
            fixture_targets=targets,contact_points=np.asarray(metrics.closest_points),contact_active=np.asarray(metrics.active),
            contact_pair_segments=np.asarray(metrics.pair_segments),contact_segment_fibre=np.asarray(data.segment_fibre),
            contact_segments=np.asarray(data.segments),particle_section=np.asarray(data.particle_section))
        payload={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'}
        payload.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
        temporary=output/'checkpoint.partial.npz'
        np.savez_compressed(temporary,step=step,signature=signature,runtime=json.dumps(runtime,sort_keys=True),accumulator=np.asarray(accumulator),observer_start=observer_start,observer_initial_energy=observer_initial_energy,observer_initial_grid_work=observer_initial_grid_work,observer_initial_abs_work=observer_initial_abs_work,**payload,
            **{'contact_'+k:np.asarray(v) for k,v in history._asdict().items() if k!='law'},
            contact_law_active=np.asarray(history.law.active),contact_law_valid=np.asarray(history.law.valid))
        temporary.replace(output/'checkpoint.npz')
        print(f'step={step} patches={int(metrics.patch_count)} grip_error={error:.3e}',flush=True)
    if not resume:save(0)
    stops=np.unique(np.rint(np.linspace(start,count,min(saved_frames,count-start)+1)).astype(int))
    previous=start
    for stop in stops[1:]:
        state,history,accumulator=advance(state,history,accumulator,int(stop-previous))
        save(int(stop));previous=int(stop)
    (output/'summary.json').write_text(json.dumps(dict(status='raw_process_candidate_only',steps=count,
        full_protocol_completed=count>=plan['full_steps'],weaving_topology_validated=False,matrix_present=False,
        no_kinematic_particle_interpolation=True,scope=plan['scope']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--per-family',type=int,default=2)
    parser.add_argument('--dx',type=float,default=.000075);parser.add_argument('--pitch',type=float,default=.0027)
    parser.add_argument('--diameter',type=float,default=.000225);parser.add_argument('--particle-spacing',type=float,default=.00006)
    parser.add_argument('--upgrade-work-observer',action='store_true')
    parser.add_argument('--grid',choices=['dense','tube'],default='dense')
    parser.add_argument('--saved-frames',type=int,default=8);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run',action='store_true');parser.add_argument('--steps',type=int)
    parser.add_argument('--output',default='outputs/triaxial-weaving-process');args=parser.parse_args()
    parameters=dict(per_family=args.per_family,dx=args.dx,pitch=args.pitch,diameter=args.diameter,particle_spacing=args.particle_spacing)
    if args.run:run(args.output,steps=args.steps,saved_frames=args.saved_frames,resume=args.resume,grid=args.grid,upgrade_work_observer=args.upgrade_work_observer,**parameters)
    else:print(json.dumps(json_value(plan_process(**parameters)[1]),indent=2))
