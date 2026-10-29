"""Bounded prefixes of the full supported-placement/dry-weaving TL protocol.

External fixtures and releasable normal bearings act on actual elastic fibres.
Interior coordinates are never assigned a woven shape. Completion requires the
entire original guided protocol, all bearings released, and original guards.
"""
import argparse,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import numpy as np


class ContactContext(NamedTuple):
    data: object
    graph: object
    history: object
    failed: object
    time: object


def run(output, *, max_steps=64, per_family=8, frame_every=0, resume_from=None, reference_layout='active-stencil',
        requested_dt_s=None,stop_time_s=None,integrator_fork=False):
    setup_started=time.perf_counter()
    import jax
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_tube_layout
    from mpm_jax.fibre_tl_reference import build_tl_reference_layout
    from mpm_jax.fibre_tlmpm import make_solver,TLState
    from mpm_jax.fibre_coupling import build_contact_data,section_centres,closest_segments
    from mpm_jax.fibre_contact import fibre_contact_parameters
    from mpm_jax.fibre_patch_contact import initial_patch_history
    from mpm_jax.fibre_hertz_variational import evaluate_variational_hertz_patches,patch_potential
    from mpm_jax.contact_patches import patch_neighbours,contact_patches
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle,BrittleHistory
    from .triaxial_weaving_guided_geometry import plan_guided
    from .triaxial_weaving_process import json_value
    if max_steps<1 or frame_every<0:raise ValueError('positive step bound and nonnegative frame stride required')
    geometry,plan=plan_guided(per_family=per_family);meta=geometry['metadata']
    x=np.asarray(geometry['x']);volume=np.asarray(geometry['volume']);ids=np.asarray(geometry['fibre_id'])
    mass=jnp.asarray(geometry['mass']);cross=int(meta['cross_section_particles'])
    duration=float(plan['duration_s']);dt=float(plan['dt']) if requested_dt_s is None else float(requested_dt_s)
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite requested dt required')
    prefix_end=duration if stop_time_s is None else float(stop_time_s)
    if not np.isfinite(prefix_end) or not 0<prefix_end<=duration:raise ValueError('prefix end must lie within the full physical protocol')
    if integrator_fork and not resume_from:raise ValueError('an integrator fork requires an explicit parent checkpoint')
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],
        tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx'])
    if reference_layout not in ('tube','active-stencil'):raise ValueError('unknown reference layout')
    builder=build_tl_reference_layout if reference_layout=='active-stencil' else build_tube_layout
    layout=builder(x,ids,dx=plan['dx'],padding_cells=6,max_nodes=40_000_000)
    layout_arrays={k:np.asarray(getattr(layout,k)) for k in ('dense_indices','lower','shape','strides','offsets','origin')}
    reference_node_count=layout.node_count
    guides=jnp.asarray([g['particle_ids'] for g in meta['placement_guides']],jnp.int32)
    guide_weights=mass[guides];guide_mass=jnp.sum(guide_weights,axis=1)
    reference_centres=jnp.sum(guide_weights[:,:,None]*jnp.asarray(x)[guides],axis=1)/guide_mass[:,None]
    releases=jnp.asarray([g['release_time_s'] for g in meta['placement_guides']])
    fids=jnp.asarray([g['fibre_id'] for g in meta['placement_guides']],jnp.int32)
    times=jnp.asarray(meta['fixture_times']);positions=jnp.asarray(meta['fixture_positions']);start_y=positions[0,2*fids,1]
    def guide_target(t,velocity=False):
        tg=jnp.minimum(t,releases);i=jnp.clip(jnp.searchsorted(times,tg,side='right')-1,0,len(times)-2)
        span=times[i+1]-times[i];phase=jnp.clip((tg-times[i])/span,0.,1.)
        a=positions[i,2*fids,1];b=positions[i+1,2*fids,1]
        if velocity:return jnp.where(t<releases,.5*jnp.pi/span*jnp.sin(jnp.pi*phase)*(b-a),0.)
        return a+.5*(1-jnp.cos(jnp.pi*phase))*(b-a)-start_y
    @jax.jit
    def bearing_observation(points,t):
        centres=jnp.sum(guide_weights[:,:,None]*points[guides],axis=1)/guide_mass[:,None]
        engaged=t<releases;offset=centres-reference_centres
        y_error=jnp.max(jnp.where(engaged,abs(offset[:,1]-guide_target(t)),0.),initial=0.)
        xz_drift=jnp.max(jnp.where(engaged,jnp.linalg.norm(offset[:,jnp.array([0,2])],axis=1),0.),initial=0.)
        return jnp.array([y_error,xz_drift,jnp.sum(engaged)])
    law=fibre_contact_parameters(effective_modulus=plan['young_Pa']/(2*(1-plan['poisson']**2)),
        effective_radius=meta['diameter_m']/2,work_of_adhesion=0.,model='hertz')
    patch_capacity=max(64,1<<int(max(1,2*len(geometry['contact_candidates'])-1).bit_length()))
    limits=dict(capacity=16*patch_capacity,max_patches=patch_capacity,max_iterations=128)
    release_by_fibre=jnp.asarray([min(g['release_time_s'] for g in meta['placement_guides'] if g['fibre_id']==f['id']) for f in meta['fibres']])
    def evaluate(points,context):
        contact_state=SimpleNamespace(x=points,mass=mass,history=SimpleNamespace(failed=context.failed),valid=jnp.asarray(True))
        force,updated,metrics=evaluate_variational_hertz_patches(contact_state,context.history,context.data,law,context.graph,**limits)
        centres,_=section_centres(points,mass,context.data);near=context.data.coverage_pairs
        ends=context.data.segments[near];owners=context.data.segment_fibre[near]
        first,second,_,_,distance=closest_segments(centres[ends[:,0,0]],centres[ends[:,0,1]],centres[ends[:,1,0]],centres[ends[:,1,1]])
        gap=distance-jnp.sum(context.data.fibre_radius[owners],axis=1)
        carrier_conflict=jnp.any((gap<=.1*meta['diameter_m'])&jnp.any(context.time<release_by_fibre[owners],axis=1))
        selection=contact_patches(gap,gap<0,context.graph,**dict(capacity=limits['capacity'],max_patches=patch_capacity,max_iterations=128))
        cells=selection.cells;live=cells<len(near)
        patch_points=jnp.concatenate(((first+second)/2,jnp.zeros((1,3))))[cells]
        low=jnp.full((limits['capacity']+1,3),jnp.inf).at[selection.labels].min(jnp.where(live[:,None],patch_points,jnp.inf))
        high=jnp.full((limits['capacity']+1,3),-jnp.inf).at[selection.labels].max(jnp.where(live[:,None],patch_points,-jnp.inf))
        roots=jnp.nonzero(live&(selection.labels==jnp.arange(limits['capacity'])),size=patch_capacity,fill_value=limits['capacity'])[0]
        extent=jnp.linalg.norm(jnp.where((roots<limits['capacity'])[:,None],high[roots]-low[roots],0.),axis=1)
        spacing=jnp.max(jnp.linalg.norm(centres[context.data.segments[:,1]]-centres[context.data.segments[:,0]],axis=1))
        region_valid=jnp.all(extent<=jnp.maximum(2*spacing,4*jnp.max(metrics.contact_radius,initial=0.)))
        metrics=metrics._replace(valid=metrics.valid&selection.valid&region_valid&~carrier_conflict)
        return force,updated,metrics
    def interaction(points,context):
        force,_,metrics=evaluate(points,context)
        return jnp.sum(metrics.potential),force,metrics.valid
    state,step,particles,energy,checks=make_solver(x,volume,layout,material,geometry['masks'],dt,
        dict(fixture_times=meta['fixture_times'],fixture_positions=meta['fixture_positions']),duration=duration,
        guide_particles=guides,guide_displacement=guide_target,guide_velocity=lambda t:guide_target(t,True),
        guide_active=lambda t:t<releases,particle_fibre_ids=ids,particle_interaction=interaction)
    del layout
    base_dt=float(checks['dt']);events=np.unique(np.r_[meta['fixture_times'],np.asarray(releases),duration,prefix_end])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_root=Path(__file__).resolve().parents[1]
    relevant=['mpm_jax/fibre_tlmpm.py','mpm_jax/fibre_tl_guides.py','mpm_jax/fibre_hertz_variational.py',
        'mpm_jax/fibre_patch_contact.py','mpm_jax/fibre_coupling.py','mpm_jax/contact_patches.py','mpm_jax/fibre_contact.py',
        'mpm_jax/fibre_mpm.py','mpm_jax/fibre_tl_reference.py','mpm_jax/materials/isotropic_brittle.py',
        'examples/triaxial_weaving_tl_process.py','examples/triaxial_weaving_guided_geometry.py',
        'examples/triaxial_weaving_batch_plan.py','examples/triaxial_weaving_batch_geometry.py','examples/triaxial_weaving_geometry.py','examples/triaxial_weaving_process.py']
    binding=dict(schema='weaving-tl-variational-hertz-v1',source_sha256={f:hashlib.sha256((source_root/f).read_bytes()).hexdigest() for f in relevant},
        physical_plan_sha256=hashlib.sha256(json.dumps(json_value(plan),sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        input_sha256={name:hashlib.sha256(np.ascontiguousarray(array,dtype=dtype).tobytes()).hexdigest()
            for name,array,dtype in [('x',x,'<f8'),('volume',volume,'<f8'),('mass',np.asarray(mass),'<f8'),
                ('fibre_id',ids,'<i4'),('grip_masks',np.asarray(geometry['masks']),'|b1'),('guide_particles',np.asarray(guides),'<i4')]},
        material=dict(young_Pa=material.young,poisson=material.poisson,density_kg_m3=material.density,
            tensile_strength_Pa=material.tensile_strength,fracture_energy_J_m2=material.fracture_energy,
            characteristic_length_m=plan['dx']),runtime=dict(jax=jax.__version__,numpy=np.__version__,float64=bool(jax.config.jax_enable_x64)),
        full_duration_s=duration,requested_dt_s=dt,base_dt_s=base_dt,per_family=per_family,particles=len(x),
        reference_layout=reference_layout,reference_node_count=reference_node_count,
        reference_dense_node_ids_sha256=hashlib.sha256(layout_arrays['dense_indices'].tobytes()).hexdigest(),
        initial_elastic_critical_dt_s=float(checks['estimated_initial_critical_dt']),
        initial_elastic_largest_eigenvalue=float(checks['initial_largest_acoustic_eigenvalue']),
        acoustic_power_iteration_relative_change=float(checks['acoustic_64_iteration_relative_change']),
        critical_dt_scope='initial elastic reference operator only; not a contact or full-process dt certificate',
        guide_count=len(guides),guide_work='RATTLE impulse dotted with complete Verlet kick midpoint velocity',
        guide_force_convention='sum of RATTLE impulses divided by actual dt; force on material, opposite machine reaction',
        guide_release='complete final controlled interval at exact release event, then remove rows',
        terminal_end_fixtures='held at final machine positions; not a freely detached product',
        contact='nonadhesive local circular equivalent-curvature Hertz potential; full geometric gradient; no friction',
        completion='entire full-duration protocol and all guides released; bounded prefixes are incomplete')
    (output/'binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    delivery_plan={**json_value(plan), 'integrator_requested_dt_s':dt,'integrator_base_dt_s':base_dt,
        'base_steps_without_event_splits':int(checks['planned_steps']),'bounded_prefix_target_s':prefix_end}
    (output/'plan.json').write_text(json.dumps(delivery_plan,indent=2)+'\n')
    frame_every=frame_every or max(1,int(np.ceil(float(checks['planned_steps'])/60)))
    window_reference=None;context=None;rebuilds=0;absolute_work=0.;topology_jump_abs=0.;started=time.perf_counter();last_row={}
    advance_times=[]
    start_step=0;restored_baseline=None;last_reaction=None
    guide_cumulative_impulse=np.zeros(len(guides));guide_peak_segment=np.zeros(len(guides))
    if resume_from:
        parent=Path(resume_from);parent_summary=json.loads((parent/'summary.json').read_text())
        cp_path=parent/'checkpoint.npz'
        if hashlib.sha256(cp_path.read_bytes()).hexdigest()!=parent_summary['checkpoint_sha256']:
            raise RuntimeError('parent checkpoint SHA mismatch')
        with np.load(cp_path,allow_pickle=False) as cp:
            parent_binding=json.loads(str(cp['binding_json'].item()))
            if parent_binding!=binding:
                ignored={'base_dt_s','requested_dt_s'} if integrator_fork else set()
                if {k:v for k,v in parent_binding.items() if k not in ignored}!={k:v for k,v in binding.items() if k not in ignored}:
                    raise RuntimeError('parent source/physical/layout binding mismatch; no silent legacy checkpoint migration')
            history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields))
            state=TLState(*(history if k=='history' else jnp.asarray(cp[k]) for k in TLState._fields))
            if not bool(state.valid):raise RuntimeError('parent checkpoint is invalid')
            start_step=int(cp['step']);absolute_work=float(cp['absolute_machine_work_J'])
            restored_baseline=float(cp['energy_baseline_J']);topology_jump_abs=float(cp['patch_branch_energy_difference_abs_total_J'])
            guide_cumulative_impulse=np.asarray(cp['guide_cumulative_impulse_Ns']).copy()
            restored_sha={}
            for key in state._fields:
                if key=='history':continue
                before=np.asarray(cp[key]);after=np.asarray(getattr(state,key))
                assert before.dtype==after.dtype and before.tobytes()==after.tobytes()
                restored_sha[key]=hashlib.sha256(after.tobytes()).hexdigest()
            for key in state.history._fields:
                before=np.asarray(cp['history_'+key]);after=np.asarray(getattr(state.history,key))
                assert before.dtype==after.dtype and before.tobytes()==after.tobytes()
                restored_sha['history_'+key]=hashlib.sha256(after.tobytes()).hexdigest()
        if prefix_end<=float(state.time):raise ValueError('prefix end must follow the accepted parent time')
        (output/'parent.json').write_text(json.dumps(dict(checkpoint_sha256=parent_summary['checkpoint_sha256'],step=start_step,time_s=float(state.time),
            explicit_integrator_fork=integrator_fork,parent_dt_s=parent_binding['base_dt_s'],new_dt_s=base_dt,
            all_state_history_restored_bitwise=True,restored_sha256=restored_sha),indent=2)+'\n')
    @jax.jit
    def contact_metrics(points,context):return evaluate(points,context)[2]
    guide_impulse_at_start=guide_cumulative_impulse.copy()
    def rebuild(points,history):
        data=build_contact_data(geometry,coverage_positions=np.asarray(points),coverage_skin=4*plan['dx'])
        graph=jnp.asarray(patch_neighbours(data.coverage_pairs,data.segment_fibre),jnp.int32)
        return ContactContext(data,graph,initial_patch_history(data,max_patches=patch_capacity),history.failed,state.time)
    def frame(index):
        points,velocity=particles(state)
        extra={} if last_reaction is None else last_reaction
        np.savez_compressed(output/f'frame_{index:07d}.npz',x=np.asarray(points),v=np.asarray(velocity),F=np.asarray(state.F),
            failed=np.asarray(state.history.failed),damage=np.asarray(state.history.damage),
            reference=x,reference_volume=volume,fibre_id=ids,time=np.asarray(state.time),step=index,
            guide_reaction_available=np.asarray(last_reaction is not None),
            guide_cumulative_impulse_Ns=guide_cumulative_impulse,guide_peak_mean_force_this_segment_N=guide_peak_segment,**extra)
    points=particles(state,positions_only=True);context=rebuild(points,state.history);window_reference=np.asarray(points);rebuilds+=1
    previous=contact_metrics(points,context);refresh_force=True
    baseline=float(sum(energy(state,context))) if restored_baseline is None else restored_baseline
    frame(start_step)
    def checkpoint():
        cp={k:np.asarray(getattr(state,k)) for k in state._fields if k!='history'}
        cp.update({'history_'+k:np.asarray(getattr(state.history,k)) for k in state.history._fields})
        cp.update(step=np.asarray(last_row.get('step',start_step)),binding_json=np.asarray(json.dumps(binding,sort_keys=True)),
            absolute_machine_work_J=absolute_work,energy_baseline_J=baseline,patch_branch_energy_difference_abs_total_J=topology_jump_abs)
        cp.update({'reference_grid_'+k:v for k,v in layout_arrays.items()})
        cp.update(guide_cumulative_impulse_Ns=guide_cumulative_impulse,
            guide_impulse_this_segment_Ns=guide_cumulative_impulse-guide_impulse_at_start,
            guide_peak_mean_force_this_segment_N=guide_peak_segment)
        np.savez_compressed(output/'checkpoint.npz',**cp)
        return hashlib.sha256((output/'checkpoint.npz').read_bytes()).hexdigest()
    def reject(reason):
        summary=dict(completed=False,state='rejected_candidate_previous_state_preserved',reason=reason,
            actual_time_s=float(state.time),steps=last_row.get('step',start_step),checkpoint_sha256=checkpoint())
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        raise RuntimeError(reason)
    @jax.jit
    def branch_energy(points,metrics,data):
        centres,_=section_centres(points,mass,data);pairs=jnp.maximum(metrics.pair_segments,0)
        return jnp.sum(patch_potential(centres,data.segments[pairs],
            data.fibre_radius[data.segment_fibre[pairs]],metrics.active,jnp.reshape(law.effective_modulus,())))
    with (output/'observations.jsonl').open('w') as observations:
        for index in range(start_step+1,start_step+max_steps+1):
            iteration_started=time.perf_counter()
            current=float(state.time)
            if current>=prefix_end-1e-8*base_dt:break
            following=events[events>current+1e-8*base_dt]
            h=min(base_dt,float(following[0]-current))
            points=particles(state,positions_only=True)
            # Conservative direct particle movement bound; the original contact
            # coverage guard is still independently checked at every force call.
            if np.max(np.linalg.norm(np.asarray(points)-window_reference,axis=1))>plan['dx']*.5:
                before=float(sum(energy(state,context)))
                context=rebuild(points,state.history);window_reference=np.asarray(points);rebuilds+=1
                previous=contact_metrics(points,context);refresh_force=True
                after=float(sum(energy(state,context)))
                if abs(after-before)>1e-10*max(abs(before),absolute_work,1e-24):
                    reject('coverage refresh changed potential at identical physical state')
            context=context._replace(failed=state.history.failed,time=state.time+.5*h)
            advance_started=time.perf_counter()
            candidate,row=step(state,context,h,refresh_force);jax.block_until_ready(candidate.displacement)
            advance_times.append(time.perf_counter()-advance_started)
            if not bool(candidate.valid) or float(row['max_damage'])>1e-10:
                reject(f'physical/support/contact/momentum/strain guard rejected candidate step {index}; previous state remains accepted')
            new_points=particles(candidate,positions_only=True);metrics=contact_metrics(new_points,context)
            bearing_y_error,bearing_xz_drift,active_guide_count=np.asarray(bearing_observation(new_points,candidate.time))
            if float(bearing_y_error)>.02*plan['dx'] or float(bearing_xz_drift)>meta['guide_maximum_XZ_drift_m']:
                reject(f'finite bearing geometry guard rejected candidate step {index}')
            jump=float(jnp.sum(metrics.potential))-float(branch_energy(new_points,previous,context.data))
            candidate_absolute_work=absolute_work+float(row['machine_abs_work_J'])
            residual=float(row['residual_J'])+baseline
            if abs(residual)>.10*max(candidate_absolute_work,1e-30):
                reject(f'original 10 percent work guard rejected candidate step {index}')
            topology_jump_abs+=abs(jump);absolute_work=candidate_absolute_work
            state=candidate;previous=metrics;refresh_force=False
            last_reaction=dict(guide_mean_force_on_material_N=np.asarray(row['guide_mean_force_on_material_N']),
                guide_position_impulse_Ns=np.asarray(row['guide_position_multiplier_kg_m'])/h,
                guide_velocity_impulse_Ns=np.asarray(row['guide_velocity_multiplier_Ns']),guide_reaction_dt_s=np.asarray(h),
                guide_active_during_last_step=np.asarray(state.time-.5*h<releases))
            guide_cumulative_impulse+=last_reaction['guide_position_impulse_Ns']+last_reaction['guide_velocity_impulse_Ns']
            guide_peak_segment=np.maximum(guide_peak_segment,abs(last_reaction['guide_mean_force_on_material_N']))
            last_row=dict(step=index,time_s=float(state.time),kinetic_J=float(row['kinetic_J']),elastic_J=float(row['elastic_J']),
                contact_J=float(row['interaction_J']),work_remainder_J=residual,absolute_machine_work_J=absolute_work,
                contact_patch_count=int(metrics.patch_count),coverage_pairs=len(context.data.coverage_pairs),
                patch_branch_energy_difference_J=jump,patch_branch_energy_difference_abs_total_J=topology_jump_abs,
                active_guides=int(active_guide_count),max_log_strain=float(row['max_log_strain']),
                guide_mean_force_max_N=float(row['guide_mean_force_max_N']),actual_step_dt_s=h,
                interaction_nodal_force_l2_N=float(row['interaction_nodal_force_l2_N']),
                interaction_nodal_force_max_N=float(row['interaction_nodal_force_max_N']),
                maximum_active_bearing_y_error_m=float(bearing_y_error),maximum_active_bearing_xz_drift_m=float(bearing_xz_drift),
                mechanics_advance_wall_seconds=advance_times[-1],iteration_wall_seconds=time.perf_counter()-iteration_started,
                momentum_relative_error=float(row['momentum_relative_error']),angular_momentum_relative_error=float(row['angular_momentum_relative_error']))
            observations.write(json.dumps(last_row)+'\n');observations.flush()
            if index%frame_every==0:frame(index)
    completed=float(state.time)>=duration-1e-8*base_dt and not bool(jnp.any(state.time<releases))
    frame(last_row.get('step',start_step))
    summary=dict(completed=completed,state='complete' if completed else 'accepted_bounded_prefix',
        full_protocol_duration_s=duration,actual_time_s=float(state.time),steps=last_row.get('step',start_step),
        requested_prefix_end_s=prefix_end,
        wall_seconds=time.perf_counter()-started,coverage_rebuilds=rebuilds,terminal=last_row,
        setup_wall_seconds=started-setup_started,
        mechanics_first_step_seconds=advance_times[0] if advance_times else None,
        mechanics_median_subsequent_step_seconds=float(np.median(advance_times[1:])) if len(advance_times)>1 else None,
        checkpoint_sha256=checkpoint())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--max-steps',type=int,default=64)
    p.add_argument('--per-family',type=int,default=8);p.add_argument('--frame-every',type=int,default=0);p.add_argument('--resume-from')
    p.add_argument('--reference-layout',choices=['active-stencil','tube'],default='active-stencil')
    p.add_argument('--dt',type=float);p.add_argument('--stop-time',type=float);p.add_argument('--integrator-fork',action='store_true')
    args=p.parse_args();run(args.output,max_steps=args.max_steps,per_family=args.per_family,frame_every=args.frame_every,resume_from=args.resume_from,
        reference_layout=args.reference_layout,requested_dt_s=args.dt,stop_time_s=args.stop_time,integrator_fork=args.integrator_fork)
