"""Bounded genuine3D continuation; explicit owner-registry migration only."""
import json,hashlib,time,argparse
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='input');parser.add_argument('--output',default='output');parser.add_argument('--steps',type=int,default=50);parser.add_argument('--skip-rejected-diagnostic',action='store_true');args=parser.parse_args();root=Path(args.input);out=Path(args.output);out.mkdir(exist_ok=False)
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.triaxial_failure import loading_motion,descriptor,json_value,restore_contact_model,restore_contact_history,_graph_observation
    from examples.triaxial_fibre_geometry import make_geometry
    from mpm_jax.fibre_mpm import FibreState,FibreConfig,build_tube_layout,reference_grips,step_with_diagnostics,validate_state
    from mpm_jax.fibre_coupling import build_contact_data,FibreCouplingHistory
    from mpm_jax.fibre_contact import fibre_contact_parameters
    from mpm_jax.fibre_patch_contact import initial_patch_history,evaluate_hertz_patches
    from mpm_jax.contact_patches import patch_neighbours
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    def save(name,value):(out/name).write_text(json.dumps(json_value(value),indent=2))
    binding=json.loads((root/'binding.json').read_text())
    for name,digest in binding['inputs'].items():
        path=(Path('/diagnostic')/name[11:]) if name.startswith('diagnostic/') else ((Path('/parent')/name[7:]) if name.startswith('parent/') else root/name)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    plan=json.loads((root/'plan.json').read_text());kwargs=json.loads((root/'context.json').read_text())['kwargs'];names=('dimensions','diameter','pitch','crimp','clearance','particle_spacing','packing','density','grip_length','max_particles');geometry=make_geometry(theta_deg=-kwargs['theta'],**{k:kwargs[k] for k in names});config=FibreConfig(plan['dt']);material=isotropic_brittle(**plan['material']);regions=plan['contact_regions'];assert regions['mode']=='connected' and plan['law']=='hertz'
    old=build_contact_data(geometry,search_window=plan['search_window'],coverage_skin=plan['coverage_skin_m']);data=build_contact_data(geometry,search_window=plan['search_window'],coverage_skin=plan['coverage_skin_m'],register_cross_family_coverage=True)
    assert descriptor(np.asarray(data.coverage_pairs))==binding['coverage_descriptor'];assert np.array_equal(np.asarray(data.coverage_pairs),np.asarray(old.coverage_pairs));assert np.array_equal(np.asarray(data.seed_fibres)[:old.pair_count],np.asarray(old.seed_fibres));assert np.array_equal(np.asarray(data.seed_segments)[:old.pair_count],np.asarray(old.seed_segments))
    neighbours=patch_neighbours(np.asarray(data.coverage_pairs),np.asarray(data.segment_fibre));assert descriptor(neighbours)==binding['neighbour_descriptor']
    law=fibre_contact_parameters(effective_modulus=material.young/(2*(1-material.poisson**2)),effective_radius=geometry['metadata']['diameter_m']/2,work_of_adhesion=plan['work_of_adhesion_J_per_m2'],model='hertz');contact_law=law
    def restore(path):
        with np.load(path,allow_pickle=False) as cp:p={k:cp[k].copy() for k in cp.files}
        state=FibreState(**{k:jnp.asarray(p['state_'+k]) for k in FibreState._fields if k!='history'},history=BrittleHistory(*(jnp.asarray(p['history_'+k]) for k in BrittleHistory._fields)))
        law=restore_contact_model(contact_law,p);template=initial_patch_history(data,max_patches=regions['max_patches']);contact=FibreCouplingHistory(restore_contact_history(template.law,p,jnp,parameters=law),jnp.asarray(p['contact_radius']),jnp.asarray(p['contact_segments']),jnp.asarray(p['contact_valid']))
        checks={k:np.array_equal(np.asarray(v),p['state_'+k]) for k,v in state._asdict().items() if k!='history'};checks.update({'history_'+k:np.array_equal(np.asarray(v),p['history_'+k]) for k,v in state.history._asdict().items()});checks.update(contact_active=np.array_equal(np.asarray(contact.law.active),p['contact_active']),contact_law_valid=np.array_equal(np.asarray(contact.law.valid),p['contact_law_valid']),contact_radius=np.array_equal(np.asarray(contact.radius),p['contact_radius']),contact_segments=np.array_equal(np.asarray(contact.segments),p['contact_segments']),contact_valid=np.array_equal(np.asarray(contact.valid),p['contact_valid']));assert all(checks.values());return state,contact,p,checks
    def evaluate_contact(state,history,data,law):return evaluate_hertz_patches(state,history,data,law,neighbours,capacity=regions['capacity'],max_patches=regions['max_patches'],max_iterations=regions['max_iterations'])
    if not args.skip_rejected_diagnostic:
        rejected,rejected_contact,rejected_payload,_=restore('/diagnostic/checkpoint.npz');assert not bool(rejected_payload['aggregate'][4])
        evaluation=jax.jit(lambda s,c:(evaluate_contact(s,c,old,law),evaluate_contact(s,c,data,law)))
        started=time.perf_counter();before,after=evaluation(rejected,rejected_contact);jax.block_until_ready((before,after));diagnostic_seconds=time.perf_counter()-started
        save('rejected-state-contact-repair.json',dict(old_valid=bool(before[2].valid),new_valid=bool(after[2].valid),old_uncovered=int(before[2].uncovered_near_pairs),new_uncovered=int(after[2].uncovered_near_pairs),force_bitexact=np.array_equal(np.asarray(before[0]),np.asarray(after[0])),force_max_difference=float(np.max(abs(np.asarray(before[0])-np.asarray(after[0])))),old_pair_count=old.pair_count,new_pair_count=data.pair_count,evaluation_seconds=diagnostic_seconds,all_new_metrics={k:np.asarray(v) for k,v in after[2]._asdict().items()}))
        assert bool(after[2].valid) and np.array_equal(np.asarray(before[0]),np.asarray(after[0]))
        del rejected,rejected_contact,rejected_payload,before,after,evaluation
    else:
        receipt=json.loads((root/'previous-contact-repair.json').read_text());assert receipt['new_valid'] and receipt['new_uncovered']==0 and receipt['force_bitexact']
    state,contact,payload,checks=restore(Path('/parent')/binding['parent_checkpoint_file'] if 'parent_checkpoint_file' in binding else root/'checkpoint.npz');assert bool(payload['aggregate'][4]) and bool(state.valid) and bool(contact.valid);assert bool(np.all(payload['history_valid']));assert np.array_equal(payload['state_mass'],geometry['mass']);assert np.array_equal(payload['state_fibre_id'],geometry['fibre_id'])
    start_step=int(payload['aggregate'][0]);assert start_step==binding['accepted_steps'];save('carry-migration.json',dict(parent_checkpoint_sha256=binding['inputs'].get('checkpoint.npz',binding['inputs'].get('parent/'+binding.get('parent_checkpoint_file',''))),old_source=plan['source'],new_source=binding['source'],checks=checks,start_step=start_step,registry_migration_only=True,rejected_carry_never_resumed=True))
    bounds=np.asarray([f['reference_tube_bounds_m'] for f in geometry['metadata']['fibres']]);started=time.perf_counter();layout=build_tube_layout(geometry['x'],geometry['fibre_id'],dx=plan['grid']['dx_m'],padding_cells=plan['grid']['padding_cells'],reference_bounds=bounds,max_nodes=plan['grid']['max_nodes']);validate_state(state,layout)
    for key,field in [('node_map','node_map'),('dense_indices','dense_indices')]:assert descriptor(np.asarray(getattr(layout,field)))==binding['layout'][key]
    save('layout.json',dict(node_map=descriptor(np.asarray(layout.node_map)),dense_indices=descriptor(np.asarray(layout.dense_indices)),seconds=time.perf_counter()-started))
    intervals=plan['grip_x_intervals_m'];masks=(geometry['x'][:,0]<=intervals[0][1],geometry['x'][:,0]>=intervals[1][0]);assert [int(mask.sum()) for mask in masks]==plan['grip_particle_counts'];motion=lambda t:loading_motion(t,plan['total_fixture_extension_m'],plan['load_time_s'],jnp);boundary=reference_grips(masks,(lambda t:motion(t)[1][0],lambda t:motion(t)[1][1]))

    def contact_statistics(metrics):
        minimum_gap = metrics.minimum_coverage_gap if hasattr(metrics, 'minimum_coverage_gap') else jnp.min(metrics.gap, initial=jnp.inf)
        minimum_force = jnp.min(metrics.normal_force, initial=0.0)
        maximum_force = jnp.max(metrics.normal_force, initial=0.0)
        return jnp.array([jnp.sum(metrics.search_boundary), metrics.uncovered_near_pairs, metrics.same_family_near_pairs, metrics.coverage_motion_exceeded, metrics.maximum_nonrigid_displacement, minimum_gap, jnp.sum(metrics.active), minimum_force, maximum_force, jnp.sum(metrics.failed_section_contact), jnp.sum(metrics.unsupported_end_contact), jnp.sum(metrics.degenerate_geometry), jnp.linalg.norm(metrics.force_residual), jnp.linalg.norm(metrics.torque_residual), jnp.sum(metrics.potential), metrics.valid], dtype=jnp.float64)
    @jax.jit
    def advance_block(state, contact, aggregate, impulses, work, ke, last, steps):
    
        def body(_, carry):
            current, attachment, totals, group_impulse, actual_work, kinetic_correction, last_metrics = carry
    
            def attempt(_):
                force, next_contact, metrics = evaluate_contact(current, attachment, data, contact_law)
                record = contact_statistics(metrics)
    
                def mechanical_step(_):
                    candidate, diag = step_with_diagnostics(current, material, layout, config, force, boundary=boundary)
                    speed = motion(candidate.time)[1]
                    tension = 0.5 * (diag.group_impulses[1, 0] - diag.group_impulses[0, 0]) / config.dt
                    first = jnp.where((totals[1] < 0) & jnp.any(candidate.history.failed), candidate.time, totals[1])
                    peak = jnp.maximum(totals[2], tension)
                    peak_time = jnp.where(tension > totals[2], candidate.time, totals[3])
                    accepted = totals.at[:4].set(jnp.array([totals[0] + 1, first, peak, peak_time]))
                    return jax.lax.cond(diag.valid, lambda _: (candidate, next_contact, accepted, group_impulse + diag.group_impulses, actual_work + jnp.sum(diag.group_impulses * speed, axis=1), kinetic_correction + diag.group_work, record), lambda _: (current, attachment, totals.at[4:].set(jnp.array([0.0, 2.0])), group_impulse, actual_work, kinetic_correction, record), None)
                return jax.lax.cond(metrics.valid, mechanical_step, lambda _: (current, attachment, totals.at[4:].set(jnp.array([0.0, 1.0])), group_impulse, actual_work, kinetic_correction, record), None)
            return jax.lax.cond(totals[4] > 0, attempt, lambda _: carry, None)
        return jax.lax.fori_loop(0, steps, body, (state, contact, aggregate, impulses, work, ke, last))

    carry=(state,contact,jnp.asarray(payload['aggregate']),jnp.asarray(payload['total_impulse']),jnp.asarray(payload['clamp_work']),jnp.asarray(payload['grid_ke_work']),jnp.asarray(payload['last_contact']))
    def checkpoint(carry):
        state,contact,aggregate,impulse,work,ke,last=carry;step=int(aggregate[0]);values={k:v for k,v in payload.items() if not k.startswith(('state_','history_','contact_'))}
        values.update({'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'});values.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()});values.update(contact_active=np.asarray(contact.law.active),contact_law_valid=np.asarray(contact.law.valid),contact_radius=np.asarray(contact.radius),contact_segments=np.asarray(contact.segments),contact_valid=np.asarray(contact.valid),aggregate=np.asarray(aggregate),total_impulse=np.asarray(impulse),clamp_work=np.asarray(work),grid_ke_work=np.asarray(ke),last_contact=np.asarray(last),checkpoint_kind=np.array('bounded-complete-step-v1'),source_binding=np.array(json.dumps(binding,sort_keys=True)),signature=np.array(hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()),parent_signature=payload['signature'])
        temp=out/f'.checkpoint_{step:07d}.npz';np.savez_compressed(temp,**values);temp.replace(out/f'checkpoint_{step:07d}.npz')
        np.savez_compressed(out/f'frame_{step:07d}.npz',x=np.asarray(state.x),damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed),speed=np.linalg.norm(np.asarray(state.v),axis=1),time=np.asarray(state.time))
        save('latest.json',dict(step=step,time=float(state.time),checkpoint_sha256=hashlib.sha256((out/f'checkpoint_{step:07d}.npz').read_bytes()).hexdigest(),accepted=bool(aggregate[4]),contact_metrics=np.asarray(last),failed_particles=int(np.asarray(state.history.failed).sum()),max_damage=float(np.asarray(state.history.damage).max())))
    started=time.perf_counter();advance_block.lower(*carry,jnp.asarray(1)).compile();save('compile.json',dict(seconds=time.perf_counter()-started));wall=time.perf_counter()
    for _ in range((args.steps+24)//25):
        count=min(25,start_step+args.steps-int(carry[2][0]));carry=advance_block(*carry,jnp.asarray(count));jax.block_until_ready(carry);checkpoint(carry)
        print(json.dumps(dict(step=int(carry[2][0]),time=float(carry[0].time),valid=bool(carry[2][4]),wall=time.perf_counter()-wall)),flush=True)
        if not bool(carry[2][4]):raise RuntimeError('Original coupled physical/contact guard rejected, raw retained')
    save('runtime.json',dict(steps=int(carry[2][0])-start_step,seconds=time.perf_counter()-wall,complete=True))
    final_state,final_contact=carry[:2];connected,cuts=_graph_observation(np.asarray(final_state.x),np.asarray(final_state.history.failed),geometry,data,final_contact,plan['grid']['dx_m'],masks)
    failed_per_section=np.bincount(np.asarray(data.particle_section),weights=np.asarray(final_state.history.failed),minlength=data.section_count)
    save('failure-observation.json',dict(step=int(carry[2][0]),time_s=float(final_state.time),sampled_section_graph_connects_grips=connected,active_fibre_x_support_cuts=cuts,fully_failed_sections=int(np.sum(failed_per_section==geometry['metadata']['cross_section_particles'])),failed_particles=int(np.asarray(final_state.history.failed).sum()),maximum_damage=float(np.asarray(final_state.history.damage).max()),scope='Original saved-state sampled section graph and active fibre x-grid support, not a full3D continuous-surface fracture certificate'))


if __name__=='__main__':main()
