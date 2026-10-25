"""Single realCP16360->16560 implicit-inertia diagnostic, no fine replay."""
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='input');p.add_argument('--output',default='output');a=p.parse_args();a.output=Path(a.output);a.output.mkdir(exist_ok=False);root=Path(a.input)
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreState,validate_state,stored_energy
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    from mpm_jax.fibre_hash_layout import build_hash_tube_layout
    from examples.triaxial_weaving_batch_plan import plan_batch
    from examples.triaxial_weaving_execution import allocate
    from examples.triaxial_weaving_geometry import fixture_position
    from examples.triaxial_weaving_implicit_candidate import make_solver
    receipt=json.loads((root/'receipt.json').read_text());arrays=[]
    for step in [16360,16560]:
        file=root/f'checkpoint_{step:07d}.npz';assert hashlib.sha256(file.read_bytes()).hexdigest()==receipt[str(step)]['sha256']
        with np.load(file,allow_pickle=False) as cp:arrays.append({k:cp[k].copy() for k in cp.files})
    parent,reference=arrays;geometry,plan=plan_batch(per_family=8,nominal_cfl=.08)
    assert json.loads(str(parent['binding']))['physical_plan']==plan
    fields={k:jnp.asarray(parent['state_'+k]) for k in FibreState._fields if k!='history'};fields['history']=BrittleHistory(*(jnp.asarray(parent['history_'+k]) for k in BrittleHistory._fields));state=FibreState(**fields)
    exact={k:bool(np.array_equal(parent['state_'+k],np.asarray(getattr(state,k)))) for k in FibreState._fields if k!='history'};exact.update({'history_'+k:bool(np.array_equal(parent['history_'+k],np.asarray(getattr(state.history,k)))) for k in BrittleHistory._fields});assert all(exact.values())
    (a.output/'input-exact.json').write_text(json.dumps(exact,indent=2))
    setup=time.perf_counter();layout,data,graph,global_resource=allocate(state,geometry,plan['dx'],plan['dt'],200,int(parent['packed_capacity']),lookup='direct',direct_capacity=int(parent['direct_map_capacity']));assert global_resource['real_coverage_pairs']==0
    dt=50*plan['dt'];material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx']);meta=geometry['metadata'];span=200*plan['dt'];x=np.asarray(state.x);v=np.asarray(state.v);ids=np.asarray(state.fibre_id);t=float(state.time);times=meta['fixture_times'];positions=np.asarray(meta['fixture_positions'])
    seed=[x,x+span*v];seed_ids=[ids,ids];base=fixture_position(t,times,positions)
    for fraction in [.5,1.]:
        target=fixture_position(t+span*fraction,times,positions)
        for mask,displacement in zip(geometry['masks'],target-base):seed.append(x[mask]+displacement);seed_ids.append(ids[mask])
    seed=np.concatenate(seed);seed_ids=np.concatenate(seed_ids);arguments=[];resources=[]
    for fibre in meta['fibres']:
        i=fibre['id'];first,last=fibre['particle_slice'];sl=slice(first,last);values={k:getattr(state,k)[sl] for k in ['x','v','F','C','mass','volume','tau']}
        values.update(fibre_id=jnp.zeros(last-first,jnp.int32),history=BrittleHistory(*(getattr(state.history,k)[sl] for k in BrittleHistory._fields)),time=state.time,valid=state.valid,boundary_impulse=jnp.zeros(3),external_impulse=jnp.zeros(3),boundary_work=jnp.asarray(0.))
        s=FibreState(**values);selected=seed[seed_ids==i];l,r=build_hash_tube_layout(selected,np.zeros(len(selected),np.int32),dx=plan['dx'],padding_cells=3,capacity_factor=1.25,direct_map_capacity='auto',max_active_nodes=8_000_000);validate_state(s,l);arguments.append((s,l,jnp.asarray(positions[:,2*i:2*i+2])));resources.append(r)
    solver=make_solver(material,dt,plan['dx'],jnp.asarray(times),meta['cross_section_particles']);layout_seconds=time.perf_counter()-setup
    compile_start=time.perf_counter();signatures=set()
    for args in arguments:
        s,l,pos=args;signature=(len(s.x),l.node_count,l.node_map.shape)
        if signature not in signatures:solver.lower(*args).compile();signatures.add(signature)
    compile_seconds=time.perf_counter()-compile_start
    def execute(args):result=solver(*args);jax.block_until_ready(result);return result
    wall=time.perf_counter();cpu=time.process_time()
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(execute,arguments))
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    values={k:jnp.concatenate([getattr(row[0],k) for row in records]) for k in ['x','v','F','C','mass','volume','tau']}
    values.update(fibre_id=state.fibre_id,history=BrittleHistory(*(jnp.concatenate([getattr(row[0].history,k) for row in records]) for k in BrittleHistory._fields)),time=records[0][0].time,valid=jnp.all(jnp.stack([row[0].valid for row in records])),boundary_impulse=state.boundary_impulse+jnp.sum(jnp.stack([row[0].boundary_impulse for row in records]),axis=0),external_impulse=state.external_impulse,boundary_work=state.boundary_work+jnp.sum(jnp.stack([row[0].boundary_work for row in records])))
    final=FibreState(**values);metrics=np.stack([np.asarray(row[2]) for row in records]);work=float(metrics[:,:,5].sum());absolute=float(metrics[:,:,6].sum());path=max(float(row[1].max()) for row in records)
    payload={'state_'+k:np.asarray(getattr(final,k)) for k in FibreState._fields if k!='history'};payload.update({'history_'+k:np.asarray(getattr(final.history,k)) for k in BrittleHistory._fields});payload.update({k:parent[k] for k in ['law_active','law_valid','contact_radius','contact_segments','contact_valid']});np.savez_compressed(a.output/'candidate-state-not-original-checkpoint.npz',**payload,solver_metrics=metrics,input_accumulator=parent['accumulator'],implicit_machine_work=work,implicit_absolute_machine_work=absolute,parent_checkpoint_sha256=receipt['16360']['sha256'],candidate_dt=dt)
    comparison={}
    for k,offset in [('state_v',0.),('state_F',np.eye(3)),('state_tau',0.),('history_kappa',0.)]:
        candidate=payload[k];fine=reference[k];comparison[k]=float(np.linalg.norm(candidate-fine)/max(np.linalg.norm(fine-offset),1e-300))
    comparison['max_position_difference_over_h']=float(np.max(np.linalg.norm(payload['state_x']-reference['state_x'],axis=1))/plan['dx'])
    def kinetic(cp):return .5*float(np.sum(cp['state_mass'][:,None]*cp['state_v']**2))+.5*plan['dx']**2/4*float(np.sum(cp['state_mass'][:,None,None]*cp['state_C']**2))
    fine_state=state._replace(F=jnp.asarray(reference['state_F']),history=BrittleHistory(*(jnp.asarray(reference['history_'+k]) for k in BrittleHistory._fields)))
    K=kinetic(payload);Kfine=kinetic(reference);U=float(stored_energy(final,material));Ufine=float(stored_energy(fine_state,material));fine_work=float(reference['accumulator'][2]-parent['accumulator'][2]);fine_abs=float(reference['accumulator'][4]-parent['accumulator'][4])
    comparison.update(K=abs(K-Kfine)/max(Kfine,1e-300),U=abs(U-Ufine)/max(Ufine,1e-300),absolute_fixture_work=abs(absolute-fine_abs)/max(fine_abs,1e-300))
    target=fixture_position(float(final.time),times,positions);grip=max(float(np.linalg.norm(payload['state_x'][mask]-geometry['x'][mask]-(end-initial),axis=1).max()) for mask,end,initial in zip(geometry['masks'],target,np.asarray(positions)[0]))
    gates=dict(nonlinear_and_original_physical_guards=bool(final.valid),all_finite=all(np.isfinite(v).all() for v in payload.values()),same_time=abs(float(final.time)-float(reference['state_time']))<1e-14,mass_volume_IDs_exact=all(np.array_equal(payload[k],reference[k]) for k in ['state_mass','state_volume','state_fibre_id']),history_irreversible=bool(np.all(payload['history_kappa']>=parent['history_kappa']) and np.all(payload['history_damage']>=parent['history_damage'])),grips=grip<=1e-10,contact_zero_certificate=path<.00015,fields=all(comparison[k]<=.05 for k in ['state_v','state_F','state_tau','history_kappa','K','U','absolute_fixture_work']),position=comparison['max_position_difference_over_h']<=.01)
    original_row=receipt['16560']['observation'];reference_wall=original_row['advance_seconds']+original_row['rebuild_seconds'];net_candidate_wall=layout_seconds+wall;gates['net_warm_window_speedup']=net_candidate_wall<reference_wall
    report=dict(scope='Independent implicit-inertia4steps vs saved original200steps; no original trajectory replay, contact-free only, no whole-process accuracy claim',gates=gates,admitted=all(gates.values()),comparison=comparison,implicit_dt=dt,original_dt=plan['dt'],physical_interval=span,implicit_steps=4,original_steps=200,workers=4,reference_production_workers=8,layout_seconds=layout_seconds,compile_seconds=compile_seconds,actual_dispatch_wall_seconds=wall,actual_dispatch_cpu_seconds=cpu,cpu_wall_ratio=cpu/wall,reference_measured_advance_plus_rebuild_seconds=reference_wall,net_warm_window_seconds=net_candidate_wall,warm_speedup=reference_wall/net_candidate_wall,cold_total_seconds=layout_seconds+compile_seconds+wall,max_nonlinear_relative_residual=float(metrics[:,:,0].max()),maximum_Newton_iterations=float(metrics[:,:,1].max()),maximum_CG_total=float(metrics[:,:,2].max()),negative_curvature_count=float(metrics[:,:,3].sum()),absolute_path_bound_m=path,maximum_grip_error_m=grip,maximum_log_stretch=float(metrics[:,:,7].max()),K=K,Kfine=Kfine,U=U,Ufine=Ufine,machine_work=work,reference_machine_work=fine_work,absolute_machine_work=absolute,reference_absolute_machine_work=fine_abs,global_resource=global_resource,local_resources=resources,source_note='Original constitutive law and APIC gradient; new implicit time integrator, exact stable polar directional derivative and Newton-CG. Not explicit-CFL admission.')
    (a.output/'implicit-matched-review.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['global_resource','local_resources']},indent=2),flush=True)
    if not report['admitted']:raise RuntimeError('Independent candidate not admitted; complete raw and reasons retained')


if __name__=='__main__':main()
