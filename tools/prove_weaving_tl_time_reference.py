"""Single realCP16360->16560 reanchored-TL-implicit-inertia diagnostic, no fine replay."""
import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from examples.triaxial_weaving_process import json_value
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
    from examples.triaxial_weaving_tl_implicit import make_solver as coarse_factory
    from examples.triaxial_weaving_tl_fine_reference import make_solver
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
    dt=plan['dt'];material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx']);meta=geometry['metadata'];span=200*plan['dt'];x=np.asarray(state.x);v=np.asarray(state.v);ids=np.asarray(state.fibre_id);t=float(state.time);times=meta['fixture_times'];positions=np.asarray(meta['fixture_positions'])
    seed=[x,x+span*v];seed_ids=[ids,ids];base=fixture_position(t,times,positions)
    for fraction in [.5,1.]:
        target=fixture_position(t+span*fraction,times,positions)
        for mask,displacement in zip(geometry['masks'],target-base):seed.append(x[mask]+displacement);seed_ids.append(ids[mask])
    seed=np.concatenate(seed);seed_ids=np.concatenate(seed_ids);arguments=[];resources=[]
    for fibre in meta['fibres']:
        i=fibre['id'];first,last=fibre['particle_slice'];sl=slice(first,last);values={k:getattr(state,k)[sl] for k in ['x','v','F','C','mass','volume','tau']}
        values.update(fibre_id=jnp.zeros(last-first,jnp.int32),history=BrittleHistory(*(getattr(state.history,k)[sl] for k in BrittleHistory._fields)),time=state.time,valid=state.valid,boundary_impulse=jnp.zeros(3),external_impulse=jnp.zeros(3),boundary_work=jnp.asarray(0.))
        s=FibreState(**values);selected=seed[seed_ids==i];l,r=build_hash_tube_layout(selected,np.zeros(len(selected),np.int32),dx=plan['dx'],padding_cells=3,capacity_factor=1.25,direct_map_capacity='auto',max_active_nodes=8_000_000);validate_state(s,l);arguments.append((s,l,jnp.asarray(positions[:,2*i:2*i+2])));resources.append(r)
    initializer=coarse_factory(material,50*dt,plan['dx'],jnp.asarray(times),meta['cross_section_particles'],steps=0)
    exact_reference=[];initial_arguments=[]
    for i,args in enumerate(arguments):
        initialization=initializer(*args);jax.block_until_ready(initialization)
        with np.load(Path('/coarse')/f'tl-reference-basis-{i:02d}.npz',allow_pickle=False) as saved:context={key:jnp.asarray(saved[key]) for key in saved.files}
        checks={key:bool(np.array_equal(np.asarray(context[key]),np.asarray(initialization[6][key]))) for key in ['mass','node_reference','indices','weights','gradients','reference','F0','volume','particle_mass','clamped','owners','anchor_time']};assert all(checks.values()),(i,checks)
        exact_reference.append(checks);initial_arguments.append((*args,initialization[3],context))
    arguments=initial_arguments
    (a.output/'same-reference-initialization.json').write_text(json.dumps(exact_reference,indent=2))
    solver=make_solver(material,dt,plan['dx'],jnp.asarray(times),meta['cross_section_particles'],steps=200,initial_bc_dt=plan['dt']);layout_seconds=time.perf_counter()-setup
    compile_start=time.perf_counter();signatures=set()
    for args in arguments:
        s,l,pos,initial_nodes,context=args;signature=(len(s.x),l.node_count,l.node_map.shape)
        if signature not in signatures:solver.lower(*args).compile();signatures.add(signature)
    compile_seconds=time.perf_counter()-compile_start
    def execute(args):result=solver(*args);jax.block_until_ready(result);return result
    wall=time.perf_counter();cpu=time.process_time()
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(execute,arguments))
    wall=time.perf_counter()-wall;cpu=time.process_time()-cpu
    (a.output/'timing.json').write_text(json.dumps(dict(layout_seconds=float(layout_seconds),compile_seconds=float(compile_seconds),dispatch_wall_seconds=float(wall),dispatch_cpu_seconds=float(cpu)),indent=2))
    values={k:jnp.concatenate([getattr(row[0],k) for row in records]) for k in ['x','v','F','C','mass','volume','tau']}
    values.update(fibre_id=state.fibre_id,history=BrittleHistory(*(jnp.concatenate([getattr(row[0].history,k) for row in records]) for k in BrittleHistory._fields)),time=records[0][0].time,valid=jnp.all(jnp.stack([row[0].valid for row in records])),boundary_impulse=state.boundary_impulse+jnp.sum(jnp.stack([row[0].boundary_impulse for row in records]),axis=0),external_impulse=state.external_impulse,boundary_work=state.boundary_work+jnp.sum(jnp.stack([row[0].boundary_work for row in records])))
    final=FibreState(**values);metrics=np.stack([np.asarray(row[2]) for row in records]);projection=np.stack([np.asarray(row[4]) for row in records]);extra=np.stack([np.asarray(row[5]) for row in records]);work=float(metrics[:,:,5].sum()+extra[:,2].sum());absolute=float(metrics[:,:,6].sum()+extra[:,3].sum());path=max(float(row[1].max()) for row in records)
    for i,row in enumerate(records):
        nodal={name:np.asarray(value) for name,value in row[3]._asdict().items() if name!='history'};nodal.update({'history_'+name:np.asarray(value) for name,value in row[3].history._asdict().items()});np.savez_compressed(a.output/f'tl-node-carry-{i:02d}.npz',**nodal)
        np.savez_compressed(a.output/f'tl-reference-basis-{i:02d}.npz',**{name:np.asarray(value) for name,value in row[6].items()})
    payload={'state_'+k:np.asarray(getattr(final,k)) for k in FibreState._fields if k!='history'};payload.update({'history_'+k:np.asarray(getattr(final.history,k)) for k in BrittleHistory._fields});payload.update({k:parent[k] for k in ['law_active','law_valid','contact_radius','contact_segments','contact_valid']});np.savez_compressed(a.output/'candidate-state-not-original-checkpoint.npz',**payload,solver_metrics=metrics,input_accumulator=parent['accumulator'],implicit_machine_work=work,implicit_absolute_machine_work=absolute,parent_checkpoint_sha256=receipt['16360']['sha256'],candidate_dt=dt)
    comparison={}
    for k,offset in [('state_v',0.),('state_F',np.eye(3)),('state_tau',0.),('history_kappa',0.)]:
        candidate=payload[k];fine=reference[k];comparison[k]=float(np.linalg.norm(candidate-fine)/max(np.linalg.norm(fine-offset),1e-300))
    comparison['max_position_difference_over_h']=float(np.max(np.linalg.norm(payload['state_x']-reference['state_x'],axis=1))/plan['dx'])
    def kinetic(cp):return .5*float(np.sum(cp['state_mass'][:,None]*cp['state_v']**2))+.5*plan['dx']**2/4*float(np.sum(cp['state_mass'][:,None,None]*cp['state_C']**2))
    fine_state=state._replace(F=jnp.asarray(reference['state_F']),history=BrittleHistory(*(jnp.asarray(reference['history_'+k]) for k in BrittleHistory._fields)))
    particle_APIC_surrogate_K=kinetic(payload);K=float(extra[:,1].sum());Kfine=kinetic(reference);U=float(stored_energy(final,material));Ufine=float(stored_energy(fine_state,material));fine_work=float(reference['accumulator'][2]-parent['accumulator'][2]);fine_abs=float(reference['accumulator'][4]-parent['accumulator'][4])
    comparison.update(K=abs(K-Kfine)/max(Kfine,1e-300),U=abs(U-Ufine)/max(Ufine,1e-300),absolute_fixture_work=abs(absolute-fine_abs)/max(fine_abs,1e-300))
    target=fixture_position(float(final.time),times,positions);grip=max(float(np.linalg.norm(payload['state_x'][mask]-geometry['x'][mask]-(end-initial),axis=1).max()) for mask,end,initial in zip(geometry['masks'],target,np.asarray(positions)[0]))
    total_mass=float(np.sum(parent['state_mass']));wave_speed=plan['elastic_wave_speed_m_s'];momentum_error=np.sum(payload['state_mass'][:,None]*(payload['state_v']-parent['state_v']),axis=0)-(payload['state_boundary_impulse']-parent['state_boundary_impulse'])
    def angular(cp):
        C=cp['state_C'];affine=plan['dx']**2/4*np.stack((C[:,2,1]-C[:,1,2],C[:,0,2]-C[:,2,0],C[:,1,0]-C[:,0,1]),axis=1)
        return np.sum(cp['state_mass'][:,None]*(np.cross(cp['state_x'],cp['state_v'])+affine),axis=0)
    angular_change=extra[:,7:10].sum(axis=0)-angular(parent);mid_torque=extra[:,13:16].sum(axis=0);old_torque=metrics[:,:,15:18].sum(axis=(0,1));energy_scale=plan['young_Pa']*float(np.sum(parent['state_volume']))
    velocity_projection_relative=float(np.sqrt(projection[:,0].sum()/max(projection[:,1].sum(),1e-300)));affine_projection_relative=float(np.sqrt(projection[:,2].sum()/max(projection[:,3].sum(),1e-300)));initial_K_error=float(abs(projection[:,6].sum()-projection[:,4].sum())/max(projection[:,4].sum(),1e-300))
    gates=dict(initial_velocity_projection=velocity_projection_relative<=.05,initial_energy_projection=initial_K_error<=.05,reference_mass=float(abs(projection[:,10].sum()-projection[:,11].sum()))/total_mass<=1e-12,affine_reproduction=float(projection[:,14].max())<=1e-12,momentum=np.linalg.norm(momentum_error)/(total_mass*wave_speed)<=1e-8,kick_work=np.max(abs(metrics[:,:,18]))/energy_scale<=1e-10,nonlinear_and_original_physical_guards=bool(final.valid),all_finite=all(np.isfinite(v).all() for v in payload.values()),same_time=abs(float(final.time)-float(reference['state_time']))<1e-14,mass_volume_IDs_exact=all(np.array_equal(payload[k],reference[k]) for k in ['state_mass','state_volume','state_fibre_id']),history_irreversible=bool(np.all(payload['history_kappa']>=parent['history_kappa']) and np.all(payload['history_damage']>=parent['history_damage'])),grips=grip<=1e-10,contact_zero_certificate=path<.00015,fields=all(comparison[k]<=.05 for k in ['state_v','state_F','state_tau','history_kappa','K','U','absolute_fixture_work']),position=comparison['max_position_difference_over_h']<=.01)
    original_row=receipt['16560']['observation'];reference_wall=original_row['advance_seconds']+original_row['rebuild_seconds'];net_candidate_wall=layout_seconds+wall;gates['net_warm_window_speedup']=net_candidate_wall<reference_wall
    report=dict(initial_velocity_projection_relative=velocity_projection_relative,initial_affine_projection_relative=affine_projection_relative,initial_nodal_K_relative_to_input=initial_K_error,input_APIC_K=float(projection[:,4].sum()),initial_nodal_fit_K=float(projection[:,5].sum()),initial_nodal_constrained_K=float(projection[:,6].sum()),initial_boundary_work=float(projection[:,7].sum()),initial_boundary_projection_loss=float(projection[:,9].sum()),initial_total_fit_loss=float(projection[:,4].sum()-projection[:,5].sum()),initial_BC_velocity_projection_loss=float(projection[:,9].sum()),particle_APIC_surrogate_K=particle_APIC_surrogate_K,authoritative_TL_kinetic_energy='Lumped reference-node kinetic, not reconstructed particleAPIC surrogate',reference_mass_error=float(abs(projection[:,10].sum()-projection[:,11].sum())),reference_basis_partition_error=float(projection[:,12].max()),reference_basis_constant_gradient_error_times_h=float(projection[:,13].max()),reference_basis_affine_reproduction_error=float(projection[:,14].max()),reference_anchor='TrueCP.x,u0=0,Ftotal=(I+Gref*u)F0, originalVp0/mass/history, noJ0volume rescale',gradient_kind='Fixed old MLS4*w*d/h^2, not trueB-spline derivative; exact energy adjoint and affine reproduction',reused_TL_framework='fibre_tlmpm.py@1e4d41c TLState/reference mass/energy/reaction design; F1explicitVerlet is notF7implicit validation',momentum_closure_absolute=float(np.linalg.norm(momentum_error)),momentum_closure_over_Mc=float(np.linalg.norm(momentum_error)/(total_mass*wave_speed)),angular_change=angular_change.tolist(),boundary_midpoint_angular_impulse=mid_torque.tolist(),boundary_old_grid_angular_impulse=old_torque.tolist(),angular_midpoint_closure_absolute=float(np.linalg.norm(angular_change-mid_torque)),max_nodal_kick_work_closure_absolute=float(np.max(abs(metrics[:,:,18]))),material_time_quadrature_defect=float(metrics[:,:,19].sum()),APIC_transfer_defect=float(metrics[:,:,20].sum()),scope='NewTL200step temporal reference, same frozen coarse basis and fitted initialstate; oldAPIC comparison retained separately',gates=gates,admitted=all(gates.values()),comparison=comparison,implicit_dt=dt,original_dt=plan['dt'],physical_interval=span,implicit_steps=200,original_steps=200,workers=4,reference_production_workers=8,layout_seconds=layout_seconds,compile_seconds=compile_seconds,actual_dispatch_wall_seconds=wall,actual_dispatch_cpu_seconds=cpu,cpu_wall_ratio=cpu/wall,reference_measured_advance_plus_rebuild_seconds=reference_wall,net_warm_window_seconds=net_candidate_wall,warm_speedup=reference_wall/net_candidate_wall,cold_total_seconds=layout_seconds+compile_seconds+wall,max_nonlinear_relative_residual=float(metrics[:,:,0].max()),maximum_absolute_momentum_residual=float(metrics[:,:,9].max()),max_residual_over_Mc=float(np.max(metrics[:,:,9]/metrics[:,:,11])),parked_yarn_absolute_residual_max=float(metrics[16:,:,9].max()),parked_yarn_Newton_iterations_max=float(metrics[16:,:,1].max()),moving_yarn_relative_residual_max=float(metrics[:16,:,0].max()),nonlinear_scale_floor='1e-6*M*c, same1e-7 relative tolerance; parked absolute target1e-13*M*c, moving normalization unchanged',maximum_Newton_iterations=float(metrics[:,:,1].max()),maximum_CG_total=float(metrics[:,:,2].max()),negative_curvature_count=float(metrics[:,:,3].sum()),absolute_path_bound_m=path,maximum_grip_error_m=grip,maximum_log_stretch=float(metrics[:,:,7].max()),K=K,Kfine=Kfine,U=U,Ufine=Ufine,machine_work=work,reference_machine_work=fine_work,absolute_machine_work=absolute,reference_absolute_machine_work=fine_abs,global_resource=global_resource,local_resources=resources,source_note='Original constitutive law and APIC gradient; Independent persistent reference-node implicit midpoint, retaining realCPprestrainF0 and originalVp0, exact stable polar directional derivative and Newton-CG. Not explicit-CFL admission.')
    (a.output/'implicit-matched-review.json').write_text(json.dumps(json_value(report),indent=2));print(json.dumps(json_value({k:v for k,v in report.items() if k not in ['global_resource','local_resources']}),indent=2),flush=True)
    coarse_path=Path('/coarse/candidate-state-not-original-checkpoint.npz')
    with np.load(coarse_path,allow_pickle=False) as coarse:
        self_compare={}
        for key,offset in [('state_v',0.),('state_F',np.eye(3)),('state_tau',0.),('history_kappa',0.)]:
            self_compare[key]=float(np.linalg.norm(coarse[key]-payload[key])/max(np.linalg.norm(payload[key]-offset),1e-300))
        self_compare['max_position_difference_over_h']=float(np.max(np.linalg.norm(coarse['state_x']-payload['state_x'],axis=1))/plan['dx'])
        coarse_report=json.loads(Path('/coarse/implicit-matched-review.json').read_text())
        self_compare.update(K=abs(coarse_report['K']-K)/max(K,1e-300),U=abs(coarse_report['U']-U)/max(U,1e-300),absolute_fixture_work=abs(float(coarse['implicit_absolute_machine_work'])-absolute)/max(absolute,1e-300))
    inherited_physical={key:value for key,value in gates.items() if key not in ['fields','position','net_warm_window_speedup']}
    same_method_admitted=all(inherited_physical.values()) and all(value<=(.01 if key=='max_position_difference_over_h' else .05) for key,value in self_compare.items())
    time_report=dict(scope='SameTL temporal comparison only,200fine steps vs stored4coarse steps; no oldAPIC replay, same fittedinitial/referenceF0/Vp0/m/basis',same_reference_and_initialization_exact=all(all(row.values()) for row in exact_reference),physical_gates=inherited_physical,comparison=self_compare,admitted=bool(same_method_admitted),old_APIC_comparison=report['comparison'],old_APIC_admitted=bool(report['admitted']),coarse_dt=50*dt,fine_dt=dt,coarse_steps=4,fine_steps=200,fine_runtime=dict(layout_and_initialization=layout_seconds,compile=compile_seconds,dispatch=wall,CPU=cpu),coarse_original_runtime=dict(dispatch=coarse_report['actual_dispatch_wall_seconds'],layout=coarse_report['layout_seconds']))
    (a.output/'same-TL-time-review.json').write_text(json.dumps(json_value(time_report),indent=2));print(json.dumps(json_value(time_report),indent=2),flush=True)
    if not same_method_admitted:raise RuntimeError('SameTL temporal qualification failed; no dt scan, raw preserved')


if __name__=='__main__':main()
