"""One actual checkpoint: independent same-world-lattice fibre dispatch proof.

The existing reference operator already gives every cord a separate material
velocity field. Global contact remains a coupling force, never a grid merge.
This short proof is restricted to a certified contact-free checkpoint window.
"""
import argparse,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreState,FibreConfig,stencil,validate_state,_step
    from mpm_jax.fibre_grips import sparse_reference_grips
    from mpm_jax.fibre_hash_layout import build_hash_tube_layout
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    from examples.triaxial_weaving_batch_plan import plan_batch
    from examples.triaxial_weaving_geometry import fixture_position
    from examples.triaxial_weaving_execution import allocate
    geometry,plan=plan_batch(per_family=8,nominal_cfl=.08);a.output.mkdir(exist_ok=False)
    with np.load(a.checkpoint,allow_pickle=False) as cp:
        assert int(cp['step'])==360
        binding=json.loads(str(cp['binding']));assert binding['physical_plan']==plan
        values={k:jnp.asarray(cp['state_'+k]) for k in FibreState._fields if k!='history'}
        values['history']=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields));state=FibreState(**values)
        exact={k:bool(np.array_equal(cp['state_'+k],np.asarray(getattr(state,k)))) for k in FibreState._fields if k!='history'}
        exact.update({'history_'+k:bool(np.array_equal(cp['history_'+k],np.asarray(getattr(state.history,k)))) for k in BrittleHistory._fields});assert all(exact.values())
        assert np.array_equal(cp['state_mass'],geometry['mass']) and np.array_equal(cp['state_volume'],geometry['volume'])
        assert not np.any(cp['law_active']) and not np.any(cp['contact_radius'])
        capacity=int(cp['packed_capacity']);direct_capacity=int(cp['direct_map_capacity'])
        parent_aux={k:cp[k].copy() for k in ['law_active','law_valid','contact_radius','contact_segments','contact_valid','accumulator']}
    (a.output/'input-exact.json').write_text(json.dumps(exact,indent=2))
    layout,data,graph,resource=allocate(state,geometry,plan['dx'],plan['dt'],200,capacity,lookup='direct',direct_capacity=direct_capacity)
    assert resource['real_coverage_pairs']==0,'Coupled-contact checkpoint needs synchronized contact force evaluation, not this proof'
    validate_state(state,layout)
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx'])
    config=FibreConfig(dt=plan['dt']);meta=geometry['metadata'];times=jnp.asarray(meta['fixture_times']);positions=jnp.asarray(meta['fixture_positions']);nsteps=16
    def boundary_for(masks,pos):
        def speed(i):return lambda t:(fixture_position(t,times,pos,jnp)[i]-fixture_position(t-config.dt,times,pos,jnp)[i])/config.dt
        return sparse_reference_grips(masks,tuple(speed(i) for i in range(len(masks))))
    global_boundary=boundary_for(geometry['masks'],positions)
    def advance(s,l,boundary,pos):
        def body(_,carry):
            old,path,work=carry
            new,diag=_step(old,material,l,config,None,None,boundary)
            maximum=jnp.max(jnp.abs(jnp.log(jnp.linalg.svd(new.F,compute_uv=False))))
            courant=config.dt*(plan['elastic_wave_speed_m_s']+jnp.max(jnp.linalg.norm(new.v,axis=1)))/plan['dx']
            new=new._replace(valid=new.valid&(maximum<=.12)&jnp.all(new.history.damage<=1e-10)&(courant<=.25))
            targets=(fixture_position(new.time,times,pos,jnp)-fixture_position(old.time,times,pos,jnp))/config.dt
            group_work=jnp.sum(diag.group_impulses*targets,axis=1);machine_work=jnp.sum(group_work)
            delta=jnp.array([0.,jnp.abs(diag.boundary_work),machine_work,machine_work-diag.boundary_work,jnp.sum(jnp.abs(group_work)),0.,0.,0.])
            updated=(work+delta).at[7].set(jnp.maximum(work[7],courant))
            return new,path+jnp.linalg.norm(new.x-old.x,axis=1),updated
        return jax.lax.fori_loop(0,nsteps,body,(s,jnp.zeros(len(s.x)),jnp.zeros(8)))
    @jax.jit
    def global_block(s,l):return advance(s,l,global_boundary,positions)
    @jax.jit
    def local_block(s,l,pos):
        cap=3*meta['cross_section_particles'];index=np.arange(len(s.x))
        return advance(s,l,boundary_for((index<cap,index>=len(index)-cap),pos),pos)
    initial_time=float(state.time);span=200*config.dt
    x=np.asarray(state.x);v=np.asarray(state.v);ids=np.asarray(state.fibre_id)
    seeds=[x,x+span*v];seed_ids=[ids,ids]
    base=fixture_position(initial_time,meta['fixture_times'],meta['fixture_positions'])
    for fraction in [.5,1.]:
        target=fixture_position(initial_time+span*fraction,meta['fixture_times'],meta['fixture_positions'])
        for mask,displacement in zip(geometry['masks'],target-base):seeds.append(x[mask]+displacement);seed_ids.append(ids[mask])
    seeds=np.concatenate(seeds);seed_ids=np.concatenate(seed_ids)
    support=jax.jit(stencil)(state.x,state.fibre_id,layout);jax.block_until_ready(support)
    local=[];node_sets=[];allocations=[];slices=[]
    for fibre in meta['fibres']:
        i=fibre['id'];first,last=fibre['particle_slice'];sl=slice(first,last);slices.append(sl)
        fields={k:getattr(state,k)[sl] for k in ['x','v','F','C','mass','volume','tau']}
        fields.update(fibre_id=jnp.zeros(last-first,jnp.int32),history=BrittleHistory(*(getattr(state.history,k)[sl] for k in BrittleHistory._fields)),time=state.time,valid=state.valid,boundary_impulse=jnp.zeros(3),external_impulse=jnp.zeros(3),boundary_work=jnp.asarray(0.))
        s=FibreState(**fields);selected=seeds[seed_ids==i]
        l,r=build_hash_tube_layout(selected,np.zeros(len(selected),np.int32),dx=plan['dx'],padding_cells=3,capacity_factor=1.25,direct_map_capacity='auto',max_active_nodes=8_000_000)
        validate_state(s,l);sp=jax.jit(stencil)(s.x,s.fibre_id,l);jax.block_until_ready(sp)
        assert np.array_equal(sp.weights,np.asarray(support.weights)[sl]) and np.array_equal(sp.dpos,np.asarray(support.dpos)[sl])
        assert np.array_equal(l.lower[0],layout.lower[i]) and np.array_equal(l.origin,layout.origin)
        # Actual nonzero-particle quadratic support, expressed on one world grid.
        offsets=np.stack(np.meshgrid(*([np.arange(3)]*3),indexing='ij'),-1).reshape(-1,3)
        world=(np.floor(x[sl]/plan['dx']-.5).astype(np.int64)[:,None,:]+offsets).reshape(-1,3)
        assert np.max(abs(world))<10000
        key=(world[:,0]+10000)*20001**2+(world[:,1]+10000)*20001+world[:,2]+10000
        node_sets.append(np.unique(key));local.append((s,l,positions[:,2*i:2*i+2]));allocations.append(r)
    overlaps=[]
    for i in range(24):
        for j in range(i+1,24):
            if meta['fibres'][i]['family_id']==meta['fibres'][j]['family_id']:
                count=len(np.intersect1d(node_sets[i],node_sets[j],assume_unique=True));overlaps.append(dict(pair=[i,j],shared_world_support_nodes=count));assert count==0
    # Compile before wall/CPU samples. Distinct local particle/map shapes each
    # compile once, then all dispatches use the same cached mathematical kernel.
    warm=global_block(state,layout);jax.block_until_ready(warm)
    signatures=set()
    for s,l,pos in local:
        signature=(len(s.x),l.node_count,l.node_map.shape)
        if signature not in signatures:jax.block_until_ready(local_block(s,l,pos));signatures.add(signature)
    samples={};results={}
    for mode in ['global','fibre_sequential','fibre_parallel']:
        wall=time.perf_counter();cpu=time.process_time()
        if mode=='global':result=global_block(state,layout);jax.block_until_ready(result)
        elif mode=='fibre_sequential':
            result=[]
            for arguments in local:
                record=local_block(*arguments);jax.block_until_ready(record);result.append(record)
        else:
            def execute(arguments):result=local_block(*arguments);jax.block_until_ready(result);return result
            with ThreadPoolExecutor(max_workers=8) as pool:result=list(pool.map(execute,local))
        samples[mode]=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
        samples[mode]['cpu_wall_ratio']=samples[mode]['cpu_seconds']/samples[mode]['wall_seconds'];samples[mode]['seconds_per_step']=samples[mode]['wall_seconds']/nsteps
        results[mode]=result;print(json.dumps(dict(mode=mode,timing=samples[mode])),flush=True)
    original=results['global'][0];comparisons={}
    def payload(s):
        result={'state_'+k:np.asarray(getattr(s,k)) for k in FibreState._fields if k!='history'};result.update({'history_'+k:np.asarray(getattr(s.history,k)) for k in BrittleHistory._fields});return result
    global_payload=payload(original);global_payload.update(parent_aux);global_acc=np.asarray(parent_aux['accumulator'])+np.asarray(results['global'][2]);global_acc[7]=max(float(parent_aux['accumulator'][7]),float(results['global'][2][7]));global_payload['accumulator']=global_acc
    np.savez_compressed(a.output/'final-global.npz',**global_payload)
    for mode in ['fibre_sequential','fibre_parallel']:
        records=results[mode];assert all(bool(row[0].valid) for row in records)
        assert max(float(row[1].max()) for row in records)<.0003/2
        fields={k:jnp.concatenate([getattr(row[0],k) for row in records]) for k in ['x','v','F','C','mass','volume','tau']}
        fields.update(fibre_id=state.fibre_id,history=BrittleHistory(*(jnp.concatenate([getattr(row[0].history,k) for row in records]) for k in BrittleHistory._fields)),time=records[0][0].time,valid=jnp.all(jnp.stack([row[0].valid for row in records])),boundary_impulse=state.boundary_impulse+jnp.sum(jnp.stack([row[0].boundary_impulse for row in records]),axis=0),external_impulse=state.external_impulse+jnp.sum(jnp.stack([row[0].external_impulse for row in records]),axis=0),boundary_work=state.boundary_work+jnp.sum(jnp.stack([row[0].boundary_work for row in records])))
        merged=FibreState(**fields);output=payload(merged);output.update(parent_aux);local_acc=np.asarray(parent_aux['accumulator'])+np.sum(np.stack([np.asarray(row[2]) for row in records]),axis=0);local_acc[7]=max(float(parent_aux['accumulator'][7]),max(float(row[2][7]) for row in records));output['accumulator']=local_acc;np.savez_compressed(a.output/('final-'+mode+'.npz'),**output)
        comparison={}
        scales={'state_x':plan['dx'],'state_v':plan['elastic_wave_speed_m_s'],'state_F':1.,'state_C':1/plan['dt'],'state_tau':plan['young_Pa'],'state_mass':float(state.mass.sum()),'state_volume':float(state.volume.sum()),'state_boundary_impulse':float(state.mass.sum())*plan['elastic_wave_speed_m_s'],'state_external_impulse':float(state.mass.sum())*plan['elastic_wave_speed_m_s'],'state_boundary_work':max(abs(float(original.boundary_work)),1e-30)}
        for k,value in global_payload.items():
            difference=float(np.max(abs(output[k].astype(float)-value.astype(float)),initial=0));scale=scales.get(k,1.);
            if k=='accumulator':
                acc_scale=np.array([max(abs(global_acc[4]),1e-30)]*5+[float(state.mass.sum())*plan['elastic_wave_speed_m_s']]*2+[1.]);normalized=float(np.max(abs(output[k]-value)/acc_scale))
            else:normalized=difference/scale
            comparison[k]=dict(exact=bool(np.array_equal(output[k],value)),max_absolute_difference=difference,dimensionless=normalized)
            comparison[k]['passes_predeclared_gate']=normalized<=1e-12
        comparisons[mode]=comparison
    admitted=all(v['passes_predeclared_gate'] for mode in comparisons.values() for v in mode.values())
    report=dict(admitted=admitted,scope='One real fresh-batch fineCP360;16steps, no contact only. Future coupled contact needs synchronized global force before every per-fibre step.',checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),input_arrays_exact=exact,same_world_grid=True,same_stencil_weights_and_offsets=True,same_family_support_intersections=overlaps,global_resource=resource,local_resources=allocations,local_occupied_nodes_sum=sum(r['occupied_nodes'] for r in allocations),local_capacity_sum=sum(r['active_nodes'] for r in allocations),local_map_bytes_sum=sum(r['direct_map_bytes'] for r in allocations),global_capacity=capacity,compile_shape_count=len(signatures),steps=nsteps,timing=samples,parallel_speedup=samples['global']['wall_seconds']/samples['fibre_parallel']['wall_seconds'],comparison=comparisons,absolute_path_bound_m=max(float(row[1].max()) for row in results['fibre_parallel']))
    (a.output/'parallel-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['comparison','same_family_support_intersections','local_resources','input_arrays_exact']},indent=2))
    if not admitted:raise RuntimeError('Original predeclared audit tolerance failed; full raw/timing saved, numerical acceptance remains independent')


if __name__=='__main__':main()
