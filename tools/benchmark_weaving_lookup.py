"""Read-only same-CP hash/direct operator timing; no canonical trajectory is rerun."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreState,FibreConfig,stencil,step_with_diagnostics,validate_state
    from mpm_jax.fibre_grips import sparse_reference_grips
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    from examples.triaxial_weaving_process import plan_process
    from examples.triaxial_weaving_geometry import fixture_position
    from examples.triaxial_weaving_execution import allocate
    geometry,plan=plan_process(per_family=1);a.output.mkdir(exist_ok=False)
    with np.load(a.checkpoint,allow_pickle=False) as cp:
        assert int(cp['step'])==1000
        values={k:jnp.asarray(cp['state_'+k]) for k in FibreState._fields if k!='history'};values['history']=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields));state=FibreState(**values);capacity=int(cp['packed_capacity']);parent_binding=json.loads(str(cp['binding']))
        assert all(np.array_equal(cp['state_'+k],np.asarray(getattr(state,k))) for k in FibreState._fields if k!='history')
        assert np.array_equal(cp['state_mass'],geometry['mass']) and np.array_equal(cp['state_volume'],geometry['volume'])
    layouts=[];resources=[]
    for mode in ['hash','direct']:
        layout,data,graph,resource=allocate(state,geometry,plan['dx'],plan['dt'],200,capacity,lookup=mode)
        assert resource['real_coverage_pairs']==0;validate_state(state,layout);layouts.append(layout);resources.append(resource)
    compiled=jax.jit(stencil);supports=[compiled(state.x,state.fibre_id,l) for l in layouts]
    for key in ['indices','weights','dpos']:assert np.array_equal(np.asarray(getattr(supports[0],key)),np.asarray(getattr(supports[1],key))),key
    material=isotropic_brittle(young=plan['young_Pa'],poisson=plan['poisson'],density=plan['density_kg_m3'],tensile_strength=plan['tensile_strength_Pa'],fracture_energy=plan['fracture_energy_J_m2'],characteristic_length=plan['dx']);config=FibreConfig(dt=plan['dt']);meta=geometry['metadata'];times=meta['fixture_times'];positions=meta['fixture_positions']
    def speed(group):return lambda t:(fixture_position(t,times,positions,jnp)[group]-fixture_position(t-config.dt,times,positions,jnp)[group])/config.dt
    boundary=sparse_reference_grips(geometry['masks'],tuple(speed(i) for i in range(len(geometry['masks']))))
    @jax.jit
    def block(s,layout,n):
        def body(_,carry):
            old,path=carry;new,_=step_with_diagnostics(old,material,layout,config,boundary=boundary)
            maximum=jnp.max(jnp.abs(jnp.log(jnp.linalg.svd(new.F,compute_uv=False))));new=new._replace(valid=new.valid&(maximum<=.12)&jnp.all(new.history.damage<=1e-10))
            return new,path+jnp.linalg.norm(new.x-old.x,axis=1)
        return jax.lax.fori_loop(0,n,body,(s,jnp.zeros(len(s.x))))
    timings=[];final=[];paths=[]
    for layout in layouts:
        warm=block(state,layout,4);jax.block_until_ready(warm);samples=[]
        for _ in range(3):
            start=time.perf_counter();result=block(state,layout,64);jax.block_until_ready(result);samples.append((time.perf_counter()-start)/64)
        assert bool(result[0].valid);validate_state(result[0],layout)
        assert float(result[1].max())<.0003/2,'Absolute path bound no longer certifies contact-free diagnostic'
        timings.append(samples);final.append(result[0]);paths.append(float(result[1].max()))
    scales={'x':plan['dx'],'v':plan['elastic_wave_speed_m_s'],'F':1.,'C':1/plan['dt']};comparison={}
    for key,scale in scales.items():
        difference=float(np.max(abs(np.asarray(getattr(final[0],key))-np.asarray(getattr(final[1],key)))));assert difference/scale<=1e-12,key;comparison[key]=dict(max_abs=difference,dimensionless=difference/scale)
    report=dict(scope='Same exact input CP1000,64-step contact-free diagnostic only; no source-matched restart or final-scale claim',input_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),input_arrays_exact=True,parent_source_sha256=parent_binding['source_sha256'],same_stencil_indices_weights_dpos=True,hash_seconds_per_step=timings[0],direct_seconds_per_step=timings[1],median_speedup=float(np.median(timings[0])/np.median(timings[1])),absolute_path_bounds_m=paths,comparison=comparison,execution=resources)
    (a.output/'lookup-benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='parent_source_sha256'},indent=2))


if __name__=='__main__':main()
