"""Bounded admission for moving support and sparse fixtures, then timed1000-step prefix."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_layout,stencil,reference_grips
    from mpm_jax.fibre_grips import sparse_reference_grips
    from mpm_jax.contact_patches import contact_patches
    x=np.array([[0.,0.,0.],[1.,1.,1.],[2.,0.,0.],[3.,1.,1.]]);ids=np.array([0,0,1,1],np.int32);layout=build_layout(x,ids,dx=.1,padding_cells=2);support=stencil(jnp.asarray(x),jnp.asarray(ids),layout)
    mass=jnp.where(jnp.arange(layout.node_count)%7,1.,0.);velocity=jnp.asarray(np.random.default_rng(4).normal(size=(layout.node_count,3)))
    masks=[np.array([True,False,False,False]),np.array([False,False,False,True])];speeds=[lambda t:jnp.array([1.,2.,3.]),lambda t:jnp.array([-1.,.5,2.])];state=SimpleNamespace(mass=jnp.ones(4))
    old=reference_grips(masks,speeds)(velocity,mass,state,support,layout,0.);new=sparse_reference_grips(masks,speeds)(velocity,mass,state,support,layout,0.)
    assert np.array_equal(old.velocity,new.velocity) and bool(old.valid&new.valid)
    assert np.allclose(old.group_impulses,new.group_impulses,rtol=1e-12,atol=1e-12) and np.allclose(old.group_work,new.group_work,rtol=1e-12,atol=1e-12)
    overlap=[masks[0],masks[0]];assert not bool(sparse_reference_grips(overlap,speeds)(velocity,mass,state,support,layout,0.).valid)
    gap=jnp.array([-.001,.1,-.0005]);graph=np.full((3,8),-1,np.int32);graph[0,0]=2;graph[2,0]=0
    original=contact_patches(gap,gap<0,jnp.asarray(graph),capacity=16,max_patches=4)
    padded_gap=jnp.concatenate([gap,jnp.ones(13)]);padded_graph=jnp.asarray(np.pad(graph,((0,13),(0,0)),constant_values=-1));padded=contact_patches(padded_gap,padded_gap<0,padded_graph,capacity=16,max_patches=4)
    assert int(original.patch_count)==int(padded.patch_count)==1 and int(original.active_count)==int(padded.active_count)==2
    assert np.array_equal(np.asarray(original.representatives)[:1],np.asarray(padded.representatives)[:1])
    from examples.triaxial_weaving_adaptive_process import run
    run(a.output,steps=64,save_stride=8)
    comparison={};all_exact=True
    for file in sorted(a.output.glob('frame_*.npz')):
        with np.load(file) as actual,np.load(a.baseline/file.name) as expected:
            assert set(actual.files)==set(expected.files)
            row={}
            for key in actual.files:
                left,right=actual[key],expected[key];exact=bool(np.array_equal(left,right));all_exact&=exact
                if left.dtype.kind in 'biu' or key in ('mass','damage','time','fixture_targets'):assert exact,key
                if left.dtype.kind=='f':assert np.allclose(left,right,rtol=1e-12,atol=1e-14),key
                row[key]=dict(exact=exact,max_abs=float(abs(left-right).max(initial=0.)) if left.dtype.kind in 'fiu' else 0.)
            comparison[file.name]=row
    run(a.output,steps=1000,save_stride=200,resume=True)
    observations=json.loads((a.output/'observations.json').read_text());steady=[r for r in observations if r['step'] in [464,664,864]]
    report=dict(scope='Moving-grid/fixture equivalence and timed opening prefix only',sparse_fixture_velocity_exact=True,sparse_fixture_impulse_and_work_close=True,overlapping_fixture_rejected=True,inactive_contact_padding_preserves_active_components=True,raw64_all_arrays_bitwise_exact=all_exact,raw64_comparison=comparison,steady_blocks=steady)
    (a.output/'adaptive-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='raw64_comparison'},indent=2))


if __name__=='__main__':main()
