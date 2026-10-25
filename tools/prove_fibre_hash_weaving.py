"""Actual-JAX sparse-index proof plus a real64-step weaving/raw comparison.

No proof is inferred from this source existing; run only inside a bounded
approved compute capsule with the original authenticated dense64 snapshots.
"""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import build_layout,build_tube_layout,stencil,grid_coordinates
    from mpm_jax.fibre_hash_layout import build_hash_tube_layout
    x=np.array([[0.,0.,0.],[1.,1.,1.],[2.,0.,0.],[3.,1.,1.]]);ids=np.array([0,0,1,1],np.int32)
    dense=build_layout(x,ids,dx=.1,padding_cells=2);tube=build_tube_layout(x,ids,dx=.1,padding_cells=2)
    hashed,resource=build_hash_tube_layout(x,ids,dx=.1,padding_cells=2)
    assert np.array_equal(tube.dense_indices,hashed.dense_indices)
    compiled_stencil=jax.jit(stencil)
    ds=compiled_stencil(jnp.asarray(x),jnp.asarray(ids),dense);ts=compiled_stencil(jnp.asarray(x),jnp.asarray(ids),tube)
    hs=compiled_stencil(jnp.asarray(x),jnp.asarray(ids),hashed)
    assert bool(ds.valid&ts.valid&hs.valid)
    for name in ['indices','weights','dpos']:assert np.array_equal(np.asarray(getattr(ts,name)),np.asarray(getattr(hs,name))),name
    assert np.array_equal(np.asarray(ds.indices),np.asarray(hashed.dense_indices)[np.asarray(hs.indices)])
    for left,right in zip(grid_coordinates(tube,jnp.arange(tube.node_count)),grid_coordinates(hashed,jnp.arange(hashed.node_count))):assert np.array_equal(left,right)
    hole=x.copy();hole[0]=[.5,.5,.5]
    assert not bool(stencil(jnp.asarray(hole),jnp.asarray(ids),hashed).valid)
    far=x.copy();far[0]=[10.,10.,10.]
    assert not bool(stencil(jnp.asarray(far),jnp.asarray(ids),hashed).valid)
    from examples.triaxial_weaving_process import run
    run(a.output,per_family=1,steps=64,saved_frames=8,grid='hash')
    comparison={};all_exact=True
    for file in sorted(a.output.glob('frame_*.npz')):
        with np.load(file) as actual,np.load(a.baseline/file.name) as expected:
            assert set(actual.files)==set(expected.files)
            fields={}
            for key in actual.files:
                left,right=actual[key],expected[key];exact=bool(np.array_equal(left,right));all_exact&=exact
                if left.dtype.kind in 'biu' or key in ('mass','damage','time','fixture_targets'):assert exact,key
                difference=float(np.max(np.abs(left-right),initial=0.)) if left.dtype.kind in 'fiu' else 0.
                if left.dtype.kind=='f':assert np.allclose(left,right,rtol=1e-12,atol=1e-14),key
                fields[key]=dict(exact=exact,max_abs=difference)
            comparison[file.name]=fields
    report=dict(scope='Index equivalence and64-step prefix only; not final-scale or weave qualification',
        stencil_indices_weights_dpos_exact=True,global_nodes_exact=True,incomplete_quadratic_stencil_rejected=True,
        all_raw_arrays_bitwise_exact=all_exact,declared_float_tolerance=dict(rtol=1e-12,atol=1e-14),
        small_layout=resource,raw_comparison=comparison)
    (a.output/'hash-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='raw_comparison'},indent=2))


if __name__=='__main__':main()
