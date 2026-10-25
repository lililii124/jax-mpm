"""Focused compact/full return-map checks; no MPM trajectory or server launch."""
import argparse
import json
from pathlib import Path
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mpm_jax.materials import initial_history,rht
    from mpm_jax.materials.rht import (_RETURN_BATCH_SIZE,_return_branch,_return_branch_full,
        _pressure_trial,_at_multiplier,_tolerance)
    jax.config.update('jax_enable_x64',True)
    batch=_RETURN_BATCH_SIZE
    p=rht()
    fixtures=[('small-original-path',3,np.array([0]),1,False),
        ('sparse-tail-padding',batch+7,np.array([0,batch-1,batch+6]),1,True),
        ('cross-batch-padding-stage1',batch+7,np.r_[np.arange(batch+2),batch+6],1,False),
        ('cross-batch-padding-stage2',batch+7,np.r_[np.arange(batch+2),batch+6],2,False),
        ('no-yield-initial-compaction',batch+3,np.array([],dtype=int),1,False)]
    records=[]
    for name,n,active,stage,vector_dt in fixtures:
        history=initial_history(n,jnp.float64,p,p.density)
        history=history._replace(temperature=history.temperature+jnp.arange(n)*1e-4,
            work=history.work+jnp.arange(n)*1e-5)
        rho=jnp.full(n,p.density)
        inactive=n-2 if n>3 else 1
        rho=rho.at[inactive].set(1.1*p.density)
        energy=jnp.zeros(n)
        trial=jnp.zeros((n,3,3))
        if len(active):
            trial=trial.at[active,0,1].set(3e8).at[active,1,0].set(3e8)
            rho=rho.at[active[0]].set(.999*p.density)
            if len(active)>1:rho=rho.at[active[-1]].set(1.005*p.density)
        dt=jnp.linspace(1.,1.001,n)*1e-6 if vector_dt else 1e-6
        context=_pressure_trial(rho,energy,history,p)
        initial,f0=_at_multiplier(jnp.zeros_like(rho),rho,energy,dt,trial,history,p,context,stage)
        tol=_tolerance(trial.dtype)*jnp.maximum(p.fc,jnp.max(jnp.abs(trial),axis=(-2,-1)))
        yielding=np.asarray(f0>tol)
        assert np.count_nonzero(yielding)==len(active),(name,np.flatnonzero(yielding))
        assert float(initial.alpha[inactive])!=float(history.alpha[inactive]),'Inactive initial compaction must be retained'
        def full(rho,energy,dt,trial,history):
            local=_pressure_trial(rho,energy,history,p)
            return _return_branch_full(rho,energy,dt,trial,history,p,local,stage)
        def compact(rho,energy,dt,trial,history):
            local=_pressure_trial(rho,energy,history,p)
            return _return_branch(rho,energy,dt,trial,history,p,local,stage)
        started=time.monotonic();reference=jax.jit(full)(rho,energy,dt,trial,history)
        jax.block_until_ready(reference.stress);full_seconds=time.monotonic()-started
        started=time.monotonic();result=jax.jit(compact)(rho,energy,dt,trial,history)
        jax.block_until_ready(result.stress);compact_seconds=time.monotonic()-started
        errors={};bitwise=True
        for field,a,b,c in zip(result._fields,reference,result,initial):
            a,b,c=np.asarray(a),np.asarray(b),np.asarray(c)
            assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(b).all(),(name,field)
            bitwise &= np.array_equal(a,b)
            if a.dtype.kind in 'biu':
                np.testing.assert_array_equal(a,b,err_msg=name+':'+field)
                errors[field]=0.
            else:
                error=float(np.max(np.abs(a-b)))
                scale=max(1.,float(np.max(np.abs(a))))
                assert error<=1e-12*scale,(name,field,error,scale)
                errors[field]=error
            # Non-yielding output must be the full initial update, not old history.
            np.testing.assert_array_equal(b[~yielding],c[~yielding],err_msg=name+':inactive:'+field)
        if n<=batch or not len(active):assert bitwise,name
        if len(active):assert np.count_nonzero(np.asarray(result.valid)&yielding)>0,name
        # In particular real point0 must survive every padded scatter.
        if len(active):np.testing.assert_allclose(np.asarray(result.stress[0]),np.asarray(reference.stress[0]),rtol=1e-12,atol=1e-5)
        records.append(dict(name=name,n=n,yielding=int(yielding.sum()),stage=stage,vector_dt=vector_dt,
            full_seconds=full_seconds,compact_seconds=compact_seconds,all_fields_bitwise_equal=bool(bitwise),
            field_max_absolute_errors=errors,non_yielding_equal_full_initial=True,point0_padding_preserved=True,
            valid_count=int(np.count_nonzero(np.asarray(result.valid)))))
    output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,scope='Focused return-map optimization checks only; no MPM/time/energy qualification.',
        batch_size=batch,cases=records,tolerances_and56_bisections64_roots_unchanged=True,
        mpm_steps_executed=0,real_state106_comparison_complete=False)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.output.resolve())
