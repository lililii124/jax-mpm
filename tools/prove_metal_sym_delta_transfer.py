"""Small arbitrary-field operator qualification; never starts impact dynamics."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.metal_frozen_transfer import support
from tools.metal_sym_delta_transfer_numpy import sym_delta_remap


def qualify():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fragment_v6_sym_delta_transfer import transfer_sym_delta
    from mpm_jax.fragment_multifield_v6_mpm import transfer_to_grid
    rng=np.random.default_rng(137)
    x=rng.uniform(-.3,.3,(40,3));v=rng.normal(size=(40,3));C=rng.normal(size=(40,3,3));m=rng.uniform(.5,1.5,40)
    body=np.repeat(np.arange(2),20);dx=.25
    frozen=support(x,body,m,dx);w,r,ids,wm,gm=frozen
    increment=np.random.default_rng(71).normal(size=(len(gm),3))*.2
    nv,nC,d=sym_delta_remap(x,v,C,m,dx,frozen,increment)
    state=SimpleNamespace(x=jnp.asarray(x),v=jnp.asarray(v),C=jnp.asarray(C),mass=jnp.asarray(m),body_id=jnp.asarray(body))
    stencil=SimpleNamespace(weights=jnp.asarray(w),dpos=jnp.asarray(r),indices=jnp.asarray(ids))
    grid=SimpleNamespace(layout=SimpleNamespace(dx=dx,node_count=len(gm),fibre_count=2))
    _,old,_=transfer_to_grid(state,stencil,grid)
    jv,jC,correction,jd=jax.jit(lambda u:transfer_sym_delta(state,old,u,stencil,grid))(old+jnp.asarray(increment))
    max_v=float(np.max(abs(nv-np.asarray(jv))));max_C=float(np.max(abs(nC-np.asarray(jC))))
    assert max_v<1e-13 and max_C<1e-13
    assert np.linalg.norm(d['momentum_defect'])<1e-13 and np.linalg.norm(d['angular_defect'])<1e-13
    assert np.max(abs(np.asarray(jd.linear_defect_by_field)))<1e-13
    assert np.max(abs(np.asarray(jd.angular_defect_by_field)))<1e-13
    assert abs(d['ledger_identity_residual_J'])<1e-12
    assert abs(d['orthogonal_identity_residual_J'])<1e-12
    assert abs(d['correction_identity_residual_J'])<1e-12
    assert abs(float(np.dot(m,np.asarray(correction)))-d['g2p_signed_J'])<1e-12
    # A physically admissible arbitrary non-affine velocity, zero affine state,
    # no grid increment: any nonzero symmetric fit creates kinetic energy.
    zero=np.zeros_like(C)
    energy=lambda vv,cc:float(.5*np.sum(m[:,None]*vv*vv)+dx**2/8*np.sum(m[:,None,None]*cc*cc))
    kinetic_initial=energy(v,zero);cv,cC=v.copy(),zero.copy();history=[]
    for i in range(128):
        cv,cC,zd=sym_delta_remap(x,cv,cC,m,dx,frozen)
        history.append(dict(remap=i+1,kinetic_J=energy(cv,cC),increment_J=zd['particle_delta_J'],
            ledger_identity_residual_J=zd['ledger_identity_residual_J']))
    assert np.array_equal(cv,v)
    assert history[0]['increment_J']>1e-6*kinetic_initial
    result=dict(scope='Operator-only, arbitrary nonaffine fields with two independent physical bodies. No impact continuation.',
        numpy_jax_max_v_error=max_v,numpy_jax_max_C_error=max_C,
        numpy_diagnostics={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in d.items()},
        jax_diagnostics={name:np.asarray(getattr(jd,name)).tolist() for name in jd._fields},
        zero_force=dict(initial_kinetic_J=kinetic_initial,first_increment_J=history[0]['increment_J'],
            maximum_kinetic_J=max(row['kinetic_J'] for row in history),final_kinetic_J=history[-1]['kinetic_J'],
            velocity_bitwise_unchanged=True,rows=history),
        conservation_and_accounting_passed=True,zero_force_kinetic_nonincrease_passed=False,
        verdict='Do not admit this candidate as energy-stable. Signed numerical-reservoir closure does not remove force-free affine energy creation.',
        dynamic_trial_prepared=False,production_restart_allowed=False,public_media_allowed=False)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier evidence')
    result=qualify()
    result['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ('tools/prove_metal_sym_delta_transfer.py','tools/metal_sym_delta_transfer_numpy.py','mpm_jax/fragment_v6_sym_delta_transfer.py')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='zero_force'},indent=2))
    print('zero-force initial/first-gain/final:',result['zero_force']['initial_kinetic_J'],result['zero_force']['first_increment_J'],result['zero_force']['final_kinetic_J'])
