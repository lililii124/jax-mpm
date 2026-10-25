"""Actual frozen-C Cayley failure and separate-gradient ballistic consistency."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def qualify():
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fragment_multifield_v6_mpm import cayley_kinematics
    from mpm_jax.fragment_v6_geometric_kinematics import geometric_kinematics
    from tools.metal_geometric_kinematics_numpy import geometric_kinematics as numpy_map
    L0=np.diag([1e7,-5e6,2e6]);T=1e-8;exact=np.eye(3)+T*L0
    exact_L=L0@np.linalg.inv(exact)
    old_step=jax.jit(lambda F,h:cayley_kinematics(F,jnp.asarray(L0),h)[0])
    new_step=jax.jit(geometric_kinematics)
    rng=np.random.default_rng(179)
    arbitrary_F=np.eye(3)+rng.normal(size=(13,3,3))*.04
    arbitrary_L=rng.normal(size=(13,3,3))*1e7
    arbitrary_increment=rng.normal(size=(13,3,3))*1e6
    reference=numpy_map(arbitrary_F,arbitrary_L,arbitrary_increment,1e-9)
    actual=new_step(jnp.asarray(arbitrary_F),jnp.asarray(arbitrary_L),jnp.asarray(arbitrary_increment),1e-9)
    errors=[float(np.max(abs(np.asarray(value)-expected))) for value,expected in zip(actual[:4],reference)]
    assert errors[0]<1e-13 and errors[1]<1e-7 and errors[2]<1e-13 and errors[3]<1e-13
    rows=[]
    for steps in (8,16,32,64):
        old_F=jnp.eye(3);new_F=jnp.eye(3);L=jnp.asarray(L0);h=T/steps
        for _ in range(steps):
            old_F=old_step(old_F,h)
            result=new_step(new_F,L,jnp.zeros((3,3)),h)
            assert bool(result.valid)
            new_F,L=result.F,result.L
        rows.append(dict(steps=steps,old_F=np.asarray(old_F).tolist(),
            old_F_linf_error=float(np.max(abs(np.asarray(old_F)-exact))),
            geometric_F_linf_error=float(np.max(abs(np.asarray(new_F)-exact))),
            geometric_L_relative_error=float(np.max(abs(np.asarray(L)-exact_L)))/np.max(abs(exact_L))))
    assert all(row['old_F_linf_error']>.005 for row in rows)
    assert all(row['geometric_F_linf_error']<1e-13 for row in rows)
    assert all(row['geometric_L_relative_error']<1e-13 for row in rows)
    # A rotating/stretching affine field demonstrates why the physical gradient
    # cannot simply overwrite C under the existing fixed APIC kinetic metric.
    C=np.array([[1.,3.,0.],[-2.,.5,0.],[0.,0.,1.]])
    following=C@np.linalg.inv(np.eye(3)+.1*C)
    spin=lambda a:np.array([a[2,1]-a[1,2],a[0,2]-a[2,0],a[1,0]-a[0,1]])
    fixed_D_angular_change=spin(following)-spin(C)
    fixed_D_affine_metric_change=float(np.sum(following**2)-np.sum(C**2))
    assert np.linalg.norm(fixed_D_angular_change)>1e-3
    return dict(scope='Pure zero-force ballistic affine kinematics; no constitutive/contact/cloud continuation.',
        initial_gradient_per_s=L0.tolist(),duration_s=T,exact_F=exact.tolist(),exact_final_spatial_gradient_per_s=exact_L.tolist(),
        old_limit_F=np.diag(np.exp(T*np.diag(L0))).tolist(),rows=rows,
        actual_old_cayley_F_is_inconsistent_with_delta_particle_motion=True,
        separate_physical_gradient_map_passed=True,
        arbitrary_noncommuting_numpy_jax_errors=dict(zip(('F','L_per_s','rotation','strain'),errors)),
        naive_C_overwrite=dict(unscaled_affine_spin_change=fixed_D_angular_change.tolist(),
            unscaled_affine_squared_norm_change=fixed_D_affine_metric_change,
            forbidden='Do not replace conserved-transfer C by transported physical L under fixed D.'),
        production_integration_prepared=False,production_restart_allowed=False,
        remaining='Independent L carry must be initialized from the authenticated seed, used for material kinematics/compression, and coupled to compatible stress work. C remains a transfer coefficient. No dynamic admission follows from this algebraic map alone.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',required=True,type=Path);a=parser.parse_args()
    if a.output.exists():raise FileExistsError('Preserve earlier evidence')
    result=qualify()
    names=('tools/prove_metal_affine_kinematics.py','tools/metal_geometric_kinematics_numpy.py','mpm_jax/fragment_v6_geometric_kinematics.py','mpm_jax/fragment_v6_delta_candidate.py','mpm_jax/fragment_multifield_v6_mpm.py')
    result['source_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
