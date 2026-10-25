"""Small new point comparisons against exact current MAT111, never old fixtures.

No beam/impact solver is run. Supply the original MAT111.f; compilation extracts
only its unchanged scalar modules, excluding host SDK adapters.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import os


def compile_reference(source, output):
    from mpm_jax.materials.hjc import MAT111_SOURCE_SHA256

    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MAT111_SOURCE_SHA256:
        raise ValueError('Exact current MAT111 source required')
    output.mkdir(parents=True, exist_ok=False)
    scalar = output / 'current_mat111_scalar.f'
    scalar.write_text(raw.decode().split('c PART 03:')[0])
    library = (output / ('oracle.dll' if os.name == 'nt' else 'oracle.so')).resolve()
    subprocess.run(['gfortran', '-shared', '-O0', '-ffixed-line-length-none',
                    str(scalar.resolve()), '-o', str(library)],
                   cwd=output, check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).hjc_point
    function.argtypes = [ctypes.POINTER(ctypes.c_double)] * 4
    return function, scalar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    from mpm_jax.materials.hjc import hjc, update_hjc, MAT111_SOURCE_SHA256, MAT111_PROFILE_SCOPE
    from mpm_jax.materials.history import initial_history
    function, f = compile_reference(args.source, args.output)
    pointer = ctypes.POINTER(ctypes.c_double)
    cases = [('D1_004_compressive_shear', .04, .02, .003, 1e-6, 0.),
             ('D1_0004_below_threshold', .004, .02, .003, 1e-6, 0.),
             ('D1_0004_active_damage', .004, .2, .03, 1e-6, 0.),
             ('subreference_rate', .04, 0., 1e-5, 1., 0.),
             ('tension', .04, -.0001, .003, 1e-6, 0.),
             ('zero_strength', .04, -.00025, .003, 1e-6, 1.),
             ('zero_trial_hydrostatic', .04, .02, 0., 1e-6, 0.),
             ('compacted_unload', .04, .099, .003, 1e-6, .25)]
    rows = []
    for name,d1,mu,shear,dt,damage in cases:
        p = hjc(profile='mat111-compatible', D1=d1)
        strain = np.diag([-.0001, .00005, .00005]) if shear else np.zeros((3,3))
        strain[0,1] = strain[1,0] = shear
        old = initial_history(1,jnp.float64,p,p.density)._replace(damage=jnp.array([damage]))
        compacted = name=='compacted_unload'
        if compacted:
            old = old._replace(compaction=jnp.ones(1), plastic_volume=jnp.full(1,.1),max_compression=jnp.full(1,.2))
        props=np.array([p.shear,p.crush_pressure,p.crush_strain,p.lock_pressure,p.lock_strain,
            p.K1,p.K2,p.K3,p.tensile_strength,p.compressive_strength,dt,p.A,p.B,p.C,p.N,
            p.reference_rate,p.maximum_strength,p.D1,p.D2,p.minimum_fracture_strain],dtype=np.float64)
        x=np.zeros(14); x[:6]=np.array([strain[0,0],strain[1,1],strain[2,2],2*strain[0,1],2*strain[1,2],2*strain[2,0]])/dt
        x[13]=1/(1+mu)
        history=np.array([damage,0.,float(compacted),.2 if compacted else 0.,0.,.1 if compacted else 0.,0.])
        result=np.zeros(18)
        function(*(a.ctypes.data_as(pointer) for a in (props,x,history,result)))
        with jax.disable_jit():
            actual=update_hjc(old,jnp.asarray(strain[None]),jnp.array([x[13]]),dt,p)
        expected_stress=np.array([[result[0],result[3],result[5]],[result[3],result[1],result[4]],[result[5],result[4],result[2]]])
        comparisons={'stress':(np.asarray(actual.stress[0]),expected_stress),
            'plastic_strain':(np.asarray(actual.plastic_strain[0]),history[1]),
            'damage':(np.asarray(actual.damage[0]),history[0]),
            'pressure':(np.asarray(actual.pressure[0]),result[6]),
            'plastic_volume':(np.asarray(actual.plastic_volume[0]),history[5]),
            'compaction':(np.asarray(actual.compaction[0]),history[2])}
        errors={k:float(np.max(np.abs(a-b))) for k,(a,b) in comparisons.items()}
        passed=all(np.allclose(a,b,rtol=2e-11,atol=1e-6 if k in ('stress','pressure') else 1e-12) for k,(a,b) in comparisons.items())
        rows.append(dict(case=name,passed=passed,maximum_absolute_error=errors,
            inputs=dict(D1=d1, mu=mu, strain=strain.tolist(), dt=dt, initial_damage=damage, compacted=compacted),
            oracle={k:np.asarray(b).tolist() for k,(a,b) in comparisons.items()},
            oracle_damage=float(history[0]),oracle_ep=float(history[1])))
    report=dict(schema='hjc-current-mat111-point-probe-v1',source_sha256=MAT111_SOURCE_SHA256,
        extracted_scalar_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        profile='mat111-compatible',scope=MAT111_PROFILE_SCOPE,cases=rows,
        all_points_passed=all(r['passed'] for r in rows), mpm_solver_executed=False,
        jax_material_points_executed=True,
        historical_finalcheck_scope='D1=.04 fixed-axis explicit-solid projectile impact only; applies to MAT111 reference, not this JAX profile',
        jax_impact_validation_passed=False, beam_validation_passed=False,
        compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0])
    (args.output/'point-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_points_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
