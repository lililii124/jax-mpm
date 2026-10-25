"""Compare sequential JAX HJC histories with the supplied MAT111 scalar path."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.probe_hjc_mat111_points import compile_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    from mpm_jax.materials.hjc import MAT111_SOURCE_SHA256, hjc, update_hjc
    from mpm_jax.materials.history import initial_history

    jax.config.update('jax_enable_x64', True)
    function, scalar = compile_reference(args.source, args.output)
    pointer = ctypes.POINTER(ctypes.c_double)
    advance = jax.jit(update_hjc)
    cases = [
        ('compaction_unload_reload', .04,
         np.r_[np.linspace(.002, .14, 30), np.linspace(.14, .03, 20), np.linspace(.03, .18, 30)],
         np.zeros(80)),
        ('compressed_shear_reversal', .04, np.full(64, .02),
         np.r_[np.full(16, .0005), np.full(32, -.0005), np.full(16, .0005)]),
        ('low_d1_active_damage', .004, np.full(64, .2),
         np.r_[np.full(32, .001), np.full(32, -.001)]),
        ('tension_recompression', .04,
         np.r_[np.linspace(-.00002, -.001, 20), np.linspace(-.001, .04, 20)], np.full(40, .0001)),
    ]
    fields = ('pressure', 'damage', 'plastic_strain', 'compaction', 'max_compression', 'plastic_volume')
    records = []
    for name, d1, compressions, shears in cases:
        dt = 1e-6
        p = hjc(profile='mat111-compatible', D1=d1)
        properties = np.array([p.shear, p.crush_pressure, p.crush_strain, p.lock_pressure,
            p.lock_strain, p.K1, p.K2, p.K3, p.tensile_strength, p.compressive_strength,
            dt, p.A, p.B, p.C, p.N, p.reference_rate, p.maximum_strength,
            p.D1, p.D2, p.minimum_fracture_strain], dtype=np.float64)
        native_history = np.zeros(7)
        native_stress = np.zeros(6)
        old_J = 1.
        current = initial_history(1, jnp.float64, p, p.density)
        inputs, expected, actual = [], [], []
        errors = {field: 0. for field in ('stress', *fields)}
        passed = True
        for mu, shear in zip(compressions, shears):
            J = 1 / (1 + mu)
            strain = np.eye(3) * (np.log(J / old_J) / 3)
            strain[0, 1] = strain[1, 0] = shear
            data = np.zeros(14)
            data[:3] = np.diag(strain) / dt
            data[3:6] = np.array([2 * shear, 0., 0.]) / dt
            data[6:12] = native_stress
            data[12] = -np.sum(native_stress[:3]) / 3
            data[13] = J
            result = np.zeros(18)
            function(*(array.ctypes.data_as(pointer) for array in
                       (properties, data, native_history, result)))
            current = advance(current, jnp.asarray(strain[None]), jnp.array([J]), dt, p)
            target = {
                'stress': np.array([[result[0], result[3], result[5]],
                                    [result[3], result[1], result[4]],
                                    [result[5], result[4], result[2]]]),
                'pressure': result[6], 'damage': native_history[0],
                'plastic_strain': native_history[1], 'compaction': native_history[2],
                'max_compression': native_history[3], 'plastic_volume': native_history[5],
            }
            for field, value in target.items():
                observed = np.asarray(getattr(current, field))[0]
                errors[field] = max(errors[field], float(np.max(np.abs(observed - value))))
                passed &= bool(np.allclose(observed, value, rtol=2e-10,
                    atol=1e-5 if field in ('stress', 'pressure') else 1e-11))
            passed &= bool(np.asarray(current.valid).all())
            inputs.append(np.r_[strain.ravel(), J])
            expected.append(np.r_[target['stress'].ravel(), *(target[field] for field in fields)])
            actual.append(np.r_[np.asarray(current.stress[0]).ravel(),
                *(float(getattr(current, field)[0]) for field in fields)])
            old_J, native_stress = J, result[:6].copy()
        output = args.output / (name + '.npz')
        np.savez_compressed(output, inputs=inputs, expected=expected, actual=actual, dt=dt, D1=d1)
        records.append(dict(case=name, file=output.name, steps=len(compressions), passed=passed,
            maximum_absolute_errors=errors, sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    root = Path(__file__).resolve().parents[1]
    report = dict(schema='hjc-mat111-sequential-reference-v1', source_sha256=MAT111_SOURCE_SHA256,
        extracted_scalar_sha256=hashlib.sha256(scalar.read_bytes()).hexdigest(),
        jax_source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('mpm_jax/materials/hjc.py', 'mpm_jax/materials/mat111_point.py')},
        compiler=subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
        jax_version=jax.__version__, fields=['stress_3x3_row_major', *fields],
        scope='Finite fixed-axis material-point paths; structural response and erosion unverified',
        caller_mapping='x(13)=-sum(previous_Cauchy_stress(1:3))/3, MAT111.f line568',
        all_passed=all(row['passed'] for row in records), cases=records)
    (args.output/'history-comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['all_passed'] else 1)


if __name__ == '__main__':
    main()
