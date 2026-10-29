"""Run fixed-mass SQP using a previously frozen rational MPM basis.

python -m tools.optimize_sandwich_rational CONFIG.json --iterations 4
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')

import numpy as np
from tools.build_sandwich_rational import digest, verify_sources


def build_frozen_model(freeze_path, basis_path):
    from examples.blast_sandwich_design import Settings, source_signature
    from examples.blast_sandwich_reduced import build_reduced
    import mpm_jax.reduced_tl as reduced

    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    if freeze.get('schema_version') != 2:
        raise ValueError('An input-bound version2 freeze receipt is required')
    verify_sources(freeze['source_sha256'])
    if digest(basis_path) != freeze['basis']['sha256']:
        raise ValueError('Frozen basis/operator artifact changed')
    with np.load(basis_path, allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in ('basis', 'nodes', 'mass', 'stiffness', 'loads', 'back')}
    V, nodes = data['basis'], data['nodes']
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError('Nonfinite frozen data')
    numeric_hash = hashlib.sha256(V.tobytes()).hexdigest()
    if numeric_hash != freeze['basis']['numeric_basis_sha256']:
        raise ValueError('Frozen numeric basis changed')
    if V.ndim != 3 or V.shape != (len(nodes), 3, freeze['basis']['rank']):
        raise ValueError('Invalid frozen basis shape')
    settings = Settings(xy_step=.01, z_step=.001, controls=(24, 24, 3), stage='fixed-mass')
    if source_signature(settings)[0] != freeze['full_mpm_source_signature']:
        raise ValueError('Frozen physics/settings do not match this optimization problem')
    original_basis, original_normalization = reduced.clamped_basis, reduced.normalize_basis

    def frozen_basis(candidate_nodes, candidate_settings, order):
        np.testing.assert_array_equal(candidate_nodes, nodes)
        _, _, fixed = original_basis(candidate_nodes, candidate_settings, order)
        if np.max(np.abs(V[fixed])) > 1e-12:
            raise ValueError('Frozen basis violates the clamped boundary')
        return V.copy(), ['frozen-physics-direction-'+str(i) for i in range(V.shape[-1])], fixed

    def identity_normalization(basis, nodal_mass):
        np.testing.assert_array_equal(basis, V)
        flat = basis.reshape(-1, V.shape[-1])
        gram = flat.T @ (np.repeat(nodal_mass, 3)[:, None] * flat)
        if np.max(np.abs(gram-np.eye(V.shape[-1]))) >= 1e-10:
            raise ValueError('Frozen reference-mass normalization changed')
        return basis, np.eye(V.shape[-1]), float(np.linalg.cond(gram))

    # Restore the ordinary basis constructors immediately after assembly.
    # This standalone entry point constructs one model at a time.
    reduced.clamped_basis, reduced.normalize_basis = frozen_basis, identity_normalization
    try:
        model = build_reduced(settings, order=2, dt=1.25e-7, time_chunk=16)
    finally:
        reduced.clamped_basis, reduced.normalize_basis = original_basis, original_normalization
    if model['projected']['basis_sha256'] != numeric_hash:
        raise ValueError('Model changed the frozen basis')
    np.testing.assert_allclose(np.asarray(model['projected']['loads']).T, data['loads'], rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(np.asarray(model['projected']['back']), data['back'], rtol=1e-11, atol=1e-14)
    return model, freeze, data


def run(configuration, *, iterations=4, resume=False, check_only=False):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from examples.blast_sandwich_design import atomic_json
    from examples.blast_sandwich_reduced import optimize_sqp
    from mpm_jax.reduced_case_pullback import configure_sqp_evaluator

    configuration = Path(configuration).resolve()
    config = json.loads(configuration.read_text(encoding='utf-8'))
    if config.get('version') != 1 or iterations < 1:
        raise ValueError('Expected configuration version1 and a positive iteration budget')
    def path(name): return (configuration.parent/config[name]).resolve()
    model, freeze, frozen = build_frozen_model(path('freeze_receipt'), path('basis'))
    control_path = path('initial_controls')
    if control_path.suffix == '.json':
        q = np.asarray(json.loads(control_path.read_text(encoding='utf-8'))['q'], dtype=np.float64)
    else:
        with np.load(control_path, allow_pickle=False) as saved:
            q = np.asarray(saved['q'], dtype=np.float64)
    control_hash = hashlib.sha256(q.tobytes()).hexdigest()
    if q.shape != model['q0'].shape or not np.isfinite(q).all() or control_hash != config['initial_controls_sha256']:
        raise ValueError('Initial control field differs from configuration')
    configure_sqp_evaluator(model, 'case-pullback')
    signature, binding = model['signature']()
    inputs = dict(configuration_sha256=digest(configuration),
                  freeze_receipt_sha256=digest(path('freeze_receipt')),
                  basis_artifact_sha256=digest(path('basis')),
                  basis_numeric_sha256=model['projected']['basis_sha256'],
                  initial_controls_sha256=control_hash, signature=signature,
                  entry_point_sha256=digest(__file__), binding=binding)
    output = path('output')
    output.mkdir(parents=True, exist_ok=True)
    if check_only:
        vertex = freeze['operator_vertex']
        if control_hash != freeze['construction_configuration'][vertex+'_controls_sha256']:
            raise ValueError('Operator replay needs the frozen material vertex controls')
        M, K, mass = jax.jit(model['projected']['operators'])(jnp.asarray(q))
        M, K = np.asarray(M), np.asarray(K)
        errors = {}
        for name, actual in (('mass', M), ('stiffness', K)):
            expected = frozen[name]
            error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            if not np.isfinite(error) or error >= 1e-9:
                raise ValueError('Frozen '+name+' operator replay failed')
            errors[name] = error
        if abs(float(mass)/1.8468-1) >= 1e-10:
            raise ValueError('Fixed-mass constraint changed')
        report = dict(passed=True, inputs=inputs, operator_relative_l2=errors, mass_kg=float(mass),
                      scope='Frozen basis, operators and public optimizer setup; no new optimization or MPM qualification.')
        atomic_json(output/'setup-check.json', report)
        return report
    receipt_path = output/'optimization-inputs.json'
    if receipt_path.exists():
        if not resume or json.loads(receipt_path.read_text(encoding='utf-8')) != inputs:
            raise ValueError('Optimization inputs changed or fresh output is required')
    elif resume:
        raise ValueError('Cannot resume without the original input receipt')
    else:
        atomic_json(receipt_path, inputs)
    result = optimize_sqp(model, output/'optimizer', iterations=iterations, resume=resume,
                          initial_controls=q, objective_scale=1000., evaluator='case-pullback')
    return dict(valid=result['valid'], feasible=result['feasible'], metrics=result['metrics'],
                best_feasible_objective=result['best_feasible_objective'],
                signature=result['signature'], convergence_claim=False, full_mpm_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.resume and args.check_only:
        parser.error('--check-only and --resume are mutually exclusive')
    print(json.dumps(run(args.configuration, iterations=args.iterations,
                         resume=args.resume, check_only=args.check_only), indent=2))


if __name__ == '__main__':
    main()
