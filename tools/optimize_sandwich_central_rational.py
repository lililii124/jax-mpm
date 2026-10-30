"""Bounded central-load fixed-mass material optimization on a frozen basis.

This is an exploratory reduced-model optimizer. It verifies the pinned three-
load freeze first, then uses only frozen load row 1 and a D2-symmetric control
parameterization. It does not perform or claim nonlinear MPM qualification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from tools.build_sandwich_rational import digest, material_fields_digest
from tools.optimize_sandwich_rational import build_frozen_model


CASE_INDEX = 1
BASELINE_MASS_KG = 1.8468


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    tmp.replace(path)


def symmetry_map(shape):
    nx, ny, nz = shape
    lookup = {}
    full_to_reduced = np.empty(nx * ny * nz, dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                orbit = (min(i, nx - 1 - i), min(j, ny - 1 - j), k)
                if orbit not in lookup:
                    lookup[orbit] = len(lookup)
                full_to_reduced[np.ravel_multi_index((i, j, k), shape)] = lookup[orbit]
    return full_to_reduced, len(lookup)


def run(config_path):
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from examples.blast_sandwich_design import Settings
    from mpm_jax.reduced_tl import newmark, response

    jax.config.update('jax_enable_x64', True)
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('version') != 1 or config.get('load_case') != 'central':
        raise ValueError('This entry point requires version 1 and the central case')
    if int(config['iterations']) < 1 or int(config['iterations']) > 12:
        raise ValueError('This exploratory task is limited to twelve SQP iterations')

    def path(key):
        return (config_path.parent / config[key]).resolve()

    freeze_path, basis_path = path('freeze_receipt'), path('basis')
    controls_path, material_path, tangent_path = path('selected_controls'), path('selected_material'), path('selected_tangent')
    output = path('output')
    output.mkdir(parents=True, exist_ok=False)
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if digest(freeze_path) != config['freeze_receipt_sha256']:
        raise ValueError('Freeze receipt hash mismatch')
    if digest(basis_path) != config['basis_sha256']:
        raise ValueError('Frozen basis artifact hash mismatch')
    if digest(controls_path) != freeze['inputs']['selected_controls']['sha256']:
        raise ValueError('Frozen selected-controls input hash mismatch')
    if digest(material_path) != freeze['inputs']['selected_material']['sha256']:
        raise ValueError('Frozen selected-material input hash mismatch')
    if digest(tangent_path) != freeze['inputs']['selected_tangent']['sha256']:
        raise ValueError('Frozen tangent input hash mismatch')
    if material_fields_digest(material_path) != freeze['material_fields_sha256']['selected']:
        raise ValueError('Frozen material fields do not match the model receipt')
    with np.load(controls_path, allow_pickle=False) as saved:
        selected_q = np.asarray(saved['q'], dtype=np.float64)
    selected_q_sha = hashlib.sha256(selected_q.tobytes()).hexdigest()
    if selected_q_sha != freeze['selected_controls_sha256']:
        raise ValueError('Frozen q949 controls do not match the freeze receipt')

    model, freeze, frozen = build_frozen_model(freeze_path, basis_path)
    original_loads = np.asarray(model['projected']['loads'])
    if original_loads.shape != (3, len(frozen['mass'])):
        raise ValueError('Expected the pinned three-load frozen model before case selection')
    np.testing.assert_allclose(original_loads.T, frozen['loads'], rtol=1e-11, atol=1e-14)
    if freeze['operator_vertex'] != 'selected' or freeze['basis']['rank'] != frozen['basis'].shape[-1]:
        raise ValueError('Frozen operator vertex or rank is inconsistent')

    # Replay the original selected-vertex operators before using q0 for exploration.
    M0, K0, mass0 = jax.jit(model['projected']['operators'])(jnp.asarray(selected_q))
    M0, K0 = np.asarray(M0), np.asarray(K0)
    replay_errors = {}
    for name, actual in (('mass', M0), ('stiffness', K0)):
        expected = frozen[name]
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        if not np.isfinite(error) or error >= 1e-9:
            raise ValueError('Frozen ' + name + ' operator replay failed')
        replay_errors[name] = error
    if abs(float(mass0) / BASELINE_MASS_KG - 1) >= 1e-10:
        raise ValueError('Selected material vertex does not satisfy fixed mass')

    settings = model['settings']
    control_shape = tuple(settings.controls)
    mapping, reduced_count = symmetry_map(control_shape)
    if mapping.size != np.size(model['q0']) or reduced_count * 4 != mapping.size:
        raise ValueError('Expected a 24x24x3 fourfold-symmetric control grid')
    geometry = model['base']['geometry']
    nx, ny, _ = geometry['plan']['cells']
    footprint = np.asarray(geometry['footprints'][CASE_INDEX]).reshape(nx, ny, 2, 2)
    footprint_symmetry_error = float(max(
        np.max(np.abs(footprint - footprint[::-1, :, ::-1, :])),
        np.max(np.abs(footprint - footprint[:, ::-1, :, ::-1])),
    ))
    if footprint_symmetry_error > 1e-13:
        raise ValueError('Central pressure footprint is not x/y reflection symmetric')

    # Bind the exact frozen three-case signature, the selected central load row,
    # and the symmetry map into a separate optimization signature.
    original_signature = model['signature']
    selected_load = original_loads[CASE_INDEX:CASE_INDEX + 1].copy()
    model['projected']['loads'] = jnp.asarray(selected_load)
    load_sha = hashlib.sha256(np.ascontiguousarray(selected_load).tobytes()).hexdigest()
    mapping_sha = hashlib.sha256(mapping.tobytes()).hexdigest()

    def signature():
        source_signature, binding = original_signature()
        binding = dict(binding)
        binding['optimization_load_selection'] = dict(
            case='central', original_frozen_load_index=CASE_INDEX,
            selected_load_sha256=load_sha, source_frozen_three_load_model=True,
            other_loads_evaluated=False)
        binding['control_symmetry'] = dict(
            group='D2 x/y reflections', full_shape=list(control_shape),
            reduced_variables=reduced_count, full_to_reduced_sha256=mapping_sha,
            diagonal_reflection='q[i,j,k] = q[nx-1-i,ny-1-j,k]')
        payload = dict(source_signature=source_signature, binding=binding)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return key, binding

    model['signature'] = signature
    source_signature, source_binding = signature()

    # q0 is the actual model's zero-control uniform material field, projected
    # through the exact symmetry map. No q949 material distribution is used as
    # the optimization starting point.
    q0_full = np.asarray(model['q0'], dtype=np.float64)
    if not np.isfinite(q0_full).all() or np.max(np.abs(q0_full)) != 0:
        raise ValueError('Expected the existing uniform q0 material initialization')
    q0_reduced = q0_full.reshape(-1)[np.unique(mapping, return_index=True)[1]]
    if not np.array_equal(q0_reduced[mapping], q0_full.reshape(-1)):
        raise ValueError('Uniform q0 does not satisfy the central symmetry map')
    q0_sha = hashlib.sha256(q0_full.tobytes()).hexdigest()

    # A continuation is explicit and pinned in the new task configuration.
    continuation = config.get('continuation')
    if continuation is None:
        start_full, start_label = q0_full.copy(), 'uniform model q0'
        start_artifact_sha = None
    else:
        if continuation.get('role') != 'saved-best-so-far-controls':
            raise ValueError('Unrecognized continuation role')
        candidate_path = path('continuation_controls')
        if digest(candidate_path) != continuation['artifact_sha256']:
            raise ValueError('Continuation controls artifact hash mismatch')
        with np.load(candidate_path, allow_pickle=False) as saved:
            if 'q' not in saved:
                raise ValueError('Continuation artifact has no q array')
            start_full = np.asarray(saved['q'], dtype=np.float64)
        start_sha = hashlib.sha256(np.ascontiguousarray(start_full).tobytes()).hexdigest()
        if start_sha != continuation['controls_sha256']:
            raise ValueError('Continuation controls numeric hash mismatch')
        if start_full.shape != q0_full.shape or not np.isfinite(start_full).all():
            raise ValueError('Continuation controls shape or values are invalid')
        _, _, start_mass = jax.jit(model['projected']['operators'])(jnp.asarray(start_full))
        if abs(float(start_mass) / BASELINE_MASS_KG - 1.) > 1e-10:
            raise ValueError('Continuation controls violate the fixed-mass invariant')
        start_label = 'saved best-so-far from ' + str(continuation['source_task'])
        start_artifact_sha = continuation['artifact_sha256']
    start_reduced = start_full.reshape(-1)[np.unique(mapping, return_index=True)[1]]
    if not np.array_equal(start_reduced[mapping], start_full.reshape(-1)):
        raise ValueError('Continuation controls do not map exactly to reduced variables')
    start_sha = hashlib.sha256(start_full.tobytes()).hexdigest()

    projected, dt, chunk = model['projected'], model['dt'], model['time_chunk']
    load = jnp.asarray(selected_load[0])

    def case_outputs(mass_matrix, stiffness):
        _, u, _, energy = newmark(mass_matrix, stiffness, load, settings, dt)
        metrics = response(projected, u, settings, chunk, model['ks_relative_error'])
        err = jnp.max(jnp.abs(energy[:, 0] + energy[:, 1] - energy[:, 2]))
        rel = err / jnp.maximum(jnp.max(jnp.abs(energy)), 1e-30)
        return jnp.asarray([metrics[1], metrics[2]]), (metrics, rel)

    @jax.jit
    def case_pullback(mass_matrix, stiffness):
        values, pullback, aux = jax.vjp(case_outputs, mass_matrix, stiffness, has_aux=True)
        seeds = (jnp.asarray([1., 0.]), jnp.asarray([0., 1.]))
        rows = [pullback(seed) for seed in seeds]
        return values, jnp.stack([r[0] for r in rows]), jnp.stack([r[1] for r in rows]), aux[0], aux[1]

    assemble = jax.jit(projected['operators'])

    @jax.jit
    def operator_pullback(q, gm, gk):
        _, pullback = jax.vjp(projected['operators'], q)
        return pullback((gm, gk, jnp.zeros(())))[0]

    cached = {}

    def evaluate_reduced(x):
        x = np.asarray(x, dtype=np.float64)
        if 'x' in cached and np.array_equal(x, cached['x']):
            return cached['value']
        qfull = x[mapping]
        M, K, mass = assemble(jnp.asarray(qfull))
        values, gm, gk, metrics, energy_error = case_pullback(M, K)
        values, metrics = np.asarray(values), np.asarray(metrics)
        full_grad = np.stack([np.asarray(operator_pullback(jnp.asarray(qfull), gm[i], gk[i]))
                              for i in range(2)])
        grad = np.stack([np.bincount(mapping, weights=row, minlength=reduced_count)
                         for row in full_grad])
        mass = float(mass)
        mass_relative = mass / BASELINE_MASS_KG - 1.
        energy_error = float(energy_error)
        if abs(mass_relative) > 1e-10:
            raise ValueError('Fixed-mass root violated its 1e-10 relative invariant')
        if not np.isfinite(values).all() or not np.isfinite(grad).all() or energy_error >= 1e-8:
            raise ValueError('Central response, gradient, or energy balance is invalid')
        result = dict(values=values, gradient=grad, metrics=metrics, mass_kg=mass,
                      mass_relative=mass_relative, energy_balance_relative=energy_error,
                      full_controls=qfull.copy(), controls_sha256=hashlib.sha256(qfull.tobytes()).hexdigest())
        cached.clear(); cached.update(x=x.copy(), value=result)
        return result

    def save_candidate(label, row, *, iteration=None, feasible=None):
        _, density, young, _ = model['base']['properties'](jnp.asarray(row['full_controls']))
        np.savez_compressed(output / (label + '-material.npz'),
            q=row['full_controls'], controls=row['full_controls'],
            positions=np.asarray(model['base']['support'].positions),
            reference_volume=np.asarray(model['base']['support'].volume), density=np.asarray(density),
            young=np.asarray(young), fraction=np.asarray(model['base']['properties'](jnp.asarray(row['full_controls']))[0]),
            designable=np.asarray(geometry['design']), skin=np.asarray(geometry['skin']),
            load_case=np.asarray('central'), source_signature=np.asarray(source_signature))
        atomic_json(output / (label + '-controls.json'), dict(
            q=row['full_controls'].tolist(), controls_sha256=row['controls_sha256'],
            load_case='central', initial_material=start_label,
            symmetry_group='D2 x/y reflections', source_signature=source_signature,
            nonlinear_blast_qualified=False, full_mpm_verified=False))
        atomic_json(output / (label + '.json'), dict(
            role='exploratory-central-only-reduced-model-candidate', iteration=iteration,
            feasible=feasible, objective=float(row['values'][0]), stress_bound=float(row['values'][1]),
            actual_peak_back_m=float(row['metrics'][3]), actual_peak_skin_Pa=float(row['metrics'][4]),
            mass_kg=row['mass_kg'], mass_relative=row['mass_relative'],
            energy_balance_relative=row['energy_balance_relative'], controls_sha256=row['controls_sha256'],
            symmetry_group='D2 x/y reflections', full_mpm_verified=False,
            nonlinear_blast_qualified=False, complete_three_case_qualification=False))

    inputs = dict(configuration_sha256=digest(config_path), freeze_receipt_sha256=digest(freeze_path),
        basis_artifact_sha256=digest(basis_path), selected_controls_sha256=selected_q_sha,
        original_source_signature=freeze['full_mpm_source_signature'], optimization_signature=source_signature,
        selected_case='central', selected_frozen_load_index=CASE_INDEX, selected_load_sha256=load_sha,
        full_frozen_load_count=3, other_load_cases_evaluated=False,
        initial_controls=start_label, initial_controls_sha256=start_sha,
        continuation_artifact_sha256=start_artifact_sha,
        symmetry_mapping_sha256=mapping_sha, reduced_control_count=reduced_count,
        entry_point_sha256=digest(__file__), operator_replay_relative_l2=replay_errors,
        frozen_mass_kg=float(mass0), central_pressure_symmetry_error=footprint_symmetry_error)
    atomic_json(output / 'optimization-inputs.json', inputs)

    initial = evaluate_reduced(start_reduced)
    init_constraints = np.asarray([1. - initial['values'][0], 1. - initial['values'][1]])
    initial_feasible = bool(np.min(init_constraints) >= 0 and
        initial['metrics'][3] <= settings.displacement_limit and
        initial['metrics'][4] <= settings.stress_limit and
        abs(initial['mass_relative']) <= 1e-10 and initial['energy_balance_relative'] < 1e-8)
    save_candidate('uniform-baseline' if continuation is None else 'continuation-start',
                   initial, iteration=0, feasible=initial_feasible)
    atomic_json(output / 'baseline.json', dict(
        role='central-load continuation start' if continuation is not None else 'uniform-q0 central-load baseline', constraints=init_constraints.tolist(),
        objective=float(initial['values'][0]), stress_bound=float(initial['values'][1]),
        mass_kg=initial['mass_kg'], metrics=initial['metrics'].tolist(),
        controls_sha256=start_sha, feasible=initial_feasible,
        full_mpm_verified=False, nonlinear_blast_qualified=False))

    direction = np.random.default_rng(20261030).normal(size=reduced_count)
    direction /= np.linalg.norm(direction)
    h = 1e-3
    plus = evaluate_reduced(q0_reduced + h * direction)
    minus = evaluate_reduced(q0_reduced - h * direction)
    analytic = initial['gradient'] @ direction
    finite_difference = (plus['values'] - minus['values']) / (2 * h)
    gradient_relative = np.abs(analytic - finite_difference) / np.maximum(
        np.maximum(np.abs(analytic), np.abs(finite_difference)), 1e-8)
    mass_probe_relative = max(abs(plus['mass_relative']), abs(minus['mass_relative']))
    gradient_check = dict(direction_seed=20261030, step=h, analytic=analytic.tolist(),
        central_difference=finite_difference.tolist(), relative_error=gradient_relative.tolist(),
        mass_probe_relative_change=mass_probe_relative,
        passed=bool(np.max(gradient_relative) < .03 and mass_probe_relative <= 1e-10),
        case='central', selected_frozen_load_index=CASE_INDEX, reduced_variables=reduced_count)
    atomic_json(output / 'gradient-check.json', gradient_check)
    if not gradient_check['passed']:
        raise ValueError('Central-only directional gradient/mass check failed')

    history = []
    best = dict(x=start_reduced.copy(), row=initial, feasible=initial_feasible)

    def constraints(x):
        r = evaluate_reduced(x)
        return np.asarray([1. - r['values'][0], 1. - r['values'][1]])

    def constraint_jac(x):
        return -evaluate_reduced(x)['gradient']

    def callback(x):
        nonlocal best
        row = evaluate_reduced(x)
        feas = bool(np.min(constraints(x)) >= 0 and row['metrics'][3] <= settings.displacement_limit
                    and row['metrics'][4] <= settings.stress_limit and abs(row['mass_relative']) <= 1e-10
                    and row['energy_balance_relative'] < 1e-8)
        if (feas and not best['feasible']) or (feas == best['feasible'] and row['values'][0] < best['row']['values'][0]):
            best = dict(x=x.copy(), row=row, feasible=feas)
            save_candidate('best-so-far', row, iteration=len(history)+1, feasible=feas)
        record = dict(iteration=len(history)+1, objective=float(row['values'][0]),
            stress_bound=float(row['values'][1]), constraints=constraints(x).tolist(), feasible=feas,
            mass_kg=row['mass_kg'], energy_balance_relative=row['energy_balance_relative'],
            controls_sha256=row['controls_sha256'])
        history.append(record)
        atomic_json(output / ('iteration-%03d.json' % len(history)), record)
        atomic_json(output / 'history.json', history)
        atomic_json(output / 'progress.json', dict(iterations_completed=len(history),
            best_feasible=best['feasible'], best_objective=float(best['row']['values'][0]),
            current=record, stopped=False))

    started = time.time()
    result = minimize(lambda x: 1000. * float(evaluate_reduced(x)['values'][0]), start_reduced,
        jac=lambda x: 1000. * evaluate_reduced(x)['gradient'][0], method='SLSQP',
        bounds=[(-12., 12.)] * reduced_count,
        constraints=[dict(type='ineq', fun=constraints, jac=constraint_jac)], callback=callback,
        options=dict(maxiter=int(config['iterations']), ftol=1e-7, disp=True))
    final_row = evaluate_reduced(np.asarray(result.x))
    final_constraints = np.asarray([1. - final_row['values'][0], 1. - final_row['values'][1]])
    final_feasible = bool(np.min(final_constraints) >= 0 and
        final_row['metrics'][3] <= settings.displacement_limit and
        final_row['metrics'][4] <= settings.stress_limit and abs(final_row['mass_relative']) <= 1e-10 and
        final_row['energy_balance_relative'] < 1e-8)
    save_candidate('final-iterate', final_row, iteration=len(history), feasible=final_feasible)
    if (final_feasible and not best['feasible']) or (final_feasible == best['feasible']
            and final_row['values'][0] < best['row']['values'][0]):
        best = dict(x=np.asarray(result.x).copy(), row=final_row, feasible=final_feasible)
        save_candidate('best-so-far', final_row, iteration=len(history), feasible=final_feasible)
    final_row = evaluate_reduced(best['x'])
    report = dict(role='exploratory central-only fixed-mass material optimization',
        optimizer_success=bool(result.success), optimizer_status=int(result.status),
        optimizer_message=str(result.message), iterations_completed=len(history),
        iteration_budget=int(config['iterations']), wall_seconds=time.time()-started,
        selected_case='central', selected_frozen_load_index=CASE_INDEX,
        other_load_cases_evaluated=False, initial_material='uniform model q0',
        initial_objective=float(initial['values'][0]), final_objective=float(final_row['values'][0]),
        best_candidate_feasible=best['feasible'], constraints=constraints(best['x']).tolist(),
        mass_kg=final_row['mass_kg'], mass_relative=final_row['mass_relative'],
        energy_balance_relative=final_row['energy_balance_relative'], controls_sha256=final_row['controls_sha256'],
        source_signature=source_signature, optimization_inputs_sha256=digest(output/'optimization-inputs.json'),
        complete_three_case_qualification=False, full_mpm_verified=False,
        nonlinear_blast_qualified=False, convergence_claim=False, global_optimum_claim=False,
        gradient_check_passed=gradient_check['passed'], frozen_operator_replay=replay_errors,
        symmetry=dict(group='D2 x/y reflections', footprint_max_error=footprint_symmetry_error,
            mapping_sha256=mapping_sha, reduced_variables=reduced_count),
        mass_constraint=dict(target_kg=BASELINE_MASS_KG, tolerance_relative=1e-10,
            finite_difference_probe_relative_change=mass_probe_relative))
    atomic_json(output/'summary.json', report)
    atomic_json(output/'progress.json', dict(iterations_completed=len(history),
        best_feasible=best['feasible'], best_objective=float(final_row['values'][0]),
        current=report, stopped=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.configuration), indent=2))


if __name__ == '__main__':
    main()
