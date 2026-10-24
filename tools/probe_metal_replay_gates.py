"""Indicative replay-only gate probe from original accepted metal frame 4.

Run with cwd/PYTHONPATH at the immutable producer checkout. Never resumes the
failed run, changes its receipt, or writes to its cache. Instrumented verdicts
are never attributed to the original rejection, including on exact equality.
"""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

PRODUCER = '07f35f678d60c881701d22dfd0ab4629ea8ee6c4'
RECEIPT = '561fc8257641b42e8db37dea2317e15e681646131498996e30f5b486cb33b889'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cache, out = a.cache.resolve(), a.output.resolve()
    if out == cache or cache in out.parents or out.exists():
        raise ValueError('Use a new output outside the immutable cache')
    if sys.platform != 'linux':
        raise RuntimeError('Numerical replay is remote Linux only')
    # Physical private-memory ceiling, avoiding JAX thread-stack reservations.
    # The workflow must also monitor the process tree and impose a timeout.
    import threading
    import time
    import psutil
    def memory_watch():
        process = psutil.Process()
        while True:
            if process.memory_full_info().uss > 6*1024**3:
                os.write(2, b'Metal diagnostic exceeded 6 GiB USS; terminated\n')
                os._exit(137)
            time.sleep(1)
    threading.Thread(target=memory_watch, daemon=True).start()
    signal.alarm(1200)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == PRODUCER
    assert sha(cache/'receipt.json') == RECEIPT
    receipt = json.loads((cache/'receipt.json').read_text())
    for name, record in receipt['files'].items():
        file = (cache/name).resolve()
        assert cache in file.parents and file.stat().st_size == record['bytes'] and sha(file) == record['sha256'], name
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import jax
    import jax.numpy as jnp
    import jaxlib
    from dataclasses import replace
    from examples import metal_sphere_impact as example
    from mpm_jax import impact_mpm as m
    from mpm_jax.materials.shock_metal import shock_metal
    jax.config.update('jax_enable_x64', True)
    folder = cache/'result'
    plan = json.loads((folder/'plan.json').read_text())
    assert plan['source'] == example.source_record()
    runtime = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
                   jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    saved_receipt = json.loads((folder/'checkpoint.json').read_text())
    for name, digest in saved_receipt['files'].items():
        assert sha(folder/name) == digest, name
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as cp:
        assert json.loads(str(cp['runtime'])) == runtime and str(cp['signature']) == signature
        assert int(cp['frame']) == 4 and int(cp['steps']) == 169
        state = m.ImpactState(*(jnp.asarray(cp['state_'+name]) for name in m.ImpactState._fields))
    assert bool(state.valid) and float(state.time) == plan['output_times_s'][4]
    with np.load(folder/'reference.npz', allow_pickle=False) as f:
        geometry = {name: f[name] for name in f.files}
    assert {name: example.descriptor(v) for name, v in geometry.items()} == plan['geometry_fields']
    gp = plan['grid']
    grid = m.build_grid(geometry['x'], geometry['body_id'], dx=gp['dx_m'], padding_cells=gp['padding_cells'],
                        reference_bounds=gp['bounds_m'], max_nodes=gp['max_nodes'])
    assert grid.layout.node_count == gp['nodes']
    assert np.array_equal(np.asarray(grid.layout.shape), gp['shapes'])
    assert np.array_equal(np.asarray(grid.layout.lower), gp['lower'])
    shape, low = np.asarray(grid.layout.shape)[1], np.asarray(grid.layout.lower)[1]
    xx, zz = (low[0]+np.arange(shape[0]))*gp['dx_m'], (low[2]+np.arange(shape[2]))*gp['dx_m']
    half, width = np.asarray(plan['definition']['plate']['size_xz_m'])/2, plan['definition']['boundary']['clamp_width_m']
    edge = np.broadcast_to((np.abs(xx) >= half[0]-width)[:, None, None] | (np.abs(zz) >= half[1]-width)[None, None, :], tuple(shape)).ravel()
    clamped = np.zeros((grid.layout.node_count, 3), bool)
    start = int(np.asarray(grid.layout.offsets)[1])
    clamped[start:start+len(edge)] = edge[:, None]
    grid = replace(grid, clamped=jnp.asarray(clamped))
    material, config = shock_metal(**plan['material']), m.ImpactConfig(**plan['config'])
    m.validate_state(state, material, grid)
    target = jnp.asarray(plan['output_times_s'][5])
    out.mkdir(parents=True)
    # Preserve the original _step expression; extra carry remembers only the
    # preceding state. Exact comparison below detects compilation differences.
    @jax.jit
    def replay(initial):
        def condition(c):
            current, previous, dt, count = c
            return current.valid & (current.time < target) & (count < 200)
        def body(c):
            current, previous, dt, count = c
            dt = jnp.minimum(m.stable_timestep(current, material, grid, config), target-current.time)
            return m._step(current, material, grid, config, dt)[0], current, dt, count+1
        return jax.lax.while_loop(condition, body, (initial, initial, jnp.asarray(0.), jnp.asarray(0)))
    candidate, old, dt, count = replay(state)
    candidate.x.block_until_ready()
    for name, value in [('old', old), ('candidate', candidate)]:
        example.save_npz(out/(name+'.npz'), **{k: np.asarray(v) for k, v in value._asdict().items()})
    with np.load(folder/'failed_state.npz', allow_pickle=False) as original:
        equal = {k: np.array_equal(np.asarray(v), original[k]) for k, v in candidate._asdict().items()}
    example.save_json(out/'replay.json', dict(producer=PRODUCER, runtime=runtime, original_receipt_sha256=RECEIPT,
        dt=float(dt), substeps=int(count), target=float(target), original_candidate_equal=equal,
        original_rejection_cause_established=False,
        scope='Replay-only indicative diagnostic; original candidate differs may reflect compiler fusion. No original-cause attribution. Phase extrapolation is not validated phase physics. Never resumable.'))
    if bool(candidate.valid) or int(count) >= 200:
        raise RuntimeError('Replay did not reach a rejected substep within its budget')
    # Expose each original conjunction independently through an AST-only return
    # extension. No assignment, gate, tolerance or numerical source is changed.
    tree = ast.parse(inspect.getsource(m._step))
    function = tree.body[0]
    valid_assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'valid' for t in node.targets))
    terms = []
    def split(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            split(node.left); split(node.right)
        else:
            terms.append(node)
    split(valid_assignment.value)
    expressions = {ast.unparse(node): node for node in terms}
    extras = "{'limit': limit, 'dt': dt, 'energy_residual': residual, 'energy_scale': energy_scale, 'momentum_residual': momentum_residual, 'momentum_scale': momentum_scale, 'contact_valid': contact_valid, 'material_valid': material_valid, 'old_eos_valid': old_thermo.valid, 'new_eos_valid': thermo.valid, 'returned_valid': returned.valid, 'before_contact': before_contact, 'mass': mass, 'normals': contact_normals(state, mass, support, grid), 'rotation': rotation, 'strain': strain, 'new_J': new_J, 'old_temperature': old_thermo.temperature}"
    gates = ast.Dict(keys=[ast.Constant(k) for k in expressions], values=list(expressions.values()))
    function.body[-1].value = ast.Tuple(elts=[ast.Name(id='current', ctx=ast.Load()), gates, ast.parse(extras, mode='eval').body], ctx=ast.Load())
    namespace = dict(m.__dict__)
    exec(compile(ast.fix_missing_locations(tree), '<isolated-metal-diagnostic>', 'exec'), namespace)
    checked, gates, raw = jax.jit(namespace['_step'], static_argnames=('config',))(old, material, grid, config, dt)
    checked.x.block_until_ready()
    example.save_npz(out/'instrumented_candidate.npz', **{k: np.asarray(v) for k, v in checked._asdict().items()})
    checked_equal = {k: np.array_equal(np.asarray(v), np.asarray(getattr(candidate, k))) for k, v in checked._asdict().items()}
    example.save_npz(out/'constituents.npz', **{k: np.asarray(v) for k, v in raw.items()})
    # Nested validity inputs and intermediate arrays, independently evaluated;
    # these are raw diagnostics and do not replace the original outer verdict.
    def expose(function, names):
        nested = ast.parse(inspect.getsource(function))
        node = nested.body[0]
        node.body[-1].value = ast.Tuple(elts=[node.body[-1].value,
            ast.Dict(keys=[ast.Constant(k) for k in names], values=[ast.Name(id=k, ctx=ast.Load()) for k in names])], ctx=ast.Load())
        ns = dict(function.__globals__)
        exec(compile(ast.fix_missing_locations(nested), '<nested-diagnostic>', 'exec'), ns)
        return jax.jit(ns[function.__name__])
    contact_probe = expose(m.paired_contact, ['occupied', 'closing', 'constrained_normal', 'direction_floor', 'valid'])
    _, contact_raw = contact_probe(raw['before_contact'], raw['mass'], raw['normals'], grid.pairs, grid.clamped)
    return_probe = expose(m.deviatoric_return, ['proper', 'symmetric', 'deviatoric', 'active', 'residual', 'tol', 'loss', 'valid'])
    _, return_raw = return_probe(old.tau, old.plastic_strain, raw['strain'], raw['rotation'], raw['new_J'], dt, raw['old_temperature'], material)
    example.save_npz(out/'contact_checks.npz', **{k: np.asarray(v) for k, v in contact_raw.items()})
    example.save_npz(out/'return_checks.npz', **{k: np.asarray(v) for k, v in return_raw.items()})
    example.save_json(out/'gates.json', dict(gates={k: bool(v) for k, v in gates.items()},
        diagnostic_candidate_equal=checked_equal, attribution_allowed=False, original_rejection_cause_established=False,
        scope='Instrumented replay-only indicative gates; never original rejection attribution',
        failed_channels=[k for k, v in gates.items() if not bool(v)],
        source_unchanged=True, production_eligible=False, physical_validation_passed=False))
    write_probe_report(out, folder, candidate, checked, raw, gates, contact_raw, return_raw, dt, np, example, old)


def write_probe_report(out, folder, candidate, checked, raw, gates, contact_raw, return_raw, dt, np, example, old=None):
    """Quantify comparisons and gate distances without an original-cause claim."""
    def compare(left, right):
        result = {}
        for name, value in left._asdict().items():
            a = np.asarray(value)
            b = np.asarray(right[name] if isinstance(right, dict) else getattr(right, name))
            assert a.shape == b.shape and a.dtype == b.dtype
            byte_a = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
            byte_b = np.ascontiguousarray(b).view(np.uint8).reshape(-1, b.dtype.itemsize)
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
            scale = max(float(np.abs(a.astype(np.float64)).max()), float(np.abs(b.astype(np.float64)).max()), 1e-300)
            result[name] = dict(shape=list(a.shape), dtype=str(a.dtype), elements=int(a.size),
                bitwise_mismatch_count=int(np.any(byte_a != byte_b, axis=1).sum()),
                numeric_mismatch_count=int(np.count_nonzero(a != b)), maxabs=float(delta.max()),
                maxabs_relative_to_global_field_scale=float(delta.max())/scale,
                both_finite=bool(np.isfinite(a).all() and np.isfinite(b).all()))
        return result
    # Load a single original field at a time to limit private-memory peaks.
    with np.load(folder/'failed_state.npz', allow_pickle=False) as original:
        original_vs_replay = {}
        original_vs_probe = {}
        for name in candidate._fields:
            from collections import namedtuple
            field = namedtuple('Field', [name])
            reference = {name: original[name]}
            original_vs_replay.update(compare(field(getattr(candidate, name)), reference))
            original_vs_probe.update(compare(field(getattr(checked, name)), reference))
    probe_vs_replay = compare(checked, candidate)
    example.save_json(out/'comparisons.json', dict(original_vs_replay=original_vs_replay,
        original_vs_instrumented_probe=original_vs_probe, instrumented_probe_vs_replay=probe_vs_replay,
        original_rejection_cause_established=False, scope='Every mismatch retained; no tolerance admits original-cause attribution.'))
    def threshold_ratio(value, threshold):
        value, threshold = float(value), float(threshold)
        ratio = value/threshold if threshold > 0 else None
        return dict(value=value, threshold=threshold, ratio=ratio,
            passed=value <= threshold, signed_margin=threshold-value,
            descriptive_band=('within 1e-6 relative of threshold' if ratio is not None and abs(ratio-1) <= 1e-6
                              else 'above threshold by factor >10' if ratio is not None and ratio > 10
                              else 'below threshold by factor >10' if ratio is not None and ratio < .1
                              else 'intermediate distance from threshold'))
    margins = dict(
        energy=threshold_ratio(abs(float(raw['energy_residual'])), 1e-10*float(raw['energy_scale'])),
        momentum=threshold_ratio(np.linalg.norm(np.asarray(raw['momentum_residual'])), 1e-10*float(raw['momentum_scale'])),
        cfl=threshold_ratio(float(dt), float(raw['limit'])*(1+1e-12)))
    active = np.asarray(return_raw['active'])
    residual, tol = np.abs(np.asarray(return_raw['residual'])), np.asarray(return_raw['tol'])
    margins['active_return_residual'] = dict(active_particles=int(active.sum()),
        failing_active_particles=int(np.count_nonzero(active & (residual > tol))),
        maximum_active_residual_threshold_ratio=float(np.max(np.where(active, residual/tol, 0))),
        minimum_active_signed_margin=float(np.min((tol-residual)[active])) if active.any() else None)
    return_checks = {name: dict(failing_particles=int(np.count_nonzero(~np.asarray(return_raw[name]))))
                     for name in ('proper', 'symmetric', 'deviatoric', 'valid')}
    if old is not None:
        tau = np.asarray(old.tau)
        symmetric_error = np.max(np.abs(tau-np.swapaxes(tau, -1, -2)), axis=(-2, -1))
        trace_error = np.abs(np.trace(tau, axis1=-2, axis2=-1))
        rotation = np.asarray(raw['rotation'])
        rotation_tol = max(1e-10, 128*np.finfo(rotation.dtype).eps)
        orthogonal_error = np.max(np.abs(np.swapaxes(rotation, -1, -2)@rotation-np.eye(3)), axis=(-2, -1))
        determinant_error = np.abs(np.linalg.det(rotation)-1)
        for key, error, threshold in [('stress_symmetry', symmetric_error, tol),
                                      ('stress_trace', trace_error, tol),
                                      ('rotation_orthogonality', orthogonal_error, rotation_tol),
                                      ('rotation_determinant', determinant_error, rotation_tol)]:
            margins[key] = dict(maximum_error_threshold_ratio=float(np.max(error/threshold)),
                minimum_signed_margin=float(np.min(threshold-error)),
                comparison='strict error < threshold; NumPy recomputation is an additional indicative diagnostic')
    occupied = np.asarray(contact_raw['occupied'])
    normal_length = np.linalg.norm(np.asarray(raw['normals']), axis=-1)
    contact_checks = dict(occupied_pairs=int(occupied.sum()),
        occupied_pairs_with_zero_normal=int(np.count_nonzero(occupied & (normal_length == 0))),
        occupied_closing_constrained_pairs=int(np.count_nonzero(occupied & np.asarray(contact_raw['closing']) & np.asarray(contact_raw['constrained_normal']))),
        minimum_occupied_normal_length=float(normal_length[occupied].min()) if occupied.any() else None,
        nested_contact_valid=bool(contact_raw['valid']), outer_probe_contact_valid=bool(raw['contact_valid']))
    example.save_json(out/'probe-report.json', dict(
        schema='metal-replay-indicative-gates-v1', original_rejection_cause_established=False,
        attribution_allowed=False, production_eligible=False, physical_validation_passed=False,
        failed_replay_probe_channels=[k for k, v in gates.items() if not bool(v)],
        gates={k: bool(v) for k, v in gates.items()}, thresholds=margins,
        return_checks=return_checks, contact_checks=contact_checks,
        nested_return_valid_equal_outer_probe=bool(np.array_equal(np.asarray(return_raw['valid']), np.asarray(raw['returned_valid']))),
        scope='Gates and margins belong only to the separately compiled instrumented replay step. Original monolithic advance_to and added-carry replay can change compiler fusion and rounding. Descriptive threshold distances are heuristics, not error bounds or evidence of original failure cause. No gate or tolerance was edited. Phase extrapolation is not validated phase physics.'))


if __name__ == '__main__':
    main()
