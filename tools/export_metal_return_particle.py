"""Bounded replay-only export of the failing return particle from accepted frame 4.

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
    checked_equal = {k: np.array_equal(np.asarray(v), np.asarray(getattr(candidate, k))) for k, v in checked._asdict().items()}
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
    return_probe = expose(m.deviatoric_return, ['proper', 'symmetric', 'deviatoric', 'active', 'residual', 'tol', 'loss', 'valid', 'lo', 'hi', 'qtrial', 'dep', 'qnew'])
    _, return_raw = return_probe(old.tau, old.plastic_strain, raw['strain'], raw['rotation'], raw['new_J'], dt, raw['old_temperature'], material)
    example.save_json(out/'gates.json', dict(gates={k: bool(v) for k, v in gates.items()},
        diagnostic_candidate_equal=checked_equal, attribution_allowed=False, original_rejection_cause_established=False,
        scope='Instrumented replay-only indicative gates; never original rejection attribution',
        failed_channels=[k for k, v in gates.items() if not bool(v)],
        source_unchanged=True, production_eligible=False, physical_validation_passed=False))
    export_particle(out, old, raw, return_raw, dt, material, m, expose, jax, jnp, np, example)



def export_particle(out, old, raw, returned, dt, material, m, expose, jax, jnp, np, example):
    """Export only small inputs; 128 iterations is an in-memory hypothesis."""
    active = np.asarray(returned['active'])
    invalid = ~np.asarray(returned['valid'])
    indices = np.flatnonzero(active & invalid)
    if len(indices) == 0 or len(indices) > 16:
        raise RuntimeError('Expected one to sixteen failed active replay return particles')
    original = m.deviatoric_return
    source = inspect.getsource(original)
    before = 'jax.lax.fori_loop(0,64,bisect,'
    assert source.count(before) == 1
    candidate_source = source.replace(before, 'jax.lax.fori_loop(0,128,bisect,')
    namespace = dict(original.__globals__)
    # Source files remain unchanged. This function is a separate numerical
    # hypothesis and its outputs never replace the accepted/rejected replay.
    import linecache
    candidate_filename = '<128-bisection-hypothesis>'
    linecache.cache[candidate_filename] = (len(candidate_source), None, candidate_source.splitlines(True), candidate_filename)
    exec(compile(candidate_source, candidate_filename, 'exec'), namespace)
    names = ['proper', 'symmetric', 'deviatoric', 'active', 'residual', 'tol', 'loss', 'valid', 'lo', 'hi', 'qtrial', 'dep', 'qnew']
    original_single = expose(original, names)
    candidate_single = expose(namespace['deviatoric_return'], names)
    reports = []
    for index in indices:
        index = int(index)
        case = dict(tau=np.asarray(old.tau[index]), ep=np.asarray(old.plastic_strain[index]),
            strain=np.asarray(raw['strain'][index]), rotation=np.asarray(raw['rotation'][index]),
            J=np.asarray(raw['new_J'][index]), dt=np.asarray(dt), T=np.asarray(raw['old_temperature'][index]))
        example.save_npz(out/f'particle-{index}.npz', **case)
        inputs = tuple(jnp.asarray(case[name]) for name in ('tau', 'ep', 'strain', 'rotation', 'J', 'dt', 'T'))+(material,)
        step64, checks64 = original_single(*inputs)
        step128, checks128 = candidate_single(*inputs)
        step128.kirchhoff_stress.block_until_ready()
        scalar = lambda data: {name: np.asarray(value).tolist() for name, value in data.items()}
        report = dict(particle_index=index, material=material._asdict(),
            input_shapes={name:list(value.shape) for name,value in case.items()},
            input_sha256=example.digest(out/f'particle-{index}.npz'),
            batch_replay64=scalar({name:value[index] for name,value in returned.items()}),
            separately_compiled_single64=scalar(checks64), separately_compiled_single128=scalar(checks128),
            single64_valid=bool(step64.valid), single128_valid=bool(step128.valid),
            hypothesis_source64_sha256=hashlib.sha256(source.encode()).hexdigest(),
            hypothesis_source128_sha256=hashlib.sha256(candidate_source.encode()).hexdigest(),
            original_rejection_cause_established=False, attribution_allowed=False,
            scope='Exact small inputs from this indicative replay only. Single-particle compilation can differ from batch replay. The in-memory128-iteration hypothesis leaves source files, material, EOS and every original tolerance unchanged; it is not an admitted impact-prefix correction.')
        example.save_json(out/f'particle-{index}.json', report)
        reports.append(dict(index=index, single64_valid=bool(step64.valid), single128_valid=bool(step128.valid)))
    example.save_json(out/'particle-export.json', dict(indices=[int(i) for i in indices], results=reports,
        full_old_state_exported=False, original_rejection_cause_established=False, numerical_source_files_unchanged=True))


if __name__ == '__main__':
    main()
