"""Preserved125ns short original/probe equivalence controller; no numerical changes."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_fields(original, probe):
    """Prospectively fixed float64 tolerance; static/discrete carries exact."""
    result = {}
    for name, a in original._asdict().items():
        a = np.asarray(a); b = np.asarray(getattr(probe, name))
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Field shape/dtype/finite mismatch: '+name)
        exact = a.tobytes() == b.tobytes()
        discrete = a.dtype.kind in 'biu' or name in ('mass', 'volume', 'time')
        tolerance = 64*np.finfo(np.float64).eps*max(1e-12, float(np.max(np.abs(a), initial=0.)))
        error = float(np.max(np.abs(a.astype(float)-b.astype(float)), initial=0.))
        accepted = exact if discrete else error <= tolerance
        result[name] = dict(bitwise_equal=exact, max_abs_error=error, tolerance=tolerance, accepted=bool(accepted))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    if args.output.exists():
        raise FileExistsError('Keep all earlier evidence')
    args.output.mkdir(parents=True)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    parent = args.parent.resolve(); cp_path = parent/'result/frame-001.npz'
    for name, expected in admission['parent_files'].items():
        if sha(parent/name) != expected:
            raise ValueError('Pinned parent differs: '+name)
    before = {name: sha(name) for name in admission['numerical_source']}
    if before != admission['numerical_source'] or sha('mpm_jax/fragment_v6_energy_probe.py') != admission['probe_sha256']:
        raise ValueError('Frozen source differs')
    with np.load(cp_path, allow_pickle=False) as saved:
        with np.load(parent/'result/checkpoint.npz', allow_pickle=False) as static:
            reference, edges = static['reference'], static['edges']
        # At0ns every bond is alive. Original formal driver performs its first
        # post-initial split at125ns; later500ns alive must never be restored.
        alive = np.ones(len(edges), dtype=bool)
        initial_fields = {k: dict(shape=list(saved[k].shape), dtype=str(saved[k].dtype), sha256=hashlib.sha256(saved[k].tobytes()).hexdigest()) for k in saved.files}
        if {k:v['sha256'] for k,v in initial_fields.items()} != admission['initial_array_sha256']:
            raise ValueError('Planned saved array hashes differ')
        with np.load(parent/'result/frame-000.npz', allow_pickle=False) as zero:
            if not np.array_equal(reference,zero['x']) or np.any(zero['damage']) or not np.array_equal(saved['physical_body'],zero['physical_body']) or len(np.unique(saved['body_id'])) != 2:
                raise ValueError('Original first epoch topology reconstruction differs')
        if sha(cp_path) != admission['checkpoint_sha256'] or not bool(saved['valid']) or float(saved['time']) != 125000*1e-12:
            raise ValueError('Exact valid125ns saved carry required')
    write('saved-initial-arraywise.json', dict(checkpoint_sha256=sha(cp_path), fields=initial_fields, source_before=before, runtime_gate_pending=True))
    if args.preflight_only:
        return
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import jaxlib
    from mpm_jax import fragment_multifield_v6_mpm as solver
    from mpm_jax import fragment_v6_energy_probe as probe
    from mpm_jax.materials.linear_metal import linear_metal
    from examples.metal_fragment_multifield_compact import split_fields
    runtime = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    if runtime != admission['runtime'] or runtime != json.loads((parent/'result/runtime.json').read_text()):
        raise ValueError('Exact producer runtime required')
    assumptions = json.loads((parent/'result/assumptions.json').read_text())
    if (assumptions['topology_epoch_ns'], assumptions['contact_iterations'], assumptions['contact_tolerance_m_s']) != (125., 1024, .005):
        raise ValueError('Unchanged physics protocol required')
    material = linear_metal(**assumptions['geometry']['material'])
    with np.load(cp_path, allow_pickle=False) as cp:
        if set(cp.files) != set(solver.ImpactState._fields)|{'physical_body'} or not all(np.isfinite(cp[k]).all() for k in cp.files):
            raise ValueError('Parent carry/finite fields differ')
        state = solver.ImpactState(*(jnp.asarray(cp[k]) for k in solver.ImpactState._fields))
        physical = cp['physical_body']
        if len(solver.ImpactState._fields) != 35 or any(np.asarray(getattr(state,k)).tobytes() != cp[k].tobytes() for k in solver.ImpactState._fields):
            raise ValueError('All35 initial state arrays must equal saved carry bitwise')
    if not bool(state.valid) or round(float(state.time)*1e12) != 125000 or abs(float(state.time)-125000*1e-12)>1e-18 or len(state.x) != 26336:
        raise ValueError('Only exact125ns formal accepted state')
    # Same next physical epoch as original continuation, not a shortened10ns pool.
    x = np.asarray(state.x); labels, alive, bodies = split_fields(x, reference, edges, alive, np.asarray(state.damage), physical, max_fields=8192)
    predicted = x+(125000*1e-12)*np.asarray(state.v); dx = .00025
    lower = np.full((len(bodies),3), np.inf); upper = np.full_like(lower, -np.inf)
    np.minimum.at(lower, labels, np.minimum(x,predicted)); np.maximum.at(upper, labels, np.maximum(x,predicted))
    bounds = np.stack((lower-2*dx, upper+2*dx), axis=-1)
    grid = solver.build_grid(x, labels, dx=dx, padding_cells=3, reference_bounds=bounds, max_nodes=2_000_000, field_physical_body=bodies, contact_sweeps=(x,predicted))
    if len(grid.pairs)>2_000_000:
        raise ValueError('Candidate pair cap')
    fixed = (physical==1)&((np.abs(reference[:,0])>=.01-.001)|(np.abs(reference[:,2])>=.01-.001))
    support = solver.stencil(jnp.asarray(x), jnp.asarray(labels), grid.layout)
    mask = np.zeros((grid.layout.node_count,3),bool); nodes = np.asarray(support.indices)[fixed].ravel(); nodes = nodes[(nodes>=0)&(nodes<grid.layout.node_count)]; mask[nodes] = True
    grid = replace(grid, clamped=jnp.asarray(mask))
    before_epoch_reset = {k:np.asarray(v).copy() for k,v in state._asdict().items()}
    state = state._replace(body_id=jnp.asarray(labels), contact_lambda=jnp.zeros(len(grid.pairs), dtype=state.x.dtype), contact_cache_dt=jnp.asarray(0.))
    epoch_changes = [k for k,v in state._asdict().items() if np.asarray(v).tobytes() != before_epoch_reset[k].tobytes()]
    if not set(epoch_changes).issubset({'body_id','contact_lambda','contact_cache_dt'}):
        raise ValueError('Unexpected original epoch reset mutation')
    write('epoch-reset.json',dict(changed_fields=epoch_changes, preserved_other_32_fields=True, original_epoch_start_ns=125, original_epoch_end_ns=250))
    solver.validate_state(state,material,grid)
    np.savez(args.output/'epoch-initial.npz',**{k:np.asarray(v) for k,v in state._asdict().items()},physical_body=physical,reference=reference,edges=edges,alive=alive)
    write('grid-spec.json',dict(epoch_start_ns=125,epoch_end_ns=250,target_ns=135,dx_m=dx,nodes=grid.layout.node_count,pairs=len(grid.pairs),pairs_ordered_bytes_sha256=hashlib.sha256(np.asarray(grid.pairs).tobytes()).hexdigest(),fields=len(bodies),source_before=before))
    config = solver.ImpactConfig(contact_iterations=1024, contact_tolerance=.005)
    target = jnp.asarray(135000*1e-12); begun = time.monotonic()
    original, n0 = solver.advance_to(state,material,grid,config,target,max_steps=512)
    original.x.block_until_ready()
    np.savez(args.output/'original-final.npz', **{k:np.asarray(v) for k,v in original._asdict().items()})
    diagnosed, n1, total, absolute, maximum = probe.advance_to_diagnosed(state,material,grid,config,target,max_steps=512)
    diagnosed.x.block_until_ready()
    np.savez(args.output/'probe-final.npz', **{k:np.asarray(v) for k,v in diagnosed._asdict().items()})
    fields = compare_fields(original, diagnosed); total = np.asarray(total); absolute = np.asarray(absolute)
    delta = float(diagnosed.numerical_reservoir-state.numerical_reservoir)
    tolerance = 1e-10*141.4599609375
    passed = bool(original.valid)&bool(diagnosed.valid)&(int(n0)==int(n1)==8)&(len(fields)==35)&all(v['bitwise_equal'] for v in fields.values())&(float(maximum)<=tolerance)&(abs(delta-total[5])<=tolerance)&(abs(total[5]-total[:5].sum())<=tolerance)
    for candidate in (original,diagnosed):
        solver.validate_state(candidate,material,grid)
        passed = passed and float(candidate.maximum_contact_closing_speed)<=.005 and float(candidate.maximum_contact_complementarity_residual)<=.005
    after = {name:sha(name) for name in admission['numerical_source']}
    passed = passed and after==before and sha('mpm_jax/fragment_v6_energy_probe.py')==admission['probe_sha256']
    write('equivalence.json', dict(admitted=bool(passed), runtime=runtime, steps_original=int(n0), steps_probe=int(n1), fields=fields, signed_J=dict(zip(probe.CHANNELS,total.tolist())), absolute_J=dict(zip(probe.CHANNELS,absolute.tolist())), maximum_step_identity_residual_J=float(maximum), delta_numerical_reservoir_J=delta, energy_identity_tolerance_J=tolerance, elapsed_s=time.monotonic()-begun, nodes=grid.layout.node_count, pairs=len(grid.pairs), parent_checkpoint_sha256=sha(cp_path), numerical_source_unchanged=after==before, public_media_allowed=False, production_restart_allowed=False, scope='Probe attribution admission only125..135ns; cannot explain the whole125..250ns span or earlier loss; neither output is a production checkpoint.'))
    if not passed:
        raise RuntimeError('Original/probe equivalence or energy attribution failed; raw states retained')


if __name__ == '__main__':
    main()
