"""Run a source-bound full-MPM sandwich trajectory with optional force blocks."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
from types import FunctionType

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')

import numpy as np
from examples.blast_sandwich_design import (
    CASE_NAMES, ROW_FIELDS, Settings, atomic_json, atomic_npz,
    build_model, source_signature)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _with_blocks(model, block_size):
    # Clone only the function bindings: boundary and time-integration code stay exact.
    import mpm_jax.compact_tl as compact
    from mpm_jax.blocked_elastic import make_internal_force

    def internal(F, history, reference, material, config, body):
        if history is not None or config.material != 'elastic':
            raise ValueError('This execution mode requires reference-grid elastic material')
        return make_internal_force(reference, material, block_size)(F)

    step = compact.step
    replacement = FunctionType(step.__code__, dict(step.__globals__, forces=internal),
                               step.__name__, step.__defaults__, step.__closure__)
    advance = model['advance']
    if 'step' not in advance.__code__.co_freevars:
        raise ValueError('Unexpected integrator closure')
    def cell(value):
        return (lambda: value).__closure__[0]
    cells = tuple(cell(replacement) if name == 'step' else value
                  for name, value in zip(advance.__code__.co_freevars, advance.__closure__))
    new_advance = FunctionType(advance.__code__, advance.__globals__, advance.__name__,
                              advance.__defaults__, cells)
    return dict(model, advance=new_advance)


def _forward(model, q, case, output, execution, *, resume=False, stop_step=None, save_every=100):
    import jax
    import jax.numpy as jnp
    import jaxlib
    from mpm_jax.total_lagrangian import TLState, sample_nodal, deformation
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature, sources = source_signature(model['settings'])
    binding = dict(source_signature=signature, controls=np.asarray(q).tolist(), case=CASE_NAMES[case],
                   jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__, execution=execution)
    binding_text = json.dumps(binding, sort_keys=True)
    if np.asarray(q).shape != model['q0'].shape or not np.isfinite(q).all():
        raise ValueError('Invalid control shape or values')
    phi, density, young, material = model['properties'](jnp.asarray(q))
    reference = model['reference_grid'](model['support'], density)
    state = model['initial_state'](reference, material, model['pressure'](0)*model['load_shape'][case])
    aggregate = model['aggregate_initial']()
    start = 0
    rows = [np.zeros(len(ROW_FIELDS))]
    rows[0][7] = rows[0][-1] = 1.
    path = output/'checkpoint.npz'
    if resume:
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['binding']) != binding_text:
                raise ValueError('Resume requires identical geometry, design, source and runtime')
            values = {}
            for k, template in state._asdict().items():
                if k == 'history':continue
                value = saved['state_'+k]
                if value.shape != template.shape or value.dtype != template.dtype or not np.isfinite(value).all():
                    raise ValueError('Invalid saved state field: '+k)
                values[k] = jnp.asarray(value)
            state = TLState(**values, history=None)
            aggregate = jnp.asarray(saved['aggregate'])
            start = int(saved['step'])
            rows = list(saved['rows'])
            if (not 0 <= start <= model['plan']['steps'] or len(rows) != start+1
                    or aggregate.shape != (14,) or not np.isfinite(aggregate).all()
                    or not bool(state.valid) or not np.isfinite(saved['rows']).all()
                    or abs(float(state.time)-start*model['plan']['dt_s']) > 1e-10*model['settings'].horizon):
                raise ValueError('Incomplete or invalid forward checkpoint')
            if not np.allclose(state.F,deformation(state.displacement,reference),rtol=0,atol=1e-13):
                raise ValueError('Saved deformation differs from its nodal displacement')
    elif any(output.iterdir()):
        raise ValueError('Use a fresh output directory or --resume')
    limit = model['plan']['steps'] if stop_step is None else stop_step
    if not start <= limit <= model['plan']['steps'] or save_every < 1:
        raise ValueError('Stop step must lie within the unchanged absolute time plan')
    one = jax.jit(lambda state, aggregate, n: model['advance'](state, aggregate, reference, material, case, n))
    def save(step_number):
        payload = {'state_'+k: np.asarray(v) for k, v in state._asdict().items() if k != 'history'}
        atomic_npz(path, **payload, step=step_number, aggregate=np.asarray(aggregate),
                   rows=np.asarray(rows), binding=np.asarray(binding_text))
        temporary = output/'history.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream);writer.writerow(ROW_FIELDS);writer.writerows(rows)
        temporary.replace(output/'history.csv')
        displacement = np.asarray(sample_nodal(state.displacement, reference))
        atomic_npz(output/f'frame_{step_number:07d}.npz', reference=np.asarray(reference.positions),
                   position=np.asarray(reference.positions)+displacement, displacement=displacement,
                   F=np.asarray(state.F), phi=np.asarray(phi), density=np.asarray(density),
                   young=np.asarray(young), volume=np.asarray(reference.volume), skin=model['geometry']['skin'],
                   time=np.asarray(state.time))
        atomic_json(output/'status.json', dict(step=step_number, planned_steps=model['plan']['steps'],
            completed=step_number == model['plan']['steps'], valid=bool(state.valid),
            time_s=float(state.time), binding=binding, qualification=model['plan']['qualification']))
    if not resume:save(0)
    for n in range(start, limit):
        state, aggregate, row = one(state, aggregate, jnp.asarray(n))
        row = np.asarray(row)
        if not np.isfinite(row).all() or not bool(state.valid):
            raise RuntimeError('Nonfinite state or nonpositive Jacobian; last saved complete checkpoint retained')
        rows.append(row)
        if (n+1)%save_every == 0 or n+1 == limit:save(n+1)
    result = dict(case=CASE_NAMES[case], completed=limit == model['plan']['steps'], steps=limit,
        time_s=float(state.time), mass_kg=float(reference.particle_mass.sum()),
        peak_back_displacement_m=float(aggregate[5]), peak_skin_stress_Pa=float(aggregate[6]),
        minimum_J=float(aggregate[7]), maximum_energy_residual_J=float(aggregate[8]),
        pressure_impulse_z_Ns=float(aggregate[11]), support_impulse_z_Ns=float(aggregate[12]),
        peak_back_time_s=float(aggregate[13]), source_sha256=sources,
        nonlinear_blast_qualified=False, plan=model['plan'], execution=execution)
    atomic_json(output/'summary.json', result)
    return result

def run(configuration, *, resume=False, stop_step=None, import_checkpoint=None):
    import jax
    import jaxlib
    jax.config.update('jax_enable_x64', True)
    import mpm_jax.blocked_elastic as blocked
    if resume and import_checkpoint is not None:
        raise ValueError('Resume and checkpoint import are separate operations')
    configuration = Path(configuration).resolve()
    config = json.loads(configuration.read_text(encoding='utf-8'))
    if config.get('version') != 1 or config.get('backend') not in ('legacy', 'blocked'):
        raise ValueError('Expected version1 and a declared execution backend')
    options = dict(config['settings'])
    options['controls'] = tuple(options['controls'])
    settings = Settings(**options)
    physical_signature, physical_sources = source_signature(settings)
    if physical_signature != config['physical_source_signature']:
        raise ValueError('Physical source/settings changed')
    controls_path = (configuration.parent/config['controls']).resolve()
    if controls_path.suffix == '.json':
        q = np.asarray(json.loads(controls_path.read_text(encoding='utf-8'))['q'], dtype=np.float64)
    else:
        with np.load(controls_path, allow_pickle=False) as saved:
            q = np.asarray(saved['q'], dtype=np.float64)
    q_sha = hashlib.sha256(q.tobytes()).hexdigest()
    if q_sha != config['controls_sha256'] or not np.isfinite(q).all():
        raise ValueError('Control data changed')
    case = CASE_NAMES.index(config['case'])
    block_size = config.get('block_size', 4096)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError('Positive integer block size required')
    output = (configuration.parent/config['output']).resolve()
    receipt_path = output/'execution.json'
    trajectory = output/'trajectory'
    execution = dict(backend=config['backend'], block_size=block_size,
                     entry_point_sha256=_digest(__file__), kernel_sha256=_digest(blocked.__file__))
    inputs = dict(configuration_sha256=_digest(configuration),
                  controls_file_sha256=_digest(controls_path), controls_sha256=q_sha,
                  physical_source_signature=physical_signature, physical_sources=physical_sources,
                  case=config['case'], runtime=dict(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__),
                  execution=execution)
    parent = None
    if resume:
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['inputs'] != inputs:
            raise ValueError('Execution inputs changed; incompatible resume')
        if not (trajectory/'checkpoint.npz').is_file():
            raise ValueError('A complete saved checkpoint is required')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Fresh output directory required')
    imported = None
    if import_checkpoint is not None:
        path = Path(import_checkpoint).resolve()
        raw = path.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            imported = {key: saved[key].copy() for key in saved.files}
        original_binding = json.loads(str(imported['binding']))
        expected = dict(source_signature=physical_signature, controls=q.tolist(), case=config['case'],
                        jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=np.__version__)
        if original_binding != expected:
            raise ValueError('Import requires an unchanged legacy physical/runtime/control binding')
        parent = dict(checkpoint=str(path), checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                      step=int(imported['step']), time_s=float(imported['state_time']),
                      original_binding=original_binding,
                      note='Preserve earlier frames at the parent trajectory for complete-history rendering.')
        imported['binding'] = np.asarray(json.dumps(dict(expected, execution=execution), sort_keys=True))
    model = build_model(settings)
    if config['backend'] == 'blocked':
        model = _with_blocks(model, block_size)
    if q.shape != model['q0'].shape:
        raise ValueError('Control shape differs from the physical model')
    if not resume:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(receipt_path, dict(inputs=inputs, parent=parent))
        if imported is not None:
            trajectory.mkdir()
            atomic_npz(trajectory/'checkpoint.npz', **imported)
    return _forward(model, q, case, trajectory, execution,
                    resume=resume or imported is not None, stop_step=stop_step,
                    save_every=int(config.get('save_every', 1000)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-step', type=int)
    parser.add_argument('--import-checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.configuration, resume=args.resume, stop_step=args.stop_step,
                         import_checkpoint=args.import_checkpoint), indent=2))
