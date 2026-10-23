"""A 25 microsecond saved-frame diagnostic within the unchanged full JC plan."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_triaxial_study as study

engine = study.engine
save, read_json, digest = study.save, study.read_json, study.digest
LOGS = ROOT/'.local/jc-refinement-ci'
OUTPUT = ROOT/'outputs/jc-refinement/grid100-ppc3/jc-necking/johnson-cook'
CACHE = ROOT/'.local/jc-refinement-cache'
DELIVERY = ROOT/'.local/jc-refinement-delivery'
PARTICLES, FRAMES, SUBSTEPS, END_FRAME = 341584, 60, 965, 10
DURATION, CFL = 150e-6, .15
CACHE_BUDGET = 1_000_000_000
LABEL = 'jc-necking / johnson-cook'
STATE_FIELDS = ('x', 'v', 'F', 'C', 'mass', 'volume', 'time', 'valid')
HISTORY_FIELDS = ('stress', 'plastic_tensor', 'backstress', 'plastic_strain', 'plastic_rate',
    'damage', 'temperature', 'plastic_volume', 'max_compression', 'compaction', 'alpha',
    'hardening', 'postpeak', 'bulking', 'energy', 'work', 'pressure', 'density', 'failed', 'valid')
FRAME_FIELDS = {'x', 'speed', 'damage', 'temperature', 'time', 'contact_impulse', 'failed'}


def encode(value):
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError('NaN is not a valid resolved input')
        return {'__float__': 'inf' if value > 0 else '-inf'}
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'__float__'}:
            if value['__float__'] not in ('inf', '-inf'):
                raise ValueError('Invalid unbounded-boundary marker')
            return float(value['__float__'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def descriptor(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return {'shape': list(np.shape(value)), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def source_record():
    record = engine.shared.source_record()
    record.pop('review_source_sha256', None)
    record['runner_source_sha256'] = record['files']['tools/run_jc_refinement.py']
    record['scope'] = 'Exact checked-out source for a start-only refinement diagnostic; not a completed tensile result.'
    return record


def cache_key(source):
    values = [str(source[name]) for name in ('commit', 'run_id', 'run_attempt')]
    if not all(re.fullmatch('[A-Za-z0-9]+', value) for value in values):
        raise ValueError('Unsafe cache identity')
    return 'jc-refinement-v1-'+'-'.join(values)+f'-frame{END_FRAME}'


def environment():
    return {'python': platform.python_version(), 'system': platform.system(),
        'machine': platform.machine(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('jax', 'jaxlib', 'numpy', 'scipy', 'psutil')},
        'cpu': Path('/proc/cpuinfo').read_text().split('\n\n')[0] if Path('/proc/cpuinfo').is_file() else platform.processor()}


def emit(**values):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            for key, value in values.items():
                if '\n' in str(value):
                    raise ValueError('Multiline workflow output')
                stream.write(f'{key}={value}\n')


def prepare():
    if platform.python_version() != '3.12.13':
        raise ValueError('This diagnostic pins Python 3.12.13')
    if OUTPUT.exists() or CACHE.exists():
        raise ValueError('Start-only diagnostic requires fresh output and cache directories')
    source = source_record()
    save(LOGS/'numerical-source.json', source)
    save(LOGS/'request.json', {'grid': 100, 'ppc': 3, 'particles_expected': PARTICLES,
        'frames': FRAMES, 'full_duration_s': DURATION, 'CFL': CFL, 'stop_after_frame': END_FRAME,
        'target_steps': FRAMES*SUBSTEPS, 'segment_steps': END_FRAME*SUBSTEPS,
        'runtime_stop_only': True, 'start_only': True, 'cross_job_resume_verified': False,
        'production_eligible': False, 'cache_key': cache_key(source), 'started_epoch': time.time()})
    emit(cache_key=cache_key(source))


def state_arrays(state):
    import numpy as np
    values = {key: np.asarray(value) for key, value in state._asdict().items() if key != 'history'}
    values.update({'history_'+key: np.asarray(value) for key, value in state.history._asdict().items()})
    return values


def worker():
    import jax
    import numpy as np
    from examples.jc_necking import make_case, reference_lattice
    from examples.common import run as common_run
    from mpm_jax.particle_fields import material_metadata

    jax.config.update('jax_enable_x64', True)
    source = read_json(LOGS/'numerical-source.json')
    if source_record() != source:
        raise ValueError('Source changed before constructing the case')
    case = make_case(grid=100, ppc=3, frames=FRAMES, end_time=DURATION, cfl=CFL)
    substeps = round(case.duration/(FRAMES*case.configs[0].dt))
    arrays = state_arrays(case.states[0])
    payload = {'configs': [asdict(item) for item in case.configs],
        'materials': [material_metadata(item) for item in case.materials],
        'frames': FRAMES, 'substeps': substeps, 'label': LABEL, 'definition': case.description}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    coarse_reference, coarse_spacing, _ = reference_lattice(100, 2)
    coarse_volume = len(coarse_reference)*coarse_spacing**3
    comparison = {'grid': 100, 'baseline_ppc': 2, 'baseline_particles': len(coarse_reference),
        'baseline_reference_descriptor': descriptor(coarse_reference),
        'baseline_reference_volume_m3': coarse_volume, 'baseline_mass_kg': 7830.*coarse_volume,
        'refined_ppc': 3, 'refined_particles': len(arrays['x']),
        'refined_reference_volume_m3': float(arrays['volume'].sum()), 'refined_mass_kg': float(arrays['mass'].sum()),
        'scope': 'Both discrete reference geometries are sampled from the same source. Boundary centre clipping changes mass and volume; neither is rescaled to match the other. This comparison does not run a second solver.'}
    del coarse_reference
    plan = {'schema': 'jc-refinement-first-segment-v1', 'completed': False, 'production_eligible': False,
        'particles': len(arrays['x']), 'full_frames': FRAMES, 'full_duration_s': DURATION,
        'full_steps': FRAMES*substeps, 'stop_after_frame': END_FRAME, 'segment_steps': END_FRAME*substeps,
        'segment_time_s': END_FRAME*substeps*case.configs[0].dt,
        'signature_data': encode(payload), 'common_signature': signature,
        'initial_state_fields': {key: descriptor(value) for key, value in arrays.items()},
        'initial_mass_kg': float(arrays['mass'].sum()), 'initial_reference_volume_m3': float(arrays['volume'].sum()),
        'sampling_comparison': comparison,
        'initial_max_speed_m_per_s': float(np.linalg.norm(arrays['v'], axis=1).max()),
        'source': source, 'environment': environment(),
        'float_encoding': 'Only unbounded boundary endpoints use explicit __float__ markers; decoding reproduces the common.run signature exactly.',
        'refinement_scope': 'Grid, material, waist, grips, CFL and full 150 microsecond plan are unchanged; only particle sampling is ppc3. Circular boundary clipping changes discrete mass/volume relative to ppc2; no mass renormalization is applied.',
        'state_history_scope': 'Latest checkpoint retains every State/History array. frames_data retains x/speed/D/T/failed at each saved prefix frame; it does not contain every-frame F or plastic-strain fields.',
        'continuation_scope': 'This command only starts the first segment. Carry files preserve the current common.run resume inputs, but cross-job reconstruction/continuation is not yet tested.'}
    validate_plan(plan)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save(OUTPUT/'plan.json', plan)
    save(OUTPUT/'numerical-source.json', source)
    save(OUTPUT/'environment.json', plan['environment'])
    del arrays
    common_run(case.states, case.materials, case.configs, frames=FRAMES, substeps=substeps,
        output=OUTPUT, label=LABEL, observer=case.observer, metadata=case.description,
        render_output=False, gif=False, resume=False, stop_after_frame=END_FRAME)


def validate_plan(plan):
    import numpy as np
    payload = decode(plan['signature_data'])
    if len(payload['configs']) != 1 or len(payload['materials']) != 1:
        raise ValueError('The refinement has exactly one steel body')
    config, material, definition = payload['configs'][0], payload['materials'][0], payload['definition']
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    expected_material = {'mu': 200e9/(2*(1+.29)), 'bulk': 200e9/(3*(1-2*.29)),
        'A': 792e6, 'B': 510e6, 'n': .26, 'C': .014, 'm': 1.03, 'reference_rate': 1.,
        'reference_temperature': 298., 'melt_temperature': 1793., 'heat_capacity': 477.,
        'heat_fraction': 1., 'density': 7830., 'failure': [.05, 3.44, -2.12, .002, .61],
        'failure_enabled': True}
    boundaries = config['boundaries']
    geometry = {'bar_diameter_m': .0016, 'waist_radius_m': .00075,
        'transition_half_length_m': .00075, 'gauge_length_m': .0056,
        'arc_radius_m': (.00075**2+(.0008-.00075)**2)/(4*(.0008-.00075)),
        'requested_particles_per_cell_per_axis': 3, 'transverse_lattice_count': 48,
        'grip_speed_m_per_s': 10., 'ramp_time_s': 20e-6, 'duration_s': DURATION,
        'reference_duration_s': DURATION, 'acoustic_CFL_requested': CFL}
    checks = {
        'unchanged_full_plan': payload['frames'] == FRAMES and payload['substeps'] == SUBSTEPS
            and plan['full_steps'] == FRAMES*SUBSTEPS and plan['full_duration_s'] == DURATION
            and plan['full_frames'] == FRAMES and payload['label'] == LABEL,
        'fixed_runtime_stop': plan['stop_after_frame'] == END_FRAME and plan['segment_steps'] == END_FRAME*SUBSTEPS
            and math.isclose(plan['segment_time_s'], 25e-6, rel_tol=1e-14),
        'grid_and_time_step': config['n_grid'] == 100 and config['domain_size'] == .01 and config['dim'] == 3
            and math.isclose(config['dt'], DURATION/(FRAMES*SUBSTEPS), rel_tol=1e-14),
        'material': set(material) == set(expected_material) and all(np.allclose(material[key], value, rtol=1e-14, atol=0)
            for key, value in expected_material.items()),
        'geometry_loading': all(np.allclose(definition[key], value, rtol=1e-14, atol=0) for key, value in geometry.items())
            and np.allclose(definition['bar_ends_x_m'], [.0018, .0082], rtol=0, atol=1e-17)
            and np.allclose(definition['grip_inner_faces_x_m'], [.0022, .0078], rtol=0, atol=1e-17),
        'grips': len(boundaries) == 2 and all(boundary['components'] is None and boundary['start'] == 0
            and boundary['end'] == math.inf and boundary['ramp_time'] == 20e-6 for boundary in boundaries)
            and boundaries[0]['velocity'] == [-10., 0., 0.] and boundaries[1]['velocity'] == [10., 0., 0.]
            and np.allclose(boundaries[0]['lower'], [0., 0., 0.], atol=0)
            and np.allclose(boundaries[0]['upper'], [.0022, .01, .01], rtol=0, atol=1e-17)
            and np.allclose(boundaries[1]['lower'], [.0078, 0., 0.], rtol=0, atol=1e-17)
            and boundaries[1]['upper'] == [.01, .01, .01],
        'failure_and_external_loading': config['failure_mode'] == 'ballistic' and config['material'] == 'johnson-cook'
            and config['gravity'] == [0., 0., 0.] and config['boundary_cells'] == 0,
        'sampling_count': plan['particles'] == PARTICLES,
        'signature_matches_actual_inputs': signature == plan['common_signature'],
        'initial_fields_complete': set(plan['initial_state_fields']) == set(STATE_FIELDS) | {'history_'+name for name in HISTORY_FIELDS},
        'initial_array_shapes_and_dtypes': valid_descriptors(plan['initial_state_fields'], plan['particles']),
        'partial_scope': plan['completed'] is False and plan['production_eligible'] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Resolved refinement plan mismatch: {[key for key, value in checks.items() if not value]}")
    return checks


def valid_descriptors(fields, count):
    expected = {'x': ([count, 3], 'float64'), 'v': ([count, 3], 'float64'),
        'F': ([count, 3, 3], 'float64'), 'C': ([count, 3, 3], 'float64'),
        'mass': ([count], 'float64'), 'volume': ([count], 'float64'),
        'time': ([], 'float64'), 'valid': ([], 'bool')}
    for name in HISTORY_FIELDS:
        shape = [count, 3, 3] if name in ('stress', 'plastic_tensor', 'backstress') else [count]
        dtype = 'bool' if name in ('postpeak', 'failed', 'valid') else 'float64'
        expected['history_'+name] = (shape, dtype)
    return set(fields) == set(expected) and all(fields[key].get('shape') == shape
        and fields[key].get('dtype') == dtype and re.fullmatch('[a-f0-9]{64}', fields[key].get('sha256', ''))
        for key, (shape, dtype) in expected.items())


def contained_file(folder, name):
    folder = Path(folder).resolve()
    candidate = folder/name
    path = candidate.resolve()
    if not path.is_relative_to(folder) or path == folder or candidate.is_symlink() or not path.is_file():
        raise ValueError('Checkpoint file must be a real file inside this result')
    if any(parent.is_symlink() for parent in candidate.parents if parent != folder and parent.is_relative_to(folder)):
        raise ValueError('Checkpoint parents must not be symlinks')
    return path


def finite_record(value):
    if isinstance(value, dict):
        return all(finite_record(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_record(item) for item in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def final_metrics(arrays, reference, spacing):
    import numpy as np
    levels, layers = np.unique(reference[:, 0], return_inverse=True)
    central = np.abs(levels-.005) <= .51*spacing
    F = arrays['F']
    J = np.linalg.det(F)
    normal = J[:, None]*np.swapaxes(np.linalg.inv(F), 1, 2)[:, :, 0]
    traction = np.einsum('pi,pi->p', arrays['history_stress'][:, 0, :], normal)
    active = ~arrays['history_failed'].astype(bool)
    force = np.bincount(layers, weights=active*traction*arrays['volume']/spacing, minlength=len(levels))
    if not central.any():
        raise ValueError('No central material section')
    return {'central_material_section_force_N': float(force[central].mean()),
        'maximum_equivalent_plastic_strain': float(arrays['history_plastic_strain'].max()),
        'max_damage': float(arrays['history_damage'].max()),
        'failed_particles': int(np.count_nonzero(arrays['history_failed'])),
        'central_active_particles': int(active[central[layers]].sum()),
        'min_J': float(J.min()), 'max_J': float(J.max()),
        'mass_kg': float(arrays['mass'].sum()), 'reference_volume_m3': float(arrays['volume'].sum())}


def audit_prefix(folder, source):
    import numpy as np
    folder = Path(folder)
    plan, checkpoint, status, summary, segment = [read_json(folder/name) for name in
        ('plan.json', 'checkpoint.json', 'status.json', 'summary.json', 'segment.json')]
    checks = validate_plan(plan)
    if source != plan['source'] or source != read_json(folder/'numerical-source.json'):
        raise ValueError('Resolved plan and executed source differ')
    payload = decode(plan['signature_data'])
    config = payload['configs'][0]
    dt, count = config['dt'], plan['particles']
    signature, frame = plan['common_signature'], plan['stop_after_frame']
    if not (checkpoint['signature'] == signature and checkpoint['frame'] == frame and len(checkpoint['states']) == 1
            and status == summary == segment and status.get('completed') is False
            and status.get('segment_complete') is True and status.get('phase') == 'checkpoint-ready'
            and status.get('signature') == signature and status.get('frame') == frame
            and status.get('frames') == FRAMES and status.get('steps_completed') == frame*SUBSTEPS
            and status.get('target_steps') == FRAMES*SUBSTEPS and status.get('particles') == [count]
            and math.isclose(status['time'], frame*SUBSTEPS*dt, rel_tol=1e-14)
            and math.isclose(status['target_time'], DURATION, rel_tol=1e-14)):
        raise ValueError('Result does not identify the requested valid partial endpoint')
    state_path = contained_file(folder, checkpoint['states'][0])
    if not re.fullmatch(r'checkpoint_[01]_0\.npz', state_path.name):
        raise ValueError('Unexpected checkpoint state name')
    with np.load(state_path, allow_pickle=False) as archive:
        if set(archive.files) != set(plan['initial_state_fields']):
            raise ValueError('Checkpoint is missing State or History fields')
        arrays = {}
        for key in archive.files:
            value = archive[key]
            expected = plan['initial_state_fields'][key]
            if list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype'] or not np.isfinite(value).all():
                raise ValueError(f'Malformed checkpoint field {key}')
            arrays[key] = value
    if not bool(arrays['valid']) or not arrays['history_valid'].all() or not (np.linalg.det(arrays['F']) > 0).all():
        raise ValueError('Invalid state/history or nonpositive deformation determinant')
    for key in ('mass', 'volume'):
        if not (arrays[key] > 0).all() or descriptor(arrays[key]) != plan['initial_state_fields'][key]:
            raise ValueError('Checkpoint mass/reference volume changed')
    if not ((arrays['history_damage'] >= 0) & (arrays['history_damage'] <= 1)).all() or (arrays['history_plastic_strain'] < 0).any():
        raise ValueError('Checkpoint damage/plastic strain is outside its physical range')
    expected_times = np.arange(frame+1)*SUBSTEPS*dt
    epsilon = np.finfo(arrays['time'].dtype).eps
    accumulated = frame*SUBSTEPS*epsilon
    clock_tolerance = max(dt*1e-5, 1e-15, 2*accumulated/(1-accumulated)*abs(expected_times[-1]))
    if not math.isclose(float(arrays['time']), expected_times[-1], rel_tol=0, abs_tol=clock_tolerance):
        raise ValueError('Accumulated checkpoint clock does not match its saved frame')
    rows, observations = checkpoint['rows'], checkpoint['observations']
    if len(rows) != frame+1 or len(observations) != frame+1 or not finite_record(checkpoint):
        raise ValueError('Diagnostic/observation prefix is incomplete')
    if np.shape(checkpoint['contact_impulse']) != (3,) or np.shape(checkpoint['external_impulse']) != (3,) or checkpoint['solver_seconds'] < 0:
        raise ValueError('Checkpoint cumulative quantities are incomplete')
    prefix_paths = [folder/'frames_data'/f'frame_{i:05d}.npz' for i in range(frame+1)]
    if sorted((folder/'frames_data').glob('*.npz')) != prefix_paths:
        raise ValueError('Saved frame prefix is missing frames or contains unexpected future frames')
    records, cut_details = [], []
    previous_failed = np.zeros(count, dtype=bool)
    previous_damage = np.zeros(count)
    reference = None
    spacing = float(payload['definition']['particle_spacing_m'])
    for index, path in enumerate(prefix_paths):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FRAME_FIELDS:
                raise ValueError('Saved frame fields differ from the complete ballistic prefix')
            values = {key: archive[key] for key in archive.files}
        shapes = {'x': (count, 3), 'speed': (count,), 'damage': (count,), 'temperature': (count,),
                  'failed': (count,), 'time': (), 'contact_impulse': (3,)}
        if any(values[key].shape != shape or not np.isfinite(values[key]).all() for key, shape in shapes.items()):
            raise ValueError('Malformed or nonfinite saved frame')
        if not math.isclose(float(values['time']), expected_times[index], rel_tol=0, abs_tol=1e-15):
            raise ValueError('Saved prefix time changed')
        if not ((values['damage'] >= 0) & (values['damage'] <= 1)).all() or not np.isin(values['failed'], (0, 1)).all():
            raise ValueError('Invalid damage/failure prefix')
        failed = values['failed'].astype(bool)
        if np.any(previous_failed & ~failed) or np.any(values['damage'] < previous_damage-64*np.finfo(float).eps):
            raise ValueError('Failure or damage decreased in the saved history')
        previous_failed, previous_damage = failed, values['damage']
        if index == 0:
            reference = values['x'].copy()
            if descriptor(reference) != plan['initial_state_fields']['x']:
                raise ValueError('Initial saved geometry differs from the resolved reference')
            for field, initial in [('damage', 'history_damage'), ('temperature', 'history_temperature')]:
                if descriptor(values[field]) != plan['initial_state_fields'][initial]:
                    raise ValueError('Initial saved history differs from the resolved input')
            if failed.any() or np.any(values['speed'] != 0):
                raise ValueError('Initial rod is not intact and stationary')
        diagnostic, observed = rows[index], observations[index]
        required_observation = {'time', 'external_work_trapezoidal', 'boundary_impulse_x', 'boundary_impulse_y', 'boundary_impulse_z',
            'failed_particles', 'max_damage', 'maximum_equivalent_plastic_strain', 'central_material_section_force_N', 'central_active_particles'}
        if not required_observation.issubset(observed):
            raise ValueError('Observation prefix lacks quantities required by continuation')
        if diagnostic['body'] != 0 or not math.isclose(diagnostic['time'], expected_times[index], rel_tol=0, abs_tol=1e-15) or not math.isclose(observed['time'], expected_times[index], rel_tol=0, abs_tol=1e-15):
            raise ValueError('Diagnostic/observer time differs from the complete prefix')
        if not (diagnostic['min_J'] > 0 and math.isclose(diagnostic['mass'], plan['initial_mass_kg'], rel_tol=1e-12)
                and observed['failed_particles'] == int(failed.sum())
                and math.isclose(observed['max_damage'], float(values['damage'].max()), rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError('Saved diagnostic/observer disagrees with valid prefix data')
        active = np.flatnonzero(~failed)
        order = active[np.argsort(values['x'][active, 0])]
        bases, lower, upper, cuts = engine.axial_support_intervals(values['x'][order, 0], .0001, 100)
        details = [{'current_x_m': [float(values['x'][order[k], 0]), float(values['x'][order[k+1], 0])],
            'reference_x_m': [float(reference[order[k], 0]), float(reference[order[k+1], 0])],
            'positive_node_intervals': [[int(lower[k]), int(upper[k])], [int(lower[k+1]), int(upper[k+1])]]} for k in cuts]
        cut_details.append({'frame': index, 'time_s': float(values['time']), 'cuts': details})
        records.append({'frame': index, 'time_s': float(values['time']), 'active_particles': int(active.size),
            'failed_particles': int(failed.sum()), 'maximum_damage': float(values['damage'].max()),
            'maximum_equivalent_plastic_strain_recorded': observed['maximum_equivalent_plastic_strain'],
            'central_material_section_force_N_recorded': observed['central_material_section_force_N'],
            'central_active_particles_recorded': observed['central_active_particles'],
            'positive_x_stencil_cuts': len(cuts), 'positive_x_support_components': len(cuts)+int(active.size > 0)})
        if index == frame:
            for field, key in [('x', 'x'), ('damage', 'history_damage'), ('temperature', 'history_temperature')]:
                np.testing.assert_array_equal(values[field], arrays[key])
            np.testing.assert_array_equal(failed, arrays['history_failed'])
            np.testing.assert_allclose(values['speed'], np.linalg.norm(arrays['v'], axis=1), rtol=1e-14, atol=0)
            np.testing.assert_array_equal(values['contact_impulse'], checkpoint['contact_impulse'])
    metrics = final_metrics(arrays, reference, spacing)
    for key in ('central_material_section_force_N', 'maximum_equivalent_plastic_strain', 'max_damage', 'failed_particles', 'central_active_particles'):
        if not math.isclose(metrics[key], observations[-1][key], rel_tol=1e-10, abs_tol=0):
            raise ValueError(f'Final checkpoint independently disagrees with observer {key}')
    for key in ('min_J', 'max_J'):
        if not math.isclose(metrics[key], rows[-1][key], rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'Final checkpoint independently disagrees with diagnostic {key}')
    momentum = np.sum(arrays['mass'][:, None]*arrays['v'], axis=0)
    reaction = momentum-np.asarray(checkpoint['external_impulse'])
    np.testing.assert_allclose([observations[-1]['boundary_impulse_'+axis] for axis in 'xyz'], reaction, rtol=1e-10, atol=0)
    if observations[-1]['external_work_trapezoidal'] != checkpoint['external_work']:
        raise ValueError('Final work differs from the cumulative checkpoint value')
    checks.update({'source_matches': True, 'valid_partial_checkpoint': True, 'all_state_and_history_fields_checked': True,
        'complete_saved_prefix_checked': True, 'constant_mass_reference_volume': True,
        'final_ep_and_section_force_recomputed': True,
        'intact_at_25_microseconds': all(row['failed_particles'] == 0 for row in records),
        'no_support_cut_before_material_failure': all(row['positive_x_stencil_cuts'] == 0 or row['failed_particles'] > 0 for row in records)})
    report = {'schema': 'jc-refinement-prefix-review-v1', 'passed': all(checks.values()), 'checks': checks,
        'completed': False, 'full_protocol_completed': False, 'production_eligible': False,
        'segment_complete': True, 'frame': frame, 'steps': frame*SUBSTEPS, 'time_s': float(expected_times[-1]),
        'full_target_steps': FRAMES*SUBSTEPS, 'full_target_time_s': DURATION,
        'initial_mass_kg': plan['initial_mass_kg'], 'initial_reference_volume_m3': plan['initial_reference_volume_m3'],
        'sampling_comparison': plan.get('sampling_comparison'),
        'final_checkpoint_metrics': metrics, 'positive_support_history': cut_details,
        'first_saved_support_cut': next((item for item in records if item['positive_x_stencil_cuts']), None),
        'first_saved_failed_material': next((item for item in records if item['failed_particles']), None),
        'review_source_sha256': digest(Path(__file__)), 'producer_source': source,
        'scope': 'Only the first 25 microseconds of the unchanged full 150 microsecond plan. Positive x-stencil cuts prove missing axial shared support, not a full 3D graph. Ep/section-force histories are recorded by the original observer; only the latest complete State independently reconstructs those fields. No complete fracture, convergence or cross-job resume claim.'}
    names = ['plan.json', 'numerical-source.json', 'checkpoint.json', 'status.json', 'summary.json', 'segment.json', state_path.name]
    names += [path.relative_to(folder).as_posix() for path in prefix_paths]
    report['input_sha256'] = {name: digest(folder/name) for name in names}
    with (folder/'prefix-history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    for name, table in [('diagnostics-prefix.csv', rows), ('observables-prefix.csv', observations)]:
        with (folder/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=dict.fromkeys(key for row in table for key in row), lineterminator='\n')
            writer.writeheader(); writer.writerows(table)
    report['prefix_history_sha256'] = digest(folder/'prefix-history.csv')
    save(folder/'prefix-review.json', report)
    return report


def file_record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': digest(path)}


def carry_names(folder):
    checkpoint = read_json(Path(folder)/'checkpoint.json')
    names = ['plan.json', 'numerical-source.json', 'environment.json', 'checkpoint.json',
        'status.json', 'summary.json', 'segment.json', 'prefix-review.json', 'prefix-history.csv',
        'diagnostics-prefix.csv', 'observables-prefix.csv', *checkpoint['states']]
    names += [f'frames_data/frame_{index:05d}.npz' for index in range(checkpoint['frame']+1)]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate carry files')
    for name in names:
        contained_file(folder, name)
    return names


def seal_carry(folder, source, cache):
    folder, cache = Path(folder), Path(cache)
    report = read_json(folder/'prefix-review.json')
    if not (study.strict_review(report) and study.inputs_match(folder, report['input_sha256'])
            and report.get('producer_source') == source
            and report.get('review_source_sha256') == source['files']['tools/run_jc_refinement.py']
            and read_json(folder/'plan.json')['source'] == source
            and read_json(folder/'numerical-source.json') == source):
        raise ValueError('Only a fully checked valid partial prefix can enter the carry cache')
    names = carry_names(folder)
    inventory = {name: file_record(folder/name) for name in names}
    total = sum(item['bytes'] for item in inventory.values())
    if total+1024**2 > CACHE_BUDGET:
        raise ValueError('Complete carry exceeds the 1 GB cache budget; no partial checkpoint is saved')
    if cache.exists():
        raise ValueError('Refinement carry destination is not fresh')
    stage = Path(tempfile.mkdtemp(prefix='jc-carry-', dir=cache.parent))
    for name, record in inventory.items():
        target = stage/'result'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (folder/name).open('rb') as original, target.open('wb') as output:
            shutil.copyfileobj(original, output, length=study.READ_BYTES)
        if file_record(target) != record:
            raise ValueError('Carry copy changed source bytes')
    receipt = {'schema': 'jc-refinement-carry-v1', 'cache_key': cache_key(source), 'source': source,
        'source_fingerprint': source['fingerprint'], 'common_signature': read_json(folder/'plan.json')['common_signature'],
        'frame': END_FRAME, 'full_frames': FRAMES, 'segment_steps': END_FRAME*SUBSTEPS,
        'full_target_steps': FRAMES*SUBSTEPS, 'completed': False, 'segment_complete': True,
        'production_eligible': False, 'cross_job_resume_verified': False,
        'files': {'result/'+name: record for name, record in inventory.items()}, 'payload_bytes': total,
        'cache_budget_bytes': CACHE_BUDGET, 'excluded': ['trajectory_*.npy', 'inactive checkpoint state slot'],
        'scope': 'Lossless original latest checkpoint plus all frames_data prefix required by common.run. No source/credentials. Cache is temporary; a future same-source controller must validate every byte and initial geometry/configuration before attempting resume.'}
    save(stage/'carry-receipt.json', receipt)
    if total+(stage/'carry-receipt.json').stat().st_size > CACHE_BUDGET:
        raise ValueError('Carry receipt exceeds cache budget')
    stage.replace(cache)
    verify_carry(cache, source, digest(cache/'carry-receipt.json'))
    return receipt


def verify_carry(cache, source, receipt_sha):
    cache = Path(cache)
    receipt = read_json(cache/'carry-receipt.json')
    if not (digest(cache/'carry-receipt.json') == receipt_sha and receipt['source'] == source
        and receipt['cache_key'] == cache_key(source) and receipt['completed'] is False
        and receipt['cross_job_resume_verified'] is False and receipt['frame'] == END_FRAME
        and receipt['source_fingerprint'] == source['fingerprint']):
        raise ValueError('Carry source/identity/receipt mismatch')
    actual = {path.relative_to(cache).as_posix() for path in cache.rglob('*') if path.is_file()}
    if any(path.is_symlink() for path in cache.rglob('*')) or actual != set(receipt['files']) | {'carry-receipt.json'}:
        raise ValueError('Carry contains missing or additional files')
    if not study.inputs_match(cache, {name: record['sha256'] for name, record in receipt['files'].items()}):
        raise ValueError('Carry raw file hash mismatch')
    if any((cache/name).stat().st_size != record['bytes'] for name, record in receipt['files'].items()):
        raise ValueError('Carry raw file size mismatch')
    if sum(record['bytes'] for record in receipt['files'].values()) != receipt['payload_bytes'] or sum(path.stat().st_size for path in cache.rglob('*') if path.is_file()) > CACHE_BUDGET:
        raise ValueError('Carry byte accounting exceeds budget')
    plan, review = read_json(cache/'result/plan.json'), read_json(cache/'result/prefix-review.json')
    if not (plan['source'] == review['producer_source'] == source
            and review['review_source_sha256'] == source['files']['tools/run_jc_refinement.py']
            and receipt['common_signature'] == plan['common_signature'] and study.strict_review(review)):
        raise ValueError('Carry plan/review source chain is inconsistent')
    return receipt


def audit_worker():
    source = read_json(LOGS/'numerical-source.json')
    if source_record() != source:
        raise ValueError('Source changed after the partial solve')
    report = audit_prefix(OUTPUT, source)
    if not study.strict_review(report):
        raise ValueError('First-segment physical or prefix review failed; do not cache as admitted continuation data')
    receipt = seal_carry(OUTPUT, source, CACHE)
    save(LOGS/'carry-receipt.json', receipt)


def run():
    requested = read_json(LOGS/'request.json')
    if source_record() != read_json(LOGS/'numerical-source.json'):
        raise ValueError('Source changed before execution')
    planned_mmap = (FRAMES+1)*PARTICLES*49
    required = planned_mmap+2*PARTICLES*544+(END_FRAME+1)*PARTICLES*49+2*CACHE_BUDGET+2*1024**3
    free = shutil.disk_usage(ROOT).free
    save(LOGS/'disk-admission.json', {'free_bytes': free, 'required_bytes': required, 'passed': free >= required,
        'scope': 'Full planned mmap, two state slots, frame prefix, carry/cache temporary and 2 GiB margin; no physical reduction.'})
    if free < required:
        raise ValueError('Insufficient free disk for the complete first-segment carry')
    seconds = min(265*60, int(requested['started_epoch']+320*60-time.time()))
    if seconds <= 0:
        raise TimeoutError('No worker budget remains after preserving audit/cache/archive time')
    result = engine.command('jc-refinement-solve', ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s', sys.executable, __file__, 'worker'], monitor=True)
    save(LOGS/'solve-memory.json', {**result['memory'], 'wall_seconds': result['wall_seconds'],
        'worker_time_limit_seconds': seconds, 'scope': 'Case construction, full-plan recording, compilation and the saved-frame-10 solve.'})
    remaining = int(requested['started_epoch']+340*60-time.time())
    if remaining < 60:
        raise TimeoutError('Insufficient audit/carry time; checkpoint remains diagnostic evidence')
    engine.command('jc-refinement-prefix-audit', ['timeout', '--signal=INT', '--kill-after=60s', f'{min(20*60, remaining)}s', sys.executable, __file__, 'audit-worker'], monitor=True)
    if source_record() != read_json(LOGS/'numerical-source.json'):
        raise ValueError('Source changed before cache handoff')
    emit(receipt_sha=digest(CACHE/'carry-receipt.json'))


def confirm_cache():
    source = read_json(LOGS/'numerical-source.json')
    if os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != cache_key(source):
        raise ValueError('The exact scoped carry cache was not saved')
    receipt_sha = digest(CACHE/'carry-receipt.json')
    if os.environ.get('RECEIPT_SHA') != receipt_sha:
        raise ValueError('Cache receipt differs from the admitted prefix')
    verify_carry(CACHE, source, receipt_sha)
    save(LOGS/'cache-confirmation.json', {'verified': True, 'cache_key': cache_key(source),
        'receipt_sha256': receipt_sha, 'permanent_delivery': False})


def package():
    if DELIVERY.exists():
        raise ValueError('Diagnostic delivery already exists')
    confirmation = read_json(LOGS/'cache-confirmation.json') if (LOGS/'cache-confirmation.json').is_file() else {}
    source = read_json(LOGS/'numerical-source.json') if (LOGS/'numerical-source.json').is_file() else None
    confirmed = (source is not None and confirmation.get('verified') is True
        and confirmation.get('cache_key') == cache_key(source) and (CACHE/'carry-receipt.json').is_file()
        and confirmation.get('receipt_sha256') == digest(CACHE/'carry-receipt.json'))
    budget = 8*1024**2 if confirmed else study.TOTAL_BYTES
    stage = Path(tempfile.mkdtemp(prefix='jc-refinement-delivery-', dir=DELIVERY.parent))
    files, omitted, states, used = {}, {}, {}, 0
    checkpoint = read_json(OUTPUT/'checkpoint.json') if (OUTPUT/'checkpoint.json').is_file() else {}
    metadata = sorted({path for parent in (LOGS, OUTPUT) if parent.exists() for path in parent.rglob('*')
        if path.is_file() and path.suffix in ('.json', '.csv')})
    candidates = list(metadata)
    if not confirmed:
        candidates += [contained_file(OUTPUT, name) for name in checkpoint.get('states', [])]
        candidates += sorted((OUTPUT/'frames_data').glob('*.npz'))
    for path in candidates:
        name = path.relative_to(ROOT).as_posix()
        record = file_record(path)
        if used+record['bytes'] > budget-1024**2 or (path.suffix == '.npz' and not study.valid_npz(path)):
            omitted[name] = {**record, 'reason': 'Whole file exceeds bounded delivery budget or is not a complete CRC-valid NPZ. No partial checkpoint is presented.'}
            continue
        if path.suffix == '.npz':
            state = study.split_complete_file(path, stage, 'state-parts/'+name)
            states[name] = state
            for part in state['parts']:
                files[part['path']] = {'bytes': part['bytes'], 'sha256': part['sha256']}
        else:
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files[name] = file_record(target)
            if files[name] != record:
                raise ValueError('Diagnostic metadata copy mismatch')
        used += record['bytes']
    manifest = {'schema': 'jc-refinement-diagnostic-delivery-v1', 'completed': False,
        'full_protocol_completed': False, 'production_eligible': False, 'media_created': False,
        'cache_saved_exactly': confirmed, 'files': files, 'state_files': states, 'omitted': omitted,
        'payload_bytes': used, 'total_budget_bytes': budget, 'full_target_time_s': DURATION,
        'requested_prefix_time_s': 25e-6,
        'scope': 'First-segment diagnostics only. A verified cache permits an 8 MiB records-only artifact; otherwise complete state files are prioritized within 192 MiB. A prefix is neither full tensile fracture nor demonstrated cross-job resume.'}
    save(stage/'delivery-manifest.json', manifest)
    if used+(stage/'delivery-manifest.json').stat().st_size > budget:
        raise ValueError('Diagnostic manifest exceeds archive budget')
    stage.replace(DELIVERY)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run', 'audit-worker', 'confirm-cache', 'package'))
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux':
        parser.error('This start-only diagnostic runs on the independent Linux Actions runner')
    LOGS.mkdir(parents=True, exist_ok=True)
    engine.LOGS = LOGS
    try:
        {'prepare': prepare, 'worker': worker, 'run': run, 'audit-worker': audit_worker,
         'confirm-cache': confirm_cache, 'package': package}[args.action]()
    except BaseException as error:
        save(LOGS/f'{args.action}-failure.json', {'completed': False, 'production_eligible': False,
            'action': args.action, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
