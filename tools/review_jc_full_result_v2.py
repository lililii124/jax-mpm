"""Read-only typed and coordinate audit of the immutable completed JC delivery."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_jc_continuation as carry
from jc_grid_separation import inspect_state


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_json(value):
    if isinstance(value, dict):
        return {key: completed_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [completed_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def typed_value(name, value, table):
    if table == 'observables' and name == 'gap_exceeds_grid_support':
        require(type(value) is bool or type(value) is str and value in ('True', 'False'),
                'Invalid Boolean: ' + name)
        return value if type(value) is bool else value == 'True'
    if table == 'diagnostics' and name == 'strain_energy' and value in ('', None):
        return None
    require(value is not None and type(value) is not bool and value != '', 'Missing numeric: ' + name)
    number = float(value)
    require(math.isfinite(number), 'Nonfinite numeric: ' + name)
    return number


def typed_row(row, table):
    require(None not in row, 'Malformed CSV row')
    return {key: typed_value(key, value, table) for key, value in row.items()}


def coordinate_neck_audit(folder, review):
    from scipy import ndimage
    folder = Path(folder)
    summary = carry.read_json(folder / 'summary.json')
    definition = summary['definition']
    spacing = definition['particle_spacing_m']
    left, right = definition['grip_inner_faces_x_m']
    middle = .5 * (left + right)
    length = right - left
    x = np.load(folder / 'trajectory_x.npy', mmap_mode='r', allow_pickle=False)
    failed = np.load(folder / 'trajectory_failed.npy', mmap_mode='r', allow_pickle=False)
    with (folder / 'observables.csv').open(encoding='utf-8', newline='') as stream:
        observations = [typed_row(row, 'observables') for row in csv.DictReader(stream)]
    reference = np.asarray(x[0])
    centers = np.arange(left + 4*spacing, right - 4*spacing, 4*spacing)
    masks = [np.abs(reference[:, 0] - center) < 2*spacing for center in centers]
    central = np.abs(centers - middle) <= .10*length
    remote = (np.abs(centers - middle) >= .25*length) & (np.abs(centers - middle) <= .35*length)
    require(central.any() and remote.any(), 'Missing silhouette sections')
    initial_widths = np.asarray([np.ptp(reference[mask, 1]) + spacing for mask in masks])
    left_grip, right_grip = reference[:, 0] <= left, reference[:, 0] >= right
    rows = []
    try:
        for frame in range(61):
            active = ~np.asarray(failed[frame], dtype=bool)
            points = np.asarray(x[frame])
            widths = np.asarray([np.ptp(points[mask & active, 1]) + spacing
                                 if np.count_nonzero(mask & active) >= 2 else 0. for mask in masks])
            windows = [np.arange(i, i+3) for i in range(len(centers)-2) if central[i:i+3].all()]
            section = min(windows, key=lambda indices: np.median(widths[indices]))
            central_width = float(np.median(widths[section]))
            remote_width = float(np.median(widths[remote]))
            initial_width = float(np.median(initial_widths[section]))
            gauge = active & (reference[:, 0] > left) & (reference[:, 0] < right)
            xy = points[gauge, :2]
            pixels = np.floor((xy - xy.min(axis=0))/spacing).astype(int)
            raster = np.zeros(tuple(pixels.max(axis=0)+3), dtype=bool)
            raster[tuple((pixels+1).T)] = True
            raster = ndimage.binary_dilation(raster, structure=np.ones((3, 3), bool))
            labels, _ = ndimage.label(raster, structure=np.ones((3, 3), bool))
            xmin, xmax = pixels[:, 0].min()+1, pixels[:, 0].max()+1
            connected = bool(set(np.unique(labels[xmin])) .intersection(set(np.unique(labels[xmax]))) - {0})
            separated = review['separation_history'][frame]['reference_grip_sets_grid_decoupled']
            ratio = central_width / initial_width
            localization = central_width / remote_width if remote_width > 0 else None
            displacement = points[:, 0] - reference[:, 0]
            coordinate_strain = float((np.mean(displacement[right_grip])
                                       - np.mean(displacement[left_grip])) / length)
            rows.append({'frame': frame, 'time_s': observations[frame]['time'],
                         'central_width_m': central_width, 'remote_width_m': remote_width,
                         'central_reference_slab_centers_m': centers[section].tolist(),
                         'central_initial_ratio': ratio, 'central_remote_ratio': localization,
                         'positive_projected_ligament': connected and central_width > spacing,
                         'separated': separated,
                         'force_N': observations[frame]['mean_grip_tensile_force_N'],
                         'engineering_strain': observations[frame]['engineering_strain'],
                         'coordinate_grip_engineering_strain': coordinate_strain,
                         'strain_difference': coordinate_strain - observations[frame]['engineering_strain'],
                         'neck_gate': bool(ratio <= .85 and localization is not None and localization <= .90
                                           and connected and central_width > spacing and not separated)})
    finally:
        x._mmap.close()
        failed._mmap.close()
    runs = [list(range(i, i+3)) for i in range(59) if all(row['neck_gate'] for row in rows[i:i+3])]
    peak = max(rows, key=lambda row: row['force_N'])
    return {'passed': bool(runs), 'consecutive_neck_frames': runs, 'history': rows,
            'force_comparison': {'peak_force_frame': peak['frame'], 'peak_force_N': peak['force_N'],
                                 'first_neck_frame': runs[0][0] if runs else None,
                                 'force_at_first_neck_N': rows[runs[0][0]]['force_N'] if runs else None},
            'method': 'Fixed reference axial slabs of thickness four particle spacings; narrowest median of three consecutive central slabs within 10% gauge length, normalized by the same reference slabs; remote median widths at 25–35%; y envelope plus one spacing. Current active XY raster at one spacing, dilated one pixel, 8-connected across gauge. Orthographic z projection.',
            'scope': 'Projected geometric ligament at particle resolution; does not establish a resolved three-dimensional material surface or experimental accuracy.'}


def audit(folder, parent_cache, parent_key, receipt_sha):
    folder, parent_cache = Path(folder), Path(parent_cache)
    request = {'parent_key': parent_key, 'parent_receipt_sha256': receipt_sha, 'from_frame': 50}
    receipt = carry.verify_carry(parent_cache, request)
    parent = parent_cache / 'result'
    before = carry.prefix_snapshot(parent, 50)
    for name, record in before['files'].items():
        require(carry.file_record(carry.contained_file(folder, name)) == record,
                'Finalization rewrote parent bytes: ' + name)
    plan = carry.read_json(folder / 'plan.json')
    frozen = carry.load_frozen_helper()
    require(all(frozen.validate_plan(plan).values()), 'Invalid original full plan')
    payload = frozen.decode(plan['signature_data'])
    summary, status = [carry.read_json(folder / name) for name in ('summary.json', 'status.json')]
    require(summary['completed'] is True and status['completed'] is True
            and status.get('failed', False) is False and status['frame'] == status['frames'] == 60
            and status['steps_completed'] == summary['steps'] == 60 * 965
            and summary['particles'] == [341584] and status['particles'] == 341584
            and summary['configs'] == completed_json(payload['configs'])
            and summary['materials'] == completed_json(payload['materials'])
            and summary['definition'] == payload['definition'], 'Completed endpoint/physical plan differs')
    config = payload['configs'][0]
    dt = config['dt']
    require(math.isclose(status['time'], 150e-6, rel_tol=1e-12)
            and math.isclose(summary['simulation_time'], 150e-6, rel_tol=1e-12), 'Wrong final time')
    with np.load(folder / 'final_state_0.npz', allow_pickle=False) as archive:
        require(set(archive.files) == set(plan['initial_state_fields']), 'Incomplete final State/History')
        arrays = {name: archive[name] for name in archive.files}
    for name, value in arrays.items():
        desc = plan['initial_state_fields'][name]
        require(list(value.shape) == desc['shape'] and str(value.dtype) == desc['dtype']
                and np.isfinite(value).all(), 'Malformed final field: ' + name)
    require(bool(arrays['valid']) and arrays['history_valid'].all()
            and (np.linalg.det(arrays['F']) > 0).all(), 'Invalid final material state')
    failed_final = arrays['history_failed'].astype(bool)
    require((arrays['history_plastic_strain'] >= 0).all()
            and np.all(arrays['history_damage'][failed_final] == 1)
            and np.all(arrays['history_stress'][failed_final] == 0), 'Failed particles retain stress or lack D=1')
    for name in ('mass', 'volume'):
        require((arrays[name] > 0).all() and frozen.descriptor(arrays[name]) == plan['initial_state_fields'][name],
                'Reference mass/volume changed')
    tolerance = max(1e-15, 4 * 60 * 965 * np.finfo(float).eps * 150e-6)
    require(abs(float(arrays['time']) - 150e-6) <= tolerance, 'Final State clock differs')
    with (folder / 'observables.csv').open(encoding='utf-8', newline='') as stream:
        observations = list(csv.DictReader(stream))
    with (folder / 'diagnostics.csv').open(encoding='utf-8', newline='') as stream:
        diagnostics = list(csv.DictReader(stream))
    require(len(observations) == len(diagnostics) == 61, 'Incomplete diagnostic history')
    for table, original in ((observations, before['observations']), (diagnostics, before['rows'])):
        for index, row in enumerate(table):
            typed_row(row, 'observables' if table is observations else 'diagnostics')
            require(abs(float(row['time']) - index * 965 * dt) < 1e-15, 'Diagnostic time differs')
            if index < 51:
                require(set(row) == set(original[index]) and all(typed_value(k, row[k], 'observables' if table is observations else 'diagnostics') == typed_value(k, v, 'observables' if table is observations else 'diagnostics')
                        for k, v in original[index].items()), 'Parent observations rewritten')
    maps = {name: np.load(folder / ('trajectory_' + name + '.npy'), mmap_mode='r', allow_pickle=False)
            for name in ('x', 'speed', 'damage', 'temperature', 'failed')}
    history = []
    previous_failed, previous_damage = np.zeros(341584, bool), np.zeros(341584)
    paths = [folder / 'frames_data' / f'frame_{i:05d}.npz' for i in range(61)]
    require(sorted((folder / 'frames_data').glob('*.npz')) == paths, 'Incomplete or extra raw frames')
    try:
        for name, value in maps.items():
            shape = (61, 341584, 3) if name == 'x' else (61, 341584)
            require(value.shape == shape and value.dtype == np.dtype('uint8' if name == 'failed' else 'float64'),
                    'Malformed full trajectory: ' + name)
        with np.load(folder / 'trajectory.npz', allow_pickle=False) as archive:
            times = archive['time']
            require(times.shape == (61,) and np.isfinite(times).all()
                    and np.max(np.abs(times - np.arange(61) * 965 * dt)) < 1e-15,
                    'Completed trajectory time index differs')
        for index, path in enumerate(paths):
            with np.load(path, allow_pickle=False) as archive:
                require(set(archive.files) == carry.FRAME_FIELDS, 'Unexpected raw frame fields')
                values = {name: archive[name] for name in archive.files}
            for name in maps:
                require(values[name].dtype == maps[name].dtype and np.isfinite(values[name]).all()
                        and np.array_equal(values[name], maps[name][index]),
                        'Raw and full trajectory differ: ' + name)
            require(values['time'].shape == () and values['time'].dtype == np.float64
                    and abs(float(values['time']) - index * 965 * dt) < 1e-15
                    and values['contact_impulse'].shape == (3,) and values['contact_impulse'].dtype == np.float64
                    and np.isfinite(values['contact_impulse']).all(),
                    'Malformed raw frame clock/impulse')
            failed, damage = values['failed'].astype(bool), values['damage']
            require(np.isin(values['failed'], (0, 1)).all() and ((damage >= 0) & (damage <= 1)).all()
                    and np.all(damage[failed] == 1)
                    and not np.any(previous_failed & ~failed) and not np.any(damage < previous_damage - 64*np.finfo(float).eps)
                    and (values['speed'] >= 0).all() and (values['temperature'] >= 0).all(), 'Invalid material history')
            if index == 0:
                reference = values['x'].copy()
                require(frozen.descriptor(reference) == plan['initial_state_fields']['x']
                        and not failed.any() and np.all(values['speed'] == 0)
                        and frozen.descriptor(damage) == plan['initial_state_fields']['history_damage']
                        and frozen.descriptor(values['temperature']) == plan['initial_state_fields']['history_temperature'],
                        'Initial reference geometry/history differs')
            require(int(observations[index]['failed_particles']) == int(failed.sum())
                    and float(observations[index]['max_damage']) == float(damage.max())
                    and float(diagnostics[index]['min_J']) > 0
                    and math.isclose(float(diagnostics[index]['mass']), plan['initial_mass_kg'], rel_tol=1e-12),
                    'Observation disagrees with raw history')
            left, right = payload['definition']['grip_inner_faces_x_m']
            separation = inspect_state(values['x'], failed, dx=config['domain_size']/config['n_grid'],
                n_grid=config['n_grid'], split_plane=.5*sum(payload['definition']['bar_ends_x_m']),
                left_grip=reference[:, 0] <= left, right_grip=reference[:, 0] >= right)
            history.append({'frame': index, 'time_s': float(values['time']), **separation})
            previous_failed, previous_damage = failed, damage
        for field, state in [('x', 'x'), ('damage', 'history_damage'), ('temperature', 'history_temperature'), ('failed', 'history_failed')]:
            np.testing.assert_array_equal(maps[field][-1], arrays[state])
        np.testing.assert_allclose(maps['speed'][-1], np.linalg.norm(arrays['v'], axis=1), rtol=1e-14, atol=0)
        metrics = frozen.final_metrics(arrays, reference, payload['definition']['particle_spacing_m'])
        for name in ('central_material_section_force_N', 'maximum_equivalent_plastic_strain',
                     'max_damage', 'failed_particles', 'central_active_particles'):
            require(math.isclose(metrics[name], float(observations[-1][name]), rel_tol=1e-10), 'Final metric differs: ' + name)
        for name in ('min_J', 'max_J'):
            require(math.isclose(metrics[name], float(diagnostics[-1][name]), rel_tol=1e-12), 'Final deformation metric differs')
        momentum = np.sum(arrays['mass'][:, None]*arrays['v'], axis=0)
        reaction = momentum - np.asarray(summary['external_impulse'])
        np.testing.assert_allclose(summary['boundary_impulse'], reaction, rtol=1e-10, atol=0)
        np.testing.assert_allclose([float(observations[-1]['boundary_impulse_'+axis]) for axis in 'xyz'],
                                   reaction, rtol=1e-10, atol=0)
        np.testing.assert_array_equal(summary['contact_impulse_on_body_1'], values['contact_impulse'])
        for index, key in ((0, 'initial'), (-1, 'final')):
            require(set(observations[index]) == set(summary['observables'][key])
                    and all(typed_value(name, observations[index][name], 'observables') == typed_value(name, value, 'observables')
                            for name, value in summary['observables'][key].items()),
                    'Summary observer endpoints differ from complete history')
    finally:
        for value in maps.values():
            value._mmap.close()
    first_failure = next((row['frame'] for row in history if row['failed_particles']), None)
    first_separation = next((row['frame'] for row in history if row['reference_grip_sets_grid_decoupled']), None)
    gates = {'material_failure_observed': first_failure is not None,
             'final_reference_grips_grid_decoupled': history[-1]['reference_grip_sets_grid_decoupled'],
             'failure_precedes_separation_at_saved_resolution': first_failure is not None
                and first_separation is not None and first_failure < first_separation,
             'no_separation_before_failed_material': all(not row['reference_grip_sets_grid_decoupled']
                or row['failed_particles'] > 0 for row in history)}
    names = ['plan.json', 'numerical-source.json', 'summary.json', 'status.json', 'final_state_0.npz',
             'observables.csv', 'diagnostics.csv', 'trajectory.npz']
    names += ['trajectory_' + name + '.npy' for name in maps]
    names += [path.relative_to(folder).as_posix() for path in paths]
    return {'schema': 'jc-full-result-review-v2', 'passed': True, 'full_protocol_completed': True,
            'fracture_gates': gates, 'resolved_saved_history_fracture': all(gates.values()),
            'production_eligible': False, 'parent_receipt_sha256': receipt_sha,
            'parent_cache_key': receipt['cache_key'], 'producer_source': receipt['source'],
            'final_metrics': metrics, 'separation_history': history,
            'first_saved_failure_frame': first_failure, 'first_saved_separation_frame': first_separation,
            'input_sha256': {name: carry.digest(folder/name) for name in names},
            'review_source_sha256': carry.digest(Path(__file__)),
            'scope': 'Completed immutable protocol and saved-state grid-decoupling audit. Does not establish experimental accuracy, fracture-energy convergence, resolved fragment surfaces, or uninterrupted-run equivalence. Same-frame failure and separation do not resolve causal ordering.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--parent-cache', type=Path, required=True)
    parser.add_argument('--parent-key', required=True)
    parser.add_argument('--receipt-sha', required=True)
    args = parser.parse_args()
    context = carry.read_json(args.records / 'context.json')
    prefix = carry.read_json(args.records / 'parent-prefix.json')
    receipt = carry.read_json(args.records / 'parent-receipt.json')
    runtime = carry.read_json(args.records / 'worker-runtime.json')
    lock = carry.read_json(args.records / 'environment-lock.json')
    require(context['request']['parent_key'] == args.parent_key
            and context['request']['parent_receipt_sha256'] == args.receipt_sha
            and carry.digest(args.records / 'parent-receipt.json') == args.receipt_sha
            and context['source'] == receipt['source'], 'Original controller/parent identity differs')
    require(carry.environment_matches(runtime['original'], runtime['current'])
            and carry.environment_matches(runtime['current'], carry.environment())
            and carry.environment_lock() == lock, 'Recorded numerical runtime differs')
    require(prefix == carry.prefix_snapshot(args.parent_cache / 'result', 50),
            'Original parent snapshot differs from restored parent')
    report = audit(args.folder, args.parent_cache, args.parent_key, args.receipt_sha)
    carry.verify_imports()
    report['original_controller_records_sha256'] = {
        name: carry.digest(args.records / name) for name in
        ('context.json', 'parent-prefix.json', 'parent-receipt.json',
         'worker-runtime.json', 'environment-lock.json', 'loaded-numerics.json')}
    require(not args.output.resolve().is_relative_to(args.folder.resolve()), 'Output must be outside immutable result')
    report['coordinate_neck_audit'] = coordinate_neck_audit(args.folder, report)
    report['production_eligible'] = False
    report['visual_neck_gate_passed'] = report['coordinate_neck_audit']['passed']
    carry.save(args.output, report)
    print(json.dumps({key: report[key] for key in ('passed', 'full_protocol_completed', 'resolved_saved_history_fracture')}))


if __name__ == '__main__':
    main()
