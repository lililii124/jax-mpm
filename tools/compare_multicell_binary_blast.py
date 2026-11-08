"""Compare complete, like-for-like binary blast replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def read_run(path, energy_tolerance, allow_incomplete=False):
    meta = json.loads((path/'inputs.json').read_text(encoding='utf-8'))
    rows = json.loads((path/'history.json').read_text(encoding='utf-8'))
    if (meta['diagnostic_only'] or not meta['connected_faces'] or meta['detached_voxels']
            or meta.get('connected_components', 1) != 1):
        raise ValueError(f'{path}: geometry was not accepted as connected')
    if meta['steps'] != meta['settings']['steps']:
        raise ValueError(f'{path}: replay does not cover the configured time horizon')
    if not rows:
        raise ValueError(f'{path}: frame history is empty')
    last = rows[-1]['step']
    if last < 0 or last > meta['steps'] or (last < meta['steps'] and not allow_incomplete):
        raise ValueError(f'{path}: replay is incomplete or exceeds the configured horizon')
    expected = (list(range(0, last + 1, meta['frame_interval'])) if last < meta['steps'] else
                list(range(0, meta['steps'], meta['frame_interval'])) + [meta['steps']])
    if [row['step'] for row in rows] != expected:
        raise ValueError(f'{path}: frame history is incomplete or has inconsistent intervals')
    times = np.array([row['time_s'] for row in rows])
    if not np.allclose(times, np.array(expected)*meta['settings']['dt'], rtol=1e-9, atol=1e-14):
        raise ValueError(f'{path}: time history does not match the configured time step')
    if any(not row['valid'] or row['min_J'] <= 0 for row in rows):
        raise ValueError(f'{path}: invalid deformation')
    fields = ('back_displacement_max_m', 'kinetic_J', 'elastic_J', 'pressure_work_J',
              'boundary_work_J', 'energy_residual_J', 'min_J', 'momentum_residual')
    if not all(np.isfinite(row[key]) for row in rows for key in fields):
        raise ValueError(f'{path}: non-finite history')
    scale = max(max(abs(row['pressure_work_J'] + row['boundary_work_J']) for row in rows),
                max(abs(row['kinetic_J']) + abs(row['elastic_J']) for row in rows), 1e-12)
    energy_error = max(abs(row['energy_residual_J']) for row in rows)/scale
    if energy_error > energy_tolerance:
        raise ValueError(f'{path}: normalized energy residual {energy_error:g} exceeds tolerance')
    return meta, rows, energy_error


def compare(baseline, candidate, energy_tolerance=.02):
    if not 0 < energy_tolerance < 1:
        raise ValueError('Energy tolerance must be between zero and one')
    a, ah, ae = read_run(baseline, energy_tolerance)
    b, bh, be = read_run(candidate, energy_tolerance)
    for field in ('settings', 'steps', 'frame_interval', 'source_sha256', 'jax_version', 'numpy_version'):
        if a[field] != b[field]:
            raise ValueError(f'Replay {field} differs')
    if a.get('empty_column_load','omit')!=b.get('empty_column_load','omit'):
        raise ValueError('Projected pressure allocation differs')
    if a.get('support_mode','single') != b.get('support_mode','single'):
        raise ValueError('Material-to-grid support mode differs')
    if not np.isclose(a['binary_mass_kg'], b['binary_mass_kg'], rtol=1e-12, atol=1e-12):
        raise ValueError('Reconstructed material mass differs')
    if a['solid_mask_sha256'] == b['solid_mask_sha256']:
        raise ValueError('Baseline and candidate have identical solid geometry')
    ap = max(row['back_displacement_max_m'] for row in ah)
    bp = max(row['back_displacement_max_m'] for row in bh)
    if ap <= 0:
        raise ValueError('Baseline has zero recorded response')
    return dict(baseline=str(baseline), candidate=str(candidate),
                baseline_geometry_sha256=a['geometry_sha256'], candidate_geometry_sha256=b['geometry_sha256'],
                mass_kg=a['binary_mass_kg'], horizon_s=ah[-1]['time_s'],
                baseline_sampled_peak_m=ap, candidate_sampled_peak_m=bp,
                sampled_peak_reduction_fraction=(ap-bp)/ap,
                baseline_max_energy_error=ae, candidate_max_energy_error=be,
                baseline_populated_column_fraction=a.get('populated_column_fraction'),
                candidate_populated_column_fraction=b.get('populated_column_fraction'),
                energy_tolerance=energy_tolerance,
                limitation=('Peak displacement is sampled only at stored frames. '
                            'Projected pressure uses '+a.get('empty_column_load','omit')+
                            ' allocation; cavity-side pressure is omitted. This is an elastic replay, not experimental validation.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--energy-tolerance', type=float, default=.02)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite comparison')
    result = compare(args.baseline, args.candidate, args.energy_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
