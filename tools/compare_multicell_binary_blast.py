"""Compare complete, like-for-like binary blast replays."""
import argparse
import json
from pathlib import Path

import numpy as np


def physical_edge_width(settings):
    widths = settings.get('fixed_edge_width')
    if widths is None:
        widths = [1.01*settings['cell_size'][axis]/settings['cell'][axis] for axis in (0, 1)]
    widths = np.asarray(widths, dtype=float)
    if widths.shape != (2,) or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError('Invalid physical fixed-edge width')
    return widths


def verify_physical_model_compatibility(proxy, binary):
    fields = ('cell', 'cell_size', 'skin', 'skin_layers', 'volume_fraction', 'skin_young',
              'core_young', 'solid_density', 'pressure', 'pulse', 'blast_radius')
    if any(proxy[key] != binary[key] for key in fields):
        raise ValueError('Gradient and binary simulations use different physical settings')
    if not np.allclose(physical_edge_width(proxy), physical_edge_width(binary), rtol=0, atol=1e-12):
        raise ValueError('Gradient and binary simulations use different fixed-edge widths')
    a, b = float(proxy['dt'])*int(proxy['steps']), float(binary['dt'])*int(binary['steps'])
    if not np.isfinite([a, b]).all() or min(a, b) <= 0 or not np.isclose(a, b, rtol=1e-10, atol=1e-14):
        raise ValueError('Gradient and binary simulations use different physical time horizons')


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
    peak_sampling = meta.get('peak_sampling', 'saved-frames')
    if peak_sampling not in ('saved-frames', 'every-step'):
        raise ValueError(f'{path}: unsupported peak sampling mode')
    point_sampling = meta.get('bottom_point_peak_sampling', 'not-recorded')
    if point_sampling not in ('not-recorded', 'every-step') or (point_sampling == 'every-step' and peak_sampling != 'every-step'):
        raise ValueError(f'{path}: unsupported bottom point peak sampling mode')
    if peak_sampling == 'every-step':
        fields += ('back_displacement_interval_peak_m',)
        if any(row['back_displacement_interval_peak_m'] + 1e-15 < row['back_displacement_max_m'] for row in rows):
            raise ValueError(f'{path}: interval peak is below the saved-frame response')
    if 'bottom_plane_sample_fraction' in meta:
        fraction = meta['bottom_plane_sample_fraction']
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError(f'{path}: invalid bottom-plane sample fraction')
        surface_fields = ['back_displacement_p90_m', 'back_displacement_median_m']
        if fraction > 0: surface_fields.append('bottom_plane_displacement_max_m')
        if fraction < 1: surface_fields.append('internal_underside_displacement_max_m')
        fields += tuple(surface_fields)
        for row in rows:
            if any(row[key] < 0 or row[key] > row['back_displacement_max_m'] + 1e-15 for key in surface_fields):
                raise ValueError(f'{path}: inconsistent surface displacement statistics')
    if not all(np.isfinite(row[key]) for row in rows for key in fields):
        raise ValueError(f'{path}: non-finite history')
    scale = max(max(abs(row['pressure_work_J'] + row['boundary_work_J']) for row in rows),
                max(abs(row['kinetic_J']) + abs(row['elastic_J']) for row in rows), 1e-12)
    energy_error = max(abs(row['energy_residual_J']) for row in rows)/scale
    if energy_error > energy_tolerance:
        raise ValueError(f'{path}: normalized energy residual {energy_error:g} exceeds tolerance')
    return meta, rows, energy_error


def compare(baseline, candidate, energy_tolerance=.02, allow_incomplete_candidate=False):
    if not 0 < energy_tolerance < 1:
        raise ValueError('Energy tolerance must be between zero and one')
    a, ah, ae = read_run(baseline, energy_tolerance)
    b, bh, be = read_run(candidate, energy_tolerance, allow_incomplete=allow_incomplete_candidate)
    for field in ('settings', 'steps', 'frame_interval', 'source_sha256', 'jax_version', 'numpy_version'):
        if a[field] != b[field]:
            raise ValueError(f'Replay {field} differs')
    if a.get('empty_column_load','omit')!=b.get('empty_column_load','omit'):
        raise ValueError('Projected pressure allocation differs')
    if a.get('support_mode','single') != b.get('support_mode','single'):
        raise ValueError('Material-to-grid support mode differs')
    if a.get('peak_sampling', 'saved-frames') != b.get('peak_sampling', 'saved-frames'):
        raise ValueError('Peak sampling mode differs')
    if a.get('bottom_point_peak_sampling', 'not-recorded') != b.get('bottom_point_peak_sampling', 'not-recorded'):
        raise ValueError('Bottom point peak sampling mode differs')
    if a.get('back_sampling', 'column-underside') != b.get('back_sampling', 'column-underside'):
        raise ValueError('Back-surface sampling mode differs')
    if not np.isclose(a['binary_mass_kg'], b['binary_mass_kg'], rtol=1e-12, atol=1e-12):
        raise ValueError('Reconstructed material mass differs')
    if a['solid_mask_sha256'] == b['solid_mask_sha256']:
        raise ValueError('Baseline and candidate have identical solid geometry')
    peak_field = ('back_displacement_interval_peak_m' if a.get('peak_sampling', 'saved-frames') == 'every-step'
                  else 'back_displacement_max_m')
    ap = max(row[peak_field] for row in ah)
    bp = max(row[peak_field] for row in bh)
    if ap <= 0:
        raise ValueError('Baseline has zero recorded response')
    candidate_complete = bh[-1]['step'] == b['steps']
    return dict(baseline=str(baseline), candidate=str(candidate),
                baseline_geometry_sha256=a['geometry_sha256'], candidate_geometry_sha256=b['geometry_sha256'],
                mass_kg=a['binary_mass_kg'], horizon_s=ah[-1]['time_s'],
                baseline_sampled_peak_m=ap, candidate_sampled_peak_m=bp,
                sampled_peak_reduction_fraction=(ap-bp)/ap,
                baseline_max_energy_error=ae, candidate_max_energy_error=be,
                baseline_populated_column_fraction=a.get('populated_column_fraction'),
                candidate_populated_column_fraction=b.get('populated_column_fraction'),
                energy_tolerance=energy_tolerance,
                candidate_complete=candidate_complete, candidate_horizon_s=bh[-1]['time_s'],
                peak_sampling=a.get('peak_sampling', 'saved-frames'),
                back_sampling=a.get('back_sampling', 'column-underside'),
                limitation=(('Peak displacement is sampled at every integration step. ' if peak_field != 'back_displacement_max_m'
                             else 'Peak displacement is sampled only at stored frames. ')+
                            ('' if candidate_complete else 'Candidate is incomplete; its reported reduction is an upper bound. ')+
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
