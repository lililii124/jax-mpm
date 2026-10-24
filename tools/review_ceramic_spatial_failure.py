"""Describe ceramic failed-particle locations from saved raw data; no solver imports."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0, 'minimum': None, 'quantiles_05_25_50_75_95': None,
                'maximum': None, 'mean': None}
    return {'count': len(values), 'minimum': values.min(axis=0).tolist(),
            'quantiles_05_25_50_75_95': np.quantile(values, [.05, .25, .5, .75, .95], axis=0).tolist(),
            'maximum': values.max(axis=0).tolist(), 'mean': values.mean(axis=0).tolist()}


def analyze(reference, x, v, failed, volume):
    reference, x, v, volume = map(np.asarray, (reference, x, v, volume))
    failed = np.asarray(failed)
    n = len(volume)
    if (reference.shape != (n, 3) or x.shape != reference.shape or v.shape != reference.shape
            or failed.shape != (n,) or not n
            or not all(np.isfinite(a).all() for a in (reference, x, v, volume))
            or not np.all(volume > 0) or not np.all((failed == 0) | (failed == 1))):
        raise ValueError('Invalid finite particle arrays or failure mask')
    failed = failed.astype(bool)
    radius = np.linalg.norm(reference[:, (0, 2)] - (.125, .125), axis=1)
    edges = {'x_low': reference[:, 0] <= .043, 'x_high': reference[:, 0] >= .207,
             'z_low': reference[:, 2] <= .043, 'z_high': reference[:, 2] >= .207}
    edge = np.logical_or.reduce(list(edges.values()))
    total_failed_volume = float(volume[failed].sum())
    count = int(failed.sum())

    def fraction(mask):
        selected = failed & mask
        return {'failed_particle_count': int(selected.sum()),
                'fraction_of_failed_count': float(selected.sum() / count) if count else None,
                'failed_reference_volume_m3': float(volume[selected].sum()),
                'fraction_of_failed_reference_volume': float(volume[selected].sum() / total_failed_volume) if count else None}

    outside = (x[:, 1] < .075) | (x[:, 1] > .1)
    bins = np.array([0., .01, .02, .03, .045, .06, .08, .10, .13])
    return {'particles': n, 'failed_particle_count': count,
            'failed_reference_volume_m3': total_failed_volume,
            'failed_fraction_of_total_reference_volume': total_failed_volume / float(volume.sum()),
            'reference_radius_m': distribution(radius[failed]),
            'reference_radius_bins_m': bins.tolist(),
            'failed_reference_radius_bin_counts': np.histogram(radius[failed], bins)[0].tolist(),
            'failed_reference_radius_bin_volume_m3': np.histogram(radius[failed], bins, weights=volume[failed])[0].tolist(),
            'within_45mm_of_impact_center': fraction(radius <= .045 + 1e-14),
            'four_edge_8mm_clamp_band_union': fraction(edge),
            'individual_clamp_bands_overlapping_at_corners': {key: fraction(mask) for key, mask in edges.items()},
            'outside_original_25mm_plate_thickness': fraction(outside),
            'failed_current_xyz_m': distribution(x[failed]),
            'failed_current_velocity_xyz_m_per_s': distribution(v[failed]),
            'failed_current_speed_m_per_s': distribution(np.linalg.norm(v[failed], axis=1)),
            'failed_displacement_xyz_m': distribution((x-reference)[failed])}


def review(folder):
    folder = Path(folder).resolve()
    names = ['checkpoint.json', 'status.json', 'trajectory_x.npy', 'trajectory_failed.npy']
    cp, status = (json.loads((folder/name).read_text(encoding='utf-8')) for name in names[:2])
    frame = int(cp['frame'])
    if (frame != status['frame'] or not 0 < frame <= 60 or status['frames'] != 60
            or len(cp['states']) != 1 or status.get('completed') is not (frame == 60)):
        raise ValueError('Checkpoint endpoint differs from original complete 60-frame plan')
    name = cp['states'][0]
    if Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('Unsafe checkpoint state path')
    names.append(name)
    before = {name: {'bytes': (folder/name).stat().st_size, 'sha256': digest(folder/name)} for name in names}
    saved_x = np.load(folder/'trajectory_x.npy', mmap_mode='r', allow_pickle=False)
    saved_failed = np.load(folder/'trajectory_failed.npy', mmap_mode='r', allow_pickle=False)
    try:
        with np.load(folder/name, allow_pickle=False) as state:
            x, v, failed, volume = (state[key] for key in ('x', 'v', 'history_failed', 'volume'))
            reference = saved_x[0]
            if (len(volume) != 386420 or saved_x.shape != (61, len(volume), 3)
                    or saved_failed.shape != (61, len(volume))
                    or not np.array_equal(saved_x[frame], x)
                    or not np.array_equal(saved_failed[frame], failed)
                    or np.any(saved_failed[0]) or not np.isclose(float(state['time']), status['time'], rtol=1e-12, atol=1e-14)):
                raise ValueError('Original saved particle ordering/endpoint differs from checkpoint')
            lower, upper = np.array((.035, .075, .035)), np.array((.215, .1, .215))
            if not np.all((reference >= lower) & (reference <= upper)):
                raise ValueError('Reference particles lie outside the original plate')
            axes = [np.unique(reference[:, i]) for i in range(3)]
            if (np.prod([len(axis) for axis in axes]) != len(volume)
                    or any(len(axis)<2 or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-10, atol=1e-14) for axis in axes)):
                raise ValueError('Reference particles do not form the original regular block')
            report = analyze(reference, x, v, failed, volume)
            observation = cp['observations'][-1]
            if (observation['failed_particle_count'] != report['failed_particle_count']
                    or not np.isclose(observation['failed_reference_volume_m3'], report['failed_reference_volume_m3'], rtol=1e-10, atol=1e-15)):
                raise ValueError('Observer failed count/reference volume differs')
    finally:
        saved_x._mmap.close()
        saved_failed._mmap.close()
    if any((folder/name).stat().st_size != record['bytes'] or digest(folder/name) != record['sha256'] for name, record in before.items()):
        raise ValueError('Raw input changed during read-only spatial review')
    report.update(schema='ceramic-spatial-failure-v1', accepted_saved_frame=frame,
                  target_saved_frame=60, full_protocol_completed=frame == 60,
                  source_consistency_passed=True, physical_admission=False,
                  sources=before, review_script_sha256=digest(__file__),
                  reference_definition={'impact_center_xz_m': [.125, .125], 'center_radius_m': .045,
                      'plate_lower_xyz_m': [.035, .075, .035], 'plate_upper_xyz_m': [.215, .1, .215],
                      'clamp_band_width_m': .008, 'center_radius_comparison_tolerance_m': 1e-14,
                      'coordinates': 'x,z in plane; y through thickness'},
                  interpretation='Descriptive failed-particle spatial statistics only. JH-2 damage D is not a crack surface. Failure uses checkpoint history_failed, with reference-volume weighting. A frame-10 prefix need not have debris outside the plate thickness. Thickness escape is measured without a particle-spacing margin and differs from the original full-plan reviewer gate.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    output = args.output or args.folder/'spatial-failure-review.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing spatial report')
    if output.resolve().is_relative_to(args.folder.resolve()) and output.name != 'spatial-failure-review.json':
        raise ValueError('Output must not replace raw evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(output), 'sha256': digest(output),
                      'frame': result['accepted_saved_frame'], 'failed': result['failed_particle_count']}))


if __name__ == '__main__':
    main()
