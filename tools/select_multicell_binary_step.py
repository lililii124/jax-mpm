"""Select a fixed-mass candidate from verified binary blast replays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tools.compare_multicell_binary_blast import compare, read_run
from tools.reconstruct_multicell_blast import mask_digest, reconstruct
from tools.multicell_feature_filter import single_voxel_run_fraction


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_early_rejection(meta, history, result, outcome, minimum_reduction):
    observed = result['candidate_sampled_peak_m']
    cutoff = result['baseline_sampled_peak_m'] * (1 - minimum_reduction)
    if (outcome['reason'] != 'peak-threshold' or outcome['step'] != history[-1]['step'] or
            not np.isclose(outcome['threshold_m'], meta['rejection_peak_m'], rtol=1e-12, atol=0.) or
            not np.isclose(outcome['observed_peak_m'], observed, rtol=1e-12, atol=0.) or
            observed < outcome['threshold_m'] or observed < cutoff):
        raise ValueError('Incomplete replay does not prove candidate rejection')


def evaluate_secondary_constraints(base_meta, base_rows, meta, rows, bottom_ratio=None, p90_ratio=None,
                                   min_bottom_coverage=None, common_bottom_coverage=None,
                                   min_common_bottom_coverage=None):
    result = {'passed': True, 'sampling': 'saved-frames'}
    for key, limit, label in (('bottom_plane_displacement_max_m', bottom_ratio, 'bottom_peak'),
                              ('back_displacement_p90_m', p90_ratio, 'p90_peak')):
        if limit is None:
            continue
        if any(key not in row for row in base_rows+rows):
            raise ValueError('Replay lacks surface statistics required by '+label)
        baseline = max(row[key] for row in base_rows)
        candidate = max(row[key] for row in rows)
        if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(candidate) or candidate < 0:
            raise ValueError('Invalid surface statistic for '+label)
        ratio = candidate/baseline
        result[label] = {'baseline_m': baseline, 'candidate_m': candidate, 'ratio': ratio, 'limit': limit}
        result['passed'] &= ratio <= limit
    if min_bottom_coverage is not None:
        fraction = meta.get('bottom_plane_sample_fraction')
        if fraction is None or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Replay lacks valid bottom-plane coverage')
        result['bottom_coverage'] = {'baseline': base_meta.get('bottom_plane_sample_fraction'),
                                     'candidate': fraction, 'minimum': min_bottom_coverage}
        result['passed'] &= fraction >= min_bottom_coverage
    if min_common_bottom_coverage is not None:
        fraction = common_bottom_coverage
        if fraction is None or not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Replay lacks valid common bottom-plane coverage')
        result['common_bottom_coverage'] = {'candidate': fraction, 'minimum': min_common_bottom_coverage}
        result['passed'] &= fraction >= min_common_bottom_coverage
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('baseline_design', type=Path)
    parser.add_argument('baseline_geometry', type=Path)
    parser.add_argument('baseline_replay', type=Path)
    parser.add_argument('proposals', type=Path)
    parser.add_argument('--replays', type=Path, nargs='+', required=True,
                        help='Replay directories, in proposed-candidate order')
    parser.add_argument('--minimum-reduction', type=float, default=1e-6)
    parser.add_argument('--max-bottom-peak-ratio', type=float)
    parser.add_argument('--max-p90-peak-ratio', type=float)
    parser.add_argument('--min-bottom-coverage', type=float)
    parser.add_argument('--min-common-bottom-coverage', type=float,
                        help='Minimum fraction of projected columns with bottom material in both designs')
    parser.add_argument('--constraint-reference-replay', type=Path)
    parser.add_argument('--constraint-reference-geometry', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.minimum_reduction < 1:
        raise ValueError('Minimum reduction must lie in [0, 1)')
    for value in (args.max_bottom_peak_ratio, args.max_p90_peak_ratio):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError('Secondary displacement ratios must be finite and positive')
    if args.min_bottom_coverage is not None and not 0 <= args.min_bottom_coverage <= 1:
        raise ValueError('Bottom coverage must lie in [0, 1]')
    if args.min_common_bottom_coverage is not None and not 0 <= args.min_common_bottom_coverage <= 1:
        raise ValueError('Common bottom coverage must lie in [0, 1]')
    if (args.constraint_reference_replay is None) != (args.constraint_reference_geometry is None):
        raise ValueError('Supply both constraint reference replay and geometry')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    manifest = json.loads((args.proposals / 'manifest.json').read_text(encoding='utf-8'))
    feature_limit = manifest.get('max_single_voxel_fraction')
    if feature_limit is not None:
        if not 0 <= feature_limit <= 1 or manifest.get('feature_filter_sha256') != sha256(Path(__file__).with_name('multicell_feature_filter.py')):
            raise ValueError('Proposal feature filter is invalid or has changed')
    if (hashlib.sha256(json.dumps(inputs['binding'], sort_keys=True).encode()).hexdigest() != inputs['signature'] or
            manifest['signature'] != inputs['signature'] or manifest['inputs_sha256'] != sha256(args.inputs)):
        raise ValueError('Proposal inputs do not match the optimizer binding')
    settings = inputs['binding']['settings']
    base_meta, base_rows, _ = read_run(args.baseline_replay, .02)
    if base_meta['geometry_sha256'] != sha256(args.baseline_geometry):
        raise ValueError('Baseline replay geometry differs')
    with np.load(args.baseline_geometry, allow_pickle=False) as data:
        solid = data['solid_mask']
        base_bottom = np.asarray(solid[0], dtype=bool).copy()
        if mask_digest(solid) != base_meta['solid_mask_sha256']:
            raise ValueError('Baseline solid mask differs from its replay')
        if not np.any(solid, axis=0).all():
            raise ValueError('Baseline does not cover the projected pressure footprint')
    constraint_meta, constraint_rows = base_meta, base_rows
    if args.constraint_reference_replay is not None:
        constraint_meta, constraint_rows, _ = read_run(args.constraint_reference_replay, .02)
        if constraint_meta['geometry_sha256'] != sha256(args.constraint_reference_geometry):
            raise ValueError('Constraint reference geometry differs from replay')
        for field in ('settings', 'steps', 'frame_interval', 'source_sha256', 'jax_version', 'numpy_version'):
            if constraint_meta[field] != base_meta[field]:
                raise ValueError('Constraint reference replay differs: '+field)
        for field, default in (('empty_column_load', 'omit'), ('support_mode', 'single'),
                               ('peak_sampling', 'saved-frames'), ('back_sampling', 'column-underside')):
            if constraint_meta.get(field, default) != base_meta.get(field, default):
                raise ValueError('Constraint reference replay differs: '+field)
        if not np.isclose(constraint_meta['binary_mass_kg'], base_meta['binary_mass_kg'], rtol=1e-12, atol=1e-12):
            raise ValueError('Constraint reference mass differs')
        with np.load(args.constraint_reference_geometry, allow_pickle=False) as data:
            reference_solid = data['solid_mask']
            if mask_digest(reference_solid) != constraint_meta['solid_mask_sha256']:
                raise ValueError('Constraint reference solid mask differs')
            if reference_solid.shape[1:] != base_bottom.shape:
                raise ValueError('Constraint reference bottom grid differs')
            base_bottom = np.asarray(reference_solid[0], dtype=bool).copy()
    with np.load(args.baseline_design, allow_pickle=False) as data:
        base_mask = reconstruct(data['rho_cell'], settings)[-1]['solid_mask_sha256']
    if (base_mask != manifest['baseline_solid_mask_sha256'] or
            base_mask != base_meta['solid_mask_sha256']):
        raise ValueError('Baseline design and replay differ')
    proposed = [row for row in manifest['steps'] if row['status'] == 'proposed']
    if len(proposed) != len(args.replays) or not proposed:
        raise ValueError('Supply one replay for every proposed candidate')
    comparisons = []
    for row, replay in zip(proposed, args.replays):
        design = args.proposals / row['design']
        geometry = args.proposals / row['geometry']
        if sha256(design) != row['design_sha256'] or sha256(geometry) != row['geometry_sha256']:
            raise ValueError('Proposal files do not match their manifest')
        meta, history, _ = read_run(replay, .02, allow_incomplete=True)
        if meta['geometry_sha256'] != row['geometry_sha256'] or meta['solid_mask_sha256'] != row['solid_mask_sha256']:
            raise ValueError('Replay does not match the proposed geometry')
        with np.load(geometry, allow_pickle=False) as data:
            solid = data['solid_mask']
            if solid.shape[1:] != base_bottom.shape:
                raise ValueError('Candidate and baseline bottom grids differ')
            common_bottom_coverage = float(np.mean(base_bottom & np.asarray(solid[0], dtype=bool)))
            cell = data['cell_mask']
            if mask_digest(solid) != row['solid_mask_sha256']:
                raise ValueError('Proposed solid mask differs from its manifest')
            if not np.any(solid, axis=0).all():
                raise ValueError('Candidate does not cover the projected pressure footprint')
            if feature_limit is not None and single_voxel_run_fraction(cell) > feature_limit:
                raise ValueError('Candidate exceeds the configured feature limit')
        with np.load(design, allow_pickle=False) as data:
            reconstructed = reconstruct(data['rho_cell'], settings)
        report = reconstructed[-1]
        if (not np.array_equal(cell, reconstructed[0]) or
                report['solid_mask_sha256'] != row['solid_mask_sha256'] or report['components'] != 1 or
                not report['faces_connected'] or report['populated_column_fraction'] != 1.):
            raise ValueError('Candidate design and connected geometry differ')
        result = compare(args.baseline_replay, replay, allow_incomplete_candidate=True)
        early_rejected = not result['candidate_complete']
        if early_rejected:
            outcome = json.loads((replay / 'outcome.json').read_text(encoding='utf-8'))
            verify_early_rejection(meta, history, result, outcome, args.minimum_reduction)
        secondary = None
        if not early_rejected and any(value is not None for value in
                                      (args.max_bottom_peak_ratio, args.max_p90_peak_ratio, args.min_bottom_coverage,
                                       args.min_common_bottom_coverage)):
            secondary = evaluate_secondary_constraints(constraint_meta, constraint_rows, meta, history,
                args.max_bottom_peak_ratio, args.max_p90_peak_ratio, args.min_bottom_coverage,
                common_bottom_coverage, args.min_common_bottom_coverage)
        comparisons.append({'step': row['step'], 'design': str(design),
                            'solid_mask_sha256': row['solid_mask_sha256'], 'early_rejected': early_rejected,
                            'secondary_constraints': secondary,
                            'comparison': result})
    completed = [row for row in comparisons if not row['early_rejected'] and
                 (row['secondary_constraints'] is None or row['secondary_constraints']['passed'])]
    best = min(completed, key=lambda row: row['comparison']['candidate_sampled_peak_m']) if completed else None
    accepted = best is not None and best['comparison']['sampled_peak_reduction_fraction'] > args.minimum_reduction
    selected = Path(best['design']) if accepted else args.baseline_design
    args.output.mkdir(parents=True)
    target = args.output / 'selected-design.npz'
    shutil.copyfile(selected, target)
    result = {'accepted': accepted, 'selected_step': best['step'] if accepted else 0.,
              'selected_design_sha256': sha256(target),
              'selected_solid_mask_sha256': best['solid_mask_sha256'] if accepted else base_mask,
              'minimum_reduction': args.minimum_reduction, 'comparisons': comparisons,
              'qualification': 'Sampled peak comparison for connected elastic solids under matched projected pressure; not experimental validation.'}
    temp = args.output / 'selection.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(args.output / 'selection.json')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
