"""Iterate full-domain shared-cell designs with binary MPM acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np

from tools.compare_multicell_binary_blast import read_run, verify_physical_model_compatibility


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(*args):
    subprocess.run([sys.executable, '-m', *map(str, args)], check=True)


def gradient_cache_path(cache, root, inputs_path, design_path):
    inputs = json.loads(inputs_path.read_text(encoding='utf-8'))
    binding = inputs['binding']
    signature = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if signature != inputs['signature']:
        raise ValueError('Gradient cache input signature differs')
    for name, expected in binding['source_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError('Gradient cache source differs: '+name)
    versions = tuple(importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy'))
    if versions != (binding['jax'], binding['jaxlib'], binding['numpy']):
        raise ValueError('Gradient cache runtime versions differ')
    parts = [sha256(inputs_path), sha256(design_path), sha256(root/'tools/gradient_multicell_at_design.py')]
    return cache/(hashlib.sha256(('\n'.join(parts)).encode()).hexdigest()+'.npz')


def validate_cached_gradient(path, inputs_path, design_path):
    with np.load(path, allow_pickle=False) as data, np.load(design_path, allow_pickle=False) as design:
        gradient = np.asarray(data['gradient'])
        objective = float(data['objective'])
        if (str(data['inputs_sha256'].item()) != sha256(inputs_path) or
                str(data['design_sha256'].item()) != sha256(design_path) or
                not np.array_equal(data['logits'], np.asarray(design['logits']).reshape(-1)) or
                not np.array_equal(data['rho_cell'], design['rho_cell']) or
                gradient.shape != np.asarray(design['logits']).reshape(-1).shape or
                not np.isfinite(gradient).all() or not np.isfinite(objective) or not np.linalg.norm(gradient) > 0):
            raise ValueError('Cached gradient is invalid or belongs to another design')


def publish_cached_gradient(source, cache_path, inputs_path, design_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        shutil.copyfile(source, temporary)
        validate_cached_gradient(temporary, inputs_path, design_path)
        try:
            os.link(temporary, cache_path)
        except FileExistsError:
            validate_cached_gradient(cache_path, inputs_path, design_path)
    finally:
        temporary.unlink(missing_ok=True)


def acceptance_receipt(root, inputs, binary_inputs, baseline_design, baseline_geometry, baseline_replay,
                       reference_geometry, reference_replay, proposals, proposed, replays, selection,
                       common_surface=False):
    sources = ('tools/optimize_multicell_binary_loop.py', 'tools/select_multicell_binary_step.py',
               'tools/compare_multicell_binary_blast.py', 'tools/reconstruct_multicell_blast.py',
               'tools/multicell_feature_filter.py')
    if common_surface:
        sources += ('tools/multicell_common_surface.py', 'mpm_jax/compact_tl.py', 'mpm_jax/multifield_support.py')
    def replay_digest(path):
        outcome = path/'outcome.json'
        return {'inputs': sha256(path/'inputs.json'), 'history': sha256(path/'history.json'),
                'outcome': sha256(outcome) if outcome.exists() else None}
    def frame_digests(path):
        rows = json.loads((path/'history.json').read_text(encoding='utf-8'))
        return {'reference': sha256(path/'reference.npz'),
                'frames': {str(row['step']): sha256(path/f"frame-{row['step']:07d}.npz") for row in rows}}
    result = {'inputs': sha256(inputs), 'binary_inputs': sha256(binary_inputs),
            'baseline_design': sha256(baseline_design), 'baseline_geometry': sha256(baseline_geometry),
            'baseline_replay': replay_digest(baseline_replay),
            'reference_geometry': sha256(reference_geometry), 'reference_replay': replay_digest(reference_replay),
            'proposal_manifest': sha256(proposals/'manifest.json'),
            'candidates': [{'design': sha256(proposals/row['design']),
                            'geometry': sha256(proposals/row['geometry']), 'replay': replay_digest(replay)}
                           for row, replay in zip(proposed, replays)],
            'selection': sha256(selection/'selection.json'),
            'selected_design': sha256(selection/'selected-design.npz'),
            'sources': {name: sha256(root/name) for name in sources}}
    if common_surface:
        result['baseline_frames'] = frame_digests(baseline_replay)
        result['reference_frames'] = frame_digests(reference_replay)
        result['candidate_frames'] = [frame_digests(path) for path in replays]
    return result


def preserve_partial_directory(path, required):
    if not path.exists():
        return
    complete = all((path / name).is_file() for name in required)
    if complete:
        for name in required:
            if name.endswith('.json'):
                try:
                    json.loads((path / name).read_text(encoding='utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    complete = False
                    break
    if complete:
        return
    index = 1
    while True:
        target = path.with_name(path.name + f'.interrupted-{index:03d}')
        if not target.exists():
            path.rename(target)
            return
        index += 1


def preserve_incomplete_replay(replay, geometry_sha256, baseline_meta):
    if not replay.exists():
        return
    meta_path, history_path = replay / 'inputs.json', replay / 'history.json'
    if not meta_path.exists():
        if any(replay.iterdir()):
            raise ValueError('Replay has files but no metadata; review before resuming')
    else:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if (meta['geometry_sha256'] != geometry_sha256 or
                meta.get('support_mode', 'single') != baseline_meta.get('support_mode', 'single') or
                meta.get('peak_sampling', 'saved-frames') != baseline_meta.get('peak_sampling', 'saved-frames') or
                meta.get('bottom_point_peak_sampling', 'not-recorded') != baseline_meta.get('bottom_point_peak_sampling', 'not-recorded') or
                any(meta[key] != baseline_meta[key] for key in ('settings', 'source_sha256', 'frame_interval'))):
            raise ValueError('Interrupted replay does not match the current calculation')
        if history_path.exists():
            _, history, _ = read_run(replay, .02, allow_incomplete=True)
            if history[-1]['step'] == meta['steps']:
                return
            outcome = replay / 'outcome.json'
            if outcome.exists():
                recorded = json.loads(outcome.read_text(encoding='utf-8'))
                field = ('back_displacement_interval_peak_m' if meta.get('peak_sampling') == 'every-step'
                         else 'back_displacement_max_m')
                observed = max(row[field] for row in history)
                if (recorded.get('reason') == 'peak-threshold' and recorded.get('step') == history[-1]['step'] and
                        observed >= recorded.get('threshold_m', float('inf'))):
                    return
                raise ValueError('Interrupted replay outcome is inconsistent')
    index = 1
    while True:
        target = replay.with_name(replay.name + f'.interrupted-{index:03d}')
        if not target.exists():
            replay.rename(target)
            return
        index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('binary_inputs', type=Path)
    parser.add_argument('baseline_design', type=Path)
    parser.add_argument('baseline_geometry', type=Path)
    parser.add_argument('baseline_replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=4)
    parser.add_argument('--steps', type=float, nargs='+', default=[.025, .0125, .00625, .003125])
    parser.add_argument('--fallback-directions', type=int, default=8)
    parser.add_argument('--explore-directions', type=int, default=0)
    parser.add_argument('--explore-seed', type=int, default=370)
    parser.add_argument('--minimum-reduction', type=float, default=1e-6)
    parser.add_argument('--objective', choices=('underside-peak', 'fixed-bottom-peak'), default='underside-peak')
    parser.add_argument('--seed-gradient', type=Path)
    parser.add_argument('--seed-proposals', type=Path)
    parser.add_argument('--gradient-cache', type=Path,
                        help='Reuse completed, matching gradients from this directory')
    parser.add_argument('--early-reject', action='store_true')
    parser.add_argument('--max-single-voxel-fraction', type=float,
                        help='Maximum fraction of solid voxels in an axis-aligned one-voxel run')
    parser.add_argument('--max-bottom-peak-ratio', type=float)
    parser.add_argument('--max-p90-peak-ratio', type=float)
    parser.add_argument('--max-common-bottom-peak-ratio', type=float)
    parser.add_argument('--max-common-bottom-p90-ratio', type=float)
    parser.add_argument('--require-common-bottom-every-step', action='store_true',
                        help='Require every-step common-bottom maximum; p90 remains saved-frame sampled')
    parser.add_argument('--min-bottom-coverage', type=float)
    parser.add_argument('--min-common-bottom-coverage', type=float,
                        help='Minimum fraction of projected columns with bottom material in both designs')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.fallback_directions < 0 or args.explore_directions < 0 or not 0 <= args.minimum_reduction < 1:
        raise ValueError('Invalid iteration or acceptance settings')
    if args.max_single_voxel_fraction is not None and not 0 <= args.max_single_voxel_fraction <= 1:
        raise ValueError('Single-voxel fraction limit must lie in [0, 1]')
    import math
    for value in (args.max_bottom_peak_ratio, args.max_p90_peak_ratio,
                  args.max_common_bottom_peak_ratio, args.max_common_bottom_p90_ratio):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('Secondary displacement ratios must be finite and positive')
    if args.min_bottom_coverage is not None and not 0 <= args.min_bottom_coverage <= 1:
        raise ValueError('Bottom coverage must lie in [0, 1]')
    if args.min_common_bottom_coverage is not None and not 0 <= args.min_common_bottom_coverage <= 1:
        raise ValueError('Common bottom coverage must lie in [0, 1]')
    use_common = (args.objective == 'fixed-bottom-peak' or args.max_common_bottom_peak_ratio is not None
                  or args.max_common_bottom_p90_ratio is not None)
    if args.objective == 'fixed-bottom-peak' and args.early_reject:
        raise ValueError('Underside-peak early rejection is incompatible with fixed-bottom objective')
    if args.require_common_bottom_every_step and args.max_common_bottom_peak_ratio is None and args.objective != 'fixed-bottom-peak':
        raise ValueError('Every-step common-bottom sampling requires a common-bottom peak limit')
    if use_common and (args.min_common_bottom_coverage is None or args.min_common_bottom_coverage <= 0):
        raise ValueError('Common-bottom response limits require positive minimum common coverage')
    if (args.seed_gradient is None) != (args.seed_proposals is None):
        raise ValueError('Seed gradient and proposals must be supplied together')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    binary = json.loads(args.binary_inputs.read_text(encoding='utf-8'))
    opt_settings = inputs['binding']['settings']
    bin_settings = binary['binding']['settings']
    verify_physical_model_compatibility(opt_settings, bin_settings)
    baseline_meta, baseline_rows, _ = read_run(args.baseline_replay, .02)
    if baseline_meta['settings'] != bin_settings or baseline_meta['geometry_sha256'] != sha256(args.baseline_geometry):
        raise ValueError('Baseline replay differs from the supplied binary inputs or geometry')
    support_mode = baseline_meta.get('support_mode', 'single')
    if support_mode not in ('single', 'split-disconnected'):
        raise ValueError('Unsupported material-to-grid support mode')
    required = []
    if args.max_bottom_peak_ratio is not None: required.append('bottom_plane_displacement_max_m')
    if args.max_p90_peak_ratio is not None: required.append('back_displacement_p90_m')
    if any(key not in row for key in required for row in baseline_rows):
        raise ValueError('Baseline lacks the requested surface statistics')
    if args.min_bottom_coverage is not None and 'bottom_plane_sample_fraction' not in baseline_meta:
        raise ValueError('Baseline lacks bottom-plane coverage')
    if use_common and (not (args.baseline_replay/'reference.npz').is_file() or
                       any(not (args.baseline_replay/f"frame-{row['step']:07d}.npz").is_file() for row in baseline_rows)):
        raise ValueError('Common-bottom response limits require every baseline frame and reference')
    if args.require_common_bottom_every_step and baseline_meta.get('bottom_point_peak_sampling') != 'every-step':
        raise ValueError('Baseline lacks every-step bottom point peaks')
    if args.min_common_bottom_coverage is not None:
        with np.load(args.baseline_geometry, allow_pickle=False) as data:
            baseline_coverage = float(np.mean(data['solid_mask'][0]))
        if baseline_coverage < args.min_common_bottom_coverage:
            raise ValueError('Common bottom coverage exceeds the baseline bottom coverage')
        if args.objective == 'fixed-bottom-peak' and (baseline_coverage <= 0 or
                args.min_common_bottom_coverage + 1e-12 < baseline_coverage):
            raise ValueError('Fixed-bottom objective must retain every initial bottom observation point')
    root = Path(__file__).resolve().parents[1]
    if any(sha256(root / name) != value for name, value in baseline_meta['source_sha256'].items()):
        raise ValueError('Baseline replay uses a different binary solver source snapshot')
    binding = {'inputs_sha256': sha256(args.inputs), 'binary_inputs_sha256': sha256(args.binary_inputs),
               'initial_design_sha256': sha256(args.baseline_design),
               'initial_geometry_sha256': sha256(args.baseline_geometry),
               'baseline_replay_inputs_sha256': sha256(args.baseline_replay / 'inputs.json'),
               'baseline_replay_history_sha256': sha256(args.baseline_replay / 'history.json'),
               'steps': args.steps, 'fallback_directions': args.fallback_directions,
               'minimum_reduction': args.minimum_reduction}
    if args.seed_gradient is not None:
        binding.update(seed_gradient_sha256=sha256(args.seed_gradient),
                       seed_manifest_sha256=sha256(args.seed_proposals / 'manifest.json'))
    if args.early_reject:
        binding['early_reject'] = True
    if args.objective != 'underside-peak':
        binding['objective'] = args.objective
    if args.require_common_bottom_every_step:
        binding['require_common_bottom_every_step'] = True
    if args.explore_directions:
        binding.update(explore_directions=args.explore_directions, explore_seed=args.explore_seed)
    if args.max_single_voxel_fraction is not None:
        binding['max_single_voxel_fraction'] = args.max_single_voxel_fraction
    for key, value in (('max_bottom_peak_ratio', args.max_bottom_peak_ratio),
                       ('max_p90_peak_ratio', args.max_p90_peak_ratio),
                       ('max_common_bottom_peak_ratio', args.max_common_bottom_peak_ratio),
                       ('max_common_bottom_p90_ratio', args.max_common_bottom_p90_ratio),
                       ('min_bottom_coverage', args.min_bottom_coverage),
                       ('min_common_bottom_coverage', args.min_common_bottom_coverage)):
        if value is not None: binding[key] = value
    use_constraints = use_common or any(value is not None for value in (args.max_bottom_peak_ratio, args.max_p90_peak_ratio,
                                                           args.max_common_bottom_peak_ratio, args.max_common_bottom_p90_ratio,
                                                           args.min_bottom_coverage, args.min_common_bottom_coverage))
    if use_constraints: binding['constraint_reference'] = 'initial-replay'
    if args.min_bottom_coverage is not None or args.min_common_bottom_coverage is not None:
        binding['proposal_coverage_filter'] = 'v1'
    folder = args.output
    if args.resume:
        if json.loads((folder / 'binding.json').read_text()) != binding:
            raise ValueError('Resume settings or baseline differ')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'binding.json', binding)
    current_design, current_geometry, current_replay = args.baseline_design, args.baseline_geometry, args.baseline_replay
    sys.path.insert(0, str(root))
    from tools.reconstruct_multicell_blast import reconstruct
    for iteration in range(1, args.iterations + 1):
        stage = folder / f'iteration-{iteration:04d}'
        stage.mkdir(exist_ok=True)
        gradient = stage / 'gradient.npz'
        if not gradient.exists():
            if iteration == 1 and args.seed_gradient is not None:
                shutil.copyfile(args.seed_gradient, gradient)
            else:
                cache_path = (gradient_cache_path(args.gradient_cache, root, args.inputs, current_design)
                              if args.gradient_cache is not None else None)
                if cache_path is not None and cache_path.exists():
                    validate_cached_gradient(cache_path, args.inputs, current_design)
                    shutil.copyfile(cache_path, gradient)
                else:
                    write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'gradient'})
                    run('tools.gradient_multicell_at_design', args.inputs, current_design, '--output', gradient)
                    if cache_path is not None:
                        publish_cached_gradient(gradient, cache_path, args.inputs, current_design)
        with np.load(gradient, allow_pickle=False) as data:
            if str(data['design_sha256'].item()) != sha256(current_design) or str(data['inputs_sha256'].item()) != sha256(args.inputs):
                raise ValueError('Stored gradient does not belong to the current design')
        proposals = stage / 'proposals'
        if args.resume:
            preserve_partial_directory(proposals, ('manifest.json',))
        if not proposals.exists():
            if iteration == 1 and args.seed_proposals is not None:
                shutil.copytree(args.seed_proposals, proposals)
            else:
                write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'proposing'})
                proposal_args = ['tools.propose_multicell_binary_steps', args.inputs, gradient, '--steps', *args.steps,
                                 '--fallback-directions', args.fallback_directions, '--output', proposals]
                if args.explore_directions:
                    proposal_args.extend(('--explore-directions', args.explore_directions,
                                          '--explore-seed', args.explore_seed))
                if args.max_single_voxel_fraction is not None:
                    proposal_args.extend(('--max-single-voxel-fraction', args.max_single_voxel_fraction))
                if args.min_bottom_coverage is not None:
                    proposal_args.extend(('--min-bottom-coverage', args.min_bottom_coverage))
                if args.min_common_bottom_coverage is not None:
                    proposal_args.extend(('--min-common-bottom-coverage', args.min_common_bottom_coverage,
                                          '--constraint-reference-geometry', args.baseline_geometry))
                run(*proposal_args)
        manifest = json.loads((proposals / 'manifest.json').read_text())
        if manifest['gradient_sha256'] != sha256(gradient) or manifest['inputs_sha256'] != sha256(args.inputs):
            raise ValueError('Proposal manifest does not match the current gradient')
        if manifest.get('max_single_voxel_fraction') != args.max_single_voxel_fraction:
            raise ValueError('Proposal feature limit differs from the optimization settings')
        if (manifest.get('explore_directions', 0) != args.explore_directions or
                (args.explore_directions and manifest.get('explore_seed') != args.explore_seed)):
            raise ValueError('Proposal exploration settings differ')
        if (manifest.get('min_bottom_coverage') != args.min_bottom_coverage or
                manifest.get('min_common_bottom_coverage') != args.min_common_bottom_coverage):
            raise ValueError('Proposal coverage filters differ from the optimization settings')
        if (args.min_common_bottom_coverage is not None and
                manifest.get('constraint_reference_geometry_sha256') != sha256(args.baseline_geometry)):
            raise ValueError('Proposal constraint reference differs')
        proposed = [row for row in manifest['steps'] if row['status'] == 'proposed']
        if not proposed:
            write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'no-feasible-proposals'})
            return
        replays = []
        current_meta, current_rows, _ = read_run(current_replay, .02)
        peak_field = ('back_displacement_interval_peak_m' if current_meta.get('peak_sampling') == 'every-step'
                      else 'back_displacement_max_m')
        cutoff = max(row[peak_field] for row in current_rows) * (1 - args.minimum_reduction)
        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError('Current baseline has no positive finite peak')
        for index, row in enumerate(proposed):
            design = proposals / row['design']; geometry = proposals / row['geometry']
            if sha256(design) != row['design_sha256'] or sha256(geometry) != row['geometry_sha256']:
                raise ValueError('Proposal files differ from their manifest')
            report_path = geometry.with_suffix('.json')
            if not report_path.exists():
                with np.load(design, allow_pickle=False) as data:
                    report = reconstruct(data['rho_cell'], opt_settings)[-1]
                if report['solid_mask_sha256'] != row['solid_mask_sha256']:
                    raise ValueError('Proposal design and geometry differ')
                report.update(geometry_sha256=sha256(geometry), inputs_sha256=sha256(args.binary_inputs),
                              optimization_signature=binary['signature'])
                write_json(report_path, report)
            replay = stage / f'replay-{index:03d}'
            if args.resume:
                preserve_incomplete_replay(replay, row['geometry_sha256'], baseline_meta)
            if not replay.exists():
                write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'replaying',
                                                   'candidate': index + 1, 'candidates': len(proposed)})
                replay_args = ['examples.multicell_binary_blast_3d', geometry, args.binary_inputs, '--output', replay,
                               '--frame-interval', baseline_meta['frame_interval'], '--empty-column-load',
                               baseline_meta.get('empty_column_load', 'omit')]
                if support_mode == 'split-disconnected':
                    replay_args.append('--split-disconnected-support')
                if baseline_meta.get('peak_sampling', 'saved-frames') == 'every-step':
                    replay_args.append('--track-subframe-peak')
                if baseline_meta.get('bottom_point_peak_sampling', 'not-recorded') == 'every-step':
                    replay_args.append('--track-bottom-point-peaks')
                if args.early_reject:
                    replay_args.extend(('--reject-if-peak-ge', cutoff))
                run(*replay_args)
            replay_meta, replay_rows, _ = read_run(replay, .02, allow_incomplete=args.early_reject)
            if replay_meta['geometry_sha256'] != sha256(geometry):
                raise ValueError('Stored replay does not match its geometry')
            if replay_meta.get('support_mode', 'single') != support_mode:
                raise ValueError('Stored replay uses a different material-to-grid support mode')
            if replay_meta.get('peak_sampling', 'saved-frames') != baseline_meta.get('peak_sampling', 'saved-frames'):
                raise ValueError('Stored replay uses a different peak sampling mode')
            if replay_meta.get('bottom_point_peak_sampling', 'not-recorded') != baseline_meta.get('bottom_point_peak_sampling', 'not-recorded'):
                raise ValueError('Stored replay uses a different bottom point peak sampling mode')
            if replay_rows[-1]['step'] < replay_meta['steps'] and not (replay / 'outcome.json').exists():
                raise ValueError('Incomplete replay has no rejection certificate')
            replays.append(replay)
        selection_dir = stage / 'selection'
        if args.resume:
            preserve_partial_directory(selection_dir, ('selection.json', 'selected-design.npz', 'acceptance-receipt.json'))
        if not selection_dir.exists():
            write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'selecting'})
            selection_args = ['tools.select_multicell_binary_step', args.inputs, current_design, current_geometry,
                              current_replay, proposals, '--replays', *replays, '--minimum-reduction', args.minimum_reduction,
                              '--objective', args.objective, '--output', selection_dir]
            for option, value in (('--max-bottom-peak-ratio', args.max_bottom_peak_ratio),
                                  ('--max-p90-peak-ratio', args.max_p90_peak_ratio),
                                  ('--max-common-bottom-peak-ratio', args.max_common_bottom_peak_ratio),
                                  ('--max-common-bottom-p90-ratio', args.max_common_bottom_p90_ratio),
                                  ('--min-bottom-coverage', args.min_bottom_coverage),
                                  ('--min-common-bottom-coverage', args.min_common_bottom_coverage)):
                if value is not None: selection_args.extend((option, value))
            if use_constraints:
                selection_args.extend(('--constraint-reference-replay', args.baseline_replay,
                                       '--constraint-reference-geometry', args.baseline_geometry))
            if args.require_common_bottom_every_step:
                selection_args.append('--require-common-bottom-every-step')
            run(*selection_args)
            receipt = acceptance_receipt(root, args.inputs, args.binary_inputs, current_design, current_geometry,
                                         current_replay, args.baseline_geometry, args.baseline_replay,
                                         proposals, proposed, replays, selection_dir, use_common)
            write_json(selection_dir/'acceptance-receipt.json', receipt)
        else:
            recorded = json.loads((selection_dir/'acceptance-receipt.json').read_text())
            actual = acceptance_receipt(root, args.inputs, args.binary_inputs, current_design, current_geometry,
                                        current_replay, args.baseline_geometry, args.baseline_replay,
                                        proposals, proposed, replays, selection_dir, use_common)
            if recorded != actual:
                raise ValueError('Stored acceptance evidence or validation code differs')
        selection = json.loads((selection_dir / 'selection.json').read_text())
        selected_design = selection_dir / 'selected-design.npz'
        if sha256(selected_design) != selection['selected_design_sha256']:
            raise ValueError('Stored selection does not match its design')
        if not selection['accepted']:
            write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'no-physical-improvement',
                                                'selected_design': str(current_design)})
            return
        matches = [i for i, row in enumerate(proposed) if row['solid_mask_sha256'] == selection['selected_solid_mask_sha256']]
        if len(matches) != 1:
            raise ValueError('Selected candidate is ambiguous')
        index = matches[0]
        current_design = selected_design
        current_geometry = proposals / proposed[index]['geometry']
        current_replay = replays[index]
        write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'accepted',
                                            'selected_design': str(current_design), 'selected_replay': str(current_replay)})
    write_json(folder / 'status.json', {'iteration': args.iterations, 'stage': 'iteration-limit',
                                        'selected_design': str(current_design), 'selected_replay': str(current_replay)})


if __name__ == '__main__':
    main()
