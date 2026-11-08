"""Iterate full-domain shared-cell designs with binary MPM acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.compare_multicell_binary_blast import read_run


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
    parser.add_argument('--minimum-reduction', type=float, default=1e-6)
    parser.add_argument('--seed-gradient', type=Path)
    parser.add_argument('--seed-proposals', type=Path)
    parser.add_argument('--early-reject', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.iterations < 1 or args.fallback_directions < 0 or not 0 <= args.minimum_reduction < 1:
        raise ValueError('Invalid iteration or acceptance settings')
    if (args.seed_gradient is None) != (args.seed_proposals is None):
        raise ValueError('Seed gradient and proposals must be supplied together')
    inputs = json.loads(args.inputs.read_text(encoding='utf-8'))
    binary = json.loads(args.binary_inputs.read_text(encoding='utf-8'))
    opt_settings = inputs['binding']['settings']
    bin_settings = binary['binding']['settings']
    fields = ('cell', 'cell_size', 'skin', 'skin_layers', 'volume_fraction', 'skin_young',
              'core_young', 'solid_density', 'pressure', 'pulse', 'blast_radius')
    if any(opt_settings[key] != bin_settings[key] for key in fields):
        raise ValueError('Gradient and binary simulations use different physical settings')
    baseline_meta, _, _ = read_run(args.baseline_replay, .02)
    if baseline_meta['settings'] != bin_settings or baseline_meta['geometry_sha256'] != sha256(args.baseline_geometry):
        raise ValueError('Baseline replay differs from the supplied binary inputs or geometry')
    support_mode = baseline_meta.get('support_mode', 'single')
    if support_mode not in ('single', 'split-disconnected'):
        raise ValueError('Unsupported material-to-grid support mode')
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
    folder = args.output
    if args.resume:
        if json.loads((folder / 'binding.json').read_text()) != binding:
            raise ValueError('Resume settings or baseline differ')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / 'binding.json', binding)
    current_design, current_geometry, current_replay = args.baseline_design, args.baseline_geometry, args.baseline_replay
    sys.path.insert(0, str(root))
    import numpy as np
    from tools.reconstruct_multicell_blast import reconstruct
    for iteration in range(1, args.iterations + 1):
        stage = folder / f'iteration-{iteration:04d}'
        stage.mkdir(exist_ok=True)
        gradient = stage / 'gradient.npz'
        if not gradient.exists():
            if iteration == 1 and args.seed_gradient is not None:
                shutil.copyfile(args.seed_gradient, gradient)
            else:
                write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'gradient'})
                run('tools.gradient_multicell_at_design', args.inputs, current_design, '--output', gradient)
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
                run('tools.propose_multicell_binary_steps', args.inputs, gradient, '--steps', *args.steps,
                    '--fallback-directions', args.fallback_directions, '--output', proposals)
        manifest = json.loads((proposals / 'manifest.json').read_text())
        if manifest['gradient_sha256'] != sha256(gradient) or manifest['inputs_sha256'] != sha256(args.inputs):
            raise ValueError('Proposal manifest does not match the current gradient')
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
            if replay_rows[-1]['step'] < replay_meta['steps'] and not (replay / 'outcome.json').exists():
                raise ValueError('Incomplete replay has no rejection certificate')
            replays.append(replay)
        selection_dir = stage / 'selection'
        if args.resume:
            preserve_partial_directory(selection_dir, ('selection.json', 'selected-design.npz'))
        if not selection_dir.exists():
            write_json(folder / 'status.json', {'iteration': iteration, 'stage': 'selecting'})
            run('tools.select_multicell_binary_step', args.inputs, current_design, current_geometry, current_replay,
                proposals, '--replays', *replays, '--minimum-reduction', args.minimum_reduction, '--output', selection_dir)
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
