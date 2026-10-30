"""Collect source-bound accepted render frames from ordered C2 run windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_frame(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        required = ('x', 'v', 'damage', 'body_id', 'particle_mass', 'time_s', 'step')
        missing = [name for name in required if name not in saved]
        if missing:
            raise ValueError(f'{path} is missing fields: {missing}')
        frame = {name: np.asarray(saved[name]) for name in required}
    n = len(frame['x'])
    if frame['x'].shape != (n, 3) or frame['v'].shape != (n, 3):
        raise ValueError(f'{path} has invalid position or velocity dimensions')
    if any(frame[name].shape != (n,) for name in ('damage', 'body_id', 'particle_mass')):
        raise ValueError(f'{path} has inconsistent particle fields')
    if not all(np.isfinite(frame[name]).all() for name in ('x', 'v', 'damage', 'particle_mass')):
        raise ValueError(f'{path} contains non-finite physical fields')
    frame['time_s'] = float(frame['time_s'])
    frame['step'] = int(frame['step'])
    if frame['time_s'] < 0 or frame['step'] < 0 or np.any(frame['particle_mass'] <= 0):
        raise ValueError(f'{path} has invalid time, step, or particle mass')
    return frame


def same_state(left: dict, right: dict) -> bool:
    return (left['step'] == right['step'] and left['time_s'] == right['time_s']
            and all(np.array_equal(left[name], right[name]) for name in
                    ('x', 'v', 'damage', 'body_id', 'particle_mass')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end-time', type=float, default=1e-6)
    parser.add_argument('--allow-prefix', action='store_true',
                        help='collect a real but incomplete history and mark it as a prefix')
    parser.add_argument('run_dirs', nargs='+', type=Path,
                        help='ordered directories, each containing run.json, accepted.npz and history/')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    if args.end_time <= 0:
        parser.error('--end-time must be positive')

    runs = []
    all_frames = []
    previous_run = None
    previous_state = None
    source_hashes = None
    for run_dir in args.run_dirs:
        run_json_path = run_dir / 'run.json'
        accepted_path = run_dir / 'accepted.npz'
        history_dir = run_dir / 'history'
        if not run_json_path.is_file() or not accepted_path.is_file() or not history_dir.is_dir():
            raise FileNotFoundError(f'{run_dir} must contain run.json, accepted.npz and history/')
        run = json.loads(run_json_path.read_text(encoding='utf-8'))
        status = run.get('status')
        if status in ('failed-preserved', 'failed-at-last-accepted-state'):
            raise ValueError(f'{run_dir} is a failed run window')
        if status == 'running' and not args.allow_prefix:
            raise ValueError(f'{run_dir} is still running; use --allow-prefix only for a labelled prefix')
        hashes = run.get('source_sha256')
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f'{run_dir}/run.json has no source hash map')
        if source_hashes is None:
            source_hashes = hashes
        elif hashes != source_hashes:
            raise ValueError('source hashes changed between windows; collect only a source-continuous chain')

        accepted_hash = sha256(accepted_path)
        with np.load(accepted_path, allow_pickle=False) as checkpoint:
            endpoint_step = int(checkpoint['step'])
            endpoint_time = float(checkpoint['time_s'])
        if int(run.get('steps', endpoint_step)) != endpoint_step or not np.isclose(
                float(run.get('time_s', endpoint_time)), endpoint_time, rtol=0, atol=1e-18):
            raise ValueError(f'{run_dir} run.json endpoint differs from accepted.npz')

        resume = run.get('resume')
        if previous_run is None:
            if resume:
                raise ValueError('the first run directory is a resumed window; include its source parent first')
        else:
            if not isinstance(resume, dict) or resume.get('checkpoint_sha256') != previous_run['accepted_sha256']:
                raise ValueError(f'{run_dir} is not hash-linked to the preceding accepted checkpoint')
            if resume.get('prior_source_sha256') != previous_run['source_sha256']:
                raise ValueError(f'{run_dir} resume receipt names a different parent source')
            if endpoint_time <= previous_run['time_s']:
                raise ValueError(f'{run_dir} does not advance physical time')

        frame_paths = sorted(history_dir.glob('frame-*.npz'))
        if not frame_paths:
            raise ValueError(f'{history_dir} contains no accepted render frames')
        records = []
        for frame_path in frame_paths:
            frame = read_frame(frame_path)
            if frame['step'] > endpoint_step or frame['time_s'] > endpoint_time + 1e-18:
                raise ValueError(f'{frame_path} is newer than its accepted checkpoint')
            record = dict(path=frame_path, state=frame, sha256=sha256(frame_path))
            records.append(record)
        if records[-1]['state']['time_s'] < records[0]['state']['time_s']:
            raise ValueError(f'{history_dir} frames are not chronological')
        if any(b['state']['time_s'] <= a['state']['time_s']
               for a, b in zip(records, records[1:])):
            raise ValueError(f'{history_dir} has non-increasing frame times')

        for record in records:
            if previous_state is not None and (
                    len(record['state']['x']) != len(previous_state['x'])
                    or not np.array_equal(record['state']['body_id'], previous_state['body_id'])
                    or not np.array_equal(record['state']['particle_mass'], previous_state['particle_mass'])):
                raise ValueError('particle identities, body labels, or physical masses changed between frames')
            if all_frames and record['state']['time_s'] == all_frames[-1]['state']['time_s']:
                if not same_state(record['state'], all_frames[-1]['state']):
                    raise ValueError('equal-time window boundary frames disagree')
                continue
            if all_frames and (record['state']['time_s'] < all_frames[-1]['state']['time_s']
                               or record['state']['step'] <= all_frames[-1]['state']['step']):
                raise ValueError('ordered windows contain out-of-order or repeated accepted frames')
            all_frames.append(record)
            previous_state = record['state']

        runs.append(dict(name=run_dir.name, status=status, steps=endpoint_step,
                         time_s=endpoint_time, accepted_sha256=accepted_hash,
                         run_json_sha256=sha256(run_json_path), source_sha256=hashes,
                         collected_source_frames=[r['path'].name for r in records]))
        previous_run = dict(accepted_sha256=accepted_hash, source_sha256=hashes,
                            time_s=endpoint_time)

    complete = (runs[-1]['time_s'] >= args.end_time - max(1e-15, args.end_time * 1e-12)
                and runs[-1]['status'] == 'completed-short-window')
    if not complete and not args.allow_prefix:
        raise ValueError('run chain has not reached the requested horizon; use --allow-prefix for a labelled prefix')
    if len(all_frames) < 2:
        raise ValueError('at least two distinct accepted frames are required')

    args.output.mkdir(parents=True)
    frame_dir = args.output / 'history'
    frame_dir.mkdir()
    frame_manifest = []
    for index, record in enumerate(all_frames):
        target = frame_dir / f'frame-{index:06d}.npz'
        shutil.copyfile(record['path'], target)
        state = record['state']
        frame_manifest.append(dict(file=f'history/{target.name}', source_sha256=record['sha256'],
                                   step=state['step'], time_s=state['time_s'],
                                   particle_count=len(state['x']),
                                   physical_mass_kg=float(state['particle_mass'].sum())))
    manifest = dict(schema='domain-impact-accepted-history-v1',
                    complete=complete, requested_end_time_s=args.end_time,
                    reached_time_s=runs[-1]['time_s'],
                    interpretation=('Complete accepted-state history to requested horizon.' if complete
                                    else 'Incomplete prefix of actual accepted states; no interpolation or invented frames.'),
                    runs=runs, frames=frame_manifest)
    (args.output / 'history-manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), complete=complete,
                          windows=len(runs), frames=len(frame_manifest),
                          first_time_s=frame_manifest[0]['time_s'],
                          last_time_s=frame_manifest[-1]['time_s'])))


if __name__ == '__main__':
    main()
