"""Bounded transport fallback; preserve whole restart before optional real frames.

The numerical worker is never imported or executed. A newly created failed
capsule has a distinct schema and is never automatically resumable, even when
its retained mechanical state was valid before a transport failure.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def controller():
    path = Path(__file__).with_name('run_triaxial_zero_recovery.py')
    spec = importlib.util.spec_from_file_location('zero_failure_storage_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_output(ready):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write('ready=' + ('true' if ready else 'false') + '\n')


def preserve(m):
    import numpy as np
    context_path = m.BASE / 'context.json'
    if not context_path.exists():
        save_output(False)
        return {'ready': False, 'reason': 'No admitted recovery context; original parent is unaffected.'}
    context = m.read(context_path)
    if m.verify_controller() != context['controller']:
        raise ValueError('Controller changed before failure preservation')
    output = m.BASE / 'failure-delivery'
    output.mkdir(exist_ok=False)
    source = m.BASE / 'delivery'
    primary, receipt_path = source / 'capsule.zip', source / 'receipt.json'
    if primary.exists() and receipt_path.exists():
        receipt = m.read(receipt_path)
        if m.record(primary) == receipt['capsule'] and primary.stat().st_size < m.ARTIFACT_LIMIT - 1024**2:
            with zipfile.ZipFile(primary) as z:
                if z.testzip() is not None:
                    raise ValueError('Complete capsule failed CRC verification')
                manifest = json.loads(z.read('capsule-manifest.json'))
                if manifest['controller'] != context['controller']:
                    raise ValueError('Complete capsule controller mismatch')
            shutil.copyfile(primary, output / 'capsule.zip')
            shutil.copyfile(receipt_path, output / 'receipt.json')
            save_output(True)
            return {'ready': True, 'scope': 'Reuse exact already-sealed whole capsule after transport failure.'}
    folder = m.BASE / 'work/formal/theta-00/hertz'
    checkpoint = folder / 'checkpoint.npz'
    if not checkpoint.exists():
        save_output(False)
        return {'ready': False, 'reason': 'No new complete checkpoint exists; original parent is retained.'}
    # This independent capsule deliberately has no accepted-capsule manifest.
    # prepare() cannot consume it as a continuation, regardless of state flags.
    with np.load(checkpoint, allow_pickle=False) as cp:
        end = int(cp['frame'])
        status = {'checkpoint_frame': end, 'accepted_steps': int(cp['aggregate'][0]),
            'aggregate_valid': bool(cp['aggregate'][4]),
            'state_valid': bool(cp['state_valid']),
            'history_valid': bool(np.all(cp['history_valid'])),
            'contact_valid': bool(cp['contact_valid']),
            'contact_law_valid': bool(np.all(cp['contact_law_valid']))}
    selected = {'diagnostic/checkpoint.npz': checkpoint,
        'records/context.json': context_path, 'records/plan.json': m.BASE / 'parent/plan.json'}
    for name in ('memory.json', 'solver.log', 'history-availability.json', 'artifact-accounting.json'):
        if (m.BASE / name).exists():
            selected['records/' + name] = m.BASE / name
    for name in ('status.json', 'failure.json', 'summary.json', 'observables.csv',
                 'diagnostics.csv', 'contact-metrics.json'):
        if (folder / name).exists():
            selected['records/current-' + name] = folder / name
    # Metadata plus the complete NPZ must fit before any optional frame payload.
    raw_reserve = sum(path.stat().st_size for path in selected.values())
    if raw_reserve > m.ARTIFACT_LIMIT - 2 * 1024**2:
        save_output(False)
        return {'ready': False, 'reason': 'Whole new checkpoint plus records exceeds the hard bound; no partial state uploaded.'}
    files, frames, omitted = {}, [], []
    target = output / 'capsule.zip'
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for name, path in selected.items():
            files[name] = m.record(path)
            z.write(path, name)
        first = context['start_frame'] if context['start_frame'] == 20 else context['start_frame'] + 1
        # One complete frame at a time; no live worker and no full history load.
        for frame in range(first, min(end, context['end_frame']) + 1):
            arrays = {}
            try:
                for field in m.FIELDS:
                    array = np.load(folder / f'trajectory_{field}.npy', mmap_mode='r', allow_pickle=False)
                    arrays[field] = array
                    value = array[frame]
                    if not np.isfinite(value).all() or (field == 'failed' and not np.all((value == 0) | (value == 1))):
                        raise ValueError('Incomplete or explicitly missing frame ' + str(frame))
                with tempfile.TemporaryDirectory(dir=m.BASE, prefix='failure-frame-') as temporary:
                    frame_path = Path(temporary) / f'frame-{frame:02d}.npz'
                    np.savez_compressed(frame_path, **{key: value[frame] for key, value in arrays.items()})
                    item = m.record(frame_path)
                    if target.stat().st_size + item['bytes'] > m.ARTIFACT_LIMIT - 2 * 1024**2:
                        omitted.append({'frame': frame, **item, 'reason': 'Whole real frame exceeds remaining hard bound.'})
                        continue
                    name = f'frames/frame-{frame:02d}.npz'
                    z.write(frame_path, name)
                    files[name] = item
                    frames.append(frame)
            except (OSError, ValueError, IndexError) as error:
                omitted.append({'frame': frame, 'reason': str(error)})
            finally:
                for array in arrays.values():
                    array._mmap.close()
        manifest = {'schema': 'triaxial-zero-failed-capsule-v1', 'restart_valid': False,
            'never_resumable': True, 'production_eligible': False, 'checkpoint': status,
            'parent_archive_sha256': context['parent_archive_sha256'],
            'controller': context['controller'], 'files': files,
            'complete_real_frames_archived': frames, 'unarchived_frames': omitted,
            'full_new_frame_history_archived': not omitted,
            'scope': 'Whole new state preserved independently of transport success. A failed diagnostic is not an accepted continuation. Original parent and missing-prefix disclosures are unchanged.'}
        z.writestr('failed-manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
    if target.stat().st_size > m.ARTIFACT_LIMIT - 1024**2:
        raise ValueError('Failed capsule exceeds hard bound; refuse upload')
    result = {'ready': True, 'capsule': m.record(target), 'never_resumable': True,
        'production_eligible': False, 'complete_real_frames_archived': frames,
        'unarchived_frames': omitted, 'artifact_limit_bytes': m.ARTIFACT_LIMIT}
    m.save(output / 'receipt.json', result)
    save_output(True)
    return result


if __name__ == '__main__':
    m = controller()
    relative = Path(__file__).resolve().relative_to(m.ROOT).as_posix()
    head = subprocess.check_output(['git', '-C', str(m.ROOT), 'rev-parse', 'HEAD']).decode().strip()
    if (head != os.environ.get('GITHUB_SHA')
            or Path(__file__).read_bytes() != subprocess.check_output(['git', '-C', str(m.ROOT), 'show', f'{head}:{relative}'])):
        raise ValueError('Failure transport helper differs from dispatched committed bytes')
    print(json.dumps(preserve(m)))
