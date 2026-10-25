"""Read-only genuine 13-state bending review; no solver or publication action."""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_phase_field_full import PINS, owned_session_uss, terminate_owned_group

OUT = ROOT / '.local/phase-bending-media/result'
DELIVERY = ROOT / '.local/phase-bending-media/delivery'
OWN = '.github/workflows/phase-bending-media.yml'
PRODUCER = '5277566890f01d867a064580efaaabea50c8efdc'
ZIP_SHA = '4558f89523908f625cd61a49e4449f5f304e921c31cca09a9dc5ce8c151aaf3f'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != os.environ['GITHUB_SHA'] or os.environ['GITHUB_REF'] != 'refs/heads/work/phase-field-bending' or os.environ['GITHUB_REPOSITORY'] != 'lililii124/jax-mpm':
        raise ValueError('Exact reviewed feature identity required')
    active = '.github/workflows/engineering-previews.yml'
    if os.environ['GITHUB_WORKFLOW_REF'].split('@')[0] != 'lililii124/jax-mpm/' + active or (ROOT / active).read_bytes() != (ROOT / OWN).read_bytes():
        raise ValueError('Registered workflow differs')
    names = [active, OWN, 'tools/finalize_phase_bending_media.py', 'tools/render_phase_bending_media.py', 'tools/encode_phase_bending_media.py', 'tools/run_phase_field_full.py', 'tools/run_woven_preview.py']
    files = {}
    for name in names:
        if (ROOT / name).read_bytes() != subprocess.check_output(['git', 'show', head + ':' + name], cwd=ROOT):
            raise ValueError('Modified media source: ' + name)
        files[name] = sha(ROOT / name)
    if {k: importlib.metadata.version(k) for k in PINS} != PINS or sys.version_info[:3] != (3, 12, 13):
        raise ValueError('Pinned host dependencies required')
    return dict(commit=head, files=files)


def prepare():
    if OUT.exists():
        raise ValueError('Fresh derivative destination required')
    context = binding()
    OUT.mkdir(parents=True)
    artifact = json.loads(subprocess.check_output(['gh', 'api', 'repos/lililii124/jax-mpm/actions/artifacts/12584382206']))
    run = json.loads(subprocess.check_output(['gh', 'api', 'repos/lililii124/jax-mpm/actions/runs/40154387841']))
    if artifact['expired'] or artifact['workflow_run']['id'] != 40154387841 or artifact['workflow_run']['head_sha'] != PRODUCER or artifact['digest'] != 'sha256:' + ZIP_SHA or run['conclusion'] != 'success' or run['head_sha'] != PRODUCER:
        raise ValueError('Immutable original raw identity differs')
    archive = OUT.parent / 'original.zip'
    with archive.open('xb') as stream:
        subprocess.run(['gh', 'api', '--allow-escape-sequences', 'repos/lililii124/jax-mpm/actions/artifacts/12584382206/zip'], stdout=stream, check=True)
    if sha(archive) != ZIP_SHA:
        raise ValueError('Official whole ZIP digest differs')
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or len(set(zipped.namelist())) != len(zipped.namelist()):
            raise ValueError('Original ZIP CRC/duplicate error')
        for name in zipped.namelist():
            if not (OUT / name).resolve().is_relative_to(OUT.resolve()):
                raise ValueError('Unsafe original member')
        zipped.extractall(OUT)
    original = json.loads((OUT / 'source-runtime.json').read_text())
    producer = ROOT / '.local/bending-producer'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=producer, text=True).strip() != PRODUCER or original['commit'] != PRODUCER:
        raise ValueError('Original numerical checkout differs')
    for name, expected in original['files'].items():
        if sha(producer / name) != expected or (producer / name).read_bytes() != subprocess.check_output(['git', 'show', PRODUCER + ':' + name], cwd=producer):
            raise ValueError('Original numerical source differs: ' + name)
    q = json.loads((OUT / 'qualification.json').read_text())
    if not q['numerical_qualification_passed'] or q['mode'] != 'medium' or q['actual_saved_states'] != 13:
        raise ValueError('Complete original qualification required')
    previous = np.zeros(12288)
    names = [f'frame_{i:06d}.npz' for i in range(0, 1201, 100)]
    if sorted(p.name for p in OUT.glob('frame_*.npz')) != names:
        raise ValueError('Exact 13 original saved states required')
    for name in names:
        with np.load(OUT / name) as frame:
            if frame['position'].shape != (12288, 3) or not all(np.isfinite(frame[k]).all() for k in frame.files) or (frame['damage'] - previous).min() < -1e-12 or frame['damage'].min() < 0 or frame['damage'].max() > 1:
                raise ValueError('Invalid original raw')
            previous = frame['damage'].copy()
    save(OUT / 'media-source.json', context)
    save(OUT / 'media-scope.json', dict(original_run=40154387841, original_artifact=12584382206, original_zip_sha256=ZIP_SHA,
        original_source=PRODUCER, actual_saved_states=13, maximum_damage=float(previous.max()), physical_acceptance=False,
        public_media_authorized=False, interpretation='Review only: no crack; original numerical qualification passed, constitutive work defect about60% input work and missing independent nodal fracture-gradient audit remain limitations.'))


def render():
    context = binding()
    if context != json.loads((OUT / 'media-source.json').read_text()):
        raise ValueError('Media source changed')
    if sys.platform != 'linux' or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('Owned Linux subreaper required')
    start = time.monotonic()
    record = dict(completed=False, peak_owned_uss=0, reason=None)
    commands = [[os.environ['PARAVIEW_PYTHON'], '--force-offscreen-rendering', str(ROOT / 'tools/render_phase_bending_media.py'), str(OUT)],
                [sys.executable, str(ROOT / 'tools/encode_phase_bending_media.py'), str(OUT)]]
    for i, command in enumerate(commands):
        child = None
        try:
            with (OUT / f'media-{i}.log').open('wb') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while child.poll() is None:
                    uss = owned_session_uss(child.pid, psutil)
                    record['peak_owned_uss'] = max(record['peak_owned_uss'], uss)
                    if uss > 5.5 * 1024**3 or psutil.virtual_memory().available < 2 * 1024**3 or time.monotonic() - start > 900:
                        raise RuntimeError('Media USS/available-memory/wall guard')
                    time.sleep(.3)
                if child.returncode:
                    raise RuntimeError('Media process failed')
        except BaseException as error:
            record['reason'] = repr(error)
            raise
        finally:
            try:
                terminate_owned_group(child)
            finally:
                save(OUT / 'media-guard.json', record)
    checks = json.loads((OUT / 'gif-checks.json').read_text())
    if checks['frames'] != 13 or not checks['decoded_exact'] or checks['unique_decoded_frames'] < 2 or binding() != context:
        raise ValueError('Actual dynamic13-frame derivative audit failed')
    record['completed'] = True
    save(OUT / 'media-guard.json', record)


def package():
    if DELIVERY.exists():
        raise ValueError('Fresh delivery required')
    DELIVERY.mkdir(parents=True)
    names = ['phase-field-bending-medium1200.gif', 'render.json', 'gif-checks.json', 'media-source.json', 'media-scope.json', 'media-guard.json', 'summary.json', 'qualification.json', 'media-0.log', 'media-1.log']
    names += [p.relative_to(OUT).as_posix() for p in (OUT / 'png').glob('*.png')]
    files = {}
    for name in names:
        source = OUT / name
        if source.is_file():
            target = DELIVERY / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files[name] = dict(bytes=target.stat().st_size, sha256=sha(target))
    if sum(v['bytes'] for v in files.values()) > 100 * 1024**2:
        raise ValueError('100MiB derivative cap')
    save(DELIVERY / 'delivery-manifest.json', dict(files=files, original_zip_sha256=ZIP_SHA,
        media_complete=(OUT / 'gif-checks.json').exists() and json.loads((OUT / 'media-guard.json').read_text())['completed']))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        stream.write('manifest_sha=' + sha(DELIVERY / 'delivery-manifest.json') + '\n')


def verify(folder):
    if sha(folder / 'delivery-manifest.json') != os.environ['EXPECTED_MANIFEST']:
        raise ValueError('Producer-pinned delivery manifest differs')
    manifest = json.loads((folder / 'delivery-manifest.json').read_text())
    for name, item in manifest['files'].items():
        if (folder / name).stat().st_size != item['bytes'] or sha(folder / name) != item['sha256']:
            raise ValueError('Retrieved whole member differs')
    if not manifest['media_complete']:
        raise RuntimeError('Incomplete media diagnostics preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'render', 'package', 'verify'))
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    if args.phase == 'verify':
        verify(args.folder)
    else:
        globals()[args.phase]()
