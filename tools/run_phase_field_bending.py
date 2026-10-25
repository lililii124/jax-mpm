"""Source-bound qualification of the uncalibrated AT2 bending candidate."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_phase_field_full import PINS, owned_session_uss, terminate_owned_group

OUT = ROOT / 'outputs/at2-bending'
OWN = '.github/workflows/phase-field-three-point-bending.yml'


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bind():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != os.environ['GITHUB_SHA'] or os.environ['GITHUB_REPOSITORY'] != 'lililii124/jax-mpm' or os.environ['GITHUB_REF'] != 'refs/heads/work/phase-field-bending':
        raise ValueError('Exact reviewed HEAD/repository/feature ref required')
    active = '.github/workflows/' + os.environ['GITHUB_WORKFLOW_REF'].split('/.github/workflows/', 1)[1].split('@', 1)[0]
    if active != '.github/workflows/engineering-previews.yml' or (ROOT / active).read_bytes() != (ROOT / OWN).read_bytes():
        raise ValueError('Registered workflow differs from reviewed template')
    names = [OWN, active, 'pyproject.toml', 'tools/run_phase_field_bending.py', 'tools/run_phase_field_full.py', 'examples/phase_field_three_point_bending.py']
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'mpm_jax').rglob('*.py')]
    files = {}
    for name in sorted(set(names)):
        raw = (ROOT / name).read_bytes()
        if raw != subprocess.check_output(['git', 'show', head + ':' + name], cwd=ROOT):
            raise ValueError('Modified source: ' + name)
        files[name] = hashlib.sha256(raw).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in PINS}
    if platform.python_version() != '3.12.13' or versions != PINS:
        raise ValueError('Exact pinned runtime required')
    return dict(commit=head, files=files, python=platform.python_version(), versions=versions)


def main():
    mode = os.environ['MODE']
    if mode not in ('smoke', 'medium') or OUT.exists():
        raise ValueError('Fresh output and explicit qualification mode required')
    OUT.mkdir(parents=True)
    source = bind()
    save('source-runtime.json', source)
    steps = 2 if mode == 'smoke' else 1200
    command = [sys.executable, '-m', 'examples.phase_field_three_point_bending', '--steps', str(steps)]
    record = dict(mode=mode, requested_steps=steps, completed=False, peak_tree_uss_bytes=0,
                  uss_limit_bytes=int(5.5 * 1024**3), seconds_limit=5400,
                  minimum_available_bytes=2 * 1024**3, guard_reason=None)
    process = None
    started = time.monotonic()
    try:
        import ctypes
        if sys.platform != 'linux' or ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            raise RuntimeError('Linux owned-session subreaper required')
        with (OUT / 'solver.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                uss = owned_session_uss(process.pid, psutil)
                record['peak_tree_uss_bytes'] = max(record['peak_tree_uss_bytes'], uss)
                elapsed = time.monotonic() - started
                reason = ('USS limit' if uss > record['uss_limit_bytes'] else
                          'available memory limit' if psutil.virtual_memory().available < record['minimum_available_bytes'] else
                          'wall limit' if elapsed > record['seconds_limit'] else None)
                if reason:
                    record['guard_reason'] = reason
                    break
                save('resource-guard.json', record)
                time.sleep(1)
            code = process.poll()
    except BaseException as error:
        record['guard_reason'] = repr(error)
        raise
    finally:
        try:
            terminate_owned_group(process)
        finally:
            record.update(returncode=process.returncode if process else None, elapsed_seconds=time.monotonic() - started)
            save('resource-guard.json', record)
    if code or record['guard_reason']:
        raise RuntimeError('Qualification stopped; partial raw retained')
    summary = json.loads((OUT / 'summary.json').read_text())
    if summary['steps'] != steps or summary['particles'] != 12288 or summary['maximum_kkt'] > 1e-7 or summary['crack_propagation_validated'] is not False:
        raise ValueError('Candidate completion metadata differs')
    names = [0, 2] if steps == 2 else list(range(0, 1201, 100))
    expected = [f'frame_{i:06d}.npz' for i in names]
    if sorted(p.name for p in OUT.glob('frame_*.npz')) != expected:
        raise ValueError('Exact actual saved-state sequence required')
    previous = np.zeros(summary['particles'])
    for name in expected:
        with np.load(OUT / name) as frame:
            if frame['position'].shape != (summary['particles'], 3) or not all(np.isfinite(frame[k]).all() for k in frame.files):
                raise ValueError('Nonfinite or wrong-shaped saved state')
            damage = frame['damage']
            if damage.min() < 0 or damage.max() > 1 or (damage - previous).min() < -1e-12:
                raise ValueError('Damage bounds/irreversibility failed')
            previous = damage.copy()
    rows = list(csv.DictReader((OUT / 'history.csv').open()))
    if int(float(rows[-1]['step'])) != steps or any(not all(np.isfinite(float(v)) for v in row.values()) for row in rows):
        raise ValueError('Finite complete history required')
    if any(float(row['support_gap']) < -1e-10 or float(row['punch_gap']) < -1e-10 or float(row['left_reaction']) < -1e-8 or float(row['right_reaction']) < -1e-8 or float(row['punch_reaction']) > 1e-8 for row in rows):
        raise ValueError('Independent contact gap/reaction-sign audit failed')
    if bind() != source:
        raise ValueError('Source/runtime changed')
    save('qualification.json', dict(mode=mode, numerical_qualification_passed=True,
        actual_saved_states=len(expected), summary=summary, public_media_authorized=False,
        interpretation='Illustrative small-strain reference-grid flat-shoe candidate; no calibrated fracture, finite sliding or quasi-static claim.'))
    record['completed'] = True
    save('resource-guard.json', record)


if __name__ == '__main__':
    main()
