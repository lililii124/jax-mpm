"""Read-only full-review diagnosis of one sealed failed ceramic cache.

Never runs or imports the mechanical solver and never admits/resumes evidence.
The exact original review report is written outside raw data even on rejection.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

import run_ceramic_continuation as c
from recover_ceramic_review import normalized

FAILED_KEY = ('ceramic-continuation-v1-68c311813b11124459b7750f52695c7caa9b6c39-'
              'df50b47ab8fcda24a924f196b1348609b34f63b9-40069099744-1-f60-failed')
RECEIPT_SHA = 'cbd552706c53db80415ffc2d24e740cb32c907baf36be9d2295eb09f001ce8ed'
OUTPUT = c.ROOT / '.local/ceramic-full-review-probe'


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_new(path, value):
    """No existing diagnostic or raw report may be replaced."""
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(normalized(value), indent=2, allow_nan=False) + '\n')


def collect_reports(folder, output, physical, spatial):
    """Save the physical rejection before spatial analysis or gate assertions."""
    report = normalized(physical.review(folder, allow_prefix=False))
    save_new(output / 'original-full-physical-review.json', report)
    checks = report.get('checks', {})
    save_new(output / 'physical-gates.json', {
        'frame': report.get('accepted_saved_frame'),
        'completed': report.get('completed'), 'passed': report.get('passed'),
        'failed_checks': [key for key, value in checks.items() if value is not True],
        'checks': checks, 'physical_admission': False, 'production_eligible': False,
        'scope': 'Diagnostic only; a returned report cannot change the failed receipt.'})
    descriptive = spatial.review(folder)
    save_new(output / 'spatial-failure-frame-60.json', descriptive)
    return report


def probe(cache=c.CACHE, output=OUTPUT):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('Diagnostic output must be separate from all failed raw data')
    if output.exists():
        raise FileExistsError('Use a new diagnostic directory; no evidence is overwritten')
    receipt_path = c.local_file(cache, 'receipt.json')
    if c.digest(receipt_path) != RECEIPT_SHA:
        raise ValueError('Failed receipt differs from admitted artifact 12552605375')
    receipt = c.read(receipt_path)
    if (receipt.get('cache_key') != FAILED_KEY or receipt.get('schema') != 'ceramic-continuation-v1'
            or receipt.get('kind') != 'failed-diagnostic'
            or receipt.get('never_resumable') is not True
            or receipt.get('prefix_verified') is not False
            or receipt.get('accepted_frame') is not None
            or receipt.get('requested_end_frame') != 60):
        raise ValueError('This extractor accepts only the sealed failed frame-60 cache')
    before = c.inventory(cache)
    if before != receipt.get('files'):
        raise ValueError('Complete failed raw inventory differs from its sealed receipt')
    source = c.read(c.local_file(cache, 'source.json'))
    if source != receipt.get('source'):
        raise ValueError('Failed cache source differs from sealed source identity')
    c.verify_source(source)
    c.verify_controller(receipt['controller'])
    pins = c.dependency_pins(c.local_file(cache, 'requirements.txt').read_text())
    if (platform.python_version() != '3.12.13' or platform.system() != 'Linux'
            or importlib.metadata.version('numpy') != pins['numpy']):
        raise ValueError('Read-only review requires original Python/Linux and NumPy versions')
    folder = cache / 'result'
    cp, status = (c.read(c.local_file(folder, name)) for name in ('checkpoint.json', 'status.json'))
    if cp.get('frame') != 60 or status.get('frame') != 60 or status.get('completed') is not True:
        raise ValueError('Saved full-plan endpoint is not frame 60')
    # Explicit completeness checks supplement the sealed inventory equality.
    for name in ('trajectory_x.npy', 'trajectory_damage.npy', 'trajectory_failed.npy',
                 'summary.json', 'final_state_0.npz', 'observables.csv', *cp['states']):
        c.local_file(folder, name)
    for frame in range(61):
        c.local_file(folder, f'frames_data/frame_{frame:05d}.npz')
    physical_path = c.SOLVER / 'tools/review_ceramic_fragmentation.py'
    if physical_path.read_bytes() != c.git(c.SOLVER, 'show', f'{c.PRODUCER}:tools/review_ceramic_fragmentation.py'):
        raise ValueError('Physical reviewer differs from the immutable producer')
    spatial_path = c.ROOT / 'tools/review_ceramic_spatial_failure.py'
    controller_paths = (Path(__file__).resolve(), spatial_path,
                        c.ROOT / 'tools/run_ceramic_continuation.py',
                        c.ROOT / 'tools/recover_ceramic_review.py')
    controller = c.git(c.ROOT, 'rev-parse', 'HEAD').decode().strip()
    if controller != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF') != c.FEATURE_REF:
        raise ValueError('Probe must run from a committed controller in the original cache scope')
    for path in controller_paths:
        name = path.relative_to(c.ROOT).as_posix()
        if path.read_bytes() != c.git(c.ROOT, 'show', f'{controller}:{name}'):
            raise ValueError('Uncommitted diagnostic controller: ' + name)
    output.mkdir(parents=True)
    save_new(output / 'diagnostic-provenance.json', {
        'failed_cache_key': FAILED_KEY, 'failed_receipt_sha256': RECEIPT_SHA,
        'source': source, 'controller_commit': controller,
        'controller_files': {p.relative_to(c.ROOT).as_posix(): c.file_record(p) for p in controller_paths},
        'physical_reviewer': c.file_record(physical_path),
        'python': platform.python_version(), 'numpy': pins['numpy'],
        'solver_executed': False, 'physical_admission': False, 'production_eligible': False})
    try:
        report = collect_reports(folder, output,
            load_module(physical_path, 'frozen_physical_review'),
            load_module(spatial_path, 'diagnostic_spatial_review'))
    finally:
        unchanged = c.inventory(cache) == before and c.digest(receipt_path) == RECEIPT_SHA
        save_new(output / 'raw-preservation.json', {'whole_raw_inventory_unchanged': unchanged,
            'receipt_sha256': RECEIPT_SHA, 'files_checked': len(before),
            'never_resumable': True, 'production_eligible': False})
        if not unchanged:
            raise ValueError('Failed raw bytes changed during diagnostic review')
    if any(name == 'jax' or name.startswith(('jax.', 'mpm_jax.', 'examples.')) for name in sys.modules):
        raise ValueError('Mechanical solver module imported during read-only diagnosis')
    print(json.dumps({'diagnostic_complete': True, 'frame': 60,
        'physical_review_passed': report.get('passed'),
        'failed_checks': [key for key, value in report.get('checks', {}).items() if value is not True],
        'failed_cache_unchanged': True, 'production_eligible': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=c.CACHE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    probe(args.cache, args.output)
