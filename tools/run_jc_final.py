"""Private exact-source frame-50 to completed frame-60 JC controller."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import run_jc_continuation as carry

ROOT = carry.ROOT
FILES = ('tools/run_jc_final.py', 'tools/review_jc_full_result.py', 'tools/check_jc_final_media.py',
         '.github/workflows/jc-final.yml', '.github/workflows/engineering-previews.yml', 'docs/JC_FINALIZATION_CONTRACT.md',
         'tools/jc_grid_separation.py', *carry.CONTROLLER_FILES)
# Cache archives retain their producer's path; changing restore.path does not
# relocate archived entries. Keep this exact path until moving the parent.
carry.CACHE = ROOT/'.local/jc-refinement-cache'
carry.PARENT = ROOT/'.local/jc-final-parent'
carry.OUTPUT = ROOT/'outputs/jc-final/grid100-ppc3/jc-necking/johnson-cook'
carry.LOGS = ROOT/'.local/jc-final-ci'
carry.CONTEXT = carry.LOGS/'context.json'
DELIVERY = ROOT/'.local/jc-final-delivery'
RETRIEVED = ROOT/'.local/jc-final-retrieved'
_CHILD_ENGINE = None


def controller_record():
    commit = carry.git(ROOT, 'rev-parse', 'HEAD').decode().strip()
    if (commit != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REPOSITORY') != carry.REPOSITORY
            or carry.git(ROOT, 'diff', 'HEAD', '--', *FILES, *carry.CONTROLLER_FILES).strip()):
        raise ValueError('Final controller must be the clean exact dispatched repository commit')
    if os.environ.get('GITHUB_REF') != 'refs/heads/work/jc-complete-refined':
        raise ValueError('Final controller requires the fixed refined completion feature ref')
    if (ROOT/'.github/workflows/engineering-previews.yml').read_bytes() != (ROOT/'.github/workflows/jc-final.yml').read_bytes():
        raise ValueError('Registered final workflow must equal frozen template bytes')
    for name in FILES:
        if carry.git(ROOT, 'show', f'{commit}:{name}') != (ROOT/name).read_bytes():
            raise ValueError('Uncommitted controller file: ' + name)
    return {'repository': carry.REPOSITORY, 'commit': commit,
            'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
            'files': {name: carry.digest(ROOT/name) for name in FILES},
            'scope': 'Completed-result orchestration/review/render only; immutable numerical source unchanged.'}


def request_values(parent_key, receipt_sha, from_frame, end_frame):
    if (not re.fullmatch(r'jc-continuation-v1-[A-Za-z0-9-]+-frame50', parent_key)
            or not re.fullmatch('[a-f0-9]{64}', receipt_sha) or (from_frame, end_frame) != (50, 60)):
        raise ValueError('Only an exact admitted frame-50 carry may complete the unchanged protocol')
    return {'parent_key': parent_key, 'parent_receipt_sha256': receipt_sha,
            'from_frame': 50, 'end_frame': 60, 'full_frames': 60, 'full_steps': 60*965,
            'full_duration_s': 150e-6, 'production_eligible': False}


# Reuse source/runtime admission and the original worker without widening the
# partial continuation controller's accepted endpoints or historical identity.
carry.controller_record = controller_record
carry.request_values = request_values


def child(action, minutes):
    global _CHILD_ENGINE
    context = carry.read_json(carry.CONTEXT)
    deadline_minutes = 305 if action == 'render-worker' else 265
    remaining = int(context['started_epoch'] + deadline_minutes*60 - time.time())
    if remaining < 60:
        raise TimeoutError('Final job budget exhausted; preserve diagnostic state')
    if _CHILD_ENGINE is None:
        _CHILD_ENGINE = carry.load_frozen_helper().engine
    _CHILD_ENGINE.command('jc-final-'+action,
        ['timeout', '--signal=INT', '--kill-after=60s', str(min(minutes*60, remaining))+'s',
         sys.executable, str(Path(__file__).resolve()), action], monitor=True)


def run():
    context = carry.read_json(carry.CONTEXT)
    # Parent byte/source admission happens again immediately before advancing.
    carry.verify_carry(carry.CACHE, context['request'])
    before = carry.prefix_snapshot(carry.CACHE/'result', 50)
    carry.save(carry.LOGS/'parent-prefix.json', before)
    # Working histories, retained parent, VTK export, ParaView installation and
    # two full delivery copies for artifact retrieval must coexist.
    required = 12*1024**3
    free = shutil.disk_usage(ROOT).free
    carry.save(carry.LOGS/'disk-admission.json', {'free_bytes': free, 'required_bytes': required,
                                               'passed': free >= required})
    if free < required:
        raise ValueError('Insufficient disk for full raw lineage, real ParaView and delivery round trip')
    carry.CACHE.rename(carry.PARENT)
    shutil.copytree(carry.PARENT/'result', carry.OUTPUT)
    os.environ['PYTHONPATH'] = str(carry.SOLVER)
    os.environ.pop('PYTHONHOME', None)
    child('audit-parent', 20)
    child('worker', 225)
    child('audit-full', 25)


def audit_full():
    from review_jc_full_result import audit
    context = carry.read_json(carry.CONTEXT)
    request = context['request']
    report = audit(carry.OUTPUT, carry.PARENT, request['parent_key'], request['parent_receipt_sha256'])
    # audit() has loaded only the verified immutable numerical helper.
    carry.runtime_check(context, None)
    report['controller_source'] = context['controller']
    carry.save(carry.OUTPUT/'full-result-review.json', report)
    carry.save(carry.LOGS/'completed-review-outcome.json', {
        'completed': True, 'full_protocol_completed': True,
        'resolved_saved_history_fracture': report['resolved_saved_history_fracture'],
        'production_eligible': False})


def install_paraview():
    carry.load_frozen_helper()
    from tools import run_woven_preview as pv
    pv.LOGS = carry.LOGS
    pv.install()
    record = carry.read_json(carry.LOGS/'paraview-install.json')
    record['executable_sha256'] = carry.digest(record['pvpython'])
    carry.save(carry.LOGS/'paraview-install.json', record)


def render_worker():
    context = carry.read_json(carry.CONTEXT)
    carry.load_frozen_helper()
    carry.runtime_check(context, None)
    review = carry.read_json(carry.OUTPUT/'full-result-review.json')
    if review.get('passed') is not True or not carry.inputs_match(carry.OUTPUT, review['input_sha256']):
        raise ValueError('Rendering requires a valid unchanged completed protocol')
    from mpm_jax.visualization import render_trajectory
    carry.verify_imports()
    render_trajectory(carry.OUTPUT, gif=True, field='damage', resolution=1200, color_range=(0, 1))
    carry.verify_imports()
    carry.runtime_check(context, None)
    install = carry.read_json(carry.LOGS/'paraview-install.json')
    pvpython = install['pvpython']
    if carry.digest(pvpython) != install['executable_sha256']:
        raise ValueError('ParaView executable changed')
    command = [pvpython, '--force-offscreen-rendering', str(ROOT/'tools/check_jc_final_media.py'),
               str(carry.OUTPUT), '--vtk']
    subprocess.run(command, cwd=ROOT, check=True)
    from check_jc_final_media import media_audit
    report = media_audit(carry.OUTPUT)
    report['paraview_install'] = install
    report['vtk_check_command'] = command
    report['renderer_source'] = {name: carry.digest(carry.SOLVER/name) for name in
        ('mpm_jax/visualization.py', 'mpm_jax/paraview_scene.py', 'mpm_jax/export.py')}
    carry.save(carry.OUTPUT/'media-audit.json', report)


def package():
    if DELIVERY.exists():
        raise ValueError('Final delivery already exists')
    DELIVERY.mkdir(parents=True)
    if carry.CONTEXT.is_file() and carry.PARENT.exists():
        context = carry.read_json(carry.CONTEXT)
        carry.verify_carry(carry.PARENT, context['request'])
    records, total = {}, 0
    roots = {'result': carry.OUTPUT, 'parent-cache': carry.PARENT if carry.PARENT.exists() else carry.CACHE,
             'controller-records': carry.LOGS}
    for prefix, root in roots.items():
        for path in sorted(root.rglob('*')) if root.exists() else []:
            if path.is_symlink():
                raise ValueError('Symlink in final raw delivery')
            if path.is_file():
                relative = prefix+'/'+path.relative_to(root).as_posix()
                records[relative] = carry.split_complete_file(path, DELIVERY, 'parts/'+relative)
                total += records[relative]['bytes']
    full = carry.read_json(carry.OUTPUT/'full-result-review.json') if (carry.OUTPUT/'full-result-review.json').is_file() else {}
    media = carry.read_json(carry.OUTPUT/'media-audit.json') if (carry.OUTPUT/'media-audit.json').is_file() else {}
    manifest = {'schema': 'jc-final-full-delivery-v1', 'files': records, 'payload_bytes': total,
                'full_protocol_completed': full.get('passed') is True,
                'resolved_saved_history_fracture': full.get('resolved_saved_history_fracture', False),
                'actual_paraview_media_verified': media.get('passed') is True,
                'artifact_round_trip_verified': False, 'production_eligible': False,
                'controller': controller_record(), 'parent_cache_retained': bool(records),
                'manifest_scope': 'All current raw outputs, exact parent cache and controller records; failed work remains diagnostic.'}
    carry.save(DELIVERY/'delivery-manifest.json', manifest)
    verify_package(DELIVERY)


def verify_package(folder):
    folder = Path(folder)
    manifest = carry.read_json(folder/'delivery-manifest.json')
    expected = {'delivery-manifest.json'}
    total = 0
    for name, record in manifest['files'].items():
        if not record.get('complete_file') or record['bytes'] < 0:
            raise ValueError('Incomplete raw file record')
        whole, size = hashlib.sha256(), 0
        for index, part in enumerate(record['parts']):
            if part['path'] != 'parts/'+name+f'.part-{index:04d}':
                raise ValueError('Unordered or substituted raw file part')
            path = carry.contained_file(folder, part['path'])
            if carry.file_record(path) != {'bytes': part['bytes'], 'sha256': part['sha256']}:
                raise ValueError('Raw file part failed byte/hash verification')
            expected.add(part['path'])
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(carry.READ_BYTES), b''):
                    whole.update(block)
                    size += len(block)
        if size != record['bytes'] or whole.hexdigest() != record['sha256']:
            raise ValueError('Reconstructed complete raw file hash differs')
        total += size
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()}
    if actual != expected or total != manifest['payload_bytes'] or any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Missing/extra delivery files or incorrect accounting')
    return manifest


def verify_retrieved():
    expected_sha = carry.digest(DELIVERY/'delivery-manifest.json')
    if carry.digest(RETRIEVED/'delivery-manifest.json') != expected_sha:
        raise ValueError('Downloaded artifact manifest differs from uploaded source')
    manifest = verify_package(RETRIEVED)
    record = {'schema': 'jc-final-artifact-round-trip-v1', 'verified': True,
              'manifest_sha256': expected_sha, 'payload_bytes': manifest['payload_bytes'],
              'full_protocol_completed': manifest['full_protocol_completed'],
              'resolved_saved_history_fracture': manifest['resolved_saved_history_fracture'],
              'actual_paraview_media_verified': manifest['actual_paraview_media_verified'],
              'artifact_name': os.environ['JC_FINAL_ARTIFACT_NAME'],
              'artifact_url': os.environ['JC_FINAL_ARTIFACT_URL'],
              'artifact_id': os.environ['JC_FINAL_ARTIFACT_ID'],
              'retention_days': 30, 'production_eligible': False}
    carry.save(carry.LOGS/'artifact-round-trip.json', record)
    if not (record['full_protocol_completed'] and record['actual_paraview_media_verified']):
        raise ValueError('Round-trip succeeded only for diagnostics; full protocol/media incomplete')
    if not record['resolved_saved_history_fracture']:
        raise ValueError('Completed protocol and actual GIF preserved, but saved-history fracture gates failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'audit-parent', 'audit-full',
        'install-paraview', 'render', 'render-worker', 'package', 'verify-retrieved'))
    parser.add_argument('--parent-key')
    parser.add_argument('--receipt-sha')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            args.from_frame, args.end_frame = 50, 60
            carry.prepare(args)
        else:
            {'run': run, 'worker': carry.worker, 'audit-parent': lambda: carry.audit_worker(parent=True),
             'audit-full': audit_full, 'install-paraview': install_paraview,
             'render': lambda: child('render-worker', 40), 'render-worker': render_worker,
             'package': package, 'verify-retrieved': verify_retrieved}[args.action]()
    except Exception:
        carry.save(carry.LOGS/(args.action+'-failure.json'), {'completed': False,
                   'production_eligible': False, 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
