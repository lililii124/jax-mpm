"""Exact-source, cache-backed saved-frame segments of the full large-plate plan."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_woven_large as probe
from tools.runner_support import cpu_record, environment, file_record

engine = probe.engine
save, digest = engine.save, engine.digest
CACHE = ROOT / '.local/woven-large-segment-cache'
LOGS = ROOT / '.local/woven-large-segment-ci'
DELIVERY = ROOT / '.local/woven-large-segment-delivery'
FINAL = ROOT / '.local/woven-large-segment-final'
CONTEXT = LOGS / 'job.json'
PYTHON_VERSION = '3.12.13'
CACHE_BUDGET = 8_000_000_000
REPOSITORY_CACHE_BUDGET = 9_000_000_000
PACKAGE_BUDGET = 192 * 1024**2
RESULT_NAMES = set(probe.REVIEW_INPUTS) | {'interpretation.json', 'memory.json', 'numerical-source.json', 'exception.json'}
RUNTIME_NAMES = {'request.json', 'source.json', 'environment.json', 'requirements.txt'}
STOPS = (20, 40, 60)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def request(architecture='plain', loading='sphere-impact'):
    if architecture not in probe.ARCHITECTURES or loading not in probe.LOADS:
        raise ValueError('Unknown architecture or loading')
    return {'schema': 'woven-large-segments-v1', 'architecture': architecture, 'loading': loading,
            'frames': 60, 'particles': 819200, 'dx_m': .0005, 'probe_steps': 0,
            'duration_s': 200e-6, 'full_steps': 14460, 'stops': list(STOPS),
            'production_eligible': False, 'scope': 'Coarse dynamic structural example; no material validation.'}


def frame_for_stage(stage):
    if stage not in ('20', '40', '60', 'finalize'):
        raise ValueError('Unknown segment stage')
    return 60 if stage == 'finalize' else int(stage)


def previous_frame(stage):
    return 60 if stage == 'finalize' else frame_for_stage(stage) - 20


def cache_key(commit, run_id, attempt, req, frame):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Cache source must be a complete commit SHA')
    if any(not re.fullmatch(r'[1-9][0-9]*', str(value)) for value in (run_id, attempt)) or frame not in STOPS:
        raise ValueError('Invalid cache run identity or frame')
    request(req['architecture'], req['loading'])
    return f"woven-large-v1-{commit}-{run_id}-{attempt}-{req['architecture']}-{req['loading']}-f{frame:02d}"


def result_folder(base, req):
    return Path(base) / 'data' / req['architecture'] / req['loading']


def source_record():
    # Inspect a literal, without importing NumPy/JAX before cache verification.
    tree = ast.parse((ROOT / 'examples/woven_large_plate.py').read_text(encoding='utf-8'))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES' for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('Numerical source declaration is ambiguous')
    names = ast.literal_eval(assignments[0].value)
    numerical = {name: digest(ROOT / name) for name in names}
    helpers = ('tools/run_woven_large_segments.py', 'tools/run_woven_large.py', 'tools/run_engineering_preview.py',
               'tools/run_woven_preview.py', 'tools/runner_support.py',
               'tools/review_woven_large.py', 'tools/install_runner_dependencies.py', 'tools/check_vtk.py',
               'tools/encode_video.py', 'mpm_jax/visualization.py', 'mpm_jax/export.py',
               'mpm_jax/paraview_scene.py', 'mpm_jax/saved_tool_motion.py', 'pyproject.toml',
               '.github/workflows/woven-large-segments.yml')
    identity = engine.identity()
    if os.environ.get('GITHUB_SHA') and identity['commit'] != os.environ['GITHUB_SHA']:
        raise ValueError('Checkout does not match the dispatched immutable commit')
    files = {name: digest(ROOT / name) for name in (*names, *helpers)}
    blobs = committed_source(files)
    return {key: identity[key] for key in ('source_kind', 'repository', 'commit', 'tree')} | {
        'numerical_source': {'files': numerical, 'fingerprint': object_hash(numerical)},
        'files': files, 'git_blob_sha': blobs,
        'review_source_sha256': digest(ROOT / 'tools/review_woven_large.py')}


def committed_source(files):
    blobs = {}
    for name in files:
        raw = (ROOT / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        expected = subprocess.check_output(['git', 'rev-parse', f'HEAD:{name}'], cwd=ROOT, text=True).strip()
        if blob != expected or hashlib.sha256(raw).hexdigest() != files[name]:
            raise ValueError('A controller or numerical source file differs from the actual Git commit: ' + name)
        blobs[name] = blob
    return blobs


def requirements_text(record):
    packages = record.get('packages')
    if (record.get('python') != PYTHON_VERSION or not isinstance(packages, dict)
            or not {'numpy', 'scipy', 'jax', 'jaxlib', 'psutil', 'pillow'}.issubset(packages)):
        raise ValueError('Runtime lock must include the pinned Python and complete scientific environment')
    for name, version in packages.items():
        if (not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name)
                or not isinstance(version, str) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version)):
            raise ValueError('Cached installation permits only distribution names and exact version pins')
    return ''.join(f'{name}=={version}\n' for name, version in sorted(packages.items()))


def cache_estimate():
    # Four histories, both complete state archives, immutable reference and layout.
    raw_history = 61 * 819200 * (3 * 8 + 3 * 4)
    snapshot = raw_history + 819200 * 740 + 64 * 1024**2
    if 3 * snapshot > CACHE_BUDGET:
        raise ValueError('The three complete snapshots exceed the uncompressed cache budget')
    return {'history_bytes': raw_history, 'estimated_snapshot_bytes': snapshot,
            'estimated_three_snapshot_bytes': 3 * snapshot, 'budget_bytes': CACHE_BUDGET,
            'scope': 'Conservative raw estimate; every sealed payload is counted again. No paid cache expansion.'}


def disk_admission(stage, free_bytes):
    snapshot = cache_estimate()['estimated_snapshot_bytes']
    # Restored raw files already occupy disk; reserve another complete cache archive.
    growth = snapshot if stage == '20' else 0
    rendering = snapshot + 61 * 819200 * 96 if stage == 'finalize' else 0
    archive = 0 if stage == 'finalize' else snapshot
    required = growth + rendering + archive + 2 * 1024**3
    record = {'free_bytes': int(free_bytes), 'required_free_bytes': required,
              'passed': bool(free_bytes >= required), 'stage': stage,
              'raw_growth_bytes': growth, 'cache_archive_reserve_bytes': archive,
              'final_copy_and_vtk_reserve_bytes': rendering, 'margin_bytes': 2 * 1024**3}
    if not record['passed']:
        raise ValueError(f'Insufficient disk space: {record}')
    return record


def repository_cache_usage(repository):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Invalid repository identity')
    token = os.environ.get('GH_TOKEN', '')
    if not token:
        raise ValueError('Read-only cache usage token is required')
    url = f'https://api.github.com/repos/{repository}/actions/cache/usage'
    query = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + token,
        'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
    with urllib.request.urlopen(query, timeout=30) as response:
        value = json.load(response)
    size = value.get('active_caches_size_in_bytes')
    if type(size) is not int or size < 0:
        raise ValueError('Cache usage response has no reliable byte count')
    return {'active_caches_size_in_bytes': size, 'observed_unix_s': time.time(),
            'repository': repository, 'scope': 'Read-only usage; no quota or billing changes.'}


def emit(values):
    path = os.environ.get('GITHUB_OUTPUT')
    if path:
        with open(path, 'a', encoding='utf-8') as stream:
            for key, value in values.items():
                if '\n' in str(value) or '\r' in str(value):
                    raise ValueError('Invalid workflow output')
                stream.write(f'{key}={value}\n')


def prepare(args):
    if CONTEXT.exists() or CACHE.exists():
        raise FileExistsError('Preparation requires a fresh job workspace')
    req = request(args.architecture, args.loading)
    stage = args.stage
    end, prior = frame_for_stage(stage), previous_frame(stage)
    source, execution = source_record(), engine.identity()
    parent_run, parent_attempt, parent_sha = args.parent_run, args.parent_attempt, args.parent_receipt_sha
    if prior:
        if not re.fullmatch(r'[0-9a-f]{64}', parent_sha):
            raise ValueError('Continuation requires the independently admitted parent receipt SHA')
        parent_key = cache_key(source['commit'], parent_run, parent_attempt, req, prior)
    else:
        if any((parent_run, parent_attempt, parent_sha)):
            raise ValueError('Fresh stage 20 must not inherit a probe or a previous result')
        parent_key = ''
    key = parent_key if stage == 'finalize' else cache_key(source['commit'], execution['run_id'], execution['run_attempt'], req, end)
    usage = repository_cache_usage(source['repository'])
    remaining = 0 if stage == 'finalize' else (80 - end) // 20
    estimate = cache_estimate()
    if usage['active_caches_size_in_bytes'] + remaining * estimate['estimated_snapshot_bytes'] > REPOSITORY_CACHE_BUDGET:
        raise ValueError('Existing caches plus all remaining raw snapshots exceed the conservative repository budget')
    LOGS.mkdir(parents=True, exist_ok=True)
    context = {'schema': 1, 'stage': stage, 'end_frame': end, 'previous_frame': prior, 'request': req,
        'source': source, 'execution': execution, 'parent_key': parent_key, 'cache_key': key,
        'parent_receipt_sha256': parent_sha, 'started_unix_s': time.time(),
        'deadline_unix_s': time.time() + 345 * 60, 'cache_estimate': estimate, 'repository_cache_usage': usage,
        'disk': disk_admission(stage, shutil.disk_usage(ROOT).free), 'production_eligible': False}
    save(CONTEXT, context)
    emit({'cache_path': CACHE.relative_to(ROOT).as_posix(), 'cache_key': key, 'parent_key': parent_key,
          'needs_restore': str(bool(prior)).lower(), 'is_finalization': str(stage == 'finalize').lower()})
    return context


def checkpoint_members(path):
    """Hash complete NPY members, including their dtype/shape headers, in bounded chunks."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.filename
            if name in result or not re.fullmatch(r'[A-Za-z0-9_]+\.npy', name):
                raise ValueError('Unexpected or duplicate checkpoint member')
            hashed, size = hashlib.sha256(), 0
            with archive.open(item) as stream:
                while block := stream.read(4 * 1024**2):
                    hashed.update(block)
                    size += len(block)
            if size != item.file_size:
                raise ValueError('Truncated checkpoint member')
            result[name] = {'bytes': size, 'sha256': hashed.hexdigest()}
    if not result:
        raise ValueError('Empty checkpoint archive')
    return result


def inventory(base, req, excluded=None, *, complete=True):
    base = Path(base)
    folder = result_folder(base, req)
    records = {}
    for path in sorted(base.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are not cache data')
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        if relative == excluded:
            continue
        result_names = RESULT_NAMES if complete else RESULT_NAMES | {name + '.tmp' for name in RESULT_NAMES}
        allowed = (path.parent == folder and path.name in result_names
            or path.parent == base / 'runtime' and path.name in RUNTIME_NAMES
            or path.parent == base / 'receipts' and re.fullmatch(r'(frame|failed)-(20|40|60)\.json', path.name))
        if not allowed:
            raise ValueError(f'Unexpected cache payload: {relative}')
        records[relative] = file_record(path)
    needed = (set(probe.REVIEW_INPUTS) | {'interpretation.json', 'memory.json', 'numerical-source.json'} if complete
              else set())
    required = {str((folder / name).relative_to(base).as_posix()) for name in needed}
    required |= {f'runtime/{name}' for name in RUNTIME_NAMES}
    if not required.issubset(records):
        raise ValueError('Cache is missing complete raw state, review, runtime or source evidence')
    if not complete and not any(path.parent == folder and path.is_file() for path in folder.glob('*')):
        raise ValueError('There are no original numerical files to preserve')
    return records


def validate_acceptance(plan, summary, status, review, req, frame, start, numerical_source):
    checks = review.get('checks')
    full = frame == 60
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError('The independent review contains missing or false checks')
    if not (plan.get('source') == summary.get('source') == review.get('producer_source') == numerical_source
            and plan.get('architecture') == req['architecture'] and plan.get('loading') == req['loading']
            and plan.get('particles') == 819200 and summary.get('particles') == [819200]
            and plan.get('frames') == 60 and plan.get('probe_steps') == 0 and plan.get('stop_steps') == 14460
            and plan.get('full_steps') == summary.get('full_steps') == status.get('full_steps') == 14460
            and plan.get('duration_s') == summary.get('target_simulation_time') == 200e-6
            and plan.get('grid', {}).get('dx_m') == .0005 and plan.get('CFL') == .15
            and plan.get('output_steps') == plan.get('frame_steps') == list(range(0, 14461, 241))):
        raise ValueError('The result is not the fixed complete physical plan')
    if not (summary.get('completed') is full and status.get('completed') is full
            and summary.get('probe_complete') is False and status.get('probe_complete') is False
            and summary.get('segment_complete') is (not full) and status.get('segment_complete') is (not full)
            and summary.get('phase') == status.get('phase') == ('completed' if full else 'checkpoint-ready')
            and summary.get('segment_start_frame') == status.get('segment_start_frame') == start
            and summary.get('segment_end_frame') == status.get('segment_end_frame') == status.get('frame') == frame
            and summary.get('runtime_stop_after_frame') == status.get('runtime_stop_after_frame') == frame
            and status.get('frames') == 60 and status.get('target_steps') == 14460
            and status.get('coupling_valid') is True and status.get('signature') == object_hash(plan)
            and summary.get('steps') == status.get('steps') == frame * 241
            and summary.get('simulation_time') == status.get('time')
            and review.get('passed') is full and review.get('segment_passed') is (not full)
            and review.get('probe_passed') is False and review.get('full_protocol_completed') is full
            and review.get('full_saved_history_checked') is True and review.get('segment_end_frame') == frame
            and review.get('allocated_states') == 61 and review.get('accepted_saved_states') == frame + 1):
        raise ValueError('Saved-prefix, checkpoint-ready or full-protocol semantics disagree')
    return {'accepted_prefix': True, 'end_frame': frame, 'start_frame': start,
            'full_protocol_completed': full, 'segment_passed': not full, 'full_review_passed': full,
            'production_eligible': False, 'media_created': False}


def inspect_result(context, *, folder=None, frame=None, start=None):
    req = context['request']
    folder = result_folder(CACHE, req) if folder is None else Path(folder)
    frame = context['end_frame'] if frame is None else frame
    start = context['previous_frame'] if start is None else start
    plan, summary, status, review = (read(folder / name) for name in ('plan.json', 'summary.json', 'status.json', 'interpretation.json'))
    result = validate_acceptance(plan, summary, status, review, req, frame, start, context['source']['numerical_source'])
    if not probe.verify_review_inputs(folder, summary, review):
        raise ValueError('Reviewer source or one of the sixteen raw input hashes differs')
    result.update({'review_sha256': digest(folder / 'interpretation.json'),
        'review_source_sha256': review['review_source_sha256'], 'plan_sha256': digest(folder / 'plan.json'),
        'signature': status['signature'], 'input_sha256': review['input_sha256']})
    return result


def validate_receipt(receipt, context, frame, key):
    if not (receipt.get('schema') == 'woven-large-cache-receipt-v1' and receipt.get('frame') == frame
            and receipt.get('kind') == 'accepted-prefix' and receipt.get('numerically_accepted') is True
            and receipt.get('never_resumable') is False
            and receipt.get('cache_key') == key and receipt.get('request') == context['request']
            and receipt.get('source') == context['source'] and receipt.get('accepted_prefix') is True
            and receipt.get('full_protocol_completed') is (frame == 60)
            and receipt.get('production_eligible') is False and receipt.get('media_created') is False):
        raise ValueError('Cache receipt identity or completed/partial claims differ')
    if not isinstance(receipt.get('files'), dict) or not receipt['files']:
        raise ValueError('Cache receipt has no complete inventory')


def verify_restore(context):
    frame = context['previous_frame']
    if not frame:
        raise ValueError('Fresh first stage has no restore operation')
    path = CACHE / 'receipts' / f'frame-{frame}.json'
    if digest(path) != context['parent_receipt_sha256']:
        raise ValueError('Parent receipt differs from the admitted SHA')
    receipt = read(path)
    validate_receipt(receipt, context, frame, context['parent_key'])
    actual = inventory(CACHE, context['request'], path.relative_to(CACHE).as_posix())
    if actual != receipt['files']:
        raise ValueError('Restored cache bytes differ from the complete sealed manifest')
    folder = result_folder(CACHE, context['request'])
    if checkpoint_members(folder / 'checkpoint.npz') != receipt['checkpoint_members']:
        raise ValueError('Restored state/history checkpoint members differ')
    if (read(CACHE / 'runtime/request.json') != context['request']
            or read(CACHE / 'runtime/source.json') != context['source']):
        raise ValueError('Restored request or immutable source differs')
    if (CACHE / 'runtime/requirements.txt').read_text(encoding='utf-8') != requirements_text(read(CACHE / 'runtime/environment.json')):
        raise ValueError('Cached installation text is not the exact safe runtime lock')
    # Prior receipts remain immutable even though their old data files have advanced.
    cursor, expected = receipt, frame
    while expected > 20:
        parent = cursor['parent']
        old_path = CACHE / 'receipts' / f'frame-{expected - 20}.json'
        if parent['frame'] != expected - 20 or digest(old_path) != parent['receipt_sha256']:
            raise ValueError('An ancestor receipt link is broken')
        old = read(old_path)
        validate_receipt(old, context, expected - 20, parent['cache_key'])
        if parent['checkpoint_sha256'] != old['checkpoint']['sha256']:
            raise ValueError('An ancestor checkpoint identity is broken')
        cursor, expected = old, expected - 20
    if cursor.get('parent') is not None:
        raise ValueError('The first full-plan segment cannot inherit a probe')
    if receipt['cumulative_raw_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Restored cache chain exceeds its budget')
    save(LOGS / 'restore-verification.json', {'verified': True, 'frame': frame,
        'cache_key': context['parent_key'], 'receipt_sha256': context['parent_receipt_sha256'],
        'checked_files': len(actual), 'checkpoint_members_checked': len(receipt['checkpoint_members'])})
    return receipt


def lock_environment(context):
    if source_record() != context['source']:
        raise ValueError('Source changed before environment admission')
    actual = environment()
    if actual['python'] != PYTHON_VERSION or actual['system'] != 'Linux' or actual['os_version_id'] != '24.04':
        raise ValueError('The pinned Python/Ubuntu runtime is required')
    if context['previous_frame']:
        verify_restore(context)
        if read(CACHE / 'runtime/environment.json') != actual:
            raise ValueError('Continuation scientific package/runtime lock differs')
    else:
        runtime = CACHE / 'runtime'
        runtime.mkdir(parents=True, exist_ok=False)
        save(runtime / 'environment.json', actual)
        save(runtime / 'source.json', context['source'])
        save(runtime / 'request.json', context['request'])
        (runtime / 'requirements.txt').write_text(requirements_text(actual), encoding='utf-8')
    save(LOGS / 'runtime-lock.json', {'verified': True, 'environment': actual, 'cpu': cpu_record(),
        'scope': 'CPU recorded independently; exact reference arrays and rebuilt layout checks prevent silent resampling.'})


def guarded(context, name, arguments, maximum_minutes, reserve_minutes=15, accepted_codes=(0,)):
    remaining = context['deadline_unix_s'] - time.time() - reserve_minutes * 60
    seconds = min(maximum_minutes * 60, int(remaining))
    if seconds < 60:
        raise TimeoutError('Job deadline leaves insufficient time for this complete operation and evidence tail')
    engine.LOGS = LOGS
    return engine.command(name, ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s', *map(str, arguments)],
                          monitor=True, accepted_codes=accepted_codes)


def worker(context):
    if context['stage'] == 'finalize':
        raise ValueError('Finalization must not advance the solver')
    from examples import woven_large_plate as driver
    if driver.source_record() != context['source']['numerical_source']:
        raise ValueError('Worker numerical imports differ from the sealed source')
    req = context['request']
    folder = result_folder(CACHE, req)
    if context['previous_frame']:
        sample, plan = driver.load_saved_case(folder)
    else:
        if folder.exists():
            raise FileExistsError('Fresh full plan cannot replace prior raw data or a probe')
        sample, plan = driver.make_plan(req['architecture'], req['loading'], frames=60, probe_steps=0, dx=.0005)
    if (plan['particles'] != 819200 or plan['full_steps'] != 14460 or plan['probe_steps'] != 0
            or plan['duration_s'] != 200e-6 or plan['source'] != context['source']['numerical_source']):
        raise ValueError('Constructed or restored physical plan differs')
    driver.run_case(sample, plan, CACHE / 'data', resume=bool(context['previous_frame']),
                    stop_after_frame=context['end_frame'])


def seal(context, acceptance, command):
    if context['stage'] == 'finalize':
        raise ValueError('Finalization cannot create another raw snapshot')
    if (acceptance.get('accepted_prefix') is not True or acceptance.get('end_frame') != context['end_frame']
            or acceptance.get('full_protocol_completed') is not (context['end_frame'] == 60)
            or acceptance.get('production_eligible') is not False or acceptance.get('media_created') is not False):
        raise ValueError('Only an independently accepted saved prefix can be sealed for continuation')
    folder = result_folder(CACHE, context['request'])
    memory = command.get('memory', {})
    if not (command.get('completed') is True and memory.get('samples', 0) > 0
            and memory.get('stopped_for_memory') is False
            and memory.get('peak_tree_uss_bytes', 13 * 1024**3) <= 12 * 1024**3
            and memory.get('minimum_system_available_bytes', 0) >= 2 * 1024**3):
        raise ValueError('A completed guarded solve is required before sealing')
    if source_record() != context['source'] or environment() != read(CACHE / 'runtime/environment.json'):
        raise ValueError('Source or runtime changed during the segment')
    save(folder / 'memory.json', memory)
    save(folder / 'numerical-source.json', context['source'])
    parent, prior_bytes = None, 0
    if context['previous_frame']:
        path = CACHE / 'receipts' / f"frame-{context['previous_frame']}.json"
        old = read(path)
        if digest(path) != context['parent_receipt_sha256']:
            raise ValueError('Parent receipt changed during advance')
        prior_bytes = old['cumulative_raw_snapshot_bytes']
        parent = {'frame': context['previous_frame'], 'receipt_sha256': digest(path),
                  'cache_key': old['cache_key'], 'checkpoint_sha256': old['checkpoint']['sha256']}
    frame = context['end_frame']
    path = CACHE / 'receipts' / f'frame-{frame}.json'
    if path.exists():
        raise FileExistsError('A sealed receipt is immutable')
    files = inventory(CACHE, context['request'])
    receipt = {'schema': 'woven-large-cache-receipt-v1', 'kind': 'accepted-prefix', 'numerically_accepted': True,
        'never_resumable': False, 'request': context['request'], 'source': context['source'],
        'execution': context['execution'], 'cache_key': context['cache_key'], 'frame': frame, 'parent': parent,
        **acceptance, 'files': files, 'checkpoint': file_record(folder / 'checkpoint.npz'),
        'checkpoint_members': checkpoint_members(folder / 'checkpoint.npz'),
        'checkpoint_member_hash_scope': 'Complete uncompressed NPY members, including dtype and shape headers.',
        'snapshot_raw_bytes': 0, 'cumulative_raw_snapshot_bytes': 0,
        'cache_persistence': 'Temporary exact-key cache, not a permanent scientific data archive.'}
    payload_bytes = sum(value['bytes'] for value in files.values())
    for _ in range(10):
        encoded = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode()
        count = payload_bytes + len(encoded)
        if receipt['snapshot_raw_bytes'] == count:
            break
        receipt['snapshot_raw_bytes'] = count
        receipt['cumulative_raw_snapshot_bytes'] = prior_bytes + count
    else:
        raise ValueError('Receipt byte count did not stabilize')
    if receipt['cumulative_raw_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Complete raw cache snapshots exceed the immutable budget')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    save(LOGS / 'sealed-receipt.json', {'cache_key': context['cache_key'], 'receipt_sha256': digest(path),
                                      'frame': frame, 'snapshot_raw_bytes': count})
    emit({'receipt_sha256': digest(path), 'cache_key': context['cache_key'], 'frame': frame})
    return receipt


def seal_failure(context):
    """Preserve available raw bytes after rejection without admitting any continuation."""
    if context['stage'] == 'finalize':
        raise ValueError('Finalization uses the existing immutable f60 cache')
    if source_record() != context['source']:
        raise ValueError('Cannot bind failed data to a changed source')
    if (LOGS / 'sealed-receipt.json').exists():
        raise ValueError('An accepted snapshot was already sealed; preserve its identity')
    files = inventory(CACHE, context['request'], complete=False)
    folder = result_folder(CACHE, context['request'])
    frame = context['end_frame']
    parent = None
    previous_bytes = 0
    if context['previous_frame']:
        path = CACHE / 'receipts' / f"frame-{context['previous_frame']}.json"
        if digest(path) != context['parent_receipt_sha256']:
            raise ValueError('Failed snapshot has a changed parent receipt')
        old = read(path)
        previous_bytes = old['cumulative_raw_snapshot_bytes']
        parent = {'receipt_sha256': digest(path), 'cache_key': old['cache_key'], 'frame': old['frame']}
    def original_record(path):
        if not path.is_file():
            return None
        try:
            return read(path)
        except (ValueError, OSError) as error:
            return {'raw_file': file_record(path), 'decode_error': type(error).__name__ + ': ' + str(error)}

    checkpoint = folder / 'checkpoint.npz'
    members, member_error = None, None
    if checkpoint.is_file():
        try:
            members = checkpoint_members(checkpoint)
        except Exception as error:
            member_error = type(error).__name__ + ': ' + str(error)
    receipt = {'schema': 'woven-large-cache-receipt-v1', 'kind': 'failed-diagnostic',
        'numerically_accepted': False, 'never_resumable': True, 'accepted_prefix': False,
        'full_protocol_completed': False, 'full_review_passed': False, 'production_eligible': False,
        'media_created': False, 'frame': frame, 'frame_scope': 'Requested boundary; failure may precede it.',
        'cache_key': context['cache_key'] + '-failed', 'request': context['request'],
        'source': context['source'], 'execution': context['execution'], 'parent': parent, 'files': files,
        'checkpoint': file_record(checkpoint) if checkpoint.is_file() else None,
        'checkpoint_members': members, 'checkpoint_member_read_error': member_error,
        'missing_result_files': sorted(name for name in probe.REVIEW_INPUTS if not (folder / name).is_file()),
        'original_status': original_record(folder / 'status.json'),
        'original_review': original_record(folder / 'interpretation.json'),
        'original_command': original_record(LOGS / 'solve-segment.json'),
        'snapshot_raw_bytes': 0, 'cumulative_raw_snapshot_bytes': 0,
        'scope': 'Temporary diagnostic backup only. No failed flag, trajectory or source was cleared or repaired.'}
    payload_bytes = sum(value['bytes'] for value in files.values())
    for _ in range(10):
        encoded = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode()
        count = payload_bytes + len(encoded)
        if receipt['snapshot_raw_bytes'] == count:
            break
        receipt['snapshot_raw_bytes'] = count
        receipt['cumulative_raw_snapshot_bytes'] = previous_bytes + count
    else:
        raise ValueError('Diagnostic receipt byte count did not stabilize')
    if receipt['cumulative_raw_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Diagnostic snapshot exceeds the existing raw cache budget')
    path = CACHE / 'receipts' / f'failed-{frame}.json'
    if path.exists():
        raise FileExistsError('Failed evidence is immutable')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    sealed = {'cache_key': receipt['cache_key'], 'receipt_sha256': digest(path), 'frame': frame,
              'receipt_relative_path': path.relative_to(CACHE).as_posix(), 'kind': 'failed-diagnostic'}
    save(LOGS / 'sealed-diagnostic.json', sealed)
    emit(sealed)
    return receipt


def run(context):
    if context['stage'] == 'finalize':
        raise ValueError('Finalization is a separate non-numerical action')
    if not read(LOGS / 'runtime-lock.json').get('verified'):
        raise ValueError('Runtime lock has not been checked')
    if context['previous_frame']:
        verify_restore(context)
        inspect_result(context, frame=context['previous_frame'], start=context['previous_frame'] - 20)
    save(LOGS / 'source-before.json', source_record())
    try:
        command = guarded(context, 'solve-segment', [sys.executable, __file__, 'worker'], 265, 40)
        folder = result_folder(CACHE, context['request'])
        options = ['--allow-segment'] if context['end_frame'] < 60 else []
        guarded(context, 'review-segment', [sys.executable, ROOT / 'tools/review_woven_large.py', folder, *options], 15, 25)
        acceptance = inspect_result(context)
        save(LOGS / 'segment-acceptance.json', acceptance)
        seal(context, acceptance, command)
    finally:
        after = source_record()
        save(LOGS / 'source-after.json', {'unchanged': after == context['source'], 'source': after})
        if after != context['source']:
            raise ValueError('Source changed during solve/review')


def confirm_cache(context):
    if context['stage'] == 'finalize':
        raise ValueError('Finalization must not save or overwrite the f60 cache')
    diagnostic = LOGS / 'sealed-diagnostic.json'
    sealed = read(diagnostic if diagnostic.exists() else LOGS / 'sealed-receipt.json')
    if os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCHED_KEY') != sealed['cache_key']:
        raise ValueError('Saved cache exact-key lookup failed; continuation is not admitted')
    if diagnostic.exists():
        path = CACHE / sealed['receipt_relative_path']
        receipt = read(path)
        if (digest(path) != sealed['receipt_sha256'] or receipt['kind'] != 'failed-diagnostic'
                or receipt['never_resumable'] is not True
                or inventory(CACHE, context['request'], sealed['receipt_relative_path'], complete=False) != receipt['files']):
            raise ValueError('Diagnostic cache changed after sealing')
        save(LOGS / 'cache-confirmation.json', {'verified': True, **sealed, 'numerically_accepted': False,
            'never_resumable': True, 'scope': 'Exact diagnostic key is present; this cache is forbidden as continuation input.'})
        return
    checked = {**context, 'previous_frame': context['end_frame'], 'parent_key': context['cache_key'],
               'parent_receipt_sha256': sealed['receipt_sha256']}
    verify_restore(checked)
    save(LOGS / 'cache-confirmation.json', {'verified': True, **sealed,
        'scope': 'Exact server key is present and local sealed bytes were checked; the next job must verify downloaded bytes again.'})


def install(context):
    if context['stage'] != 'finalize':
        raise ValueError('Rendering dependencies are only required by full-result finalization')
    engine.shared.LOGS = LOGS
    guarded(context, 'render-dependencies', [sys.executable, ROOT / 'tools/install_runner_dependencies.py', '--log-dir', LOGS], 30, 60)
    # The official installer is already bounded internally and records its exact download.
    guarded(context, 'paraview-install', [sys.executable, __file__, 'install-paraview'], 20, 45)


def integrated_tool_declaration(summary):
    spheres = summary.get('definition', {}).get('rigid_spheres', [])
    if (not isinstance(spheres, list) or len(spheres) != 1
            or spheres[0].get('motion_kind') != 'integrated-saved-states'):
        raise ValueError('A single actual integrated tool ledger is required')
    return spheres[0]


def verify_media(folder, context, numeric_review):
    import numpy as np
    from PIL import Image
    import xml.etree.ElementTree as ET
    metadata, render, video = (read(folder / name) for name in ('vtk_metadata.json', 'render.json', 'video.json'))
    if not (metadata.get('frames') == render.get('frames_rendered') == video.get('frames') == 61
            and metadata.get('particles') == 819200 and render.get('renderer') == 'ParaView'
            and render.get('field') == 'family_id' and render.get('geometry_scale') == 1
            and render.get('axes') is False and render.get('annotations') is False
            and render.get('animation') == 'simulation.gif' and video.get('render_sha256') == digest(folder / 'render.json')
            and video.get('sha256') == digest(folder / 'simulation.mp4')
            and render.get('render_overrides', {}).get('matrix_opacity') == .008
            and render.get('render_overrides', {}).get('sphere_opacity') == .45):
        raise ValueError('Full original-frame media metadata differs')
    with np.load(folder / 'trajectory.npz', allow_pickle=False) as archive:
        times = archive['time']
    pvd = [float(value.attrib['timestep']) for value in ET.parse(folder / 'particles.pvd').getroot().findall('.//DataSet')]
    np.testing.assert_array_equal(times, pvd)
    files = {}
    for index in range(61):
        path = folder / 'frames' / f'frame_{index:05d}.png'
        with Image.open(path) as image:
            if image.size != (1200, 1200):
                raise ValueError('Wrong ParaView frame resolution')
            image.verify()
        files[path.name] = file_record(path)
    if context['request']['loading'] == 'sphere-impact':
        import importlib.util
        spec = importlib.util.spec_from_file_location('saved_tool_motion_check', ROOT / 'mpm_jax/saved_tool_motion.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.integrated_sphere_states(folder, integrated_tool_declaration(read(folder / 'summary.json')), times)
    return {'passed': True, 'media_created': True, 'production_eligible': False,
        'original_png_frames': 61, 'png_files': files, 'gif': engine.gif_timing(folder / 'simulation.gif', 61),
        'render_sha256': digest(folder / 'render.json'), 'video_sha256': digest(folder / 'simulation.mp4'),
        'pvd_times_match_exactly': True, 'numeric_review_sha256': numeric_review,
        'source': context['source'], 'geometry_scale': 1, 'axes': False, 'annotations': False}


def finalize(context):
    if context['stage'] != 'finalize':
        raise ValueError('Only a separately restored full f60 cache can be finalized')
    verify_restore(context)
    if environment() != read(CACHE / 'runtime/environment.json') or source_record() != context['source']:
        raise ValueError('Finalization source/runtime differs')
    original = result_folder(CACHE, context['request'])
    inspect_result(context, frame=60, start=40)
    save(LOGS / 'final-disk-admission.json', disk_admission('finalize', shutil.disk_usage(ROOT).free))
    if FINAL.exists():
        raise FileExistsError('Finalization never overwrites existing reviewed media')
    folder = result_folder(FINAL, context['request'])
    shutil.copytree(original, folder)
    before = {name: file_record(folder / name) for name in probe.REVIEW_INPUTS}
    guarded(context, 'review-full', [sys.executable, ROOT / 'tools/review_woven_large.py', folder], 15, 35)
    acceptance = inspect_result(context, folder=folder, frame=60, start=40)
    guarded(context, 'paraview-full', [sys.executable, '-m', 'mpm_jax.visualization', folder, '--gif',
        '--field', 'family_id', '--resolution', '1200', '--matrix-opacity', '.008', '--sphere-opacity', '.45'], 35, 20)
    guarded(context, 'video-full', [sys.executable, ROOT / 'tools/encode_video.py', folder], 10, 15)
    checked = guarded(context, 'vtk-check-full', [os.environ['PARAVIEW_PYTHON'], '--force-offscreen-rendering',
        ROOT / 'tools/check_vtk.py', folder], 15, 10)
    if not checked['completed']:
        raise ValueError('Actual VTK readback did not finish')
    media = verify_media(folder, context, digest(folder / 'interpretation.json'))
    if before != {name: file_record(folder / name) for name in probe.REVIEW_INPUTS} or source_record() != context['source']:
        raise ValueError('Numeric evidence or source changed during finalization')
    save(folder / 'vtk_check.json', {'passed': True, 'sampled_frames': [0, 60], 'particles_per_frame': 819200,
        'checker_sha256': digest(ROOT / 'tools/check_vtk.py'), 'rtol': 1e-6, 'atol': 1e-7,
        'input_sha256': before, 'vtk_metadata_sha256': digest(folder / 'vtk_metadata.json'),
        'scope': 'Actual ParaView VTK readback of first/last coordinates, scalars and integer labels.'})
    save(folder / 'media-check.json', media)
    save(LOGS / 'final-acceptance.json', {**acceptance, 'media_created': True,
        'media_check_sha256': digest(folder / 'media-check.json'), 'raw_cache_key': context['parent_key'],
        'raw_cache_receipt_sha256': context['parent_receipt_sha256'],
        'permanent_raw_backup_completed': False, 'public_display_approved': False})
    verify_restore(context)


def package(context, *, delivery=None, limit=PACKAGE_BUDGET):
    delivery = DELIVERY if delivery is None else Path(delivery)
    if delivery.exists():
        raise FileExistsError('Delivery is immutable')
    confirmation = LOGS / 'cache-confirmation.json'
    compact = context['stage'] != 'finalize' and confirmation.exists() and read(confirmation).get('verified') is True
    limit = min(limit, 8 * 1024**2) if compact else limit
    reserve = 1024**2
    if limit <= reserve:
        raise ValueError('A complete manifest requires a reserved budget')
    delivery.mkdir(parents=True)
    candidates = [(path, Path('records') / path.name) for path in sorted(LOGS.glob('*'))
                  if path.is_file() and path.suffix in ('.json', '.log')]
    base = FINAL if context['stage'] == 'finalize' and result_folder(FINAL, context['request']).exists() else CACHE
    folder = result_folder(base, context['request'])
    order = ('plan.json', 'summary.json', 'status.json', 'interpretation.json', 'observables.csv', 'diagnostics.csv',
             'layout.json', 'memory.json', 'numerical-source.json', 'media-check.json', 'render.json', 'video.json',
             'vtk_metadata.json', 'vtk_check.json', 'particles.pvd', 'simulation.gif', 'simulation.mp4',
             'frames/frame_00000.png', 'frames/frame_00030.png', 'frames/frame_00060.png',
             'checkpoint.npz', 'final_state_0.npz', 'reference.npz', 'layout_arrays.npz', 'particle_fields.npz',
             'trajectory.npz', 'trajectory_x.npy', 'trajectory_speed.npy', 'trajectory_damage.npy', 'trajectory_temperature.npy')
    candidates += [(path, Path('result') / name) for name in order if (path := folder / name).is_file()]
    candidates += [(path, Path('result') / path.name) for path in sorted(folder.glob('*.tmp'))
                   if path.is_file() and path.name.removesuffix('.tmp') in RESULT_NAMES]
    candidates += [(path, Path('receipts') / path.name) for path in sorted((CACHE / 'receipts').glob('*.json'))]
    copied, omitted, used = {}, {}, 0
    for source, relative in candidates:
        if source.is_symlink():
            raise ValueError('Evidence cannot be a symlink')
        record = file_record(source)
        if used + record['bytes'] > limit - reserve or compact and source.suffix not in ('.json', '.csv', '.log'):
            omitted[relative.as_posix()] = {**record, 'reason': 'Whole-file budget; raw data remain only in the declared temporary cache when verified.'}
            continue
        target = delivery / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_record(target) != record or file_record(source) != record:
            raise ValueError('Evidence changed while copying')
        copied[relative.as_posix()] = record
        used += record['bytes']
    value = {'schema': 1, 'request': context['request'], 'source': context['source'], 'execution': context['execution'],
        'stage': context['stage'], 'files': copied, 'omitted': omitted, 'budget_bytes': limit, 'copied_bytes': used,
        'cache_verified': compact,
        'cache_key': read(confirmation)['cache_key'] if compact else context['cache_key'], 'production_eligible': False,
        'media_created': (LOGS / 'final-acceptance.json').exists(),
        'scope': 'Complete listed files only. Omitted mmap/VTK frames are not a permanent raw backup. No automatic public display.'}
    encoded = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(encoded) > reserve or used + len(encoded) > limit:
        raise ValueError('Complete manifest exceeds its reserved budget')
    (delivery / 'manifest.json').write_bytes(encoded)
    if sum(path.stat().st_size for path in delivery.rglob('*') if path.is_file()) > limit:
        raise ValueError('Actual complete package exceeds its declared budget')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'lock', 'worker', 'run', 'confirm-cache', 'seal-failure',
                                          'install', 'install-paraview', 'finalize', 'package'))
    parser.add_argument('--stage', choices=('20', '40', '60', 'finalize'), default='20')
    parser.add_argument('--architecture', choices=probe.ARCHITECTURES, default='plain')
    parser.add_argument('--loading', choices=probe.LOADS, default='sphere-impact')
    parser.add_argument('--parent-run', default='')
    parser.add_argument('--parent-attempt', default='')
    parser.add_argument('--parent-receipt-sha', default='')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
        return
    context = read(CONTEXT)
    engine.LOGS = LOGS
    if args.action == 'install-paraview':
        engine.shared.LOGS = LOGS
        engine.shared.install()
    else:
        {'verify': verify_restore, 'lock': lock_environment, 'worker': worker, 'run': run,
         'seal-failure': seal_failure,
         'confirm-cache': confirm_cache, 'install': install, 'finalize': finalize, 'package': package}[args.action](context)


if __name__ == '__main__':
    main()
