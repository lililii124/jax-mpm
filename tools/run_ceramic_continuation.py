"""Continue an exact ceramic prefix with the immutable original numerical checkout."""

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / '.local/ceramic-frozen-solver'
CACHE = ROOT / '.local/ceramic-prefix-cache'
LOGS = ROOT / '.local/ceramic-continuation-records'
CONTEXT = LOGS / 'job.json'
DELIVERY = ROOT / '.local/ceramic-continuation-delivery'
PRODUCER = '68c311813b11124459b7750f52695c7caa9b6c39'
REPOSITORY = 'lililii124/jax-mpm'
FEATURE_REF = 'refs/heads/work/ceramic-fragmentation-prefix'
FIRST_KEY = f'ceramic-prefix-v1-{PRODUCER}-40065893663-1-f10'
LEGACY_CONTROLLER_FILES = ('tools/run_ceramic_continuation.py', 'tools/recover_ceramic_review.py', '.github/workflows/engineering-previews.yml')
CONTROLLER_FILES = (*LEGACY_CONTROLLER_FILES, 'tools/review_ceramic_spatial_failure.py')
SOURCE_PATHS = ('mpm_jax', 'examples', 'tools', 'pyproject.toml', '.github/workflows/engineering-previews.yml')
RAW_LIMIT = 6_000_000_000
REPO_LIMIT = 9_000_000_000
PACKAGE_LIMIT = 192 * 1024**2


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    path = Path(path)
    before = path.stat()
    record = {'bytes': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Data changed while hashing')
    return record


def local_file(base, name):
    base = Path(base).resolve()
    if not isinstance(name, str) or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe relative data path')
    path = base/name
    if not path.resolve().is_relative_to(base) or not path.is_file():
        raise ValueError('Data file is missing or outside its root')
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(base)):
        raise ValueError('Symlinks are not numerical evidence')
    return path


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def verify_source(source):
    if (source.get('repository') != REPOSITORY or source.get('commit') != PRODUCER
            or source.get('run_id') != '40065893663' or source.get('attempt') != '1'
            or git(SOLVER, 'rev-parse', 'HEAD').decode().strip() != PRODUCER
            or git(SOLVER, 'rev-parse', 'HEAD^{tree}').decode().strip() != source.get('tree')):
        raise ValueError('Only the original admitted ceramic producer is accepted')
    names = git(SOLVER, 'ls-tree', '-r', '--name-only', PRODUCER, '--', *SOURCE_PATHS).decode().splitlines()
    expected = {name for name in names if name.endswith(('.py', '.toml', '.yml'))}
    if set(source.get('files', {})) != expected:
        raise ValueError('Original source inventory differs from the complete original Git tree')
    for name, record in source['files'].items():
        path = local_file(SOLVER, name)
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        actual_git = git(SOLVER, 'rev-parse', f'HEAD:{name}').decode().strip()
        if (file_record(path) != {'bytes': record['bytes'], 'sha256': record['sha256']}
                or blob != actual_git or blob != record['git_blob']):
            raise ValueError('A numerical source file differs from its frozen producer blob: '+name)
    untracked = git(SOLVER, 'ls-files', '--others', '--exclude-standard', '--', *SOURCE_PATHS).decode().splitlines()
    if any(name.endswith(('.py', '.toml', '.yml')) for name in untracked):
        raise ValueError('Untracked executable source in the numerical checkout')


def controller_record():
    commit = git(ROOT, 'rev-parse', 'HEAD').decode().strip()
    if commit != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF') != FEATURE_REF:
        raise ValueError('The controller must run on the fixed feature ref/cache scope')
    files = {}
    for name in CONTROLLER_FILES:
        raw = local_file(ROOT, name).read_bytes()
        if raw != git(ROOT, 'show', f'{commit}:{name}'):
            raise ValueError('Controller source differs from its committed blob')
        files[name] = hashlib.sha256(raw).hexdigest()
    return {'repository': REPOSITORY, 'commit': commit, 'run_id': os.environ['GITHUB_RUN_ID'],
            'attempt': os.environ['GITHUB_RUN_ATTEMPT'], 'ref': FEATURE_REF, 'files': files,
            'scope': 'Continuation orchestration only; numerical source remains '+PRODUCER}


def verify_controller(record):
    expected_files = (LEGACY_CONTROLLER_FILES if record.get('commit') ==
                      'cc033df244f7349d89b6cc08c74374454f44baa2' else CONTROLLER_FILES)
    if (record.get('repository') != REPOSITORY or record.get('ref') != FEATURE_REF
            or not re.fullmatch('[0-9a-f]{40}', record.get('commit', ''))
            or set(record.get('files', {})) != set(expected_files)):
        raise ValueError('Invalid historical controller source identity')
    for name, expected in record['files'].items():
        if hashlib.sha256(git(ROOT, 'show', f"{record['commit']}:{name}")).hexdigest() != expected:
            raise ValueError('Historical controller differs from its exact committed source')


def inventory(base=CACHE):
    base = Path(base)
    result = {}
    for path in sorted(base.rglob('*')):
        if path.is_symlink():
            raise ValueError('Raw cache cannot contain symlinks')
        if not path.is_file() or path == base/'receipt.json':
            continue
        name = path.relative_to(base).as_posix()
        allowed = (name in ('source.json', 'requirements.txt', 'runtime.json')
            or re.fullmatch(r'receipts/[0-9a-f]{64}\.json', name)
            or re.fullmatch(r'ancestors/[0-9a-f]{64}/(checkpoint\.json|checkpoint_[01]_0\.npz)', name)
            or name.startswith('result/') and path.suffix in ('.json', '.csv', '.npz', '.npy', '.tmp'))
        if not allowed:
            raise ValueError('Unexpected raw cache file: '+name)
        result[name] = file_record(path)
    return result


def dependency_pins(text):
    pins, editable = {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-e '):
            # Never execute the producer's editable location or URL in this job.
            editable.append(line)
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9.!+_-]+)', line)
        if not match:
            raise ValueError('Recorded dependencies must be exact pins; URLs/options are not installed')
        name = re.sub(r'[-_.]+', '-', match[1]).lower()
        if name in pins:
            raise ValueError('Duplicate dependency pin')
        pins[name] = match[2]
    if len(editable) != 1 or not {'jax', 'jaxlib', 'numpy', 'scipy', 'psutil'}.issubset(pins):
        raise ValueError('Incomplete original dependency lock')
    return pins


def verify_cache(context):
    path = CACHE/'receipt.json'
    if digest(path) != context['parent_receipt_sha256']:
        raise ValueError('Original receipt SHA differs from the independently admitted parent')
    receipt = read(path)
    if (receipt.get('cache_key') != context['parent_key'] or receipt.get('never_resumable') is not False
            or receipt.get('production_eligible') is not False):
        raise ValueError('The parent is not an accepted continuation cache')
    if receipt.get('schema') == 'ceramic-resource-prefix-v1':
        if (receipt['cache_key'] != FIRST_KEY or receipt.get('prefix_complete') is not True
                or receipt.get('requested_stop_frame') != 10 or context['from_frame'] != 10
                or receipt.get('requested_stroke_m') != .012 or receipt.get('full_target_s') != 3e-4):
            raise ValueError('Only the original frame-10 full-plan prefix may start this lineage')
    elif receipt.get('schema') == 'ceramic-review-recovery-v1':
        from recover_ceramic_review import verify_recovery
        verify_recovery(receipt, CACHE)
        if context['from_frame'] != 10:
            raise ValueError('Recovered prefix endpoint must be frame 10')
    elif receipt.get('schema') == 'ceramic-continuation-v1':
        if (receipt.get('kind') != 'accepted-prefix' or receipt.get('accepted_frame') != context['from_frame']
                or receipt.get('full_protocol_completed') is not False or receipt.get('prefix_verified') is not True):
            raise ValueError('Parent continuation endpoint is not a complete accepted partial prefix')
    else:
        raise ValueError('Unknown ceramic receipt schema')
    if inventory() != receipt.get('files'):
        raise ValueError('Restored raw cache contains changed, missing or extra bytes')
    source = read(CACHE/'source.json')
    if source != receipt.get('source'):
        raise ValueError('Original source identity changed')
    verify_source(source)
    cursor = receipt
    seen = set()
    while cursor['schema'] == 'ceramic-continuation-v1':
        verify_controller(cursor['controller'])
        if (cursor.get('kind') != 'accepted-prefix' or cursor.get('prefix_verified') is not True
                or cursor.get('never_resumable') is not False or cursor.get('full_protocol_completed') is not False):
            raise ValueError('A historical continuation is not an accepted partial prefix')
        parent = cursor['parent']
        sha = parent['receipt_sha256']
        if not re.fullmatch('[0-9a-f]{64}', sha) or sha in seen:
            raise ValueError('Invalid or cyclic ceramic lineage')
        previous = local_file(CACHE, f'receipts/{sha}.json')
        if digest(previous) != sha:
            raise ValueError('Historical receipt bytes differ')
        old = read(previous)
        old_frame = old.get('accepted_frame', old.get('requested_stop_frame'))
        if (old['source'] != source or old['cache_key'] != parent['cache_key']
                or old.get('never_resumable') is not False or old_frame != parent['frame']
                or not old_frame < cursor['accepted_frame']):
            raise ValueError('Original lineage source/key/frame differs')
        seen.add(sha)
        cursor = old
    if cursor.get('schema') == 'ceramic-review-recovery-v1':
        from recover_ceramic_review import verify_recovery
        verify_recovery(cursor, CACHE)
    elif cursor.get('schema') != 'ceramic-resource-prefix-v1' or cursor.get('cache_key') != FIRST_KEY or not cursor.get('prefix_complete'):
        raise ValueError('Lineage does not start from an independently accepted original frame-10 prefix')
    pins = dependency_pins((CACHE/'requirements.txt').read_text())
    recorded = read(CACHE/'runtime.json')
    if (recorded.get('python') != '3.12.13' or recorded.get('backend') != 'cpu' or recorded.get('x64') is not True
            or any(recorded[name] != pins[name] for name in ('jax', 'jaxlib', 'numpy'))):
        raise ValueError('Original recorded numerical runtime is inconsistent')
    return receipt, pins


def source_engine():
    sys.path.insert(0, str(SOLVER))
    engine = importlib.import_module('tools.run_engineering_preview')
    if (Path(engine.__file__).resolve() != SOLVER/'tools/run_engineering_preview.py'
            or engine.ROOT.resolve() != SOLVER.resolve()):
        raise ValueError('The resource guard was shadowed by another checkout')
    engine.LOGS = LOGS
    return engine


def verify_runtime(context):
    if platform.python_version() != '3.12.13' or platform.system() != 'Linux':
        raise ValueError('Recorded Python/Linux runtime differs')
    for name, version in context['pins'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Numerical dependency changed: '+name)
    for name, module in list(sys.modules.items()):
        if name in ('tools', 'examples', 'mpm_jax') or name.startswith(('tools.', 'examples.', 'mpm_jax.')):
            location = getattr(module, '__file__', None)
            if location and not Path(location).resolve().is_relative_to(SOLVER):
                raise ValueError('Numerical import escaped the original source checkout: '+name)


def prefix_snapshot(frame):
    folder = CACHE/'result'
    checkpoint = read(folder/'checkpoint.json')
    if checkpoint['frame'] != frame or len(checkpoint['states']) != 1:
        raise ValueError('The checkpoint frame/body count differs')
    return {'frame': frame, 'signature': checkpoint['signature'],
        'raw_frames': {f'frames_data/frame_{i:05d}.npz': file_record(local_file(folder, f'frames_data/frame_{i:05d}.npz')) for i in range(frame+1)},
        'state_files': {name: file_record(local_file(folder, name)) for name in checkpoint['states']},
        'checkpoint_json': file_record(folder/'checkpoint.json'), 'rows': checkpoint['rows'],
        'observations': checkpoint['observations'], 'solver_seconds': checkpoint['solver_seconds']}


def validate_prefix_preserved(before):
    folder = CACHE/'result'
    cp = read(folder/'checkpoint.json')
    if (cp['signature'] != before['signature'] or cp['rows'][:len(before['rows'])] != before['rows']
            or cp['observations'][:len(before['observations'])] != before['observations']
            or cp['solver_seconds'] < before['solver_seconds']
            or any(file_record(local_file(folder, name)) != value for name, value in before['raw_frames'].items())):
        raise ValueError('Continuation changed a complete original raw/observer prefix')


def mass_history_check(folder, rows, total, active, debris, frame):
    def agrees(value, expected):
        return math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=0.)
    if (len(rows) != frame+1 or not math.isfinite(total) or total <= 0
            or any(row.get('body') != 0 or not agrees(row['mass'], total)
                   or not agrees(float(row['active_mass'])+float(row['debris_mass']), total) for row in rows)
            or not agrees(rows[-1]['active_mass'], active) or not agrees(rows[-1]['debris_mass'], debris)):
        raise ValueError('Saved diagnostic mass history differs from the complete checkpoint')
    path = Path(folder)/'diagnostics.csv'
    csv_checked = path.is_file()
    if frame == 60 and not csv_checked:
        raise ValueError('Completed mass history requires diagnostics.csv')
    if csv_checked:
        with path.open(newline='', encoding='utf-8') as stream:
            table = list(csv.DictReader(stream))
        if (len(table) != len(rows) or any(
                int(record['body']) != row['body'] or float(record['time']) != row['time']
                or not agrees(record['mass'], total)
                or any(float(record[key]) != row[key] for key in ('mass', 'active_mass', 'debris_mass'))
                for record, row in zip(table, rows))):
            raise ValueError('Diagnostic CSV mass history differs from the accepted checkpoint prefix')
    return dict(passed=True, checked_states=len(rows), total_mass_kg=total,
                final_active_mass_kg=active, final_debris_mass_kg=debris,
                diagnostics_csv_checked=csv_checked,
                diagnostics_csv_sha256=digest(path) if csv_checked else None,
                relative_roundoff_tolerance=1e-12, absolute_tolerance=0.)


def audit_worker(context, frame):
    source_engine()
    verify_runtime(context)
    from examples.common import _load_state, _valid_checkpoint_states, _validate_resume_prefix, _state_arrays
    import jax
    import jaxlib
    import numpy as np
    actual_runtime = dict(python=platform.python_version(), numpy=np.__version__,
        jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(),
        x64=bool(jax.config.jax_enable_x64))
    if actual_runtime != read(CACHE/'runtime.json'):
        raise ValueError('Loaded numerical runtime differs from the original prefix')
    verify_runtime(context)
    folder = CACHE/'result'
    cp, status, report = (read(folder/name) for name in ('checkpoint.json', 'status.json', 'fragmentation-review.json'))
    checks = report.get('checks')
    if (report.get('passed') is not True or not isinstance(checks, dict) or not checks
            or not all(value is True for value in checks.values()) or report['accepted_saved_frame'] != frame
            or report['target_saved_frame'] != 60 or report['particles'] != 386420
            or any(digest(local_file(folder, name)) != sha for name, sha in report['sources'].items())):
        raise ValueError('Independent original reviewer did not accept this exact prefix')
    if (status['frame'] != frame or status['frames'] != 60 or status['completed'] is not (frame == 60)
            or status['steps_completed'] % frame or cp['frame'] != frame):
        raise ValueError('Actual saved endpoint differs from the unchanged complete plan')
    substeps = status['steps_completed']//frame
    dt = 3e-4/(60*substeps)
    restored = tuple(_load_state(local_file(folder, name)) for name in cp['states'])
    if len(restored) != 1 or not _valid_checkpoint_states(restored):
        raise ValueError('Invalid full mechanical/history checkpoint')
    _validate_resume_prefix(folder, cp, restored, 60, substeps, dt, True)
    leaves = {}
    for name, value in _state_arrays(restored[0]).items():
        value = np.asarray(value)
        array = np.ascontiguousarray(value)
        leaves[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}
    if restored[0].x.shape != (386420, 3):
        raise ValueError('The original particle resolution changed')
    mass = np.asarray(restored[0].mass)
    failed = np.asarray(restored[0].history.failed, dtype=bool)
    mass_check = mass_history_check(folder, cp['rows'], float(mass.sum()),
                                   float(mass[~failed].sum()), float(mass[failed].sum()), frame)
    result = {'prefix_verified': True, 'frame': frame, 'full_protocol_completed': frame == 60,
        'production_eligible': False, 'media_created': False, 'steps': status['steps_completed'],
        'signature': cp['signature'], 'checkpoint_leaves': leaves,
        'runtime': actual_runtime,
        'mass_history': mass_check,
        'original_review_sha256': digest(folder/'fragmentation-review.json'),
        'review_source_sha256': context['source']['files']['tools/review_ceramic_fragmentation.py']['sha256'],
        'scope': 'Original full-state/prefix validation and independent ceramic review; no new calibration or visualization approval.'}
    save(LOGS/f'prefix-{frame:02d}-audit.json', result)


def raw_forecast(context):
    files = inventory()
    frame_files = [record['bytes'] for name, record in files.items() if re.fullmatch(r'result/frames_data/frame_\d{5}\.npz', name)]
    if not frame_files:
        raise ValueError('No complete original frame data are present')
    # mmap files already have all 61 slots. Only compressed frames/CP/final output grow.
    allocated = sum(record['bytes'] for name, record in files.items() if name.endswith('.npy'))
    other = sum(record['bytes'] for name, record in files.items() if not name.endswith('.npy') and '/frames_data/' not in name)
    frame_reserve = 2*max(frame_files)*(context['end_frame']+1)
    final_reserve = allocated if context['end_frame'] == 60 else 0
    estimate = allocated+2*other+frame_reserve+final_reserve+64*1024**2
    if estimate > RAW_LIMIT:
        raise ValueError('Conservative complete raw forecast exceeds the 6 GB per-snapshot budget')
    return {'predicted_raw_bytes': estimate, 'allocated_mmap_bytes': allocated,
            'frame_reserve_bytes': frame_reserve, 'final_archive_reserve_bytes': final_reserve,
            'scope': 'Measured frame sizes with a factor-two growth reserve; final exact bytes are checked again.'}


def cache_admission(context, phase):
    import urllib.request
    token = os.environ.get('GH_TOKEN', '')
    if not token:
        raise ValueError('Read-only cache usage token is required for this step')
    if phase == 'initial':
        added = raw_forecast(context)['predicted_raw_bytes']
    else:
        added = sum(path.stat().st_size for path in CACHE.rglob('*') if path.is_file())
    if added > RAW_LIMIT:
        raise ValueError('Actual complete snapshot exceeds its raw budget')
    query = urllib.request.Request(f'https://api.github.com/repos/{REPOSITORY}/actions/cache/usage',
        headers={'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(query, timeout=30) as response:
        usage = json.load(response)['active_caches_size_in_bytes']
    if type(usage) is not int or usage < 0:
        raise ValueError('Invalid repository cache usage response')
    record = {'phase': phase, 'existing_repository_bytes': usage, 'new_raw_bytes': added,
        'repository_budget_bytes': REPO_LIMIT, 'passed': usage+added+1024**2 <= REPO_LIMIT,
        'scope': 'Read-only usage; no cache deletion or paid quota change.'}
    save(LOGS/f'cache-budget-{phase}.json', record)
    if not record['passed']:
        raise ValueError('Existing caches plus the new whole snapshot exceed the 9 GB repository budget')


def emit(**values):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            for key, value in values.items():
                if '\n' in str(value) or '\r' in str(value):
                    raise ValueError('Invalid workflow output')
                stream.write(f'{key}={value}\n')


def prepare(args):
    if CONTEXT.exists() or not 10 <= args.from_frame < args.end_frame <= 60 or args.end_frame % 10:
        raise ValueError('Use a fresh controller and a later saved endpoint through frame 60')
    if (not re.fullmatch('[0-9a-f]{64}', args.receipt_sha) or args.parent_key.endswith('-failed')
            or os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != args.parent_key):
        raise ValueError('An exact accepted cache hit and admitted receipt SHA are required')
    context = {'parent_key': args.parent_key, 'parent_receipt_sha256': args.receipt_sha,
        'from_frame': args.from_frame, 'end_frame': args.end_frame,
        'started_epoch': float(os.environ['CERAMIC_JOB_STARTED_EPOCH'])}
    if not 0 <= time.time()-context['started_epoch'] < 360*60:
        raise ValueError('Invalid whole-job clock')
    parent, pins = verify_cache(context)
    controller = controller_record()
    context.update(source=parent['source'], pins=pins, controller=controller,
        cache_key=f"ceramic-continuation-v1-{PRODUCER}-{controller['commit']}-{controller['run_id']}-{controller['attempt']}-f{args.end_frame}")
    save(CONTEXT, context)
    shutil.copyfile(CACHE/'receipt.json', LOGS/'parent-receipt.json')
    save(LOGS/'raw-forecast.json', raw_forecast(context))
    (LOGS/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name, version in sorted(pins.items())), encoding='utf-8')
    emit(cache_key=context['cache_key'])


def run(context):
    parent, _ = verify_cache(context)
    engine = source_engine()
    verify_runtime(context)
    os.environ['PYTHONPATH'] = str(SOLVER)
    if controller_record() != context['controller']:
        raise ValueError('Controller source changed')
    folder = CACHE/'result'
    forecast = raw_forecast(context)
    if shutil.disk_usage(ROOT).free < 2*forecast['predicted_raw_bytes']+2*1024**3:
        raise ValueError('Insufficient disk for full raw growth, cache archive and diagnostic reserve')
    def command(name, arguments, minutes, tail=35):
        remaining = context['started_epoch']+340*60-time.time()-tail*60
        seconds = int(min(minutes*60, remaining))
        if seconds < 60:
            raise TimeoutError('Insufficient shared job budget for this operation and the evidence tail')
        return engine.command(name, ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s', *map(str, arguments)], monitor=True)
    command('parent-review', [sys.executable, ROOT/'tools/recover_ceramic_review.py', 'review', '--folder', folder, '--allow-prefix'], 10)
    command('parent-state-audit', [sys.executable, __file__, 'audit', '--audit-frame', context['from_frame']], 10)
    command('parent-spatial-failure-review', [sys.executable, ROOT/'tools/review_ceramic_spatial_failure.py',
        folder, '--output', LOGS/f"spatial-failure-frame-{context['from_frame']}.json"], 5)
    before = prefix_snapshot(context['from_frame'])
    save(LOGS/'original-prefix.json', before)
    save(LOGS/'timing-forecast.json', {
        'completed_solver_seconds': before['solver_seconds'],
        'completed_frames': context['from_frame'],
        'remaining_frames': context['end_frame']-context['from_frame'],
        'linear_remaining_solver_seconds': before['solver_seconds']/context['from_frame']*(context['end_frame']-context['from_frame']),
        'scope': 'Observed average only; later contact and damage can change cost. The bounded walltime guard remains authoritative.'})
    # Preserve the exact referenced old state, not just a checksum of an overwritten alternating slot.
    archive = CACHE/'ancestors'/context['parent_receipt_sha256']
    archive.mkdir(parents=True, exist_ok=False)
    for name, record in {'checkpoint.json': before['checkpoint_json'], **before['state_files']}.items():
        shutil.copyfile(folder/name, archive/name)
        if file_record(archive/name) != record:
            raise ValueError('Original checkpoint archive copy differs')
    receipts = CACHE/'receipts'
    receipts.mkdir(exist_ok=True)
    shutil.copyfile(CACHE/'receipt.json', receipts/(context['parent_receipt_sha256']+'.json'))
    solve = command('ceramic-continuation', [sys.executable, '-m', 'examples.ceramic_fragmentation',
        '--resume', '--stop-after-frame', context['end_frame'], '--output', folder], 265, 45)
    validate_prefix_preserved(before)
    options = ['--allow-prefix'] if context['end_frame'] < 60 else []
    command('independent-ceramic-review', [sys.executable, ROOT/'tools/recover_ceramic_review.py', 'review', '--folder', folder, *options], 12, 30)
    command('accepted-state-audit', [sys.executable, __file__, 'audit', '--audit-frame', context['end_frame']], 12, 20)
    command('accepted-spatial-failure-review', [sys.executable, ROOT/'tools/review_ceramic_spatial_failure.py',
        folder, '--output', LOGS/f"spatial-failure-frame-{context['end_frame']}.json"], 5, 15)
    verify_source(context['source'])
    verify_runtime(context)
    if controller_record() != context['controller']:
        raise ValueError('Controller source changed during advance')
    memory = solve['memory']
    if (not solve['completed'] or memory['stopped_for_memory'] or memory['samples'] < 1
            or memory['peak_tree_uss_bytes'] > 12*1024**3 or memory['minimum_system_available_bytes'] < 2*1024**3):
        raise ValueError('Guarded computation was not admitted')
    save(LOGS/'numerical-admission.json', {'accepted': True, 'memory': memory,
        'prefix_preserved': True, 'full_protocol_completed': context['end_frame'] == 60,
        'production_eligible': False, 'media_created': False})
    seal(context, diagnostic=False)


def seal(context, *, diagnostic):
    verify_source(context['source'])
    if controller_record() != context['controller']:
        raise ValueError('Cannot seal evidence against changed controller bytes')
    if not diagnostic and read(LOGS/'numerical-admission.json').get('accepted') is not True:
        raise ValueError('An independently admitted numerical endpoint is required')
    # Preserve the exact admitted parent even if the new job failed before advancing.
    original = LOGS/'parent-receipt.json'
    if digest(original) != context['parent_receipt_sha256']:
        raise ValueError('Original receipt bytes were not retained')
    prior = CACHE/'receipts'/(context['parent_receipt_sha256']+'.json')
    prior.parent.mkdir(exist_ok=True)
    if prior.exists():
        if digest(prior) != context['parent_receipt_sha256']:
            raise ValueError('An existing historical receipt changed')
    else:
        shutil.copyfile(original, prior)
    files = inventory()
    key = context['cache_key']+('-failed' if diagnostic else '')
    receipt = {'schema': 'ceramic-continuation-v1', 'kind': 'failed-diagnostic' if diagnostic else 'accepted-prefix',
        'never_resumable': diagnostic, 'prefix_verified': not diagnostic,
        'full_protocol_completed': not diagnostic and context['end_frame'] == 60,
        'accepted_frame': None if diagnostic else context['end_frame'], 'requested_end_frame': context['end_frame'],
        'source': context['source'], 'controller': context['controller'], 'cache_key': key, 'files': files,
        'parent': {'cache_key': context['parent_key'], 'receipt_sha256': context['parent_receipt_sha256'], 'frame': context['from_frame']},
        'requested_stroke_m': .012, 'full_target_s': 3e-4, 'production_eligible': False,
        'media_created': False, 'public_visual_review_passed': False, 'permanent_raw_backup_verified': False,
        'audit': read(LOGS/f"prefix-{context['end_frame']:02d}-audit.json") if not diagnostic else None,
        'scope': 'Complete original raw cache and saved prefix; failures remain diagnostic only. Full independent display review, permanent backup and ParaView are separate.'}
    encoded = (json.dumps(receipt, indent=2, allow_nan=False)+'\n').encode()
    if sum(entry['bytes'] for entry in files.values())+len(encoded) > RAW_LIMIT:
        raise ValueError('Complete new raw snapshot exceeds the 6 GB bound')
    (CACHE/'receipt.json').write_bytes(encoded)
    save(LOGS/'new-receipt.json', receipt)
    context.update(sealed_key=key, sealed_receipt_sha256=digest(CACHE/'receipt.json'))
    save(CONTEXT, context)
    emit(cache_key=key, receipt_sha=context['sealed_receipt_sha256'])


def confirm(context):
    receipt = read(CACHE/'receipt.json')
    if (os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != context['sealed_key']
            or digest(CACHE/'receipt.json') != context['sealed_receipt_sha256']
            or inventory() != receipt['files']):
        raise ValueError('Exact new cache lookup or sealed data differs')
    save(LOGS/'cache-confirmation.json', {'exact_lookup': True, 'key': context['sealed_key'],
        'receipt_sha256': context['sealed_receipt_sha256'], 'never_resumable': receipt['never_resumable'],
        'permanent_backup': False, 'production_eligible': False})


def package():
    if DELIVERY.exists():
        raise FileExistsError('Evidence delivery already exists')
    DELIVERY.mkdir(parents=True)
    candidates = [(path, Path('records')/path.name) for path in sorted(LOGS.glob('*')) if path.is_file()]
    folder = CACHE/'result'
    current_states = read(folder/'checkpoint.json').get('states', []) if (folder/'checkpoint.json').is_file() else []
    names = ['summary.json', 'status.json', 'fragmentation-review.json', 'segment.json', 'checkpoint.json',
             'observables.csv', 'diagnostics.csv', *current_states, 'final_state_0.npz']
    candidates += [(folder/name, Path('result')/name) for name in names if (folder/name).is_file()]
    selected = {path for path, _ in candidates}
    candidates += [(path, Path('raw')/path.relative_to(CACHE)) for path in sorted(CACHE.rglob('*')) if path.is_file() and path not in selected]
    files, omitted, used = {}, {}, 0
    for path, relative in candidates:
        if path.is_symlink():
            raise ValueError('Evidence cannot contain symlinks')
        record = file_record(path)
        if used+record['bytes'] > PACKAGE_LIMIT-1024**2:
            omitted[relative.as_posix()] = {**record, 'reason': 'Whole-file budget; full raw requires the independently confirmed temporary cache.'}
            continue
        target = DELIVERY/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_record(target) != record or file_record(path) != record:
            raise ValueError('Diagnostic bytes changed during copying')
        files[relative.as_posix()] = record
        used += record['bytes']
    value = {'files': files, 'omitted': omitted, 'bytes': used, 'budget_bytes': PACKAGE_LIMIT,
        'production_eligible': False, 'media_created': False, 'scope': 'Listed complete files only; no implicit reconstruction of omitted raw data.'}
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    if len(raw) > 1024**2 or used+len(raw) > PACKAGE_LIMIT:
        raise ValueError('Complete manifest exceeds the reserved budget')
    (DELIVERY/'manifest.json').write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'audit', 'seal-failure', 'cache-admission', 'confirm-cache', 'package'))
    parser.add_argument('--parent-key', default='')
    parser.add_argument('--receipt-sha', default='')
    parser.add_argument('--from-frame', type=int, default=10)
    parser.add_argument('--end-frame', type=int, default=60)
    parser.add_argument('--audit-frame', type=int)
    parser.add_argument('--cache-phase', choices=('initial', 'before-save'), default='before-save')
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux':
        parser.error('This controller is restricted to the isolated Linux Actions runner')
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == 'prepare': prepare(args)
        elif args.action == 'package': package()
        else:
            context = read(CONTEXT)
            if args.action == 'run': run(context)
            elif args.action == 'audit': audit_worker(context, args.audit_frame)
            elif args.action == 'seal-failure': seal(context, diagnostic=True)
            elif args.action == 'cache-admission': cache_admission(context, args.cache_phase)
            elif args.action == 'confirm-cache': confirm(context)
    except BaseException as error:
        save(LOGS/f'{args.action}-failure.json', {'action': args.action, 'error_type': type(error).__name__,
            'error': str(error), 'traceback': traceback.format_exc(), 'production_eligible': False})
        raise


if __name__ == '__main__':
    main()
