"""Continue an admitted JC refinement prefix using its original solver checkout."""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / '.local/jc-frozen-solver'
CACHE = ROOT / '.local/jc-refinement-cache'
PARENT = ROOT / '.local/jc-continuation-parent'
OUTPUT = ROOT / 'outputs/jc-continuation/grid100-ppc3/jc-necking/johnson-cook'
LOGS = ROOT / '.local/jc-continuation-ci'
DELIVERY = ROOT / '.local/jc-continuation-delivery'
CONTEXT = LOGS / 'context.json'
SOLVER_COMMIT = '08d83a051a9764a5a218fd33e184f3291eb3dd1c'
REPOSITORY = 'lililii124/jax-mpm'
SOURCE_PATHS = ('mpm_jax', 'examples', 'tools', '.github/workflows', 'pyproject.toml')
CONTROLLER_FILES = ('tools/run_jc_continuation.py', 'tests/test_jc_continuation.py',
                    '.github/workflows/jc-continuation.yml', 'docs/JC_CONTINUATION.md')
PARTICLES, FRAMES, SUBSTEPS = 341584, 60, 965
DURATION = 150e-6
CACHE_BUDGET = 1_000_000_000
FRAME_FIELDS = {'x', 'speed', 'damage', 'temperature', 'time', 'contact_impulse', 'failed'}
PINNED_PACKAGES = {'jax', 'jaxlib', 'numpy', 'scipy', 'psutil'}
READ_BYTES = 4 * 1024**2
PART_BYTES = 80 * 1024**2
TOTAL_BYTES = 192 * 1024**2


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b''):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': digest(path)}


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def strict_review(report):
    checks = report.get('checks', {})
    return report.get('passed') is True and bool(checks) and all(v is True for v in checks.values())


def contained_file(folder, name):
    root = Path(folder).resolve()
    if not isinstance(name, str) or '\\' in name or Path(name).is_absolute():
        raise ValueError('Unsafe relative payload path')
    candidate = root/name
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root or not resolved.is_file():
        raise ValueError('Payload path escaped or is missing')
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p.is_relative_to(root)):
        raise ValueError('Payload symlinks are not admitted')
    return resolved


def inputs_match(folder, records):
    if not records:
        return False
    return all(digest(contained_file(folder, name)) == value for name, value in records.items())


def solver_files(root):
    entries = git(root, 'ls-tree', '-r', 'HEAD', '--', *SOURCE_PATHS).decode().splitlines()
    return {name: header.split()[2] for header, name in (line.split('\t', 1) for line in entries)
            if Path(name).suffix in ('.py', '.toml', '.yml', '.yaml')}


def verify_solver_source(source, root=SOLVER):
    """Original producer identity stays intact; a new run is a separate controller."""
    if (source.get('commit') != SOLVER_COMMIT or source.get('repository') != REPOSITORY
            or source.get('source_kind') != 'github-actions'):
        raise ValueError('Only the fixed original refinement source is accepted')
    if git(root, 'rev-parse', 'HEAD').decode().strip() != SOLVER_COMMIT:
        raise ValueError('Frozen solver checkout is not the original commit')
    if git(root, 'rev-parse', 'HEAD^{tree}').decode().strip() != source.get('tree'):
        raise ValueError('Original solver tree differs')
    if git(root, 'diff', 'HEAD', '--', *SOURCE_PATHS).strip():
        raise ValueError('Frozen solver working tree was changed')
    blobs = solver_files(root)
    if blobs != source.get('git_blob_sha') or set(blobs) != set(source.get('files', {})):
        raise ValueError('Original source inventory differs from the fixed Git tree')
    actual = {name: digest(contained_file(root, name)) for name in blobs}
    normalized = {name: hashlib.sha256((Path(root)/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in blobs}
    if (actual != source['files'] or normalized != source.get('lf_sha256')
            or hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != source.get('fingerprint')
            or source.get('runner_source_sha256') != actual['tools/run_jc_refinement.py']):
        raise ValueError('Frozen solver raw bytes/fingerprint differ from the executed first segment')
    untracked = git(root, 'ls-files', '--others', '--exclude-standard', '--', *SOURCE_PATHS).decode().splitlines()
    if any(Path(name).suffix in ('.py', '.toml', '.yml', '.yaml') for name in untracked):
        raise ValueError('Untracked executable source in frozen solver checkout')


def controller_record():
    commit = git(ROOT, 'rev-parse', 'HEAD').decode().strip()
    if commit != os.environ.get('GITHUB_SHA') or git(ROOT, 'diff', 'HEAD', '--', *CONTROLLER_FILES).strip():
        raise ValueError('Controller is not the clean dispatched commit')
    return {'repository': os.environ['GITHUB_REPOSITORY'], 'commit': commit,
        'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
        'files': {name: digest(ROOT/name) for name in CONTROLLER_FILES},
        'scope': 'Orchestration and independent prefix review only; numerical imports use the separately verified original solver commit.'}


def verify_controller_history(record):
    if (record.get('repository') != REPOSITORY or not re.fullmatch('[a-f0-9]{40}', record.get('commit', ''))
            or set(record.get('files', {})) != set(CONTROLLER_FILES)):
        raise ValueError('Unknown historical continuation controller')
    for name, expected in record['files'].items():
        actual = hashlib.sha256(git(ROOT, 'show', f"{record['commit']}:{name}")).hexdigest()
        if actual != expected:
            raise ValueError('Historical controller is not bound to its Git source')


def request_values(parent_key, receipt_sha, from_frame, end_frame):
    if not re.fullmatch(r'jc-(?:refinement|continuation)-v1-[A-Za-z0-9-]+-frame\d+', parent_key):
        raise ValueError('Unrecognized exact carry cache key')
    if not re.fullmatch('[a-f0-9]{64}', receipt_sha):
        raise ValueError('Parent receipt SHA256 is required')
    if type(from_frame) is not int or type(end_frame) is not int or from_frame not in (10, 20, 30, 40) or end_frame != from_frame+10:
        raise ValueError('This controller advances exactly ten frames, from 10/20/30/40 to 20/30/40/50; frame 60 needs a full-result audit')
    return {'parent_key': parent_key, 'parent_receipt_sha256': receipt_sha, 'from_frame': from_frame,
        'end_frame': end_frame, 'full_frames': FRAMES, 'full_steps': FRAMES*SUBSTEPS,
        'full_duration_s': DURATION, 'increment_steps': 10*SUBSTEPS,
        'completed': False, 'full_protocol_completed': False, 'production_eligible': False}


def next_key(controller, frame):
    for name in ('commit', 'run_id', 'run_attempt'):
        if not re.fullmatch('[A-Za-z0-9]+', str(controller[name])):
            raise ValueError('Unsafe continuation cache identity')
    return 'jc-continuation-v1-'+'-'.join(str(controller[k]) for k in ('commit', 'run_id', 'run_attempt'))+f'-frame{frame}'


def emit(**values):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            for key, value in values.items():
                if '\n' in str(value):
                    raise ValueError('Multiline workflow output')
                stream.write(f'{key}={value}\n')


def environment():
    return {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(),
        'packages': {name: importlib.metadata.version(name) for name in sorted(PINNED_PACKAGES)},
        'cpu': Path('/proc/cpuinfo').read_text().split('\n\n')[0] if Path('/proc/cpuinfo').is_file() else platform.processor()}


def environment_matches(original, current):
    return (set(original.get('packages', {})) == PINNED_PACKAGES
        and all(original.get(key) == current.get(key) for key in ('python', 'system', 'machine', 'packages')))


def environment_lock():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', distribution.metadata['Name']).lower()
        if name in ('pip', 'setuptools', 'wheel', 'jax-mpm'):
            continue
        version = distribution.version
        if not re.fullmatch('[a-z0-9-]+', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Unsafe installed dependency identity')
        if name in packages and packages[name] != version:
            raise ValueError('Ambiguous installed package versions')
        packages[name] = version
    return {'python': platform.python_version(), 'system': platform.system(), 'machine': platform.machine(),
        'packages': dict(sorted(packages.items())),
        'scope': 'Installed environment first completely recorded by the continuation controller; not a retrospective lock for the initial 25 microseconds.'}


def verify_inventory(cache, receipt_sha):
    cache = Path(cache)
    if digest(cache/'carry-receipt.json') != receipt_sha:
        raise ValueError('Restored carry receipt SHA differs from the supplied parent receipt')
    receipt = read_json(cache/'carry-receipt.json')
    inventory = receipt['files']
    actual = {p.relative_to(cache).as_posix() for p in cache.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in cache.rglob('*')) or actual != set(inventory) | {'carry-receipt.json'}:
        raise ValueError('Carry has missing/extra files or symlinks')
    for name, record in inventory.items():
        if file_record(contained_file(cache, name)) != record:
            raise ValueError(f'Carry bytes differ: {name}')
    total = sum(v['bytes'] for v in inventory.values())
    if total != receipt['payload_bytes'] or total+(cache/'carry-receipt.json').stat().st_size > CACHE_BUDGET:
        raise ValueError('Carry size accounting or 1 GB budget failed')
    return receipt


def verify_carry(cache, request, *, check_source=True):
    receipt = verify_inventory(cache, request['parent_receipt_sha256'])
    source = receipt['source']
    folder = Path(cache)/'result'
    plan, review, cp = [read_json(folder/name) for name in ('plan.json', 'prefix-review.json', 'checkpoint.json')]
    if len(cp['states']) != 1 or not re.fullmatch(r'checkpoint_[01]_0\.npz', cp['states'][0]):
        raise ValueError('Carry must contain the sole latest original State checkpoint')
    base = {'plan.json', 'numerical-source.json', 'environment.json', 'checkpoint.json', 'status.json',
        'summary.json', 'segment.json', 'prefix-review.json', 'prefix-history.csv',
        'diagnostics-prefix.csv', 'observables-prefix.csv', *cp['states']}
    expected_result = {'result/'+name for name in base}
    expected_result |= {f'result/frames_data/frame_{i:05d}.npz' for i in range(request['from_frame']+1)}
    actual_result = {name for name in receipt['files'] if name.startswith('result/')}
    lineage = set(receipt['files'])-actual_result
    if expected_result != actual_result or any(not re.fullmatch(
        r'lineage/frame-\d{2}-(?:receipt|parent-prefix|worker-runtime|loaded-numerics|jc-continuation-solve|environment-lock|prefix-review)\.json', name)
        for name in lineage):
        raise ValueError('Carry includes unknown files or lacks its complete saved prefix')
    if not (receipt['cache_key'] == request['parent_key'] and receipt['frame'] == request['from_frame']
        and receipt['completed'] is False and receipt['segment_complete'] is True
        and receipt['production_eligible'] is False and receipt['full_frames'] == FRAMES
        and receipt['full_target_steps'] == FRAMES*SUBSTEPS and receipt['source_fingerprint'] == source['fingerprint']
        and source == plan['source'] == read_json(folder/'numerical-source.json')
        and receipt['common_signature'] == plan['common_signature'] == cp['signature']
        and cp['frame'] == request['from_frame'] and strict_review(review)
        and review['frame'] == request['from_frame'] and review['producer_source'] == source
        and inputs_match(folder, review['input_sha256'])
        and digest(folder/'prefix-history.csv') == review['prefix_history_sha256']):
        raise ValueError('Carry identity, admitted review, common signature or frame differs')
    if receipt['schema'] == 'jc-refinement-carry-v1':
        if not (request['from_frame'] == 10 and review['review_source_sha256'] == source['files']['tools/run_jc_refinement.py']
                and receipt.get('cross_job_resume_verified') is False and not lineage):
            raise ValueError('Unknown original start-only carry')
    elif receipt['schema'] == 'jc-continuation-carry-v1':
        controller = receipt['controller']
        if check_source:
            verify_controller_history(controller)
        if not (review['review_source_sha256'] == controller['files']['tools/run_jc_continuation.py']
                and review.get('controller_source') == controller
                and receipt['prefix_preserved'] is True
                and receipt['environment_lock_file'] == f"lineage/frame-{receipt['frame']:02d}-environment-lock.json"):
            raise ValueError('Continuation reviewer/controller chain differs')
        prior = contained_file(cache, receipt['parent_receipt_file'])
        if digest(prior) != receipt['parent_receipt_sha256']:
            raise ValueError('Continuation parent receipt link is broken')
        ancestor = read_json(prior)
        if not (ancestor['frame']+10 == receipt['frame'] and ancestor['source'] == source
                and ancestor['common_signature'] == receipt['common_signature']):
            raise ValueError('Continuation parent is not the preceding physical prefix')
        verify_lineage(cache, receipt, check_source=check_source)
    else:
        raise ValueError('Unknown carry schema')
    if check_source:
        verify_solver_source(source)
    return receipt


def verify_lineage(cache, receipt, *, check_source):
    current = receipt
    seen = set()
    while current['schema'] == 'jc-continuation-carry-v1':
        frame = current['frame']
        if frame in seen or frame not in (20, 30, 40, 50):
            raise ValueError('Cyclic or unknown continuation lineage')
        seen.add(frame)
        if check_source:
            verify_controller_history(current['controller'])
        expected = f'lineage/frame-{frame-10:02d}-receipt.json'
        if current['parent_receipt_file'] != expected:
            raise ValueError('Parent receipt is not stored under its exact preceding frame')
        path = contained_file(cache, expected)
        if digest(path) != current['parent_receipt_sha256']:
            raise ValueError('Historical receipt SHA chain changed')
        parent = read_json(path)
        if not (parent['frame'] == frame-10 and parent['source'] == receipt['source']
                and parent['common_signature'] == receipt['common_signature']
                and parent['cache_key'] == current['parent_cache_key']):
            raise ValueError('Historical receipt identity changed')
        review_path = contained_file(cache, f'lineage/frame-{frame-10:02d}-prefix-review.json')
        if digest(review_path) != parent['files']['result/prefix-review.json']['sha256']:
            raise ValueError('Historical admitted review differs from its original receipt')
        if not strict_review(read_json(review_path)):
            raise ValueError('Historical prefix was not admitted')
        current = parent
    if current['schema'] != 'jc-refinement-carry-v1' or current['frame'] != 10:
        raise ValueError('Continuation chain does not terminate at the original frame-10 carry')


def prepare(args):
    if PARENT.exists() or OUTPUT.exists() or CONTEXT.exists():
        raise ValueError('Continuation requires fresh controller work directories')
    request = request_values(args.parent_key, args.receipt_sha, args.from_frame, args.end_frame)
    if os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != request['parent_key']:
        raise ValueError('Exact parent cache hit required; never restart the initial segment on a miss')
    receipt = verify_carry(CACHE, request)
    original_env = read_json(CACHE/'result/environment.json')
    if (original_env['python'] != '3.12.13' or original_env['system'] != 'Linux'
            or original_env['machine'] != platform.machine() or set(original_env['packages']) != PINNED_PACKAGES):
        raise ValueError('Original runtime cannot be reconstructed by this workflow')
    lock = read_json(contained_file(CACHE, receipt['environment_lock_file'])) if receipt.get('environment_lock_file') else None
    packages = lock['packages'] if lock is not None else original_env['packages']
    if lock is not None and (not PINNED_PACKAGES <= packages.keys()
        or any(lock[key] != original_env[key] for key in ('python', 'system', 'machine'))
        or any(packages[name] != original_env['packages'][name] for name in PINNED_PACKAGES)):
        raise ValueError('Continuation environment lock conflicts with the original recorded runtime')
    pins = []
    for name, version in sorted(packages.items()):
        if not re.fullmatch('[a-z0-9-]+', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Unsafe exact dependency version')
        pins.append(f'{name}=={version}')
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS/'requirements.txt').write_text('\n'.join(pins)+'\n', encoding='utf-8')
    controller = controller_record()
    started = float(os.environ['JC_JOB_STARTED_EPOCH'])
    if not math.isfinite(started) or not 0 <= time.time()-started < 360*60:
        raise ValueError('Missing or invalid whole-job wall-clock budget')
    context = {'request': request, 'source': receipt['source'], 'controller': controller, 'parent_environment_lock': lock,
        'cache_key': next_key(controller, request['end_frame']), 'started_epoch': started,
        'runtime_scope': 'Exact recorded Python/system/machine and five numerical package versions; CPU is recorded, not required equal. Unrecorded first-job transitive dependencies cannot be retroactively certified.'}
    save(CONTEXT, context)
    save(LOGS/'parent-receipt.json', receipt)
    emit(cache_key=context['cache_key'])


def load_frozen_helper():
    if any(name == 'tools' or name.startswith(('tools.', 'examples', 'mpm_jax')) for name in sys.modules):
        raise RuntimeError('Numerical modules were imported before selecting the frozen solver')
    sys.path.insert(0, str(SOLVER))
    spec = importlib.util.spec_from_file_location('frozen_jc_refinement', SOLVER/'tools/run_jc_refinement.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.engine.LOGS = LOGS
    return helper


def verify_imports():
    for name, module in list(sys.modules.items()):
        if name == 'tools' or name.startswith(('tools.', 'examples.', 'mpm_jax.')) or name in ('examples', 'mpm_jax'):
            path = getattr(module, '__file__', None)
            if path is not None and not Path(path).resolve().is_relative_to(SOLVER.resolve()):
                raise ValueError(f'Numerical import escaped original checkout: {name}')


def prefix_snapshot(folder, frame):
    folder = Path(folder)
    cp = read_json(folder/'checkpoint.json')
    if cp['frame'] != frame:
        raise ValueError('Parent checkpoint frame changed')
    names = ['plan.json', 'numerical-source.json', 'environment.json']
    names += [f'frames_data/frame_{i:05d}.npz' for i in range(frame+1)]
    return {'frame': frame, 'files': {name: file_record(contained_file(folder, name)) for name in names},
        'rows': cp['rows'], 'observations': cp['observations'], 'solver_seconds': cp['solver_seconds']}


def verify_prefix_preserved(folder, before):
    if any(file_record(contained_file(folder, name)) != record for name, record in before['files'].items()):
        raise ValueError('Continuation changed an original raw prefix file or initial plan')
    cp = read_json(Path(folder)/'checkpoint.json')
    n = before['frame']+1
    if cp['rows'][:n] != before['rows'] or cp['observations'][:n] != before['observations'] or cp['solver_seconds'] < before['solver_seconds']:
        raise ValueError('Continuation rewrote saved observations or reset accumulated solver time')


def runtime_check(context, frozen):
    verify_solver_source(context['source'])
    if controller_record() != context['controller']:
        raise ValueError('Controller source changed')
    current = environment()
    original = read_json((PARENT if PARENT.exists() else CACHE)/'result/environment.json')
    if not environment_matches(original, current):
        raise ValueError('Recorded numerical runtime changed across jobs')
    if context.get('parent_environment_lock') is not None and environment_lock() != context['parent_environment_lock']:
        raise ValueError('Full environment recorded by the preceding controller changed')
    verify_imports()
    return current


def worker():
    context = read_json(CONTEXT)
    frozen = load_frozen_helper()
    current_env = runtime_check(context, frozen)
    verify_carry(PARENT, context['request'])
    plan = read_json(OUTPUT/'plan.json')
    frozen.validate_plan(plan)
    import jax
    import numpy as np
    from dataclasses import asdict
    from examples.jc_necking import make_case
    from examples.common import run as common_run
    from mpm_jax.particle_fields import material_metadata
    jax.config.update('jax_enable_x64', True)
    case = make_case(grid=100, ppc=3, frames=FRAMES, end_time=DURATION, cfl=.15)
    arrays = frozen.state_arrays(case.states[0])
    if {key: frozen.descriptor(value) for key, value in arrays.items()} != plan['initial_state_fields']:
        raise ValueError('Reconstructed reference arrays differ; no cross-CPU resampling substitution is allowed')
    payload = {'configs': [asdict(c) for c in case.configs], 'materials': [material_metadata(m) for m in case.materials],
        'frames': FRAMES, 'substeps': SUBSTEPS, 'label': frozen.LABEL, 'definition': case.description}
    if frozen.encode(payload) != plan['signature_data'] or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan['common_signature']:
        raise ValueError('Reconstructed materials, geometry, loading or physical signature differ')
    del arrays
    verify_imports()
    save(LOGS/'worker-runtime.json', {'current': current_env, 'original': plan['environment'],
        'cpu_equal': current_env['cpu'] == plan['environment']['cpu'], 'scope': context['runtime_scope']})
    save(LOGS/'environment-lock.json', environment_lock())
    save(LOGS/'loaded-numerics.json', {name: str(Path(module.__file__).resolve().relative_to(SOLVER.resolve()))
        for name, module in sys.modules.items() if name in ('examples.common', 'examples.jc_necking', 'mpm_jax.core', 'mpm_jax.materials.johnson_cook')})
    common_run(case.states, case.materials, case.configs, frames=FRAMES, substeps=SUBSTEPS,
        output=OUTPUT, label=frozen.LABEL, observer=case.observer, metadata=case.description,
        render_output=False, gif=False, resume=True, stop_after_frame=context['request']['end_frame'])
    runtime_check(context, frozen)


def run():
    context = read_json(CONTEXT)
    frozen = load_frozen_helper()
    runtime_check(context, frozen)
    verify_carry(CACHE, context['request'])
    before = prefix_snapshot(CACHE/'result', context['request']['from_frame'])
    save(LOGS/'parent-prefix.json', before)
    required = (FRAMES+1)*PARTICLES*49 + 2*PARTICLES*544 + (context['request']['end_frame']+1)*PARTICLES*49 + 3*CACHE_BUDGET + 2*1024**3
    free = shutil.disk_usage(ROOT).free
    save(LOGS/'disk-admission.json', {'free_bytes': free, 'required_bytes': required, 'passed': free >= required})
    if free < required:
        raise ValueError('Insufficient disk for immutable parent, working mmap and complete next carry')
    CACHE.rename(PARENT)
    shutil.copytree(PARENT/'result', OUTPUT)
    os.environ['PYTHONPATH'] = str(SOLVER)
    os.environ.pop('PYTHONHOME', None)
    frozen.engine.command('jc-continuation-parent-audit', limited(context, 'audit-parent', 20), monitor=True)
    frozen.engine.command('jc-continuation-solve', limited(context, 'worker', 265), monitor=True)
    frozen.engine.command('jc-continuation-prefix-audit', limited(context, 'audit-worker', 20), monitor=True)
    runtime_check(context, frozen)
    emit(receipt_sha=digest(CACHE/'carry-receipt.json'))


def limited(context, action, minutes):
    deadline = context['started_epoch']+(300 if action == 'worker' else 320)*60
    seconds = min(minutes*60, int(deadline-time.time()))
    if seconds < 60:
        raise TimeoutError('No guarded child budget remains; existing state is diagnostic evidence only')
    return ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s', sys.executable, str(Path(__file__).resolve()), action]


def audit_prefix(folder, source, expected_frame, frozen):
    import numpy as np
    folder = Path(folder)
    plan, checkpoint, status, summary, segment = [read_json(folder/name) for name in
        ('plan.json', 'checkpoint.json', 'status.json', 'summary.json', 'segment.json')]
    checks = frozen.validate_plan(plan)
    if source != plan['source'] or source != read_json(folder/'numerical-source.json'):
        raise ValueError('Resolved plan and executed source differ')
    payload = frozen.decode(plan['signature_data'])
    config = payload['configs'][0]
    dt, count = config['dt'], plan['particles']
    signature, frame = plan['common_signature'], expected_frame
    if not (checkpoint['signature'] == signature and checkpoint['frame'] == frame and len(checkpoint['states']) == 1
            and status == summary == segment and status.get('completed') is False
            and status.get('segment_complete') is True and status.get('phase') == 'checkpoint-ready'
            and status.get('signature') == signature and status.get('frame') == frame
            and status.get('frames') == FRAMES and status.get('steps_completed') == frame*SUBSTEPS
            and status.get('target_steps') == FRAMES*SUBSTEPS and status.get('particles') == [count]
            and math.isclose(status['time'], frame*SUBSTEPS*dt, rel_tol=1e-14)
            and math.isclose(status['target_time'], DURATION, rel_tol=1e-14)):
        raise ValueError('Result does not identify the requested valid partial endpoint')
    state_path = contained_file(folder, checkpoint['states'][0])
    if not re.fullmatch(r'checkpoint_[01]_0\.npz', state_path.name):
        raise ValueError('Unexpected checkpoint state name')
    with np.load(state_path, allow_pickle=False) as archive:
        if set(archive.files) != set(plan['initial_state_fields']):
            raise ValueError('Checkpoint is missing State or History fields')
        arrays = {}
        for key in archive.files:
            value = archive[key]
            expected = plan['initial_state_fields'][key]
            if list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype'] or not np.isfinite(value).all():
                raise ValueError(f'Malformed checkpoint field {key}')
            arrays[key] = value
    if not bool(arrays['valid']) or not arrays['history_valid'].all() or not (np.linalg.det(arrays['F']) > 0).all():
        raise ValueError('Invalid state/history or nonpositive deformation determinant')
    for key in ('mass', 'volume'):
        if not (arrays[key] > 0).all() or frozen.descriptor(arrays[key]) != plan['initial_state_fields'][key]:
            raise ValueError('Checkpoint mass/reference volume changed')
    if not ((arrays['history_damage'] >= 0) & (arrays['history_damage'] <= 1)).all() or (arrays['history_plastic_strain'] < 0).any():
        raise ValueError('Checkpoint damage/plastic strain is outside its physical range')
    expected_times = np.arange(frame+1)*SUBSTEPS*dt
    epsilon = np.finfo(arrays['time'].dtype).eps
    accumulated = frame*SUBSTEPS*epsilon
    clock_tolerance = max(dt*1e-5, 1e-15, 2*accumulated/(1-accumulated)*abs(expected_times[-1]))
    if not math.isclose(float(arrays['time']), expected_times[-1], rel_tol=0, abs_tol=clock_tolerance):
        raise ValueError('Accumulated checkpoint clock does not match its saved frame')
    rows, observations = checkpoint['rows'], checkpoint['observations']
    if len(rows) != frame+1 or len(observations) != frame+1 or not frozen.finite_record(checkpoint):
        raise ValueError('Diagnostic/observation prefix is incomplete')
    if np.shape(checkpoint['contact_impulse']) != (3,) or np.shape(checkpoint['external_impulse']) != (3,) or checkpoint['solver_seconds'] < 0:
        raise ValueError('Checkpoint cumulative quantities are incomplete')
    prefix_paths = [folder/'frames_data'/f'frame_{i:05d}.npz' for i in range(frame+1)]
    if sorted((folder/'frames_data').glob('*.npz')) != prefix_paths:
        raise ValueError('Saved frame prefix is missing frames or contains unexpected future frames')
    records, cut_details = [], []
    previous_failed = np.zeros(count, dtype=bool)
    previous_damage = np.zeros(count)
    reference = None
    spacing = float(payload['definition']['particle_spacing_m'])
    for index, path in enumerate(prefix_paths):
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FRAME_FIELDS:
                raise ValueError('Saved frame fields differ from the complete ballistic prefix')
            values = {key: archive[key] for key in archive.files}
        shapes = {'x': (count, 3), 'speed': (count,), 'damage': (count,), 'temperature': (count,),
                  'failed': (count,), 'time': (), 'contact_impulse': (3,)}
        if any(values[key].shape != shape or values[key].dtype != np.dtype('uint8' if key == 'failed' else 'float64')
               or not np.isfinite(values[key]).all() for key, shape in shapes.items()):
            raise ValueError('Malformed or nonfinite saved frame')
        if np.any(values['speed'] < 0) or np.any(values['temperature'] < 0):
            raise ValueError('Negative saved speed or absolute temperature')
        if not math.isclose(float(values['time']), expected_times[index], rel_tol=0, abs_tol=1e-15):
            raise ValueError('Saved prefix time changed')
        if not ((values['damage'] >= 0) & (values['damage'] <= 1)).all() or not np.isin(values['failed'], (0, 1)).all():
            raise ValueError('Invalid damage/failure prefix')
        failed = values['failed'].astype(bool)
        if np.any(previous_failed & ~failed) or np.any(values['damage'] < previous_damage-64*np.finfo(float).eps):
            raise ValueError('Failure or damage decreased in the saved history')
        previous_failed, previous_damage = failed, values['damage']
        if index == 0:
            reference = values['x'].copy()
            if frozen.descriptor(reference) != plan['initial_state_fields']['x']:
                raise ValueError('Initial saved geometry differs from the resolved reference')
            for field, initial in [('damage', 'history_damage'), ('temperature', 'history_temperature')]:
                if frozen.descriptor(values[field]) != plan['initial_state_fields'][initial]:
                    raise ValueError('Initial saved history differs from the resolved input')
            if failed.any() or np.any(values['speed'] != 0):
                raise ValueError('Initial rod is not intact and stationary')
        diagnostic, observed = rows[index], observations[index]
        required_observation = {'time', 'external_work_trapezoidal', 'boundary_impulse_x', 'boundary_impulse_y', 'boundary_impulse_z',
            'failed_particles', 'max_damage', 'maximum_equivalent_plastic_strain', 'central_material_section_force_N', 'central_active_particles'}
        if not required_observation.issubset(observed):
            raise ValueError('Observation prefix lacks quantities required by continuation')
        if diagnostic['body'] != 0 or not math.isclose(diagnostic['time'], expected_times[index], rel_tol=0, abs_tol=1e-15) or not math.isclose(observed['time'], expected_times[index], rel_tol=0, abs_tol=1e-15):
            raise ValueError('Diagnostic/observer time differs from the complete prefix')
        if not (diagnostic['min_J'] > 0 and math.isclose(diagnostic['mass'], plan['initial_mass_kg'], rel_tol=1e-12)
                and observed['failed_particles'] == int(failed.sum())
                and math.isclose(observed['max_damage'], float(values['damage'].max()), rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError('Saved diagnostic/observer disagrees with valid prefix data')
        active = np.flatnonzero(~failed)
        order = active[np.argsort(values['x'][active, 0])]
        bases, lower, upper, cuts = frozen.engine.axial_support_intervals(values['x'][order, 0], .0001, 100)
        details = [{'current_x_m': [float(values['x'][order[k], 0]), float(values['x'][order[k+1], 0])],
            'reference_x_m': [float(reference[order[k], 0]), float(reference[order[k+1], 0])],
            'positive_node_intervals': [[int(lower[k]), int(upper[k])], [int(lower[k+1]), int(upper[k+1])]]} for k in cuts]
        cut_details.append({'frame': index, 'time_s': float(values['time']), 'cuts': details})
        records.append({'frame': index, 'time_s': float(values['time']), 'active_particles': int(active.size),
            'failed_particles': int(failed.sum()), 'maximum_damage': float(values['damage'].max()),
            'maximum_equivalent_plastic_strain_recorded': observed['maximum_equivalent_plastic_strain'],
            'central_material_section_force_N_recorded': observed['central_material_section_force_N'],
            'central_active_particles_recorded': observed['central_active_particles'],
            'positive_x_stencil_cuts': len(cuts), 'positive_x_support_components': len(cuts)+int(active.size > 0)})
        if index == frame:
            for field, key in [('x', 'x'), ('damage', 'history_damage'), ('temperature', 'history_temperature')]:
                np.testing.assert_array_equal(values[field], arrays[key])
            np.testing.assert_array_equal(failed, arrays['history_failed'])
            np.testing.assert_allclose(values['speed'], np.linalg.norm(arrays['v'], axis=1), rtol=1e-14, atol=0)
            np.testing.assert_array_equal(values['contact_impulse'], checkpoint['contact_impulse'])
    metrics = frozen.final_metrics(arrays, reference, spacing)
    for key in ('central_material_section_force_N', 'maximum_equivalent_plastic_strain', 'max_damage', 'failed_particles', 'central_active_particles'):
        if not math.isclose(metrics[key], observations[-1][key], rel_tol=1e-10, abs_tol=0):
            raise ValueError(f'Final checkpoint independently disagrees with observer {key}')
    for key in ('min_J', 'max_J'):
        if not math.isclose(metrics[key], rows[-1][key], rel_tol=1e-12, abs_tol=0):
            raise ValueError(f'Final checkpoint independently disagrees with diagnostic {key}')
    momentum = np.sum(arrays['mass'][:, None]*arrays['v'], axis=0)
    reaction = momentum-np.asarray(checkpoint['external_impulse'])
    np.testing.assert_allclose([observations[-1]['boundary_impulse_'+axis] for axis in 'xyz'], reaction, rtol=1e-10, atol=0)
    if observations[-1]['external_work_trapezoidal'] != checkpoint['external_work']:
        raise ValueError('Final work differs from the cumulative checkpoint value')
    checks.update({'source_matches': True, 'valid_partial_checkpoint': True, 'all_state_and_history_fields_checked': True,
        'complete_saved_prefix_checked': True, 'constant_mass_reference_volume': True,
        'final_ep_and_section_force_recomputed': True,
        'original_25us_prefix_intact': all(row['failed_particles'] == 0 for row in records[:11]),
        'no_support_cut_before_material_failure': all(row['positive_x_stencil_cuts'] == 0 or row['failed_particles'] > 0 for row in records)})
    report = {'schema': 'jc-continuation-prefix-review-v1', 'passed': all(checks.values()), 'checks': checks,
        'completed': False, 'full_protocol_completed': False, 'production_eligible': False,
        'segment_complete': True, 'frame': frame, 'steps': frame*SUBSTEPS, 'time_s': float(expected_times[-1]),
        'full_target_steps': FRAMES*SUBSTEPS, 'full_target_time_s': DURATION,
        'initial_mass_kg': plan['initial_mass_kg'], 'initial_reference_volume_m3': plan['initial_reference_volume_m3'],
        'sampling_comparison': plan.get('sampling_comparison'),
        'final_checkpoint_metrics': metrics, 'positive_support_history': cut_details,
        'first_saved_support_cut': next((item for item in records if item['positive_x_stencil_cuts']), None),
        'first_saved_failed_material': next((item for item in records if item['failed_particles']), None),
        'review_source_sha256': digest(Path(__file__)), 'producer_source': source,
        'scope': 'An admitted later partial prefix of the unchanged full 150 microsecond plan. Original first-segment plan bytes and all prior saved frames remain unchanged. Positive x-stencil cuts prove missing axial shared support, not a full 3D graph. Ep/section-force histories are recorded by the original observer; only the latest complete State independently reconstructs those fields. No complete fracture, convergence or equivalence to an uninterrupted run is asserted. Saved-frame ordering cannot resolve events inside the same output interval.'}
    names = ['plan.json', 'numerical-source.json', 'checkpoint.json', 'status.json', 'summary.json', 'segment.json', state_path.name]
    names += [path.relative_to(folder).as_posix() for path in prefix_paths]
    report['input_sha256'] = {name: digest(folder/name) for name in names}
    with (folder/'prefix-history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    for name, table in [('diagnostics-prefix.csv', rows), ('observables-prefix.csv', observations)]:
        with (folder/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=dict.fromkeys(key for row in table for key in row), lineterminator='\n')
            writer.writeheader(); writer.writerows(table)
    report['prefix_history_sha256'] = digest(folder/'prefix-history.csv')
    save(folder/'prefix-review.json', report)
    return report


def seal_carry(context, report):
    if not strict_review(report):
        raise ValueError('Rejected prefix cannot become an admitted carry')
    cp = read_json(OUTPUT/'checkpoint.json')
    names = ['plan.json', 'numerical-source.json', 'environment.json', 'checkpoint.json', 'status.json',
        'summary.json', 'segment.json', 'prefix-review.json', 'prefix-history.csv',
        'diagnostics-prefix.csv', 'observables-prefix.csv', *cp['states']]
    names += [f'frames_data/frame_{i:05d}.npz' for i in range(cp['frame']+1)]
    inventory = {'result/'+name: (contained_file(OUTPUT, name), file_record(OUTPUT/name)) for name in names}
    # Keep source/receipt lineage, never historical state slots or executable files.
    for path in sorted((PARENT/'lineage').glob('*.json')):
        inventory['lineage/'+path.name] = (path, file_record(path))
    parent_name = f"lineage/frame-{context['request']['from_frame']:02d}-receipt.json"
    inventory[parent_name] = (PARENT/'carry-receipt.json', file_record(PARENT/'carry-receipt.json'))
    old_review = PARENT/'result/prefix-review.json'
    inventory[f"lineage/frame-{context['request']['from_frame']:02d}-prefix-review.json"] = (old_review, file_record(old_review))
    for name in ('parent-prefix.json', 'worker-runtime.json', 'loaded-numerics.json', 'jc-continuation-solve.json', 'environment-lock.json'):
        path = contained_file(LOGS, name)
        inventory['lineage/'+f"frame-{cp['frame']:02d}-"+name] = (path, file_record(path))
    total = sum(record['bytes'] for _, record in inventory.values())
    if total+1024**2 > CACHE_BUDGET or CACHE.exists():
        raise ValueError('Complete next carry exceeds budget or destination already exists')
    stage = Path(tempfile.mkdtemp(prefix='jc-continuation-carry-', dir=CACHE.parent))
    for name, (original, record) in inventory.items():
        target = stage/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open('rb') as inp, target.open('wb') as out:
            shutil.copyfileobj(inp, out, READ_BYTES)
        if file_record(target) != record:
            raise ValueError('Carry changed during lossless copy')
    source, req = context['source'], context['request']
    receipt = {'schema': 'jc-continuation-carry-v1', 'source': source, 'source_fingerprint': source['fingerprint'],
        'controller': context['controller'], 'cache_key': context['cache_key'], 'frame': req['end_frame'],
        'full_frames': FRAMES, 'full_target_steps': FRAMES*SUBSTEPS, 'segment_steps': req['end_frame']*SUBSTEPS,
        'common_signature': read_json(OUTPUT/'plan.json')['common_signature'], 'completed': False,
        'full_protocol_completed': False, 'segment_complete': True, 'production_eligible': False,
        'prefix_preserved': True, 'parent_receipt_file': parent_name,
        'environment_lock_file': f"lineage/frame-{req['end_frame']:02d}-environment-lock.json",
        'parent_receipt_sha256': req['parent_receipt_sha256'], 'parent_cache_key': req['parent_key'],
        'files': {name: record for name, (_, record) in inventory.items()}, 'payload_bytes': total,
        'cache_budget_bytes': CACHE_BUDGET, 'cross_job_resume_performed': True,
        'continuation_equivalence_to_uninterrupted_run_verified': False,
        'scope': 'A valid later partial prefix, preserving all parent frame bytes and original solver identity. Numerical continuation was executed; equivalence to an uninterrupted refined run and complete fracture are not established.'}
    save(stage/'carry-receipt.json', receipt)
    stage.rename(CACHE)
    verify_carry(CACHE, {**req, 'parent_key': context['cache_key'], 'parent_receipt_sha256': digest(CACHE/'carry-receipt.json'), 'from_frame': req['end_frame']})
    save(LOGS/'carry-receipt.json', receipt)


def audit_worker(parent=False):
    context = read_json(CONTEXT)
    frozen = load_frozen_helper()
    runtime_check(context, frozen)
    if parent:
        # Work on the copy only; the restored parent is never rewritten.
        if context['request']['from_frame'] == 10:
            report = frozen.audit_prefix(OUTPUT, context['source'])
        else:
            report = audit_prefix(OUTPUT, context['source'], context['request']['from_frame'], frozen)
        save(LOGS/'parent-independent-review.json', report)
        if not strict_review(report):
            raise ValueError('Restored parent failed independent prefix review')
        return
    verify_prefix_preserved(OUTPUT, read_json(LOGS/'parent-prefix.json'))
    report = audit_prefix(OUTPUT, context['source'], context['request']['end_frame'], frozen)
    report['controller_source'] = context['controller']
    report['checks']['parent_prefix_raw_bytes_unchanged'] = True
    report['passed'] = all(report['checks'].values())
    save(OUTPUT/'prefix-review.json', report)
    runtime_check(context, frozen)
    seal_carry(context, report)


def confirm_cache():
    context = read_json(CONTEXT)
    receipt_sha = digest(CACHE/'carry-receipt.json')
    if (os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != context['cache_key']
            or os.environ.get('RECEIPT_SHA') != receipt_sha):
        raise ValueError('Next exact cache key/receipt was not confirmed retrievable')
    req = context['request']
    verify_carry(CACHE, {**req, 'parent_key': context['cache_key'], 'parent_receipt_sha256': receipt_sha, 'from_frame': req['end_frame']})
    save(LOGS/'cache-confirmation.json', {'verified': True, 'cache_key': context['cache_key'],
        'receipt_sha256': receipt_sha, 'permanent_delivery': False})


def valid_npz(path):
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and all(name.endswith('.npy') for name in archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def split_complete_file(path, destination, relative, *, part_bytes=PART_BYTES):
    if type(part_bytes) is not int or not 0 < part_bytes <= PART_BYTES:
        raise ValueError('Invalid state part size')
    before = Path(path).stat()
    whole, parts, consumed = hashlib.sha256(), [], 0
    with Path(path).open('rb') as stream:
        while consumed < before.st_size:
            target = Path(destination)/f'{relative}.part-{len(parts):04d}'
            target.parent.mkdir(parents=True, exist_ok=True)
            part, size = hashlib.sha256(), 0
            with target.open('wb') as out:
                while size < part_bytes and consumed < before.st_size:
                    block = stream.read(min(READ_BYTES, part_bytes-size, before.st_size-consumed))
                    if not block:
                        raise ValueError('State source ended before its recorded size')
                    out.write(block)
                    whole.update(block)
                    part.update(block)
                    size += len(block)
                    consumed += len(block)
            parts.append({'path': target.relative_to(destination).as_posix(), 'bytes': size, 'sha256': part.hexdigest()})
    after = Path(path).stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise ValueError('State source changed during packaging')
    return {'bytes': consumed, 'sha256': whole.hexdigest(), 'parts': parts, 'complete_file': True}


def package():
    if DELIVERY.exists():
        raise ValueError('Diagnostic package already exists')
    confirmation = read_json(LOGS/'cache-confirmation.json') if (LOGS/'cache-confirmation.json').is_file() else {}
    confirmed = confirmation.get('verified') is True and (CACHE/'carry-receipt.json').is_file() and confirmation.get('receipt_sha256') == digest(CACHE/'carry-receipt.json')
    budget = 8*1024**2 if confirmed else TOTAL_BYTES
    stage = Path(tempfile.mkdtemp(prefix='jc-continuation-delivery-', dir=DELIVERY.parent))
    files, states, omitted, used = {}, {}, {}, 0
    records = sorted({p for root in (LOGS, OUTPUT) if root.exists() for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv')})
    cp = read_json(OUTPUT/'checkpoint.json') if (OUTPUT/'checkpoint.json').is_file() else {}
    if not confirmed:
        records += [contained_file(OUTPUT, name) for name in cp.get('states', [])]
        records += sorted((OUTPUT/'frames_data').glob('*.npz'))
    for path in records:
        name = path.relative_to(ROOT).as_posix()
        record = file_record(path)
        if used+record['bytes'] > budget-1024**2 or (path.suffix == '.npz' and not valid_npz(path)):
            omitted[name] = {**record, 'reason': 'Whole complete file exceeds the bounded diagnostic budget or fails CRC; no partial file presented.'}
            continue
        if path.suffix == '.npz':
            state = split_complete_file(path, stage, 'state-parts/'+name)
            states[name] = state
            for part in state['parts']:
                files[part['path']] = {'bytes': part['bytes'], 'sha256': part['sha256']}
        else:
            target = stage/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_record(target) != record:
                raise ValueError('Diagnostic metadata changed during copy')
            files[name] = record
        used += record['bytes']
    manifest = {'schema': 'jc-continuation-diagnostic-delivery-v1', 'completed': False,
        'full_protocol_completed': False, 'production_eligible': False, 'media_created': False,
        'cache_saved_exactly': confirmed, 'files': files, 'state_files': states, 'omitted': omitted,
        'payload_bytes': used, 'total_budget_bytes': budget,
        'scope': 'Partial diagnostic only; latest complete state is prioritized after records if cache handoff failed. No full fracture, uninterrupted-run equivalence, or public-gallery eligibility.'}
    save(stage/'delivery-manifest.json', manifest)
    if used+(stage/'delivery-manifest.json').stat().st_size > budget:
        raise ValueError('Delivery metadata exceeds bounded budget')
    stage.rename(DELIVERY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'worker', 'audit-parent', 'audit-worker', 'confirm-cache', 'package'))
    parser.add_argument('--parent-key')
    parser.add_argument('--receipt-sha')
    parser.add_argument('--from-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux':
        parser.error('Continuation is restricted to the independent Linux Actions runner')
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == 'prepare':
            prepare(args)
        elif args.action == 'audit-parent':
            audit_worker(parent=True)
        else:
            globals()[args.action.replace('-', '_')]()
    except BaseException as error:
        save(LOGS/f'{args.action}-failure.json', {'completed': False, 'production_eligible': False,
            'action': args.action, 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
