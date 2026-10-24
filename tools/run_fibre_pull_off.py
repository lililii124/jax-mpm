"""Bounded full-scale prefixes of the unchanged crossed-fibre pull-off protocol."""

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_engineering_preview as engine

CACHE = ROOT/'.local/fibre-pull-off-cache'
LOGS = ROOT/'.local/fibre-pull-off-ci'
DELIVERY = ROOT/'.local/fibre-pull-off-delivery'
GIB = 1024**3
CACHE_LIMIT = 2*GIB
PACKAGE_LIMIT = 192*1024**2
REPOSITORY_CACHE_LIMIT = 9_000_000_000
PYTHON_VERSION = '3.12.13'
MODELS = ('hertz', 'jkr', 'md', 'self-consistent')
RESULT_NAMES = {'plan.json', 'summary.json', 'status.json', 'checkpoint.npz', 'final_state_0.npz',
    'particle_fields.npz', 'trajectory.npz', 'observables.csv', 'diagnostics.csv', 'contact-model.json',
    'trajectory_x.npy', 'trajectory_damage.npy', 'trajectory_failed.npy', 'trajectory_speed.npy',
    'trajectory_temperature.npy'}
RUNTIME_NAMES = {'request.json', 'source.json', 'environment.json', 'requirements.txt'}
digest, save = engine.digest, engine.save


def file_record(path):
    path = Path(path)
    before = path.stat()
    record = {'bytes': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing: '+str(path))
    return record


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_record():
    parsed = ast.parse((ROOT/'examples/fibre_pull_off.py').read_text(encoding='utf-8'))
    names = next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'SOURCE_FILES' for target in node.targets))
    names = sorted(set(names) | {'tools/run_fibre_pull_off.py', 'tools/run_engineering_preview.py',
        'tools/run_woven_preview.py', 'pyproject.toml', '.github/workflows/fibre-pull-off-checks.yml'})
    source = engine.identity()
    files, blobs, normalized = {}, {}, {}
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe source path')
        raw = (ROOT/name).read_bytes()
        committed = subprocess.check_output(['git', 'rev-parse', f'HEAD:{name}'], cwd=ROOT, text=True).strip()
        actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if committed != actual:
            raise ValueError('Numerical/controller source differs from its Git blob: '+name)
        files[name] = hashlib.sha256(raw).hexdigest()
        blobs[name] = actual
        normalized[name] = hashlib.sha256(raw.decode('utf-8').replace('\r\n', '\n').encode()).hexdigest()
    return {**source, 'files': files, 'git_blob_sha': blobs, 'lf_sha256': normalized,
        'fingerprint': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def numerical_identity(source):
    return {key: source[key] for key in ('repository', 'commit', 'tree', 'files', 'git_blob_sha', 'lf_sha256', 'fingerprint')}


def request(model):
    if model not in MODELS:
        raise ValueError('Unknown contact model')
    return {'schema': 'fibre-pull-off-full-protocol-v1', 'model': model, 'frames': 60,
        'duration_s': .03, 'dx_m': .0004, 'spacing_m': .0002, 'particles': 202240,
        'steps': 24495, 'rings': 32, 'production_eligible': False}


def cache_key(source, model, frame):
    if (model not in MODELS or not re.fullmatch('[0-9a-f]{40}', source['commit'])
            or not re.fullmatch(r'\d+', str(source['run_id']))
            or not re.fullmatch(r'\d+', str(source['run_attempt'])) or type(frame) is not int or not 0 < frame <= 60):
        raise ValueError('Invalid cache identity or saved frame')
    return f'fibre-pull-off-v1-{source["commit"]}-{model}-{source["run_id"]}-{source["run_attempt"]}-f{frame:02d}'


def result_folder(cache, req):
    if req.get('model') not in MODELS:
        raise ValueError('Invalid result model path')
    return Path(cache)/'data'/req['model']


def payload_inventory(cache, req, *, diagnostic=False):
    cache = Path(cache)
    if cache.is_symlink():
        raise ValueError('Cache root must not be a symlink')
    records = {}
    folder = result_folder(cache, req)
    for path in sorted(cache.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlinks are not allowed in numerical cache')
        if not path.is_file() or path == cache/'receipt.json':
            continue
        result_names = RESULT_NAMES | ({'rejected_checkpoint.npz'} | {name+'.tmp' for name in RESULT_NAMES | {'rejected_checkpoint.npz'}} if diagnostic else set())
        allowed = ((path.parent == folder and path.name in result_names)
            or (path.parent == cache/'runtime' and path.name in RUNTIME_NAMES)
            or (path.parent == cache/'receipts' and re.fullmatch(r'[0-9a-f]{64}\.json', path.name)))
        if not allowed:
            raise ValueError('Unexpected numerical cache file: '+str(path))
        record = file_record(path)
        if path.parent == cache/'receipts' and record['sha256'] != path.stem:
            raise ValueError('Historical receipt filename does not match its complete bytes')
        records[path.relative_to(cache).as_posix()] = record
    if sum(item['bytes'] for item in records.values()) > CACHE_LIMIT:
        raise ValueError('Complete raw cache exceeds its 2 GiB budget')
    return records


def environment():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', distribution.metadata['Name']).lower()
        if name in ('pip', 'setuptools', 'wheel', 'jax-mpm'):
            continue
        version = distribution.version
        if not re.fullmatch('[a-z0-9][a-z0-9-]*', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Unsafe package version pin')
        if name in packages and packages[name] != version:
            raise ValueError('Conflicting installed package versions')
        packages[name] = version
    if not {'jax', 'jaxlib', 'numpy', 'scipy', 'pillow', 'psutil'}.issubset(packages):
        raise ValueError('Incomplete scientific environment')
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'system': platform.system(), 'machine': platform.machine(), 'packages': dict(sorted(packages.items()))}


def requirements_text(record):
    if record['python'] != PYTHON_VERSION or not record['packages']:
        raise ValueError('Python/runtime lock is incompatible')
    result = []
    for name, version in sorted(record['packages'].items()):
        if not re.fullmatch('[a-z0-9][a-z0-9-]*', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Runtime lock may contain only pinned distribution names and versions')
        result.append(name+'=='+version)
    return '\n'.join(result)+'\n'


def output_values(**values):
    target = os.environ.get('GITHUB_OUTPUT')
    if target:
        with Path(target).open('a', encoding='utf-8') as stream:
            for key, value in values.items():
                stream.write(f'{key}={value}\n')


def repository_cache_usage(repository):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Invalid repository identity')
    token = os.environ.get('GH_TOKEN', '')
    if not token:
        raise ValueError('Read-only cache-usage credentials are required')
    request = urllib.request.Request(f'https://api.github.com/repos/{repository}/actions/cache/usage',
        headers={'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    size = value.get('active_caches_size_in_bytes')
    if type(size) is not int or size < 0:
        raise ValueError('Cache API did not return a valid existing byte count')
    return {'existing_bytes': size, 'repository': repository, 'observed_unix_s': time.time()}


def check_cache_quota(source, new_bytes):
    if type(new_bytes) is not int or new_bytes < 0:
        raise ValueError('New cache byte estimate must be a nonnegative integer')
    usage = repository_cache_usage(source['repository'])
    record = {**usage, 'new_raw_bytes': new_bytes, 'conservative_repository_limit_bytes': REPOSITORY_CACHE_LIMIT,
        'passed': usage['existing_bytes']+new_bytes <= REPOSITORY_CACHE_LIMIT,
        'scope': 'Read-only repository usage plus uncompressed new snapshot; no cache removal or paid quota changes.'}
    if not record['passed']:
        raise ValueError('Existing raw caches plus this snapshot exceed the conservative repository budget: '+json.dumps(record))
    return record


def prepare(model, endpoint, parent_key='', parent_receipt=''):
    req, source = request(model), source_record()
    if type(endpoint) is not int or not 0 < endpoint <= 60:
        raise ValueError('A complete saved frame 1..60 is required')
    if bool(parent_key) != bool(parent_receipt):
        raise ValueError('Resume needs both exact cache key and expected receipt SHA256')
    if parent_key:
        pattern = f'fibre-pull-off-v1-{source["commit"]}-{re.escape(model)}-\\d+-\\d+-f\\d{{2}}'
        if not re.fullmatch(pattern, parent_key) or not re.fullmatch('[0-9a-f]{64}', parent_receipt):
            raise ValueError('Restore cache must belong to this exact source/model')
    elif endpoint != 2:
        raise ValueError('The first full-scale resource check must stop at frame 2')
    key = cache_key(source, model, endpoint)
    quota = check_cache_quota(source, GIB)
    context = {'request': req, 'source': source, 'end_frame': endpoint,
        'parent_key': parent_key, 'parent_receipt_sha256': parent_receipt, 'cache_key': key,
        'job_deadline_monotonic': time.monotonic()+330*60, 'repository_cache_admission': quota}
    save(LOGS/'job.json', context)
    output_values(cache_key=key, parent_key=parent_key)
    return context


def verify_restored(cache, context):
    cache = Path(cache)
    path = cache/'receipt.json'
    if not path.is_file() or path.is_symlink() or digest(path) != context['parent_receipt_sha256']:
        raise ValueError('Missing or incorrect exact parent receipt')
    receipt = read(path)
    if (receipt.get('schema') != 'fibre-pull-off-cache-v1' or receipt.get('cache_key') != context['parent_key']
            or receipt.get('kind') != 'accepted-prefix' or receipt.get('never_resumable') is not False
            or receipt.get('numerical_all_checks') is not True
            or receipt.get('request') != context['request']
            or numerical_identity(receipt['source']) != numerical_identity(context['source'])
            or receipt.get('prefix_verified') is not True or receipt.get('production_eligible') is not False
            or type(receipt.get('frame')) is not int or not 0 < receipt['frame'] < context['end_frame'] <= 60):
        raise ValueError('Parent source/protocol/frame identity is not a valid continuation')
    if payload_inventory(cache, context['request']) != receipt.get('files'):
        raise ValueError('Restored raw cache bytes differ from the sealed parent snapshot')
    current, visited = receipt, set()
    while True:
        if (current['cache_key'] != cache_key(current['source'], context['request']['model'], current['frame'])
                or current['request'] != context['request'] or current.get('prefix_verified') is not True
                or current.get('kind') != 'accepted-prefix' or current.get('never_resumable') is not False
                or current.get('numerical_all_checks') is not True
                or numerical_identity(current['source']) != numerical_identity(context['source'])
                or current['prefix_review'].get('frame') != current['frame']
                or current['prefix_review'].get('prefix_verified') is not True):
            raise ValueError('Historical receipt protocol/source/frame identity differs')
        parent = current.get('parent_receipt_sha256', '')
        if not parent:
            if current['frame'] != 2 or current.get('parent_cache_key'):
                raise ValueError('Ancestry must begin with the fixed frame-2 resource prefix')
            break
        if parent in visited or not re.fullmatch('[0-9a-f]{64}', parent):
            raise ValueError('Invalid or cyclic receipt ancestry')
        path = cache/'receipts'/f'{parent}.json'
        if not path.is_file() or digest(path) != parent:
            raise ValueError('Missing original parent receipt')
        previous = read(path)
        if previous['cache_key'] != current['parent_cache_key'] or not previous['frame'] < current['frame']:
            raise ValueError('Receipt ancestry does not advance complete frames')
        visited.add(parent)
        current = previous
    if {path.stem for path in (cache/'receipts').glob('*.json')} != visited:
        raise ValueError('Unexpected unbound historical receipt')
    saved_environment = read(cache/'runtime/environment.json')
    expected_requirements = requirements_text(saved_environment)
    if (cache/'runtime/requirements.txt').read_text(encoding='utf-8') != expected_requirements:
        raise ValueError('Cached requirements are not exactly derived from the verified environment lock')
    if (read(cache/'runtime/request.json') != context['request']
            or numerical_identity(read(cache/'runtime/source.json')) != numerical_identity(context['source'])):
        raise ValueError('Cached runtime source or request differs from this continuation')
    return receipt


def lock_environment(context, cache=CACHE):
    cache = Path(cache)
    current = environment()
    if current['python'] != PYTHON_VERSION:
        raise ValueError('Use the fixed Python patch version')
    if context['parent_key']:
        verify_restored(cache, context)
        if current != read(cache/'runtime/environment.json'):
            raise ValueError('Installed runtime differs from the original numerical environment')
    else:
        if cache.exists() and any(cache.iterdir()):
            raise FileExistsError('A fresh prefix must not overwrite an existing cache')
        runtime = cache/'runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        save(runtime/'environment.json', current)
        save(runtime/'request.json', context['request'])
        save(runtime/'source.json', context['source'])
        (runtime/'requirements.txt').write_text(requirements_text(current), encoding='utf-8')
    save(LOGS/'runtime.json', current)
    return current


def command_line(context, cache=CACHE):
    arguments = [sys.executable, '-m', 'examples.fibre_pull_off', '--model', context['request']['model'],
        '--dx', '.0004', '--spacing', '.0002', '--frames', '60', '--rings', '32',
        '--stop-after-frame', str(context['end_frame']), '--output', str(Path(cache)/'data')]
    if context['parent_key']:
        arguments.append('--resume')
    return arguments


def remaining_solve_seconds(context, now=None):
    now = time.monotonic() if now is None else now
    remaining = context['job_deadline_monotonic']-now-60*60
    if remaining < 60:
        raise TimeoutError('Insufficient shared job budget for solve, review, raw cache and evidence upload')
    return int(min(265*60, remaining))


def _array_record(value):
    import numpy as np
    value = np.asarray(value)
    array = np.ascontiguousarray(value)
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
        'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def audit_prefix(folder, req, frame):
    """Independent geometry/impulse checks and byte binding; not full model validation."""
    import numpy as np
    from examples.fibre_pull_off import fixture_motion, plan_case

    folder = Path(folder)
    geometry, expected_plan = plan_case(model=req['model'])
    plan, summary, status = (read(folder/name) for name in ('plan.json', 'summary.json', 'status.json'))
    if plan != expected_plan or len(geometry['x']) != req['particles'] or plan['steps'] != req['steps']:
        raise ValueError('Actual protocol differs from the fixed full-scale request')
    expected_steps = plan['frame_steps'][frame]
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()
    completed = frame == plan['frames']
    if (summary.get('completed') is not completed or summary.get('segment_complete') is not (not completed)
            or summary.get('production_eligible') is not False or status.get('completed') is not completed
            or status.get('coupling_valid') is not True or summary.get('particles') != [req['particles']]
            or summary.get('definition') != plan or summary.get('target_steps') != req['steps']
            or summary.get('target_simulation_time') != req['duration_s']
            or status.get('phase') != ('completed' if completed else 'checkpoint-ready')
            or status.get('frame') != frame or summary.get('segment_end_frame') != frame
            or summary.get('steps') != expected_steps or status.get('steps') != expected_steps
            or status.get('signature') != signature or summary.get('source') != plan['source']):
        raise ValueError('Summary/status do not describe the requested valid saved-frame prefix')
    with (folder/'observables.csv').open(newline='', encoding='utf-8') as stream:
        observations = list(csv.DictReader(stream))
    with (folder/'diagnostics.csv').open(newline='', encoding='utf-8') as stream:
        diagnostics = list(csv.DictReader(stream))
    if len(observations) != frame+1 or len(diagnostics) != frame+1:
        raise ValueError('Incomplete saved CSV prefix')
    arrays = {}
    try:
        with np.load(folder/'checkpoint.npz', allow_pickle=False) as cp, np.load(folder/'final_state_0.npz', allow_pickle=False) as final:
            leaves = {name: _array_record(cp[name]) for name in cp.files}
            if len(cp.files) != len(set(cp.files)):
                raise ValueError('Duplicate checkpoint leaves')
            if any(cp[name].dtype.kind not in 'biufUS' or (cp[name].dtype.kind in 'biuf' and not np.isfinite(cp[name]).all()) for name in cp.files):
                raise ValueError('Checkpoint contains nonfinite or unsupported leaves')
            if (int(cp['frame']) != frame or int(cp['last_saved_frame']) != frame or not bool(cp['complete_saved_frame'])
                    or not bool(cp['totals_valid']) or int(cp['totals_reason']) != 0 or not bool(cp['state_valid'])
                    or not bool(cp['contact_valid']) or not bool(cp['metrics_valid'])
                    or str(cp['signature']) != signature or int(cp['totals_steps']) != expected_steps):
                raise ValueError('Checkpoint is not a valid complete saved frame')
            tables = json.loads(str(cp['tables']))
            for actual, table_name in ((observations, 'observables'), (diagnostics, 'diagnostics')):
                expected_rows = [{key: '' if value is None else str(value) for key, value in row.items()} for row in tables[table_name]]
                if actual != expected_rows:
                    raise ValueError('CSV rows differ from the checkpointed numerical records')
            times = np.asarray(tables['times'])
            clock = float(cp['state_time'])
            if (times.shape != (frame+1,) or clock != times[-1] or float(cp['partial_step_clock_s']) != clock
                    or summary['simulation_time'] != clock or len(tables['frame_records']) != frame+1
                    or not np.isfinite(times).all() or times[0] != 0 or not np.all(np.diff(times) > 0)):
                raise ValueError('Incomplete checkpoint clock/table prefix')
            clock_bound = 32*np.finfo(float).eps*expected_steps*max(clock, plan['dt'])
            if np.max(np.abs(times-np.asarray(plan['frame_steps'][:frame+1])*plan['dt'])) > clock_bound:
                raise ValueError('Saved times do not belong to the unchanged complete protocol')
            for index, instant in enumerate(times):
                if (float(observations[index]['time']) != instant or float(diagnostics[index]['time']) != instant
                        or int(observations[index]['accepted_steps']) != plan['frame_steps'][index]
                        or float(observations[index]['max_damage']) != 0 or int(observations[index]['failed_particles']) != 0
                        or float(diagnostics[index]['min_J']) <= 0 or float(diagnostics[index]['mass_relative_drift']) != 0):
                    raise ValueError('CSV prefix clock, material or mass record is invalid')
            for name in final.files:
                checkpoint_name = name if name.startswith('history_') else 'state_'+name
                if checkpoint_name not in cp.files or not np.array_equal(final[name], cp[checkpoint_name]):
                    raise ValueError('Final-state file differs from the checkpoint: '+name)
            required_state = {'x', 'v', 'F', 'C', 'mass', 'volume', 'fibre_id', 'tau', 'time', 'valid',
                'boundary_impulse', 'external_impulse', 'boundary_work',
                'history_kappa', 'history_damage', 'history_failed', 'history_valid'}
            if set(final.files) != required_state:
                raise ValueError('Final-state archive does not contain the complete state')
            count = req['particles']
            scalar_fields = {'time': 'float64', 'valid': 'bool', 'boundary_work': 'float64'}
            for name in final.files:
                dtype = ('bool' if name in ('valid', 'history_failed', 'history_valid') else
                         'int32' if name == 'fibre_id' else 'float64')
                shape = (() if name in scalar_fields else (3,) if name in ('boundary_impulse', 'external_impulse')
                    else (count, 3, 3) if name in ('F', 'C', 'tau') else (count, 3) if name in ('x', 'v') else (count,))
                if final[name].shape != shape or final[name].dtype != np.dtype(dtype) or not np.isfinite(final[name]).all():
                    raise ValueError('Invalid final-state shape/dtype/value: '+name)
            for name in ('mass', 'volume', 'fibre_id'):
                if not np.array_equal(final[name], geometry[name]):
                    raise ValueError('Reference mass/volume/IDs changed')
            x, v, mass, F, C = (final[name] for name in ('x', 'v', 'mass', 'F', 'C'))
            if (x.shape != (count, 3) or v.shape != x.shape or F.shape != (count, 3, 3) or C.shape != F.shape
                    or any(not np.isfinite(value).all() for value in (x, v, F, C))
                    or np.any(np.linalg.det(F) <= 0) or not final['history_valid'].all()
                    or np.any(final['history_damage'] != 0) or np.any(final['history_failed'])):
                raise ValueError('Invalid or damaged independent-fibre final state')
            expected_fields = {'phase': np.ones(count, dtype=np.int8), 'yarn_id': geometry['fibre_id'],
                'family_id': geometry['family_id'],
                'grip_id': geometry['grip_masks'][0].astype(np.int8)+2*geometry['grip_masks'][1].astype(np.int8)}
            with np.load(folder/'particle_fields.npz', allow_pickle=False) as static:
                if set(static.files) != set(expected_fields):
                    raise ValueError('Changed static particle-field set')
                for name, reference in expected_fields.items():
                    if static[name].dtype != reference.dtype or not np.array_equal(static[name], reference):
                        raise ValueError('Changed static particle labels')
            with np.load(folder/'trajectory.npz', allow_pickle=False) as index:
                if not np.array_equal(index['time'], times) or not np.array_equal(index['body'], np.zeros(count, dtype=np.int32)):
                    raise ValueError('Trajectory index disagrees with complete prefix')
            for name, dtype in (('x', 'float64'), ('damage', 'float64'), ('speed', 'float64'), ('failed', 'uint8'), ('temperature', 'float32')):
                value = np.load(folder/f'trajectory_{name}.npy', mmap_mode='r', allow_pickle=False)
                arrays[name] = value
                expected_shape = (61, count, 3) if name == 'x' else (61, count)
                if value.shape != expected_shape or value.dtype != np.dtype(dtype):
                    raise ValueError('Trajectory does not retain all planned slots: '+name)
                for i in range(frame+1):
                    if _array_record(value[i]) != tables['frame_records'][i][name] or not np.isfinite(value[i]).all():
                        raise ValueError('Missing or changed complete prefix frame: '+name)
                    if name in ('damage', 'failed') and np.any(value[i] != 0):
                        raise ValueError('Elastic demonstration developed damage')
            for name, expected in (('x', x), ('damage', final['history_damage']), ('failed', final['history_failed']),
                                    ('speed', np.linalg.norm(v, axis=1))):
                if not np.array_equal(arrays[name][frame], expected):
                    raise ValueError('Trajectory endpoint differs from the complete final state')
            if not np.array_equal(arrays['x'][0], geometry['x']):
                raise ValueError('Initial particle coordinates changed')
            impulse = cp['totals_grip_impulse']+cp['totals_contact_impulse']
            if impulse.shape != (2, 3) or cp['totals_grip_work'].shape != (2,) or cp['totals_contact_work'].shape != (2,):
                raise ValueError('Malformed per-fibre impulse/work channels')
            momenta = np.vstack([np.sum(mass[geometry['fibre_id'] == i, None]*v[geometry['fibre_id'] == i], axis=0) for i in range(2)])
            scales = np.array([np.sum(np.abs(mass[geometry['fibre_id'] == i, None]*v[geometry['fibre_id'] == i]))
                + np.sum(np.abs(cp['totals_grip_impulse'][i]))+np.sum(np.abs(cp['totals_contact_impulse'][i])) for i in range(2)])
            errors = np.max(np.abs(momenta-impulse), axis=1)
            budgets = 1e-9*np.maximum(scales, 1e-30)
            if np.any(errors > budgets):
                raise ValueError('Independent per-fibre linear momentum balance failed')
            displacement = fixture_motion(clock, plan['protocol'])[0]
            grip_errors = [float(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-target, axis=1)))
                for mask, target in zip(geometry['grip_masks'], ([0., 0., 0.], [0., displacement, 0.]))]
            grip_budget = 64*np.finfo(float).eps*max(1, expected_steps)*plan['geometry']['length_m']
            if max(grip_errors) > grip_budget:
                raise ValueError('Reference grip particles did not follow prescribed displacements')
            particle_force = cp['contact_particle_force']
            if particle_force.shape != (count, 3) or not np.isfinite(particle_force).all():
                raise ValueError('Malformed cached contact force')
            force_scale = float(np.sum(np.abs(particle_force)))
            force_error = float(np.max(np.abs(particle_force.sum(0))))
            if force_error > 1e-10*max(force_scale, 1e-30):
                raise ValueError('Cached contact force violates action/reaction')
            final_row = observations[-1]
            kinetic = .5*np.sum(mass*np.sum(v*v, axis=1))+plan['grid']['dx_m']**2/8*np.sum(mass*np.sum(C*C, axis=(1, 2)))
            if not np.isclose(kinetic, float(final_row['kinetic_energy_J']), rtol=1e-12, atol=0):
                raise ValueError('Independent translational/APIC kinetic energy differs from the recorded value')
            for i, label in enumerate(('lower', 'upper')):
                for axis, coordinate in enumerate('xyz'):
                    if not np.isclose(momenta[i, axis], float(final_row[f'{label}_momentum_{coordinate}_Ns']), rtol=1e-12, atol=0):
                        raise ValueError('CSV per-fibre momentum differs from actual saved particles')
            saved_runtime = json.loads(str(cp['runtime']))
            return {'schema': 'fibre-pull-off-prefix-audit-v1', 'prefix_verified': True,
                'full_protocol_completed': completed, 'production_eligible': False, 'media_created': False,
                'frame': frame, 'saved_states_checked': frame+1, 'planned_saved_states': 61,
                'steps': expected_steps, 'physical_time_s': clock, 'target_duration_s': .03,
                'checkpoint_leaves': leaves, 'frame_records': tables['frame_records'], 'runtime': saved_runtime,
                'body_momentum_error_Ns': errors.tolist(), 'body_momentum_budget_Ns': budgets.tolist(),
                'momentum_budget_definition': '1e-9 times each body\'s absolute particle momentum plus absolute accumulated fixture/contact impulse, with a 1e-30 Ns scale floor.',
                'grip_position_error_m': grip_errors, 'grip_position_budget_m': grip_budget,
                'contact_force_balance_N': force_error, 'kinetic_energy_J': float(kinetic),
                'local_model_independently_resolved': False, 'full_physical_review_completed': False,
                'scope': 'Complete raw prefix, static labels, accepted state/clock and independently reconstructed momentum/kinetic energy. Local constitutive forces and full-period physical admissibility are not independently re-solved here; no media or calibrated response approval.'}
    finally:
        for value in arrays.values():
            value._mmap.close()


def seal(context, audit, cache=CACHE):
    cache = Path(cache)
    req = context['request']
    if (audit.get('prefix_verified') is not True or audit.get('production_eligible') is not False
            or audit.get('frame') != context['end_frame']):
        raise ValueError('Only a verified complete prefix can be sealed')
    if context['parent_key']:
        old = cache/'receipt.json'
        if digest(old) != context['parent_receipt_sha256']:
            raise ValueError('Parent receipt was changed during continuation')
        ancestry = cache/'receipts'/f'{context["parent_receipt_sha256"]}.json'
        ancestry.parent.mkdir(exist_ok=True)
        if ancestry.exists() and digest(ancestry) != context['parent_receipt_sha256']:
            raise ValueError('Historical receipt identity collision')
        shutil.copyfile(old, ancestry)
    records = payload_inventory(cache, req)
    required = {f'data/{req["model"]}/{name}' for name in RESULT_NAMES}
    if not required.issubset(records):
        raise ValueError('A continuation cache must contain every complete raw numerical file')
    quota = check_cache_quota(context['source'], sum(item['bytes'] for item in records.values())+1024**2)
    receipt = {'schema': 'fibre-pull-off-cache-v1', 'kind': 'accepted-prefix', 'never_resumable': False,
        'numerical_all_checks': True, 'source': context['source'], 'request': req,
        'cache_key': context['cache_key'], 'parent_cache_key': context['parent_key'],
        'parent_receipt_sha256': context['parent_receipt_sha256'], 'frame': context['end_frame'],
        'prefix_verified': True, 'production_eligible': False, 'media_created': False,
        'files': records, 'prefix_review': audit, 'repository_cache_admission': quota,
        'cache_scope': 'Complete raw continuation state and all planned mmap slots. Hosted cache can expire or be evicted; it is not permanent backup.',
        'cross_cpu_bitwise_equivalence_claimed': False}
    pending = cache/'receipt.pending.json'
    save(pending, receipt)
    pending.replace(cache/'receipt.json')
    if sum(item['bytes'] for item in records.values())+(cache/'receipt.json').stat().st_size > CACHE_LIMIT:
        raise ValueError('Complete cache including receipt exceeds its raw byte budget')
    receipt_sha = digest(cache/'receipt.json')
    save(LOGS/'receipt.json', receipt)
    output_values(receipt_sha=receipt_sha)
    return receipt


def seal_failure(context, cache=CACHE):
    """Preserve failed raw bytes without converting them into a valid continuation."""
    cache = Path(cache)
    actual_source = source_record()
    if numerical_identity(actual_source) != numerical_identity(context['source']):
        raise ValueError('Cannot bind a diagnostic snapshot to changed source')
    if not cache.is_dir():
        raise ValueError('No raw cache exists to preserve')
    prior_sha = ''
    if (cache/'receipt.json').is_file():
        prior_sha = digest(cache/'receipt.json')
        ancestry = cache/'receipts'/f'{prior_sha}.json'
        ancestry.parent.mkdir(exist_ok=True)
        shutil.copyfile(cache/'receipt.json', ancestry)
    records = payload_inventory(cache, context['request'], diagnostic=True)
    quota = check_cache_quota(actual_source, sum(item['bytes'] for item in records.values())+1024**2)
    receipt = {'schema': 'fibre-pull-off-cache-v1', 'kind': 'failed-diagnostic',
        'never_resumable': True, 'numerical_all_checks': False, 'prefix_verified': False,
        'source': actual_source, 'request': context['request'], 'files': records, 'repository_cache_admission': quota,
        'cache_key': context['cache_key']+'-failed', 'requested_end_frame': context['end_frame'],
        'prior_receipt_sha256': prior_sha, 'original_parent_key': context['parent_key'],
        'production_eligible': False, 'media_created': False,
        'scope': 'Unmodified failure-state files, including incomplete temporary files. Hashes identify raw bytes only, not valid arrays or a resumable numerical state. Temporary hosted cache is not permanent backup.'}
    pending = cache/'receipt.pending.json'
    save(pending, receipt)
    pending.replace(cache/'receipt.json')
    if sum(item['bytes'] for item in records.values())+(cache/'receipt.json').stat().st_size > CACHE_LIMIT:
        raise ValueError('Failed raw snapshot exceeds the complete-cache budget')
    save(LOGS/'failed-receipt.json', receipt)
    output_values(cache_key=receipt['cache_key'], receipt_sha=digest(cache/'receipt.json'))
    return receipt


def run(context):
    engine.LOGS = LOGS
    before = source_record()
    if numerical_identity(before) != numerical_identity(context['source']):
        raise ValueError('Source changed after preparation')
    if context['parent_key']:
        verify_restored(CACHE, context)
    if environment() != read(CACHE/'runtime/environment.json'):
        raise ValueError('Numerical environment changed after it was locked')
    free = shutil.disk_usage(ROOT).free
    save(LOGS/'disk-admission.json', {'free_bytes': free, 'required_free_bytes': 4*GIB,
        'cache_raw_limit_bytes': CACHE_LIMIT, 'package_limit_bytes': PACKAGE_LIMIT,
        'scope': 'Raw cache, checkpoint temporary files, cache compression and two-GiB disk buffer.'})
    if free < 4*GIB:
        raise RuntimeError('Insufficient runner disk space before full-scale prefix')
    started = time.monotonic()
    solve_seconds = remaining_solve_seconds(context, started)
    save(LOGS/'time-budget.json', {'solve_limit_seconds': solve_seconds, 'independent_timeout': True,
        'shared_job_deadline_monotonic': context['job_deadline_monotonic'], 'reserved_review_cache_evidence_seconds': 60*60})
    command = engine.command('solve-fibre-pull-off', ['timeout', '--signal=INT', '--kill-after=30s', str(solve_seconds)+'s',
        *command_line(context)], monitor=True)
    if time.monotonic()-started > solve_seconds+60:
        raise TimeoutError('Solve exceeded the independent controller budget')
    folder = result_folder(CACHE, context['request'])
    audit_seconds = int(min(15*60, context['job_deadline_monotonic']-time.monotonic()-45*60))
    if audit_seconds < 60:
        raise TimeoutError('Insufficient shared budget for independent prefix audit and cache save')
    engine.command('audit-fibre-pull-off', ['timeout', '--signal=INT', '--kill-after=30s', str(audit_seconds)+'s',
        sys.executable, 'tools/run_fibre_pull_off.py', 'audit'], monitor=True)
    audit = read(LOGS/'prefix-review.json')
    runtime = read(CACHE/'runtime/environment.json')
    expected_runtime = {'python': runtime['python'], **{key: runtime['packages'][key] for key in ('jax', 'jaxlib', 'numpy')}}
    if audit['runtime'] != expected_runtime:
        raise ValueError('Actual numerical runtime differs from its original lock')
    summary = read(folder/'summary.json')
    before_steps = 0
    before_solver_seconds = 0.
    if context['parent_key']:
        parent = read(CACHE/'receipt.json')
        before_steps = parent['prefix_review']['steps']
        before_solver_seconds = parent['prefix_review'].get('cumulative_solver_seconds', 0.)
    actual_steps = audit['steps']-before_steps
    seconds = float(summary['solver_seconds'])-before_solver_seconds
    memory = command['memory']
    if (actual_steps <= 0 or seconds <= 0 or not command['completed'] or memory['stopped_for_memory']
            or memory['samples'] < 1 or memory['peak_tree_uss_bytes'] > 12*GIB
            or memory['minimum_system_available_bytes'] < 2*GIB):
        raise ValueError('Resource prefix did not pass actual timing/memory acceptance')
    audit.update(cumulative_solver_seconds=float(summary['solver_seconds']), measured_segment_steps=actual_steps,
        measured_segment_solver_seconds=seconds, measured_seconds_per_step=seconds/actual_steps,
        early_linear_full_solver_estimate_seconds=seconds/actual_steps*context['request']['steps'],
        estimate_scope='Measured prefix only. The initial frame-2 prefix does not reach the reference contact gap and cannot estimate active MD/BEM cost. Continue the identical protocol from the exact parent to frame 10 to measure attachment; later bending, BEM relaxation, compilation and IO can change cost.',
        memory=memory,
        maximum_step_contact_force_N=float(summary['observables']['final']['maximum_step_contact_force_N']),
        minimum_step_contact_force_N=float(summary['observables']['final']['minimum_step_contact_force_N']))
    audit['contact_exercised'] = max(abs(audit['maximum_step_contact_force_N']), abs(audit['minimum_step_contact_force_N'])) > 0
    after = source_record()
    if after != before:
        raise ValueError('Source changed during the numerical prefix or audit')
    save(LOGS/'source-after.json', after)
    save(LOGS/'prefix-review.json', audit)
    return seal(context, audit)


def confirm_cache(context):
    receipt = read(CACHE/'receipt.json')
    diagnostic = receipt.get('kind') == 'failed-diagnostic'
    expected_key = context['cache_key']+('-failed' if diagnostic else '')
    if (os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != expected_key
            or os.environ.get('EXPECTED_KEY') != expected_key):
        raise ValueError('New exact snapshot was not confirmed by cache lookup')
    if payload_inventory(CACHE, context['request'], diagnostic=diagnostic) != receipt['files'] or receipt['cache_key'] != expected_key:
        raise ValueError('Raw data changed after sealing or cache identity differs')
    sha = digest(CACHE/'receipt.json')
    if os.environ.get('RECEIPT_SHA') != sha:
        raise ValueError('Saved receipt SHA differs from the executed prefix')
    save(LOGS/'cache-confirmation.json', {'cache_confirmed': True, 'cache_key': expected_key,
        'kind': receipt['kind'], 'never_resumable': diagnostic,
        'receipt_sha256': sha, 'production_eligible': False, 'media_created': False,
        'warning': 'Cache is evictable and is not a permanent copy of the raw history.'})
    output_values(receipt_sha=sha)


def package(*, cache=CACHE, logs=LOGS, delivery=DELIVERY, limit=PACKAGE_LIMIT):
    cache, logs, delivery = map(Path, (cache, logs, delivery))
    reserve = 1024**2
    if type(limit) is not int or limit <= reserve:
        raise ValueError('Package budget must include one MiB for its complete manifest')
    if delivery.exists():
        raise FileExistsError('A diagnostic delivery must not replace an earlier one')
    delivery.mkdir(parents=True)
    context = read(logs/'job.json') if (logs/'job.json').is_file() else None
    candidates = [(path, Path('records')/path.name) for path in sorted(logs.glob('*')) if path.is_file() and path.suffix in ('.json', '.log')]
    if (cache/'receipt.json').is_file():
        candidates.append((cache/'receipt.json', Path('cache-receipt.json')))
    if context:
        folder = result_folder(cache, context['request'])
        priority = ('plan.json', 'summary.json', 'status.json', 'observables.csv', 'diagnostics.csv',
            'contact-model.json', 'checkpoint.npz', 'rejected_checkpoint.npz', 'final_state_0.npz',
            'particle_fields.npz', 'trajectory.npz', 'trajectory_x.npy', 'trajectory_damage.npy',
            'trajectory_failed.npy', 'trajectory_speed.npy', 'trajectory_temperature.npy')
        candidates.extend((folder/name, Path('result')/context['request']['model']/name) for name in priority if (folder/name).is_file())
        if folder.is_dir():
            candidates.extend((path, Path('result')/context['request']['model']/path.name)
                for path in sorted(folder.iterdir()) if path.is_file() and path.name not in priority and path.name.endswith('.tmp'))
    files, omitted, used = {}, {}, 0
    for path, relative in candidates:
        if path.is_symlink():
            raise ValueError('Evidence must not be symlinks')
        record = file_record(path)
        if used+record['bytes'] > limit-reserve:
            omitted[relative.as_posix()] = {**record, 'reason': 'Whole file exceeds remaining bounded-artifact budget; no truncated array is substituted.'}
            continue
        target = delivery/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if file_record(target) != record or file_record(path) != record:
            raise ValueError('Evidence bytes changed while packaging')
        files[relative.as_posix()] = record
        used += record['bytes']
    manifest = {'schema': 'fibre-pull-off-records-v1', 'source': context['source'] if context else None,
        'files': files, 'omitted': omitted, 'copied_bytes': used, 'budget_bytes': limit,
        'cache_confirmation': read(logs/'cache-confirmation.json') if (logs/'cache-confirmation.json').is_file() else None,
        'production_eligible': False, 'media_created': False,
        'scope': 'Complete bounded record files only. Missing raw history is explicitly omitted, not reconstructed. Cache remains an evictable continuation aid.'}
    encoded = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    if len(encoded) > reserve or used+len(encoded) > limit:
        raise ValueError('Complete manifest exceeds the reserved package budget')
    (delivery/'manifest.json').write_bytes(encoded)
    if sum(path.stat().st_size for path in delivery.rglob('*') if path.is_file()) > limit:
        raise ValueError('Actual artifact exceeds the whole-package budget')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'lock', 'run', 'audit', 'seal-failure', 'confirm-cache', 'package'))
    parser.add_argument('--model', choices=MODELS, default='hertz')
    parser.add_argument('--stop-after-frame', type=int, default=2)
    parser.add_argument('--restore-key', default='')
    parser.add_argument('--parent-receipt', default='')
    args = parser.parse_args()
    if args.action == 'package':
        package()
        return
    if os.environ.get('GITHUB_ACTIONS') != 'true' or sys.platform != 'linux':
        parser.error('Numerical/cache operations are restricted to a Linux Actions runner')
    try:
        if args.action == 'prepare':
            prepare(args.model, args.stop_after_frame, args.restore_key, args.parent_receipt)
            return
        context = read(LOGS/'job.json')
        if args.action == 'verify':
            verify_restored(CACHE, context)
        elif args.action == 'lock':
            lock_environment(context)
        elif args.action == 'run':
            run(context)
        elif args.action == 'audit':
            report = audit_prefix(result_folder(CACHE, context['request']), context['request'], context['end_frame'])
            save(LOGS/'prefix-review.json', report)
        elif args.action == 'seal-failure':
            seal_failure(context)
        else:
            confirm_cache(context)
    except BaseException as error:
        save(LOGS/f'{args.action}-exception.json', {'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc(), 'production_eligible': False, 'media_created': False})
        raise


if __name__ == '__main__':
    main()
