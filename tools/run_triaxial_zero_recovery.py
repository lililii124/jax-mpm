"""Continue the intact theta-0 frame-20 state with frozen numerical source.

Only output storage/orchestration changes. Missing old particle trajectories
are explicit NaN/255 sentinels, never reconstructed observations. The state,
contact history, work integrals and original observation tables are intact.
Each bounded capsule contains the complete restart and new real saved frames.
No Actions cache is written; capsules must be backed up before cloud retirement.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ORIGINAL_COMMIT = 'b873d277711f5e706b2d57eea1b69f25e6375dc2'
ORIGINAL_ARCHIVE_SHA = 'b1dc4b9f9c2b9cc9aa3b4f36e906b5e44eff24f920e2bffd17d6703d53ff82d7'
ORIGINAL_CHECKPOINT_SHA = 'b0613cbcbcf9e6dbccd53e2b28f8196f8242946856869510a013f554505d5ba3'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '.local/triaxial-zero-recovery'
FROZEN = ROOT / '.local/triaxial-zero-frozen'
ARTIFACT_LIMIT = 192 * 1024**2
USS_LIMIT = 6 * 1024**3
CLOUD_LIMIT = 500 * 1024**2
FEATURE_REF = 'refs/heads/work/triaxial-zero-recovery'
CONTROLLER_PATHS = ('tools/run_triaxial_zero_recovery.py',
    '.github/workflows/engineering-previews.yml',
    '.github/workflows/triaxial-zero-recovery.yml',
    'tests/test_triaxial_zero_recovery_storage.py')
FIELDS = ('x', 'speed', 'damage', 'failed', 'temperature')
RESUME_METADATA = ('status.json', 'summary.json', 'contact-model.json', 'layout.json')
PREFIX = '.local/triaxial-segment-cache/data/formal/theta-00/hertz/'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': digest(path)}


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_member(name):
    if '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe archive member')
    return name


def verify_controller():
    """Bind the actual checkout, registered workflow override and copied template."""
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip()
    if commit != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF') != FEATURE_REF:
        raise ValueError('Controller must run at the exact dispatched SHA on the fixed recovery feature ref')
    files = {}
    for name in CONTROLLER_PATHS:
        path = ROOT / name
        raw = path.read_bytes()
        if path.is_symlink() or raw != subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']):
            raise ValueError('Diagnostic orchestration differs from its committed blob: ' + name)
        files[name] = record(path)
    return {'commit': commit, 'ref': FEATURE_REF, 'files': files,
            'scope': 'Storage/orchestration only; numerical source remains ' + ORIGINAL_COMMIT}


def artifact_usage(pages):
    """Bound only this recovery's rolling pool; report other evidence read-only."""
    active = [item for page in pages for item in page['artifacts'] if not item['expired']]
    if any(type(item['size_in_bytes']) is not int or item['size_in_bytes'] < 0 for item in active):
        raise ValueError('Invalid artifact accounting response')
    if len({item['id'] for item in active}) != len(active):
        raise ValueError('Duplicate artifact in repository accounting pages')
    pool = [item for item in active if re.fullmatch(
        r'theta-zero-capsule-f\d+-\d+-\d+', item['name'])]
    return {'repository_artifact_bytes_observed': sum(item['size_in_bytes'] for item in active),
        'repository_artifact_count_observed': len(active),
        'rolling_pool_used_bytes': sum(item['size_in_bytes'] for item in pool),
        'rolling_pool_artifact_ids': [item['id'] for item in pool],
        'rolling_pool_name_scope': 'Exact theta-zero-capsule-f<frame>-<run>-<attempt> names only; diagnostics and all pre-existing studies remain in the read-only global total.',
        'other_repository_artifacts': 'Read-only observation; neither gated, deleted nor retired by this controller.'}


def authenticate_parent_archive(archive, parent):
    """Authenticate GitHub transport wrapper before interpreting capsule data."""
    proof = {'outer_archive': record(archive), 'outer_crc_checked': True}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(safe_member(name) != name for name in names):
            raise ValueError('Ambiguous or unsafe parent transport inventory')
        if z.testzip() is not None:
            raise ValueError('Parent transport CRC mismatch')
        if 'capsule.zip' not in names:
            proof['encoding'] = 'bare original artifact or bare admitted capsule'
            return Path(archive), proof
        if set(names) != {'capsule.zip', 'receipt.json'}:
            raise ValueError('Unexpected members in GitHub capsule transport wrapper')
        receipt = json.loads(z.read('receipt.json'))
        item = receipt.get('capsule', {})
        if (receipt.get('schema') != 'triaxial-zero-cloud-capsule-v1'
                or receipt.get('never_resumable') is True
                or type(item.get('bytes')) is not int or not 0 < item['bytes'] <= ARTIFACT_LIMIT
                or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', ''))
                or z.getinfo('capsule.zip').file_size != item['bytes']):
            raise ValueError('Transport receipt is rejected or does not bind one complete admitted capsule')
        inner = parent / 'authenticated-capsule.zip'
        with z.open('capsule.zip') as source, inner.open('xb') as output:
            shutil.copyfileobj(source, output, 1024**2)
        if record(inner) != {'bytes': item['bytes'], 'sha256': item['sha256']}:
            raise ValueError('Inner capsule differs from exact transport receipt size/SHA')
        proof.update(encoding='GitHub outer ZIP with authenticated inner capsule',
            inner_capsule=item, receipt_end_frame=receipt.get('end_frame'))
        return inner, proof


def prepare(archive, expected_sha, end_frame, cloud_used, *, repository_bytes=0,
            repository_count=0, usage_record=None):
    """Verify immutable parent and stage a separate restart; never change it."""
    if BASE.exists():
        raise FileExistsError('Use a fresh hosted workspace; original evidence is retained')
    if digest(archive) != expected_sha:
        raise ValueError('Parent archive SHA differs from the admitted bytes')
    controller = verify_controller()
    if type(cloud_used) is not int or cloud_used < 0 or cloud_used + ARTIFACT_LIMIT > CLOUD_LIMIT:
        raise ValueError('Theta-zero recovery rolling pool plus full reserve exceeds 500 MiB; other repository evidence is unaffected')
    accounting = {'repository_artifact_bytes_observed': repository_bytes,
        'repository_artifact_count_observed': repository_count,
        'rolling_pool_used_bytes': cloud_used}
    if usage_record is not None:
        accounting = artifact_usage(read(usage_record))
        if (accounting['rolling_pool_used_bytes'] != cloud_used
                or accounting['repository_artifact_bytes_observed'] != repository_bytes
                or accounting['repository_artifact_count_observed'] != repository_count):
            raise ValueError('Artifact accounting arguments differ from read-only API receipt')
    BASE.mkdir(parents=True)
    save(BASE / 'artifact-accounting.json', {**accounting,
        'rolling_pool_limit_bytes': CLOUD_LIMIT, 'new_artifact_reserve_bytes': ARTIFACT_LIMIT,
        'repository_total_is_not_a_gate': True, 'deletion_performed': False,
        'usage_record': record(usage_record) if usage_record is not None else None})
    parent = BASE / 'parent'
    parent.mkdir()
    archive, transport = authenticate_parent_archive(archive, parent)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(safe_member(name) != name for name in names):
            raise ValueError('Ambiguous archive inventory')
        if expected_sha == ORIGINAL_ARCHIVE_SHA:
            manifest = json.loads(z.read('delivery-manifest.json'))
            for name, item in manifest['files'].items():
                with z.open(name) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                        raise ValueError('Original artifact member mismatch: ' + name)
            cp = manifest['state_files'][PREFIX + 'checkpoint.npz']
            if cp['sha256'] != ORIGINAL_CHECKPOINT_SHA or cp['complete_file'] is not True:
                raise ValueError('Original full frame-20 state is absent')
            with (parent / 'checkpoint.npz').open('xb') as output:
                for part in cp['parts']:
                    with z.open(part['path']) as source:
                        shutil.copyfileobj(source, output, 1024**2)
            if record(parent / 'checkpoint.npz') != {'bytes': cp['bytes'], 'sha256': cp['sha256']}:
                raise ValueError('Reassembled full checkpoint mismatch')
            plan = json.loads(z.read(PREFIX + 'plan.json'))
            # The original driver reads status before entering its runtime.
            # Preserve actual admitted metadata bytes, rather than synthesizing
            # a running/accepted phase from the existence of an NPZ alone.
            for name in RESUME_METADATA:
                (parent / name).write_bytes(z.read(PREFIX + name))
            job = json.loads(z.read('.local/triaxial-segment-ci/job.json'))
            environment = json.loads(z.read('.local/triaxial-segment-ci/environment-check.json'))['environment']
            provenance = {'origin_run': 40066438924, 'origin_artifact': 12556081018,
                'origin_archive_sha256': expected_sha, 'numerical_commit': ORIGINAL_COMMIT,
                'initial_missing_particle_frames': list(range(1, 20)),
                'scope': 'Original f20 checkpoint intact; original f1-19 particle arrays were not archived and remain unavailable.'}
            start = 20
            kwargs = job['request']['plan_kwargs']
        else:
            if 'failed-manifest.json' in names or 'capsule-manifest.json' not in names:
                raise ValueError('Failed or unclassified capsule is not an admitted parent')
            manifest = json.loads(z.read('capsule-manifest.json'))
            if (manifest.get('schema') != 'triaxial-zero-capsule-v1'
                    or manifest.get('restart_valid') is not True
                    or manifest.get('numerical_commit') != ORIGINAL_COMMIT):
                raise ValueError('Rejected or unrelated capsule cannot resume')
            if set(names) != set(manifest.get('files', {})) | {'capsule-manifest.json'}:
                raise ValueError('Inner capsule has missing or unlisted members')
            for name, item in manifest['files'].items():
                with z.open(safe_member(name)) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                        raise ValueError('Capsule member differs: ' + name)
            transport['all_inner_members_sha256_and_crc_checked'] = True
            for name in ('checkpoint.npz', 'plan.json', 'context.json'):
                with z.open('restart/' + name) as source, (parent / name).open('xb') as output:
                    shutil.copyfileobj(source, output, 1024**2)
            for name in RESUME_METADATA:
                with z.open('records/' + name) as source, (parent / name).open('xb') as output:
                    shutil.copyfileobj(source, output, 1024**2)
            prior = read(parent / 'context.json')
            plan, kwargs, environment = read(parent / 'plan.json'), prior['kwargs'], prior['environment']
            provenance, start = prior['provenance'], manifest['end_frame']
            if 'receipt_end_frame' in transport and transport['receipt_end_frame'] != start:
                raise ValueError('Outer receipt and inner capsule endpoint differ')
    if type(end_frame) is not int or not start < end_frame <= min(60, start + 5):
        raise ValueError('Continue one to five complete frames of the unchanged 60-frame plan')
    if start == 20 and expected_sha == ORIGINAL_ARCHIVE_SHA and end_frame != 21:
        raise ValueError('The first recovered advance must stop at frame 21 to measure actual cost')
    context = {'schema': 'triaxial-zero-recovery-v1', 'parent_archive_sha256': expected_sha,
        'controller': controller,
        'parent_transport': transport,
        'parent_checkpoint': record(parent / 'checkpoint.npz'), 'start_frame': start,
        'parent_metadata': {name: record(parent / name) for name in RESUME_METADATA},
        'end_frame': end_frame, 'kwargs': kwargs, 'environment': environment,
        'provenance': provenance, 'artifact_limit_bytes': ARTIFACT_LIMIT,
        'rolling_pool_used_bytes': cloud_used, 'rolling_pool_limit_bytes': CLOUD_LIMIT,
        'repository_artifact_bytes_observed': repository_bytes,
        'repository_artifact_count_observed': repository_count,
        'uss_limit_bytes': USS_LIMIT, 'production_eligible': False,
        'full_original_particle_history_available': False}
    save(parent / 'plan.json', plan)
    save(BASE / 'context.json', context)
    pins = environment['packages']
    if not {'jax', 'jaxlib', 'numpy', 'scipy', 'psutil', 'pillow'}.issubset(pins):
        raise ValueError('Missing exact original environment pins')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version)
           for name, version in pins.items()):
        raise ValueError('Unsafe original dependency pin')
    (BASE / 'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name, version in pins.items()))
    print(json.dumps({'prepared': True, 'start_frame': start, 'end_frame': end_frame,
                      'physical_plan_changed': False, 'first_prefix_cost_probe': start == 20}))


def verify_frozen(plan):
    if subprocess.check_output(['git', '-C', str(FROZEN), 'rev-parse', 'HEAD']).decode().strip() != ORIGINAL_COMMIT:
        raise ValueError('Numerical checkout must be the immutable admitted producer')
    subprocess.run(['git', '-C', str(FROZEN), 'diff', '--exit-code', 'HEAD', '--',
                    'mpm_jax', 'examples', 'tools', 'pyproject.toml'], check=True,
                   stdout=subprocess.DEVNULL)
    for name, expected in plan['source']['files'].items():
        path = FROZEN / safe_member(name)
        if (path.is_symlink() or digest(path) != expected
                or path.read_bytes() != subprocess.check_output(['git', '-C', str(FROZEN), 'show', f'{ORIGINAL_COMMIT}:{name}'])):
            raise ValueError('Frozen numerical source differs: ' + name)


def restore_resume_metadata(folder, plan, context):
    """Restore real parent status/summary only after checkpoint-bound admission."""
    import numpy as np
    parent = BASE / 'parent'
    for name in RESUME_METADATA:
        if record(parent / name) != context['parent_metadata'][name]:
            raise ValueError('Parent resume metadata bytes changed: ' + name)
    if record(folder / 'checkpoint.npz') != context['parent_checkpoint']:
        raise ValueError('Copied original checkpoint bytes changed')
    status, summary = read(parent / 'status.json'), read(parent / 'summary.json')
    frame = context['start_frame']
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    with np.load(folder / 'checkpoint.npz', allow_pickle=False) as cp:
        steps, instant = int(cp['aggregate'][0]), float(cp['state_time'])
        if (int(cp['frame']) != frame or str(cp['signature']) != signature
                or steps != plan['frame_steps'][frame] or not bool(cp['aggregate'][4])
                or not all(np.all(cp[key]) for key in ('state_valid', 'history_valid', 'contact_valid', 'contact_law_valid'))):
            raise ValueError('Parent metadata cannot admit a rejected or mismatched checkpoint')
    if (status.get('completed') is not False or status.get('probe_complete') is not False
            or status.get('segment_complete') is not True or status.get('coupling_valid') is not True
            or status.get('phase') != 'checkpoint-ready' or status.get('frame') != frame
            or status.get('segment_end_frame') != frame or status.get('frames') != plan['saved_states'] - 1
            or status.get('steps') != steps or status.get('target_steps') != plan['stop_steps']
            or status.get('signature') != signature):
        raise ValueError('Original saved status does not identify the exact resumable parent')
    if (summary.get('completed') is not False or summary.get('probe_complete') is not False
            or summary.get('segment_complete') is not True or summary.get('phase') != 'checkpoint-ready'
            or summary.get('segment_end_frame') != frame or summary.get('steps') != steps
            or summary.get('definition') != {**plan, 'contact_model': read(parent / 'contact-model.json')}
            or summary.get('source') != plan['source']
            or summary.get('layout') != read(parent / 'layout.json')
            or summary.get('simulation_time') != instant):
        raise ValueError('Original saved summary differs from checkpoint/plan endpoint')
    for name in RESUME_METADATA:
        shutil.copyfile(parent / name, folder / name)
        if record(folder / name) != context['parent_metadata'][name]:
            raise ValueError('Resume metadata copy changed bytes: ' + name)
    save(BASE / 'resume-metadata-proof.json', {'parent_checkpoint': context['parent_checkpoint'],
        'parent_metadata': context['parent_metadata'], 'frame': frame, 'steps': steps,
        'signature': signature, 'copied_exact_original_bytes': True,
        'old_particle_history_reconstructed': False, 'numerical_steps_executed': 0})


def resume_file_contract(source):
    """Enumerate every original resume file read, including dynamic mmaps."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    names, fields = set(), None
    for function in ('run_case', '_run_runtime', 'open_trajectory_arrays'):
        for node in ast.walk(functions[function]):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'dtypes' for t in node.targets):
                fields = tuple(ast.literal_eval(key) for key in node.value.keys)
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in ('read_text', 'read_bytes'):
                path = node.func.value
            elif node.func.attr in ('load', 'open_memmap') and node.args:
                path = node.args[0]
            else:
                continue
            if (isinstance(path, ast.BinOp) and isinstance(path.op, ast.Div)
                    and isinstance(path.left, ast.Name) and path.left.id == 'folder'):
                if isinstance(path.right, ast.Constant) and isinstance(path.right.value, str):
                    names.add(path.right.value)
                elif not (function == 'open_trajectory_arrays' and isinstance(path.right, ast.JoinedStr)):
                    raise ValueError('Unclassified original resume path; review its complete dependency before dispatch')
    if fields is None or set(fields) != set(FIELDS):
        raise ValueError('Original saved-field storage contract changed')
    names.update(f'trajectory_{field}.npy' for field in fields)
    expected = {'plan.json', 'checkpoint.npz', *RESUME_METADATA,
                *(f'trajectory_{field}.npy' for field in FIELDS)}
    if names != expected:
        raise ValueError('Original resume read-set changed: ' + repr(sorted(names)))
    return sorted(names)


def audit_resume_directory(folder, plan, context, *, source=None):
    """Check the full original read-set and actual CP/mmap endpoint before runtime.

    This validates storage and state schema without integrating or substituting
    a physical model. Original runtime still rebuilds and verifies grid/contact.
    """
    import numpy as np
    source = (FROZEN / 'examples/triaxial_failure.py').read_text() if source is None else source
    required = resume_file_contract(source)
    for name in required:
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Original resume dependency absent: ' + name)
    if read(folder / 'plan.json') != plan:
        raise ValueError('Restored resume plan changed')
    frame, count, saved = context['start_frame'], plan['particles'], plan['saved_states']
    arrays = {}
    with np.load(folder / 'checkpoint.npz', allow_pickle=False) as cp:
        required_leaves = {
            *('state_' + key for key in ('x', 'v', 'F', 'C', 'tau', 'mass', 'volume', 'fibre_id',
                'time', 'valid', 'boundary_impulse', 'external_impulse', 'boundary_work')),
            *('history_' + key for key in ('kappa', 'damage', 'failed', 'valid')),
            'contact_active', 'contact_law_valid', 'contact_radius', 'contact_segments', 'contact_valid',
            'total_impulse', 'clamp_work', 'grid_ke_work', 'aggregate', 'last_contact', 'initial_momentum',
            'frame', 'signature', 'solver_seconds', 'jax_version', 'tables'}
        if set(cp.files) != required_leaves:
            raise ValueError('Full original Hertz restart leaf inventory differs')
        for key in sorted(required_leaves):
            value = cp[key]
            if value.dtype.hasobject or (value.dtype.kind in 'biufc' and not np.isfinite(value).all()):
                raise ValueError('Invalid finite original restart leaf: ' + key)
        tables = json.loads(str(cp['tables']))
        if (any(len(tables[key]) != frame + 1 for key in ('observables', 'diagnostics', 'times'))
                or float(cp['state_time']) != tables['times'][-1]
                or str(cp['jax_version']) != context['environment']['packages']['jax']):
            raise ValueError('Complete original restart tables/runtime disagree')
        for key in ('x', 'v', 'F', 'C', 'tau', 'mass', 'volume', 'fibre_id'):
            trailing = (3, 3) if key in ('F', 'C', 'tau') else (3,) if key in ('x', 'v') else ()
            if cp['state_' + key].shape != (count, *trailing):
                raise ValueError('Restart particle state schema differs: ' + key)
        for key in ('kappa', 'damage', 'failed', 'valid'):
            if cp['history_' + key].shape != (count,):
                raise ValueError('Restart material history schema differs: ' + key)
        patches = plan['contact_regions']['max_patches']
        for key in ('contact_active', 'contact_law_valid', 'contact_radius', 'contact_segments'):
            shape = (patches, 2) if key == 'contact_segments' else (patches,)
            if cp[key].shape != shape:
                raise ValueError('Complete connected-contact restart schema differs: ' + key)
        expected = {'x': cp['state_x'], 'damage': cp['history_damage'], 'failed': cp['history_failed'],
                    'speed': np.linalg.norm(cp['state_v'], axis=1), 'temperature': np.full(count, 293.15, np.float32)}
        for field in FIELDS:
            array = np.load(folder / f'trajectory_{field}.npy', mmap_mode='r', allow_pickle=False)
            try:
                shape = (saved, count, 3) if field == 'x' else (saved, count)
                dtype = np.dtype(np.uint8 if field == 'failed' else np.float32 if field == 'temperature' else np.float64)
                if array.shape != shape or array.dtype != dtype or not np.array_equal(array[frame], expected[field]):
                    raise ValueError('Original resume mmap endpoint/schema differs: ' + field)
                arrays[field] = {'shape': list(shape), 'dtype': str(dtype), 'actual_checkpoint_frame': frame}
            finally:
                array._mmap.close()
    proof = {'schema': 'triaxial-zero-complete-resume-directory-v1', 'required_files': required,
        'original_driver_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'mmap_endpoints': arrays, 'frame': frame, 'steps': plan['frame_steps'][frame],
        'complete_restart_tables_checked': True, 'numerical_steps_executed': 0,
        'complete_checkpoint_leaves_checked': sorted(required_leaves),
        'scope': 'All original resume file reads and complete CP/mmap endpoint checked before runtime; grid/contact physics and first solver step remain original runtime checks.'}
    save(BASE / 'resume-directory-proof.json', proof)
    return proof


def worker():
    """Numerical worker: exact original source, parameters, checkpoint and law."""
    context, plan = read(BASE / 'context.json'), read(BASE / 'parent/plan.json')
    if verify_controller() != context['controller']:
        raise ValueError('Controller changed after recovery admission')
    verify_frozen(plan)
    if platform.python_version() != '3.12.13' or platform.system() != 'Linux':
        raise ValueError('Original Python/Linux runtime required')
    for name, version in context['environment']['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Original scientific environment changed: ' + name)
    sys.path.insert(0, str(FROZEN))
    from examples import triaxial_failure as driver
    import numpy as np
    if Path(driver.__file__).resolve() != FROZEN / 'examples/triaxial_failure.py':
        raise ValueError('Numerical driver import was shadowed')
    geometry, rebuilt = driver.plan_case(**context['kwargs'])
    if rebuilt != plan:
        raise ValueError('Rebuilt exact reference geometry/plan differs from archived original')
    output = BASE / 'work'
    folder = driver.output_directory(output, plan)
    folder.mkdir(parents=True)
    save(folder / 'plan.json', plan)
    shutil.copyfile(BASE / 'parent/checkpoint.npz', folder / 'checkpoint.npz')
    restore_resume_metadata(folder, plan, context)
    count, saved, start = plan['particles'], plan['saved_states'], context['start_frame']
    with np.load(folder / 'checkpoint.npz', allow_pickle=False) as cp:
        if (int(cp['frame']) != start or not bool(cp['aggregate'][4])
                or int(cp['aggregate'][0]) != plan['frame_steps'][start]
                or not all(np.all(cp[key]) for key in ('state_valid', 'history_valid', 'contact_valid', 'contact_law_valid'))):
            raise ValueError('Original restart state is rejected or has an inconsistent endpoint')
        for field in FIELDS:
            shape = (saved, count, 3) if field == 'x' else (saved, count)
            dtype = np.uint8 if field == 'failed' else np.float32 if field == 'temperature' else np.float64
            array = np.lib.format.open_memmap(folder / f'trajectory_{field}.npy', mode='w+', dtype=dtype, shape=shape)
            # Explicitly unavailable history cannot masquerade as physical data.
            for frame in range(saved):
                array[frame] = 255 if field == 'failed' else np.nan
            if field == 'x':
                array[0], array[start] = geometry['x'], cp['state_x']
            elif field in ('damage', 'failed'):
                array[0], array[start] = 0, cp['history_' + field]
            elif field == 'speed':
                array[0], array[start] = 0., np.linalg.norm(cp['state_v'], axis=1)
            else:
                array[0] = array[start] = 293.15
            array.flush()
            array._mmap.close()
    save(BASE / 'history-availability.json', {'real_particle_frames_this_worker': [0, start],
        'unavailable_prefix_frames': list(range(1, start)),
        'sentinels': {'floating_fields': 'NaN', 'failed_uint8': 255},
        'integration_uses': 'complete original mechanical and contact checkpoint only',
        'old_observation_and_diagnostic_tables': 'complete inside checkpoint',
        'full_original_particle_history_available': False})
    audit_resume_directory(folder, plan, context)
    driver.run_case(geometry, plan, output, resume=True, stop_after_frame=context['end_frame'])
    verify_frozen(plan)


def guarded_command(arguments, log_path, memory_path, requested_frames, *,
                    wall_seconds=110 * 60, sample_seconds=1.):
    """Fail closed on unavailable monitoring and reap the owned process group."""
    import psutil
    started = time.monotonic()
    process, peak, samples, reason, error = None, 0, 0, None, None
    owned = {}
    cleanup_errors = []
    memory = {'uss_limit_bytes': USS_LIMIT, 'real_frames_requested': requested_frames,
              'scope': 'Monitor plus owned worker process tree; owned Linux session is killed in finally.'}
    try:
        if psutil.virtual_memory().available < 7 * 1024**3:
            reason = 'insufficient startup memory headroom'
            raise RuntimeError(reason)
        launch = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                  if platform.system() == 'Windows' else {'start_new_session': True})
        with Path(log_path).open('x', encoding='utf-8') as log:
            process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'JAX_PLATFORMS': 'cpu', 'JAX_ENABLE_X64': 'true',
                     'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'OPENBLAS_NUM_THREADS': '1',
                     'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '2'}, **launch)
            # Failure here is also covered by the process-group finally cleanup.
            parent = psutil.Process(process.pid)
            owned[parent.pid] = parent.create_time()
            while process.poll() is None:
                try:
                    tree = [parent, *parent.children(recursive=True)]
                    uss = psutil.Process().memory_full_info().uss
                    for child in tree:
                        try:
                            owned[child.pid] = child.create_time()
                            uss += child.memory_full_info().uss
                        except psutil.NoSuchProcess:
                            if child.pid == process.pid and process.poll() is None:
                                raise RuntimeError('live worker disappeared from resource monitor')
                    peak, samples = max(peak, uss), samples + 1
                    available = psutil.virtual_memory().available
                    reason = ('process-tree USS > 6 GiB' if uss > USS_LIMIT else
                              'system available < 2 GiB' if available < 2 * 1024**3 else
                              'segment wall budget exhausted' if time.monotonic() - started > wall_seconds else None)
                    if reason:
                        break
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    reason = 'resource monitor lost a live worker'
                    raise RuntimeError(reason)
                except psutil.AccessDenied:
                    reason = 'resource monitor access denied'
                    raise RuntimeError(reason)
                time.sleep(sample_seconds)
    except BaseException as exception:
        error = {'type': type(exception).__name__, 'message': str(exception)}
        reason = reason or 'guard exception; failing closed'
    finally:
        if process is not None:
            if platform.system() != 'Windows':
                try:
                    # Session was created by this Popen. Kill remaining children
                    # even when the direct worker already exited successfully.
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exception:
                    cleanup_errors.append(str(exception))
            else:
                # Linux is the admitted numerical runtime; this branch permits
                # small local guard checks without leaving owned children alive.
                for pid, created in reversed(list(owned.items())):
                    try:
                        child = psutil.Process(pid)
                        if child.create_time() == created:
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                    except Exception as exception:
                        cleanup_errors.append(str(exception))
            if process.poll() is None:
                try:
                    process.kill()
                except OSError as exception:
                    cleanup_errors.append(str(exception))
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                reason = reason or 'worker cleanup wait exceeded five seconds'
                try:
                    process.kill()
                except OSError as exception:
                    cleanup_errors.append(str(exception))
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append('worker did not reap after second bounded wait')
                except OSError as exception:
                    cleanup_errors.append(str(exception))
            except OSError as exception:
                cleanup_errors.append(str(exception))
        memory.update(peak_process_tree_uss_bytes=peak, samples=samples,
            stop_reason=reason, guard_exception=error, cleanup_errors=cleanup_errors,
            returncode=process.poll() if process is not None else None,
            wall_seconds=time.monotonic() - started)
        save(memory_path, memory)
    if reason or error or cleanup_errors or memory['returncode'] != 0:
        raise RuntimeError('Segment failed closed; original parent retained; inspect resource diagnostic')
    return memory


def run():
    """Enforce a 6 GiB process-tree USS ceiling independently of the solver."""
    context = read(BASE / 'context.json')
    if verify_controller() != context['controller']:
        raise ValueError('Controller changed before advance')
    return guarded_command([sys.executable, __file__, 'worker'], BASE / 'solver.log',
        BASE / 'memory.json', context['end_frame'] - context['start_frame'])


def package():
    """Stream complete checkpoint plus real new frames into one bounded ZIP."""
    import numpy as np
    context, plan = read(BASE / 'context.json'), read(BASE / 'parent/plan.json')
    if verify_controller() != context['controller']:
        raise ValueError('Controller changed before capsule sealing')
    folder = BASE / 'work/formal/theta-00/hertz'
    destination = BASE / 'delivery'
    destination.mkdir(exist_ok=False)
    archive = destination / 'capsule.zip'
    files = {}
    status = read(folder / 'status.json')
    start, end = context['start_frame'], status.get('frame')
    restart_valid = bool(status.get('coupling_valid') is True and end == context['end_frame']
                         and status.get('phase') in ('checkpoint-ready', 'completed'))
    if not restart_valid:
        raise ValueError('No valid requested endpoint; retain diagnostic logs and original parent')
    with np.load(folder / 'checkpoint.npz', allow_pickle=False) as cp:
        if (int(cp['frame']) != end or not bool(cp['aggregate'][4])
                or int(cp['aggregate'][0]) != plan['frame_steps'][end]
                or not all(np.all(cp[key]) for key in ('state_valid', 'history_valid', 'contact_valid', 'contact_law_valid'))):
            raise ValueError('Rejected or mismatched restart state is not resumable')
        rows = json.loads(str(cp['tables']))['observables']
    rupture = rupture_evidence(rows)
    save(BASE / 'rupture-evidence.json', rupture)
    save(BASE / 'restart-context.json', context)
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        selected = {'restart/checkpoint.npz': folder / 'checkpoint.npz',
            'restart/plan.json': BASE / 'parent/plan.json', 'restart/context.json': BASE / 'restart-context.json',
            'records/history-availability.json': BASE / 'history-availability.json',
            'records/rupture-evidence.json': BASE / 'rupture-evidence.json',
            'records/memory.json': BASE / 'memory.json', 'records/solver.log': BASE / 'solver.log'}
        if (BASE / 'artifact-accounting.json').exists():
            selected['records/artifact-accounting.json'] = BASE / 'artifact-accounting.json'
        if (BASE / 'resume-metadata-proof.json').exists():
            selected['records/resume-metadata-proof.json'] = BASE / 'resume-metadata-proof.json'
        if (BASE / 'resume-directory-proof.json').exists():
            selected['records/resume-directory-proof.json'] = BASE / 'resume-directory-proof.json'
        for name in ('summary.json', 'status.json', 'observables.csv', 'diagnostics.csv',
                     'layout.json', 'contact-model.json', 'contact-metrics.json', 'particle_fields.npz'):
            selected['records/' + name] = folder / name
        for name, path in selected.items():
            files[name] = record(path)
            z.write(path, name)
            if archive.stat().st_size > ARTIFACT_LIMIT - 1024**2:
                raise ValueError('Complete restart/records exceed hard artifact bound; never upload a partial capsule')
        # The first recovered capsule also preserves the exact f20 endpoint.
        first = start if start == 20 else start + 1
        for field in FIELDS:
            array = np.load(folder / f'trajectory_{field}.npy', mmap_mode='r', allow_pickle=False)
            try:
                for frame in range(first, end + 1):
                    name = f'frames/frame-{frame:02d}/{field}.npy'
                    with z.open(name, 'w', force_zip64=True) as stream:
                        np.lib.format.write_array(stream, np.asarray(array[frame]), allow_pickle=False)
                    with z.open(name) as stream:
                        h = hashlib.file_digest(stream, 'sha256').hexdigest()
                    files[name] = {'bytes': z.getinfo(name).file_size, 'sha256': h}
                    if archive.stat().st_size > ARTIFACT_LIMIT - 1024**2:
                        raise ValueError('New real frames plus full restart exceed 192 MiB; no partial upload is allowed')
            finally:
                array._mmap.close()
        readme = (
            'Theta-zero recovery capsule: complete restart and actual new particle frames.\n'
            f"Read-only global artifact observation: {context.get('repository_artifact_bytes_observed')} bytes, "
            f"{context.get('repository_artifact_count_observed')} active artifacts. This is not an admission gate.\n"
            f"This recovery's rolling capsule pool before this upload: {context.get('rolling_pool_used_bytes')} bytes; "
            f'pool cap {CLOUD_LIMIT} bytes; new-artifact reserve {ARTIFACT_LIMIT} bytes.\n'
            'Pool names match exactly theta-zero-capsule-f<frame>-<run>-<attempt>. '
            'Other repository artifacts are observed read-only and never deleted or retired by this controller.\n'
            'Original numerical source/plan/material/boundaries are unchanged. USS cap is 6 GiB.\n'
            'Original f1-19 particle trajectories are unavailable; no synthetic histories are supplied. '
            'Complete checkpoint history and original observation/work tables remain intact.\n'
            'No macroscopic rupture, physical dissipation or production admission is implied.\n')
        z.writestr('README.txt', readme)
        files['README.txt'] = {'bytes': len(readme.encode()),
            'sha256': hashlib.sha256(readme.encode()).hexdigest()}
        manifest = {'schema': 'triaxial-zero-capsule-v1', 'numerical_commit': ORIGINAL_COMMIT,
            'controller': context['controller'],
            'repository_artifact_bytes_observed': context.get('repository_artifact_bytes_observed'),
            'repository_artifact_count_observed': context.get('repository_artifact_count_observed'),
            'rolling_pool_used_bytes': context.get('rolling_pool_used_bytes'),
            'rolling_pool_limit_bytes': CLOUD_LIMIT,
            'parent_archive_sha256': context['parent_archive_sha256'], 'parent_checkpoint': context['parent_checkpoint'],
            'start_frame': start, 'end_frame': end, 'real_particle_frames': list(range(first, end + 1)),
            'restart_valid': True, 'files': files, 'physical_plan_changed': False,
            'full_original_particle_history_available': False, 'production_eligible': False,
            'raw_history_gap': context['provenance'],
            'scope': 'Exact restart and actual new saved fields. No synthetic prefix; no macroscopic rupture admission.'}
        z.writestr('capsule-manifest.json', json.dumps(manifest, indent=2, allow_nan=False))
    if archive.stat().st_size > ARTIFACT_LIMIT - 1024**2:
        raise ValueError('Final complete capsule exceeds 192 MiB; refuse upload')
    receipt = {'schema': 'triaxial-zero-cloud-capsule-v1', 'capsule': record(archive),
        'artifact_limit_bytes': ARTIFACT_LIMIT, 'rolling_pool_limit_bytes': CLOUD_LIMIT,
        'rolling_pool_used_bytes_before_upload': context.get('rolling_pool_used_bytes'),
        'repository_artifact_bytes_observed': context.get('repository_artifact_bytes_observed'),
        'repository_artifact_count_observed': context.get('repository_artifact_count_observed'),
        'parent_archive_sha256': context['parent_archive_sha256'], 'end_frame': end,
        'production_eligible': False, 'permanent_backup_verified': False,
        'next_action': 'Download, verify every member and whole archive, keep permanent local backup, then admit next prefix separately.'}
    save(destination / 'receipt.json', receipt)
    print(json.dumps(receipt))


def rupture_evidence(rows):
    """Report sufficient axial-cut evidence separately from first-point damage.

    The energy residual is explicitly not named fracture dissipation: APIC
    transfer and contact curvature can contribute, and no exact balance is
    claimed by the underlying source. All raw work/energy traces are retained.
    """
    peak = 0.
    candidates = []
    initial = rows[0]
    energy = []
    for row in rows:
        peak = max(peak, row['mean_grip_tensile_force_N'])
        residual = (row['actual_clamp_work_J'] - initial['actual_clamp_work_J']
            - (row['kinetic_energy_J'] - initial['kinetic_energy_J'])
            - (row['elastic_energy_J'] - initial['elastic_energy_J'])
            - (row['contact_potential_J'] - initial['contact_potential_J']))
        energy.append({'time_s': row['time'], 'work_minus_stored_and_kinetic_J': residual})
        if (row['active_fibre_x_support_cuts'] > 0
                and row['sampled_section_graph_connects_grips'] is False
                and row['failed_particles'] > 0 and peak > 0
                and row['mean_grip_tensile_force_N'] <= .5 * peak):
            candidates.append({'time_s': row['time'], 'prior_mean_force_peak_N': peak,
                'mean_force_N': row['mean_grip_tensile_force_N'],
                'force_drop_fraction': 1 - row['mean_grip_tensile_force_N'] / peak,
                'sufficient_axial_support_cuts': row['active_fibre_x_support_cuts'],
                'work_minus_stored_and_kinetic_J': residual})
    return {'schema': 'triaxial-zero-rupture-evidence-v1',
        'sampled_rupture_candidate': candidates[0] if candidates else None,
        'criteria': 'Positive axial support cut, disconnected section graph, material failure, and >=50% drop from prior sampled mean force peak; no criterion is relaxed.',
        'energy_residual_history': energy, 'production_eligible': False,
        'fracture_dissipation_verified': False,
        'limitations': 'Section graph is approximate; positive x-support cuts are sufficient axial separation evidence. Residual includes constitutive dissipation, numerical transfer and omitted contact-curvature work. Independent particle-state/energy review is still required; no complete macroscopic failure claim follows automatically.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'run', 'package'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha')
    parser.add_argument('--end-frame', type=int, default=21)
    parser.add_argument('--pool-used-bytes', '--cloud-used-bytes', dest='pool_used_bytes', type=int)
    parser.add_argument('--repository-artifact-bytes', type=int, default=0)
    parser.add_argument('--repository-artifact-count', type=int, default=0)
    parser.add_argument('--usage-record', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.archive, args.archive_sha, args.end_frame, args.pool_used_bytes,
            repository_bytes=args.repository_artifact_bytes,
            repository_count=args.repository_artifact_count, usage_record=args.usage_record)
    elif args.action == 'worker':
        worker()
    elif args.action == 'run':
        run()
    else:
        package()
