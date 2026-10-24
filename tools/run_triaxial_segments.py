"""Three exact-cache numerical segments and a separate review/render job."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_triaxial_study as study

engine = study.engine
save, read_json, digest = study.save, study.read_json, study.digest
CACHE = ROOT / '.local/triaxial-segment-cache'
LOGS = ROOT / '.local/triaxial-segment-ci'
DELIVERY = ROOT / '.local/triaxial-segment-delivery'
CONTEXT = LOGS / 'job.json'
CACHE_BUDGET = 8_000_000_000
PYTHON_VERSION = '3.12.13'
RESULT_NAMES = {'plan.json', 'summary.json', 'status.json', 'layout.json', 'contact-model.json',
    'particle_fields.json', 'particle_fields.npz', 'checkpoint.npz', 'final_state_0.npz',
    'trajectory.npz', 'observables.csv', 'diagnostics.csv', 'contact-metrics.json',
    'trajectory_x.npy', 'trajectory_damage.npy', 'trajectory_failed.npy',
    'trajectory_speed.npy', 'trajectory_temperature.npy'}
RUNTIME_NAMES = {'request.json', 'source.json', 'environment.json', 'requirements.txt',
                 'geometry.json', 'geometry.npz', 'plan.json', 'numerical-layout.json'}


def request(mode='tiny', theta=0, law='hertz', load_transits=10., size='24x16', contact_regions='single'):
    case = study.selection(theta, law, load_transits, size, contact_regions)
    if mode not in ('tiny', 'formal'):
        raise ValueError('Unknown segment mode')
    if mode == 'tiny' and (theta != 0 or load_transits != 10. or size != '24x16'):
        raise ValueError('Tiny uses theta=0 and fixed smoke geometry; leave size/transits at their defaults')
    kwargs = dict(theta=theta, law=law, smoke=mode == 'tiny', probe_steps=0,
        frames=4 if mode == 'tiny' else 60, max_strain=1e-7 if mode == 'tiny' else .02,
        dx=75e-6, padding_cells=6, max_nodes=case['max_nodes'], coverage_skin=.0005,
        search_window=6, work_of_adhesion=.1, particle_spacing=45e-6, grid_storage='tube',
        load_transits=.03 if mode == 'tiny' else float(load_transits),
        dimensions=[.006, .004] if mode == 'tiny' else case['dimensions_m'],
        grip_length=.001 if mode == 'tiny' else .003, diameter=235e-6, pitch=.001,
        crimp=140e-6, clearance=2e-6, young=1e10, poisson=.25, density=1500.,
        tensile_strength=1e8, fracture_energy=100., packing=1., max_particles=1_000_000)
    if contact_regions == 'connected':
        kwargs.update(contact_regions='connected', contact_capacity=16384,
                      max_contact_patches=4096, patch_iterations=128)
    return {'schema': 'triaxial-segment-request-v1', 'mode': mode, 'case': case,
            'plan_kwargs': kwargs, 'stops': [1, 2, 4] if mode == 'tiny' else [20, 40, 60],
            'production_eligible': False}


def cache_key(source, mode, frame):
    for name in ('commit', 'run_id', 'run_attempt'):
        if not re.fullmatch(r'[A-Za-z0-9]+', str(source[name])):
            raise ValueError('Unsafe cache identity')
    return f"triaxial-segments-v1-{source['commit']}-{source['run_id']}-{source['run_attempt']}-{mode}-f{frame:02d}"


def result_folder(cache, req):
    kw = req['plan_kwargs']
    return Path(cache)/'data'/('smoke' if kw['smoke'] else 'formal')/f"theta-{kw['theta']:02d}"/kw['law']


def file_record(path):
    before = Path(path).stat()
    value = {'bytes': before.st_size, 'sha256': digest(path)}
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing')
    return value


def inventory(cache, req, excluded=None, *, diagnostic=False):
    cache = Path(cache)
    prefix = result_folder(cache, req).relative_to(cache)
    records = {}
    for path in sorted(cache.rglob('*')):
        if path.is_symlink():
            raise ValueError('Cache must not contain symlinks')
        if not path.is_file():
            continue
        name = path.relative_to(cache).as_posix()
        if name == excluded:
            continue
        results = RESULT_NAMES | ({'failure.json'} if diagnostic else set())
        runtime = RUNTIME_NAMES
        if diagnostic:
            results = results | {name+'.tmp' for name in results}
            runtime = runtime | {name+'.tmp' for name in runtime}
        allowed = ((path.parent == cache/'runtime' and path.name in runtime)
            or (path.parent == cache/prefix and path.name in results)
            or (path.parent == cache/'receipts' and re.fullmatch(r'(frame|failed-frame)-\d{2}.json', path.name)))
        if not allowed:
            raise ValueError(f'Unexpected cache payload: {name}')
        records[name] = file_record(path)
    return records


def cpu_record():
    path = Path('/proc/cpuinfo')
    fields = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition(':')
            if sep and key.strip() in ('vendor_id', 'model name', 'cpu family', 'model', 'stepping', 'flags'):
                fields.setdefault(key.strip(), value.strip())
    return {'machine': platform.machine(), 'logical_cpus': os.cpu_count(), 'fields': fields,
            'kernel': platform.release(), 'scope': 'Recorded per job; CPU identity is not assumed equal across hosted jobs.'}


def environment():
    versions = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r'[-_.]+', '-', distribution.metadata['Name']).lower()
        if name in ('pip', 'setuptools', 'wheel', 'jax-mpm'):
            continue
        version = distribution.version
        if not re.fullmatch(r'[a-z0-9-]+', name) or not re.fullmatch(r'[A-Za-z0-9.!+_-]+', version):
            raise ValueError('Unsafe environment pin')
        if name in versions and versions[name] != version:
            raise ValueError('Ambiguous installed distribution')
        versions[name] = version
    if not {'jax', 'jaxlib', 'numpy', 'scipy', 'psutil', 'pillow'}.issubset(versions):
        raise ValueError('Incomplete scientific environment')
    os_release = platform.freedesktop_os_release() if hasattr(platform, 'freedesktop_os_release') and platform.system() == 'Linux' else {}
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'system': platform.system(), 'machine': platform.machine(),
            'os_id': os_release.get('ID'), 'os_version_id': os_release.get('VERSION_ID'),
            'packages': dict(sorted(versions.items()))}


def write_geometry(folder, geometry):
    import numpy as np
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Reference arrays cannot contain Python objects')
            key = f'a{len(arrays):04d}'
            arrays[key] = value
            return {'array': key, 'dtype': value.dtype.str, 'shape': list(value.shape)}
        if isinstance(value, dict):
            return {'mapping': {key: pack(item) for key, item in value.items()}}
        if isinstance(value, (list, tuple)):
            return {'sequence': [pack(item) for item in value]}
        if isinstance(value, np.generic):
            return value.item()
        return value

    description = pack(geometry)
    np.savez_compressed(Path(folder)/'geometry.npz', **arrays)
    save(Path(folder)/'geometry.json', {'structure': description, 'archive': file_record(Path(folder)/'geometry.npz'),
        'scope': 'Exact first-job reference arrays and metadata; resumed jobs do not regenerate particle sampling.'})


def read_geometry(folder):
    import numpy as np
    description = read_json(Path(folder)/'geometry.json')
    if file_record(Path(folder)/'geometry.npz') != description['archive']:
        raise ValueError('Reference geometry archive changed')
    with np.load(Path(folder)/'geometry.npz', allow_pickle=False) as archive:
        used = set()

        def unpack(value):
            if isinstance(value, dict) and 'array' in value:
                key = value['array']
                array = archive[key]
                if list(array.shape) != value['shape'] or array.dtype.str != value['dtype'] or key in used:
                    raise ValueError('Reference array descriptor mismatch')
                used.add(key)
                return array
            if isinstance(value, dict) and 'mapping' in value:
                return {key: unpack(item) for key, item in value['mapping'].items()}
            if isinstance(value, dict) and 'sequence' in value:
                return [unpack(item) for item in value['sequence']]
            return value

        geometry = unpack(description['structure'])
        if used != set(archive.files):
            raise ValueError('Unlisted reference arrays')
    return geometry


def cache_estimate(plan):
    # Two complete state archives, the reference geometry, static labels and
    # binary failed-history index fit below this uncompressed per-particle reserve.
    per_snapshot = int(plan['trajectory_estimated_bytes']) + int(plan['particles'])*864 + 32*1024**2
    total = 3*per_snapshot
    if total > CACHE_BUDGET:
        raise ValueError(f'Three conservative raw snapshots need {total} bytes, exceeding {CACHE_BUDGET}; no paid cache expansion or physical reduction is allowed')
    return {'estimated_snapshot_bytes': per_snapshot, 'estimated_three_snapshot_bytes': total,
            'workflow_cache_budget_bytes': CACHE_BUDGET,
            'scope': 'Conservative pre-solve raw-data estimate; final complete payloads are independently counted at each seal. Cache compression is not used to justify this budget.'}


def disk_admission(plan, *, stage, free_bytes, raw_already_present=False):
    snapshot = cache_estimate(plan)['estimated_snapshot_bytes']
    # Complete cache archive may coexist with the raw tree. Final review uses a
    # separate data copy; 96 bytes/point/frame conservatively reserves VTK payload.
    growth = 0 if raw_already_present else snapshot
    render = (snapshot + plan['particles']*plan['saved_states']*96 + 512*1024**2) if stage == 4 else 0
    archive = snapshot if stage <= 3 else 0
    required = growth + archive + render + 2*1024**3
    record = {'free_bytes': int(free_bytes), 'required_free_bytes': int(required),
        'raw_growth_bytes': growth, 'cache_archive_reserve_bytes': archive,
        'final_copy_and_render_reserve_bytes': render, 'unallocated_margin_bytes': 2*1024**3,
        'passed': free_bytes >= required, 'stage': stage,
        'scope': 'Preflight disk budget for whole raw data, cache compression temporary, final copy and VTK/media; no deletion or reduced sampling to pass.'}
    if not record['passed']:
        raise ValueError(f"Insufficient free disk: {int(free_bytes)} < {int(required)} bytes")
    return record


def check_endpoint(folder, plan, frame, *, final):
    import numpy as np
    folder = Path(folder)
    summary, status = [read_json(folder/name) for name in ('summary.json', 'status.json')]
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    if not (read_json(folder/'plan.json') == plan and status.get('signature') == signature
        and status.get('frame') == frame and status.get('steps') == plan['frame_steps'][frame]
        and status.get('coupling_valid') is True and summary.get('completed') is final
        and status.get('completed') is final and status.get('probe_complete') is False
        and summary.get('probe_complete') is False and summary.get('segment_end_frame') == frame
        and status.get('segment_end_frame') == frame
        and summary.get('segment_complete') is (not final) and status.get('segment_complete') is (not final)
        and status.get('phase') == ('completed' if final else 'checkpoint-ready')):
        raise ValueError('Endpoint is not a valid completed segment of the unchanged plan')
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as checkpoint:
        if not (int(checkpoint['frame']) == frame and str(checkpoint['signature']) == signature
                and bool(checkpoint['aggregate'][4]) and int(checkpoint['aggregate'][0]) == plan['frame_steps'][frame]):
            raise ValueError('Checkpoint does not identify a valid segment endpoint')
        if not all(bool(np.all(checkpoint[name])) for name in
                   ('state_valid', 'history_valid', 'contact_law_valid', 'contact_valid')):
            raise ValueError('Checkpoint contains invalid mechanical/contact history')
        for field, key in [('x', 'state_x'), ('damage', 'history_damage'), ('failed', 'history_failed')]:
            array = np.load(folder/f'trajectory_{field}.npy', mmap_mode='r', allow_pickle=False)
            try:
                np.testing.assert_array_equal(array[frame], checkpoint[key])
            finally:
                array._mmap.close()
    return {'completed_frame': frame, 'completed_steps': plan['frame_steps'][frame],
            'continuation_ready': not final, 'full_protocol_completed': final,
            'full_review_passed': False, 'production_eligible': False}


def verify_cache(cache, req, source, frame, receipt_sha, key):
    cache = Path(cache)
    relative = f'receipts/frame-{frame:02d}.json'
    receipt = read_json(cache/relative)
    if digest(cache/relative) != receipt_sha:
        raise ValueError('Parent receipt differs from the preceding job output')
    if not (receipt['cache_key'] == key and receipt['request'] == req and receipt['source'] == source
            and receipt['end_frame'] == frame and receipt['sealed'] is True
            and receipt.get('kind') == 'accepted-prefix' and receipt.get('never_resumable') is False
            and receipt['production_eligible'] is False):
        raise ValueError('Cache identity, source, request or frame mismatch')
    actual = inventory(cache, req, excluded=relative)
    if actual != receipt['files']:
        raise ValueError('Restored cache contains missing, changed or additional bytes')
    runtime = cache/'runtime'
    if read_json(runtime/'request.json') != req or read_json(runtime/'source.json') != source:
        raise ValueError('Runtime source/request mismatch')
    plan = read_json(runtime/'plan.json')
    if receipt['plan_sha256'] != digest(runtime/'plan.json'):
        raise ValueError('Plan hash mismatch')
    checkpoint = result_folder(cache, req)/'checkpoint.npz'
    if receipt['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Checkpoint hash mismatch')
    if receipt['parent']:
        parent = receipt['parent']
        path = cache/f"receipts/frame-{parent['frame']:02d}.json"
        old = read_json(path)
        if not (digest(path) == parent['receipt_sha256'] and old['checkpoint_sha256'] == parent['checkpoint_sha256']
                and old['cache_key'] == parent['cache_key'] and old['end_frame'] < frame
                and receipt['prior_snapshot_bytes'] == old['cumulative_snapshot_bytes']):
            raise ValueError('Broken parent receipt/checkpoint chain')
    elif receipt['prior_snapshot_bytes'] != 0 or frame != req['stops'][0]:
        raise ValueError('Invalid first receipt')
    size = sum(item['bytes'] for item in actual.values()) + (cache/relative).stat().st_size
    if receipt['raw_snapshot_bytes'] != size or receipt['cumulative_snapshot_bytes'] != size+receipt['prior_snapshot_bytes'] or receipt['cumulative_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Cache byte accounting mismatch')
    return receipt, plan


def seal(cache, context):
    if context['stage'] not in (1, 2, 3):
        raise ValueError('Only the three numerical jobs create cache snapshots')
    req, source, frame = context['request'], context['source'], context['end_frame']
    plan = read_json(cache/'runtime/plan.json')
    endpoint = check_endpoint(result_folder(cache, req), plan, frame, final=context['stage'] == 3)
    relative = f'receipts/frame-{frame:02d}.json'
    files = inventory(cache, req, excluded=relative)
    parent = context.get('parent')
    prior = parent['cumulative_snapshot_bytes'] if parent else 0
    receipt = {'schema': 'triaxial-segment-cache-v1', 'kind': 'accepted-prefix', 'never_resumable': False,
        'sealed': True, 'source': source,
        'segment_runner_sha256': source['files']['tools/run_triaxial_segments.py'],
        'request': req, 'cache_key': context['cache_key'], 'end_frame': frame,
        'parent': {key: parent[key] for key in ('frame', 'receipt_sha256', 'checkpoint_sha256', 'cache_key')} if parent else None,
        'plan_sha256': digest(cache/'runtime/plan.json'),
        'checkpoint_sha256': digest(result_folder(cache, req)/'checkpoint.npz'),
        'files': files, 'endpoint': endpoint, 'cpu': cpu_record(),
        'prior_snapshot_bytes': prior, 'raw_snapshot_bytes': 0, 'cumulative_snapshot_bytes': 0,
        'workflow_cache_budget_bytes': CACHE_BUDGET, 'production_eligible': False,
        'full_review_passed': False, 'cache_scope': 'Raw numerical data sealed before review and rendering. No source, credentials or VTK payload.'}
    for _ in range(10):
        save(cache/relative, receipt)
        size = sum(item['bytes'] for item in files.values()) + (cache/relative).stat().st_size
        if size == receipt['raw_snapshot_bytes']:
            break
        receipt['raw_snapshot_bytes'] = size
        receipt['cumulative_snapshot_bytes'] = size+prior
    else:
        raise ValueError('Receipt byte-count fixpoint did not converge')
    if receipt['cumulative_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Complete snapshots exceed the conservative 8 GB workflow budget')
    verify_cache(cache, req, source, frame, digest(cache/relative), context['cache_key'])
    return receipt, digest(cache/relative)


def seal_failure(context):
    """Retain rejected raw bytes without changing failure flags or enabling resume."""
    if context['stage'] not in (1, 2, 3):
        raise ValueError('Finalization already has the immutable full raw cache')
    if study.source_record() != context['source']:
        raise ValueError('Diagnostic data cannot be attributed to changed source')
    req = context['request']
    folder = result_folder(CACHE, req)
    if not folder.is_dir() or not any(path.is_file() for path in folder.iterdir()):
        raise ValueError('No original numerical files were created')
    relative = f"receipts/failed-frame-{context['end_frame']:02d}.json"
    if (CACHE/relative).exists():
        raise FileExistsError('Failed diagnostic receipts are immutable')
    files = inventory(CACHE, req, diagnostic=True)
    parent = context.get('parent')
    prior = parent['cumulative_snapshot_bytes'] if parent else 0
    receipt = {'schema': 'triaxial-segment-failed-diagnostic-v1', 'kind': 'failed-diagnostic',
        'never_resumable': True, 'numerically_accepted': False, 'sealed': True,
        'source': context['source'], 'request': req, 'cache_key': context['cache_key']+'-failed',
        'requested_end_frame': context['end_frame'], 'parent': parent, 'files': files,
        'checkpoint_sha256': digest(folder/'checkpoint.npz') if (folder/'checkpoint.npz').is_file() else None,
        'missing_result_files': sorted(name for name in RESULT_NAMES if not (folder/name).is_file()),
        'prior_snapshot_bytes': prior, 'raw_snapshot_bytes': 0, 'cumulative_snapshot_bytes': 0,
        'production_eligible': False, 'full_review_passed': False, 'media_created': False,
        'scope': 'Original failed bytes, including known atomic-write remnants. This evictable cache is neither a valid continuation nor permanent backup; no failed state or threshold was changed.'}
    for _ in range(10):
        save(CACHE/relative, receipt)
        size = sum(item['bytes'] for item in files.values())+(CACHE/relative).stat().st_size
        if size == receipt['raw_snapshot_bytes']:
            break
        receipt['raw_snapshot_bytes'] = size
        receipt['cumulative_snapshot_bytes'] = size+prior
    else:
        raise ValueError('Diagnostic receipt byte count did not stabilize')
    if receipt['cumulative_snapshot_bytes'] > CACHE_BUDGET:
        raise ValueError('Diagnostic raw snapshot exceeds the existing workflow cache budget')
    context['diagnostic_cache_key'] = receipt['cache_key']
    context['diagnostic_receipt_sha256'] = digest(CACHE/relative)
    context['diagnostic_receipt_path'] = relative
    save(CONTEXT, context)
    save(LOGS/'failed-diagnostic-receipt.json', receipt)
    emit(cache_key=receipt['cache_key'], receipt_sha=context['diagnostic_receipt_sha256'])
    return receipt


def confirm_diagnostic_cache(context):
    key = context.get('diagnostic_cache_key')
    sha = context.get('diagnostic_receipt_sha256')
    if (not key or not sha or os.environ.get('CACHE_HIT') != 'true'
            or os.environ.get('CACHE_MATCH') != key or os.environ.get('RECEIPT_SHA') != sha):
        raise ValueError('No exact diagnostic cache save was verified')
    relative = context['diagnostic_receipt_path']
    receipt = read_json(CACHE/relative)
    if (digest(CACHE/relative) != sha or receipt.get('kind') != 'failed-diagnostic'
            or receipt.get('never_resumable') is not True or receipt.get('numerically_accepted') is not False
            or receipt.get('source') != context['source'] or receipt.get('request') != context['request']
            or inventory(CACHE, context['request'], relative, diagnostic=True) != receipt['files']):
        raise ValueError('Diagnostic raw evidence changed after sealing')
    save(LOGS/'diagnostic-cache-confirmation.json', {'verified': True, 'cache_key': key,
        'receipt_sha256': sha, 'kind': 'failed-diagnostic', 'never_resumable': True,
        'scope': 'Exact server key exists; original failed bytes remain diagnostic only.'})


def emit(**values):
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            for key, value in values.items():
                if '\n' in str(value):
                    raise ValueError('Multiline action output')
                stream.write(f'{key}={value}\n')


def prepare(args):
    if platform.python_version() != PYTHON_VERSION:
        raise ValueError(f'All jobs require Python {PYTHON_VERSION}')
    req = request(args.mode, args.theta, args.law, args.load_transits, args.size,
                  getattr(args, 'contact_regions', 'single'))
    source = study.source_record()
    end = req['stops'][min(args.stage, 3)-1]
    previous = req['stops'][args.stage-2] if args.stage > 1 else None
    if CACHE.exists():
        raise ValueError('Prepare requires a fresh hosted workspace before exact cache restore')
    context = {'request': req, 'source': source, 'stage': args.stage, 'end_frame': end,
        'previous_frame': previous, 'parent_receipt_sha256': args.parent_receipt,
        'parent_key': cache_key(source, req['mode'], previous) if previous is not None else '',
        'cache_key': cache_key(source, req['mode'], end), 'job_started_epoch': time.time(),
        'cpu': cpu_record()}
    if args.stage > 1 and not re.fullmatch('[a-f0-9]{64}', args.parent_receipt):
        raise ValueError('Resume requires the preceding job receipt SHA')
    save(CONTEXT, context)
    save(LOGS/'numerical-source.json', source)
    emit(cache_key=context['cache_key'], parent_key=context['parent_key'], end_frame=end,
         cache_path=CACHE.relative_to(ROOT).as_posix())


def restore_verify(context):
    if context['stage'] == 1:
        raise ValueError('The initial stage must not restore data')
    receipt, _ = verify_cache(CACHE, context['request'], context['source'], context['previous_frame'],
        context['parent_receipt_sha256'], context['parent_key'])
    context['parent'] = {'frame': receipt['end_frame'], 'receipt_sha256': context['parent_receipt_sha256'],
        'checkpoint_sha256': receipt['checkpoint_sha256'], 'cache_key': receipt['cache_key'],
        'cumulative_snapshot_bytes': receipt['cumulative_snapshot_bytes']}
    context['restore_verified'] = True
    if context['stage'] == 4:
        if receipt['endpoint']['full_protocol_completed'] is not True or receipt['endpoint']['continuation_ready'] is not False:
            raise ValueError('Finalization must restore the complete final numerical segment')
        context['sealed_receipt_sha256'] = context['parent_receipt_sha256']
    save(CONTEXT, context)
    save(LOGS/'restore-verification.json', {'verified': True, 'parent': context['parent'],
        'files_checked': len(receipt['files'])+1, 'source': context['source'], 'cpu': cpu_record()})


def lock(context):
    current = environment()
    if current['python'] != PYTHON_VERSION or current['system'] != 'Linux' or current['os_id'] != 'ubuntu' or current['os_version_id'] != '24.04':
        raise ValueError('Unexpected Python/OS environment')
    runtime = CACHE/'runtime'
    if context['stage'] == 1:
        runtime.mkdir(parents=True, exist_ok=False)
        save(runtime/'request.json', context['request'])
        save(runtime/'source.json', context['source'])
        save(runtime/'environment.json', current)
        (runtime/'requirements.txt').write_text(''.join(f'{key}=={value}\n' for key, value in current['packages'].items()), encoding='utf-8')
    elif not context.get('restore_verified') or current != read_json(runtime/'environment.json'):
        raise ValueError('The restored runtime environment differs from the first job')
    save(LOGS/'environment-check.json', {'verified': True, 'environment': current, 'cpu': cpu_record()})


def worker(context, operation):
    from examples import triaxial_failure as driver
    runtime = CACHE/'runtime'
    if environment() != read_json(runtime/'environment.json') or study.source_record() != context['source']:
        raise ValueError('Worker source/environment changed')
    if operation == 'plan':
        if context['stage'] != 1:
            raise ValueError('A resumed stage must load, not regenerate, reference sampling')
        geometry, plan = driver.plan_case(**context['request']['plan_kwargs'])
        cache_estimate(plan)
        write_geometry(runtime, geometry)
        save(runtime/'plan.json', plan)
        save(LOGS/'cache-budget.json', cache_estimate(plan))
        return
    geometry, plan = read_geometry(runtime), read_json(runtime/'plan.json')
    if read_json(runtime/'request.json') != context['request'] or driver.current_plan_source(plan) != plan['source']:
        raise ValueError('Saved input/source identity mismatch')
    study.validate_contact_regions(plan, context['request']['case'])
    if driver.descriptor(geometry['x']) != plan['sampling_hash'] or geometry['metadata'] != plan['geometry']:
        raise ValueError('Reference geometry does not match the saved resolved plan')
    # Observe the builders actually called by the driver, without a second build
    # or changing their returned values. Coverage identity is part of history.
    install_layout_checks(runtime, allow_create=context['stage'] == 1 and operation == 'solve')
    if operation == 'reference':
        if context['request']['mode'] != 'tiny' or context['stage'] != 4:
            raise ValueError('Unsegmented reference is only the tiny admission check')
        driver.run_case(geometry, plan, ROOT/'.local/triaxial-segment-reference')
    else:
        if context['stage'] == 4:
            raise ValueError('Finalization never advances the cached numerical solution')
        driver.run_case(geometry, plan, CACHE/'data', resume=context['stage'] > 1,
                        stop_after_frame=context['end_frame'])


def structure_descriptor(value):
    import dataclasses
    import numpy as np
    from examples.triaxial_failure import descriptor
    fields = {}
    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        if item is None or isinstance(item, (bool, int, float, str)):
            fields[field.name] = item
        else:
            fields[field.name] = descriptor(np.asarray(item))
    return fields


def verify_layout_record(path, name, record, *, allow_create):
    path = Path(path)
    values = read_json(path) if path.is_file() else {}
    if name in values:
        if values[name] != record:
            raise ValueError(f'Rebuilt {name} differs from the original runtime mapping/coverage')
    elif allow_create:
        values[name] = record
        save(path, values)
    else:
        raise ValueError(f'Original {name} mapping/coverage descriptor is missing')


def install_layout_checks(runtime, *, allow_create):
    from mpm_jax import fibre_mpm, fibre_coupling
    path = Path(runtime)/'numerical-layout.json'
    for module, attribute, name in ((fibre_mpm, 'build_tube_layout', 'grid'),
                                    (fibre_coupling, 'build_contact_data', 'contact')):
        original = getattr(module, attribute)
        def checked(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            verify_layout_record(path, _name, structure_descriptor(result), allow_create=allow_create)
            return result
        setattr(module, attribute, checked)


def budget_seconds(context, final_reserve_minutes):
    remaining = context['job_started_epoch']+350*60-time.time()-final_reserve_minutes*60
    seconds = min(265*60, int(remaining))
    if seconds <= 0:
        raise TimeoutError('Insufficient hosted-job time after reserving cache/finalization/upload time')
    return seconds


def limited(context, arguments, minutes):
    seconds = min(int(minutes*60), int(context['job_started_epoch']+330*60-time.time()))
    if seconds <= 0:
        raise TimeoutError('Finalization exhausted its shared budget; 20 minutes remain reserved for archival')
    return ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s', *map(str, arguments)]


def admit_finalization(context):
    remaining = context['job_started_epoch']+350*60-time.time()
    required = (100 if context['request']['mode'] == 'formal' else 40)*60
    record = {'remaining_seconds': remaining, 'required_seconds': required,
        'passed': remaining >= required, 'scope': 'Admission for dependencies, independent reviews, media and bounded archival; raw numerical cache is already sealed.'}
    save(LOGS/'finalization-time-admission.json', record)
    if not record['passed']:
        raise TimeoutError('Insufficient shared budget for complete finalization; raw completed cache is preserved')


def confirm_cache(context):
    if context['stage'] == 4:
        raise ValueError('The finalization job restores a snapshot; it never saves a fourth one')
    if not (os.environ.get('CACHE_HIT') == 'true' and os.environ.get('CACHE_MATCH') == context['cache_key']
            and os.environ.get('RECEIPT_SHA') == context.get('sealed_receipt_sha256')):
        raise ValueError('The save action did not produce a verified exact cache hit')
    receipt_path = CACHE/f"receipts/frame-{context['end_frame']:02d}.json"
    if digest(receipt_path) != context['sealed_receipt_sha256']:
        raise ValueError('Local receipt changed after cache save')
    save(LOGS/'cache-confirmation.json', {'verified': True, 'cache_key': context['cache_key'],
        'receipt_sha256': context['sealed_receipt_sha256'],
        'scope': 'Exact actions/cache lookup after save; temporary continuation storage, not permanent delivery.'})
    emit(receipt_sha=context['sealed_receipt_sha256'])


def run(context):
    if context['stage'] == 4:
        raise ValueError('The finalization job must not advance or seal a new snapshot')
    if context['stage'] > 1 and not context.get('restore_verified'):
        raise ValueError('No exact verified restore; refusing to recompute from scratch')
    lock_env = read_json(CACHE/'runtime/environment.json')
    if environment() != lock_env or study.source_record() != context['source']:
        raise ValueError('Source/runtime changed before solve')
    if context['stage'] == 1:
        engine.command('segment-plan', study.bounded([sys.executable, __file__, 'worker', '--operation', 'plan'], 20), monitor=True)
    plan = read_json(CACHE/'runtime/plan.json')
    study.validate_contact_regions(plan, context['request']['case'])
    if context['request']['mode'] == 'formal':
        study.validate_plan(plan, context['request']['case'], context['source'])
    cache_estimate(plan)
    save(LOGS/'disk-admission.json', disk_admission(plan, stage=context['stage'],
        free_bytes=shutil.disk_usage(ROOT).free, raw_already_present=context['stage'] > 1))
    seconds = budget_seconds(context, 40)
    save(LOGS/'time-budget.json', {'job_started_epoch': context['job_started_epoch'],
        'job_work_deadline_epoch': context['job_started_epoch']+350*60,
        'maximum_solve_seconds': 265*60, 'actual_solve_limit_seconds': seconds,
        'reserved_after_solve_minutes': 40,
        'scope': 'Shared wall-clock admission, not a promise that this physical protocol fits. Timeout is incomplete evidence; the requested physical time and frame plan are unchanged.'})
    solve = engine.command('segment-solve', ['timeout', '--signal=INT', '--kill-after=60s', f'{seconds}s',
        sys.executable, __file__, 'worker', '--operation', 'solve'], monitor=True)
    save(LOGS/'solve-memory.json', solve['memory'])
    if study.source_record() != context['source'] or environment() != lock_env:
        raise ValueError('Source/runtime changed during solve')
    receipt, receipt_sha = seal(CACHE, context)
    save(LOGS/'segment-receipt.json', receipt)
    context['sealed_receipt_sha256'] = receipt_sha
    save(CONTEXT, context)
    emit(receipt_sha=receipt_sha, cache_key=context['cache_key'])


def tiny_parity(segmented, reference):
    from tools.triaxial_parity import compare_runs
    return compare_runs(segmented, reference)


def finalize(context):
    if context['stage'] != 4 or not context.get('restore_verified'):
        raise ValueError('Only the separate verified finalization job can review full history')
    verify_cache(CACHE, context['request'], context['source'], context['end_frame'],
                 context['sealed_receipt_sha256'], context['cache_key'])
    source, req = context['source'], context['request']
    plan = read_json(CACHE/'runtime/plan.json')
    original = result_folder(CACHE, req)
    save(LOGS/'final-disk-admission.json', disk_admission(plan, stage=4,
        free_bytes=shutil.disk_usage(ROOT).free, raw_already_present=True))
    destination = ROOT/'.local/triaxial-segment-final'/original.relative_to(CACHE/'data')
    if destination.exists():
        raise ValueError('Finalization destination is not fresh')
    shutil.copytree(original, destination)
    # Raw cache remains immutable and excludes the much larger VTK export.
    save(destination/'numerical-source.json', source)
    save(destination/'continuation-source.json', {
        'segment_runner_sha256': source['files']['tools/run_triaxial_segments.py'],
        'review_and_render_helper_sha256': source['runner_source_sha256'],
        'cache_key': context['cache_key'], 'sealed_receipt_sha256': context['sealed_receipt_sha256'],
        'receipt_sha256': {path.name: digest(path) for path in sorted((CACHE/'receipts').glob('*.json'))},
        'runtime_geometry_sha256': digest(CACHE/'runtime/geometry.npz'),
        'runtime_layout_sha256': digest(CACHE/'runtime/numerical-layout.json'),
        'runtime_environment_sha256': digest(CACHE/'runtime/environment.json'),
        'production_eligible': False,
        'scope': 'Segment orchestration/source chain is additional to the reused single-case review/render helper. Exact raw data were sealed and saved before this independent finalization copy.'})
    if req['mode'] == 'tiny':
        engine.command('unsegmented-tiny-reference', limited(context, [sys.executable, __file__, 'worker', '--operation', 'reference'], 15), monitor=True)
        reference = ROOT/'.local/triaxial-segment-reference'/original.relative_to(CACHE/'data')
        parity = tiny_parity(destination, reference)
        save(destination/'continuation-parity.json', parity)
        save(LOGS/'continuation-parity.json', parity)
    targets = [('segmented', destination)]
    if req['mode'] == 'tiny':
        targets.append(('reference', reference))
    for label, target in targets:
        for index, name in enumerate(study.REVIEW_FILES):
            engine.command(f'{label}-review-{index}', limited(context, [sys.executable, name, str(target)], 10), monitor=True, accepted_codes=(0, 1))
    if req['mode'] == 'formal':
        report = study.acceptance(destination, req['case'], source, plan, study.source_record() == source)
    else:
        record, state = [read_json(destination/name) for name in ('record-audit.json', 'state-review.json')]
        reference_record, reference_state = [read_json(reference/name) for name in ('record-audit.json', 'state-review.json')]
        checks = {'tiny_scope': plan['scope'] == 'smoke' and plan['saved_states'] == 5 and plan['load_transits'] == .03,
            'continuation_parity': parity.get('passed') is True and not parity.get('errors')
                and bool(parity.get('metrics')) and all(item.get('passed') is True for item in parity['metrics']),
            'comparison_policy_source': parity.get('review_source_sha256') == source['files']['tools/triaxial_parity.py'],
            'comparison_policy_schema': parity.get('schema') == 'triaxial-tiny-cross-cpu-parity-v1',
            'comparison_inputs': study.inputs_match(destination, parity.get('input_sha256', {}).get('segmented', {}))
                and study.inputs_match(reference, parity.get('input_sha256', {}).get('reference', {})),
            'source_unchanged': study.source_record() == source,
            'record_review': study.strict_review(record), 'state_review': study.strict_review(state),
            'full_saved_history_checked': state.get('full_history_checked') is True,
            'record_inputs': study.RECORD_INPUTS.issubset(record.get('input_sha256', {})) and study.inputs_match(destination, record.get('input_sha256', {})),
            'state_inputs': study.STATE_INPUTS.issubset(state.get('input_sha256', {})) and study.inputs_match(destination, state.get('input_sha256', {})),
            'review_sources': record.get('auditor_sha256') == source['review_sources'][study.REVIEW_FILES[0]] and state.get('review_source_sha256') == source['review_sources'][study.REVIEW_FILES[1]],
            'producer_source': plan['source'] == record.get('source') == state.get('producer_source'),
            'reference_record_review': study.strict_review(reference_record),
            'reference_state_review': study.strict_review(reference_state),
            'reference_full_saved_history_checked': reference_state.get('full_history_checked') is True,
            'reference_record_inputs': study.RECORD_INPUTS.issubset(reference_record.get('input_sha256', {}))
                and study.inputs_match(reference, reference_record.get('input_sha256', {})),
            'reference_state_inputs': study.STATE_INPUTS.issubset(reference_state.get('input_sha256', {}))
                and study.inputs_match(reference, reference_state.get('input_sha256', {})),
            'reference_review_sources': reference_record.get('auditor_sha256') == source['review_sources'][study.REVIEW_FILES[0]]
                and reference_state.get('review_source_sha256') == source['review_sources'][study.REVIEW_FILES[1]],
            'reference_producer_source': plan['source'] == reference_record.get('source') == reference_state.get('producer_source')}
        report = {'passed': all(checks.values()), 'checks': checks, 'production_eligible': False,
            'media_created': False, 'scope': 'Tiny cache/continuation admission only; no production or homepage eligibility.',
            'input_sha256': {name: digest(destination/name) for name in ('plan.json', 'summary.json', 'status.json', 'record-audit.json', 'state-review.json', 'continuation-parity.json')},
            'reference_review_sha256': {name: digest(reference/name) for name in ('record-audit.json', 'state-review.json')}}
    save(destination/'runner-acceptance.json', report)
    if not study.strict_review(report):
        raise ValueError('Final numerical review rejected this case')
    if req['mode'] == 'formal':
        remaining = context['job_started_epoch']+330*60-time.time()
        study.render(destination, source, time.monotonic()+remaining)
    if environment() != read_json(CACHE/'runtime/environment.json') or study.source_record() != source:
        raise ValueError('Final source/environment changed after independent review')
    context['final_folder'] = destination.relative_to(ROOT).as_posix()
    save(CONTEXT, context)
    verify_cache(CACHE, req, source, context['end_frame'], context['sealed_receipt_sha256'], context['cache_key'])


def package(context):
    case = dict(context['request']['case'])
    fallback = result_folder(CACHE, context['request'])
    if context['stage'] == 4:
        candidate = ROOT/'.local/triaxial-segment-final'/fallback.relative_to(CACHE/'data')
        if candidate.is_dir():
            fallback = candidate
    case['folder'] = context.get('final_folder', fallback.relative_to(ROOT).as_posix())
    case['segment_mode'] = context['request']['mode']
    if context['request']['mode'] == 'tiny':
        return small_package(context, case, complete_tiny=True)
    confirmed = read_json(LOGS/'cache-confirmation.json') if (LOGS/'cache-confirmation.json').is_file() else {}
    if context['stage'] <= 3 and confirmed.get('verified') is True and confirmed.get('cache_key') == context['cache_key'] and confirmed.get('receipt_sha256') == context.get('sealed_receipt_sha256'):
        return small_package(context, case, complete_tiny=False)
    return study.package(case, root=ROOT, logs=LOGS, delivery=DELIVERY)


def small_package(context, case, *, complete_tiny):
    budget = study.TOTAL_BYTES if complete_tiny else 8*1024**2
    if DELIVERY.exists():
        manifest = read_json(DELIVERY/'delivery-manifest.json')
        if not study.inputs_match(DELIVERY, {key: item['sha256'] for key, item in manifest['files'].items()}):
            raise ValueError('Existing segment delivery changed')
        return manifest
    stage = Path(tempfile.mkdtemp(prefix='segment-delivery-', dir=DELIVERY.parent))
    roots = [LOGS, CACHE] if complete_tiny else [LOGS, ROOT/case['folder']]
    if complete_tiny and context['stage'] == 4:
        roots += [ROOT/'.local/triaxial-segment-reference', ROOT/'.local/triaxial-segment-final']
    paths = sorted({path for root in roots if root.exists() for path in root.rglob('*') if path.is_file()
        and not path.is_symlink() and (complete_tiny or path.suffix in ('.json', '.csv'))})
    files, omitted, used = {}, {}, 0
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        record = file_record(path)
        if used+record['bytes'] > budget-1024**2:
            omitted[relative] = {**record, 'reason': 'Whole file exceeds remaining bounded artifact space; no partial file stored.'}
            continue
        target = stage/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as source, target.open('wb') as output:
            shutil.copyfileobj(source, output, length=study.READ_BYTES)
        if file_record(target) != record:
            raise ValueError('Segment archive copy mismatch')
        files[relative] = record
        used += record['bytes']
    manifest = {'schema': 'triaxial-segment-records-v1', 'case': case, 'stage': context['stage'],
        'production_eligible': False, 'media_created': False, 'files': files, 'omitted': omitted,
        'payload_bytes': used, 'total_budget_bytes': budget,
        'cache_key': context['cache_key'], 'sealed_receipt_sha256': context.get('sealed_receipt_sha256'),
        'full_tiny_payload_selected': complete_tiny and not omitted,
        'scope': 'Tiny admission data, including raw history and any completed reference, are preserved within 192 MiB.' if complete_tiny else
            'Successful intermediate segment metadata only, at most 8 MiB. Full checkpoint/history exist in the verified temporary cache; cache availability is not a permanent delivery claim.'}
    save(stage/'delivery-manifest.json', manifest)
    if used+(stage/'delivery-manifest.json').stat().st_size > budget:
        raise ValueError('Segment archive manifest exceeds budget')
    stage.replace(DELIVERY)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify', 'lock', 'run', 'worker', 'install', 'confirm-cache',
        'seal-failure', 'confirm-diagnostic', 'admit-finalize', 'finalize', 'package'))
    parser.add_argument('--mode', choices=('tiny', 'formal'), default='tiny')
    parser.add_argument('--stage', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--theta', type=int, default=0)
    parser.add_argument('--law', choices=('hertz', 'jkr'), default='hertz')
    parser.add_argument('--contact-regions', choices=('single', 'connected'), default='single')
    parser.add_argument('--load-transits', type=float, default=10.)
    parser.add_argument('--size', choices=tuple(study.SIZES), default='24x16')
    parser.add_argument('--parent-receipt', default='')
    parser.add_argument('--operation', choices=('plan', 'solve', 'reference'), default='solve')
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux':
        parser.error('Segment execution requires the independent Linux Actions runner')
    LOGS.mkdir(parents=True, exist_ok=True)
    engine.LOGS = study.LOGS = LOGS
    try:
        if args.action == 'prepare':
            prepare(args)
            return
        if args.action == 'package' and not CONTEXT.is_file():
            case = study.selection()
            case['folder'] = '.local/triaxial-segment-missing-result'
            study.package(case, root=ROOT, logs=LOGS, delivery=DELIVERY)
            return
        context = read_json(CONTEXT)
        if args.action == 'verify': restore_verify(context)
        elif args.action == 'lock': lock(context)
        elif args.action == 'worker': worker(context, args.operation)
        elif args.action == 'run': run(context)
        elif args.action == 'confirm-cache': confirm_cache(context)
        elif args.action == 'seal-failure': seal_failure(context)
        elif args.action == 'confirm-diagnostic': confirm_diagnostic_cache(context)
        elif args.action == 'admit-finalize': admit_finalization(context)
        elif args.action == 'finalize': finalize(context)
        elif args.action == 'package': package(context)
        elif args.action == 'install':
            engine.shared.LOGS, engine.shared.command = LOGS, engine.command
            engine.shared.install()
    except BaseException as error:
        save(LOGS/f'{args.action}-failure.json', {'completed': False, 'production_eligible': False,
            'action': args.action, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
