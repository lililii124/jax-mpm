"""Preserve an exact final cache as draft release assets, without running numerics."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import time
from urllib.parse import urlencode


REPOSITORY = 'lililii124/jax-mpm'
CACHE_PATH = Path('.local/triaxial-segment-cache')
READ_BYTES = 1024**2
ASSET_LIMIT = 1_900_000_000
PART_BYTES = 512*1024**2
MANIFEST_NAME = 'raw-export-manifest.json'
RUNTIME_FILES = {'request.json', 'source.json', 'environment.json', 'requirements.txt',
                 'geometry.json', 'geometry.npz', 'plan.json', 'numerical-layout.json'}
RESULT_FILES = {'plan.json', 'summary.json', 'status.json', 'layout.json', 'contact-model.json',
    'particle_fields.npz', 'checkpoint.npz', 'final_state_0.npz',
    'trajectory.npz', 'observables.csv', 'diagnostics.csv', 'contact-metrics.json',
    'trajectory_x.npy', 'trajectory_damage.npy', 'trajectory_failed.npy',
    'trajectory_speed.npy', 'trajectory_temperature.npy'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name, 'Unsafe relative path')
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
            and all(part not in ('', '.') for part in path.parts), 'Unsafe relative path')
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b''):
            result.update(block)
    return result.hexdigest()


def file_record(path):
    path = Path(path)
    before = path.stat()
    result = {'bytes': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during hashing')
    return result


def read_json(path):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, 'Duplicate JSON member')
            result[name] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=invalid)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def identity(commit, run, attempt, receipt_sha):
    require(re.fullmatch('[0-9a-f]{40}', commit) is not None, 'Invalid producer commit')
    require(type(run) is int and run > 0 and type(attempt) is int and attempt > 0, 'Invalid producer run')
    require(re.fullmatch('[0-9a-f]{64}', receipt_sha) is not None, 'Invalid final receipt digest')
    return {'repository': REPOSITORY, 'commit': commit, 'run_id': run, 'run_attempt': attempt,
        'receipt_sha256': receipt_sha,
        'cache_key': f'triaxial-segments-v1-{commit}-{run}-{attempt}-formal-f60',
        'release_tag': f'raw-triaxial-{run}-{attempt}-f60'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def verify_source(source, expected, source_checkout):
    require(source.get('source_kind') == 'github-actions' and source.get('repository') == REPOSITORY,
            'Unexpected producer source type')
    require(source.get('commit') == expected['commit'] and str(source.get('run_id')) == str(expected['run_id'])
            and str(source.get('run_attempt')) == str(expected['run_attempt']), 'Producer identity mismatch')
    commit = expected['commit']
    tree = git(source_checkout, 'rev-parse', commit+'^{tree}').decode().strip()
    require(source.get('tree') == tree, 'Producer Git tree mismatch')
    paths = ('mpm_jax', 'examples', 'tools', '.github/workflows', 'pyproject.toml')
    tracked = git(source_checkout, 'ls-tree', '-r', '--name-only', commit, '--', *paths).decode().splitlines()
    required = {name for name in tracked if PurePosixPath(name).suffix in ('.py', '.toml', '.yml', '.yaml')}
    files, blobs, normalized = source['files'], source['git_blob_sha'], source['lf_sha256']
    require(bool(files) and set(files) == set(blobs) == set(normalized) == required, 'Incomplete producer source inventory')
    require(source['fingerprint'] == hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'Producer source fingerprint mismatch')
    for name, sha in files.items():
        relative_name(name)
        raw = git(source_checkout, 'cat-file', 'blob', commit+':'+name)
        require(hashlib.sha256(raw).hexdigest() == sha, 'Producer raw source changed: '+name)
        require(hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == normalized[name], 'Producer LF source changed: '+name)
        require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() == blobs[name], 'Producer Git blob changed: '+name)
    return {'tree': tree, 'fingerprint': source['fingerprint'], 'checked_source_files': len(files)}


def inspect_cache(cache, expected, source_checkout):
    cache = Path(cache)
    require(cache.is_dir() and not cache.is_symlink(), 'Missing cache directory')
    paths = list(cache.rglob('*'))
    require(not any(path.is_symlink() for path in paths), 'Cache contains a symlink')
    final_name = 'receipts/frame-60.json'
    require(digest(cache/final_name) == expected['receipt_sha256'], 'Final receipt SHA mismatch')
    final = read_json(cache/final_name)
    source_proof = verify_source(final['source'], expected, source_checkout)
    request = final['request']
    require(request.get('mode') == 'formal' and request.get('stops') == [20, 40, 60], 'Not a complete formal request')
    case = request['case']
    require(case.get('law') in ('hertz', 'jkr') and type(case.get('theta')) is int
            and case['theta'] in range(0, 31, 5), 'Unexpected formal case')
    prefix = f"data/formal/theta-{case['theta']:02d}/{case['law']}"
    required = {f'runtime/{name}' for name in RUNTIME_FILES} | {f'{prefix}/{name}' for name in RESULT_FILES}
    optional = f'{prefix}/particle_fields.json'
    if optional in final['files']:
        required.add(optional)
    required |= {'receipts/frame-20.json', 'receipts/frame-40.json'}
    require(set(final['files']) == required, 'Missing or additional raw cache members')
    actual = {path.relative_to(cache).as_posix() for path in paths if path.is_file()}
    require(actual == required | {final_name}, 'Restored cache file set differs from its receipt')
    records = {}
    for name, record in final['files'].items():
        relative_name(name)
        require(file_record(cache/name) == record, 'Raw cache bytes changed: '+name)
        records[name] = record
    records[final_name] = file_record(cache/final_name)
    require(sum(record['bytes'] for record in records.values()) == final['raw_snapshot_bytes'], 'Cache byte accounting mismatch')
    require(final['cumulative_snapshot_bytes'] <= final['workflow_cache_budget_bytes'] == 8_000_000_000,
            'Cache budget differs from the producer protocol')
    chain = []
    previous = None
    for frame in (20, 40, 60):
        name = f'receipts/frame-{frame:02d}.json'
        record = read_json(cache/name)
        require(record.get('schema') == 'triaxial-segment-cache-v1' and record.get('sealed') is True
                and record.get('end_frame') == frame and record.get('production_eligible') is False,
                'Invalid sealed receipt')
        require(record['source'] == final['source'] and record['request'] == request, 'Receipt source or request changed')
        require(record['plan_sha256'] == final['plan_sha256'] and record['segment_runner_sha256'] ==
                final['source']['files']['tools/run_triaxial_segments.py'], 'Receipt producer protocol changed')
        wanted_key = expected['cache_key'][:-2]+f'{frame:02d}'
        require(record['cache_key'] == wanted_key, 'Receipt cache key mismatch')
        allowed = required - {'receipts/frame-20.json', 'receipts/frame-40.json'}
        allowed |= {f'receipts/frame-{old:02d}.json' for old in (20, 40) if old < frame}
        require(set(record['files']) == allowed, 'Ancestor receipt inventory is incomplete')
        size = sum(item['bytes'] for item in record['files'].values())+(cache/name).stat().st_size
        prior = previous['cumulative_snapshot_bytes'] if previous else 0
        require(record['raw_snapshot_bytes'] == size and record['prior_snapshot_bytes'] == prior
                and record['cumulative_snapshot_bytes'] == size+prior <= 8_000_000_000
                and record['workflow_cache_budget_bytes'] == 8_000_000_000, 'Receipt chain byte accounting mismatch')
        if previous is None:
            require(record['parent'] is None, 'Initial receipt has an unexpected parent')
        else:
            parent = record['parent']
            require(parent == {'frame': previous['frame'], 'receipt_sha256': previous['sha256'],
                'checkpoint_sha256': previous['checkpoint_sha256'], 'cache_key': previous['cache_key']}, 'Broken parent receipt chain')
        previous = {'frame': frame, 'sha256': digest(cache/name),
                    'checkpoint_sha256': record['checkpoint_sha256'], 'cache_key': record['cache_key'],
                    'cumulative_snapshot_bytes': record['cumulative_snapshot_bytes']}
        chain.append(previous)
    require(final['endpoint']['completed_frame'] == 60 and final['endpoint']['full_protocol_completed'] is True
            and final['endpoint']['continuation_ready'] is False,
            'Final endpoint is incomplete')
    require(digest(cache/f'{prefix}/checkpoint.npz') == final['checkpoint_sha256'], 'Final checkpoint differs from receipt')
    require(read_json(cache/'runtime/source.json') == final['source']
            and read_json(cache/'runtime/request.json') == request, 'Runtime identity changed')
    plan = read_json(cache/'runtime/plan.json')
    require(read_json(cache/f'{prefix}/plan.json') == plan and digest(cache/'runtime/plan.json') == final['plan_sha256'], 'Plan changed')
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    status, summary = read_json(cache/f'{prefix}/status.json'), read_json(cache/f'{prefix}/summary.json')
    require(status.get('completed') is True and summary.get('completed') is True and status.get('frame') == 60
            and status.get('signature') == signature and status.get('coupling_valid') is True
            and status.get('steps') == plan['frame_steps'][60] == final['endpoint']['completed_steps'],
            'Saved endpoint does not match the completed plan')
    require(all(record.get('probe_complete') is False and record.get('segment_complete') is False
                and record.get('segment_end_frame') == 60 for record in (summary, status))
            and status.get('phase') == 'completed', 'Incomplete or probe endpoint')
    return {'producer': expected, 'source_verification': source_proof, 'receipt_chain': chain,
        'files': records, 'raw_bytes': sum(record['bytes'] for record in records.values()),
        'scientific_acceptance': 'Not evaluated by this exporter. Original independent reviews remain separate.',
        'earlier_checkpoint_bytes_included': False,
        'scope': 'Complete sealed final cache: recorded x, damage, failed, speed and temperature histories; final State/contact history, reference geometry and runtime records. Not every-frame F/C/stress, VTK or all rendered PNGs.'}


def prepare_assets(cache, inventory, staging, *, asset_limit=ASSET_LIMIT, part_bytes=PART_BYTES):
    require(0 < part_bytes <= asset_limit <= ASSET_LIMIT, 'Invalid release asset size policy')
    staging = Path(staging)
    require(not staging.exists(), 'Asset staging must be fresh')
    staging.mkdir(parents=True)
    files, local = {}, {}
    for index, (name, record) in enumerate(sorted(inventory['files'].items())):
        relative_name(name)
        original = Path(cache)/name
        require(file_record(original) == record, 'Cache changed before export: '+name)
        stem = f'raw-{index:04d}-'+re.sub('[^A-Za-z0-9_.-]', '-', PurePosixPath(name).name)
        parts = []
        if record['bytes'] <= asset_limit:
            local[stem] = original
            parts.append({'name': stem, 'offset': 0, **record})
        else:
            with original.open('rb') as source:
                offset = 0
                whole = hashlib.sha256()
                while offset < record['bytes']:
                    asset = f'{stem}.part-{len(parts):04d}'
                    target = staging/asset
                    remaining = min(part_bytes, record['bytes']-offset)
                    with target.open('wb') as output:
                        while remaining:
                            block = source.read(min(READ_BYTES, remaining))
                            require(bool(block), 'Truncated source while creating parts')
                            whole.update(block)
                            output.write(block); remaining -= len(block)
                    item = file_record(target)
                    parts.append({'name': asset, 'offset': offset, **item})
                    local[asset] = target
                    offset += item['bytes']
                require(not source.read(1) and whole.hexdigest() == record['sha256'],
                        'Split bytes differ from the original whole-file digest')
        files[name] = {**record, 'parts': parts}
    require(len(local)+1 <= 1000, 'Release asset count exceeds the documented limit')
    manifest = {'schema': 'triaxial-raw-cache-export-v1', **{key: value for key, value in inventory.items() if key != 'files'},
        'files': files, 'asset_limit_bytes': asset_limit, 'draft_only': True, 'production_eligible': False}
    return manifest, local


def reconstruct(manifest, assets, destination):
    require(manifest.get('schema') == 'triaxial-raw-cache-export-v1' and manifest.get('draft_only') is True,
            'Unexpected raw export manifest')
    assets, destination = Path(assets), Path(destination)
    require(not destination.exists(), 'Backup destination must be fresh')
    require(assets.is_dir() and not assets.is_symlink()
            and not any(path.is_symlink() for path in assets.rglob('*')), 'Downloaded assets contain a symlink')
    require(0 < manifest['asset_limit_bytes'] <= ASSET_LIMIT, 'Invalid manifest asset size policy')
    expected = {}
    for name, record in manifest['files'].items():
        relative_name(name)
        offset = 0
        require(bool(record['parts']), 'File has no asset coverage')
        for part in record['parts']:
            relative_name(part['name'])
            require('/' not in part['name'] and part['name'] not in expected, 'Duplicate or unsafe asset name')
            require(part['offset'] == offset and 0 <= part['bytes'] <= manifest['asset_limit_bytes'], 'Noncontiguous asset coverage')
            require(file_record(assets/part['name']) == {key: part[key] for key in ('bytes', 'sha256')}, 'Downloaded asset digest mismatch')
            expected[part['name']] = part
            offset += part['bytes']
        require(offset == record['bytes'], 'Parts do not cover the whole file')
    actual = {path.relative_to(assets).as_posix() for path in assets.rglob('*') if path.is_file()}
    require(not any(path.is_symlink() for path in assets.rglob('*')), 'Downloaded assets contain a symlink')
    require(actual in (set(expected), set(expected) | {MANIFEST_NAME}), 'Missing or extra downloaded assets')
    destination.mkdir(parents=True)
    for name, record in manifest['files'].items():
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            for part in record['parts']:
                with (assets/part['name']).open('rb') as source:
                    shutil.copyfileobj(source, output, READ_BYTES)
        require(file_record(target) == {key: record[key] for key in ('bytes', 'sha256')}, 'Reconstructed whole-file digest mismatch')
    return {'complete_files': len(manifest['files']), 'raw_bytes': sum(item['bytes'] for item in manifest['files'].values())}


class GitHub:
    def __init__(self, token):
        require(bool(token) and '\n' not in token and '\r' not in token, 'Missing GitHub credential')
        self.token = token

    def request(self, method, path, body=None, *, missing_ok=False):
        connection = http.client.HTTPSConnection('api.github.com', timeout=60)
        headers = {'Authorization': 'Bearer '+self.token, 'Accept': 'application/vnd.github+json',
                   'User-Agent': 'jax-mpm-raw-export', 'X-GitHub-Api-Version': '2026-03-10'}
        encoded = None if body is None else json.dumps(body).encode()
        if encoded is not None:
            headers['Content-Type'] = 'application/json'
        try:
            connection.request(method, f'/repos/{REPOSITORY}/'+path, body=encoded, headers=headers)
            response = connection.getresponse()
            raw = response.read(16*1024**2)
            if missing_ok and response.status == 404:
                return None
            require(200 <= response.status < 300, f'GitHub API returned {response.status} for {path}')
            return json.loads(raw) if raw else None
        finally:
            connection.close()

    def assets(self, release_id):
        result = []
        for page in range(1, 12):
            batch = self.request('GET', f'releases/{release_id}/assets?per_page=100&page={page}')
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise ValueError('Unexpected release asset count')

    def upload(self, release_id, name, path, expected):
        existing = [item for item in self.assets(release_id) if item['name'] == name]
        if existing:
            require(len(existing) == 1, 'Duplicate release asset')
            item = existing[0]
        else:
            require(file_record(path) == expected and expected['bytes'] < 2**31, 'Upload source or size mismatch')
            connection = http.client.HTTPSConnection('uploads.github.com', timeout=120)
            started = time.monotonic()
            try:
                connection.putrequest('POST', f'/repos/{REPOSITORY}/releases/{release_id}/assets?'+urlencode({'name': name}))
                for key, value in {'Authorization': 'Bearer '+self.token, 'User-Agent': 'jax-mpm-raw-export',
                    'Accept': 'application/vnd.github+json', 'Content-Type': 'application/octet-stream',
                    'Content-Length': str(expected['bytes']), 'X-GitHub-Api-Version': '2026-03-10'}.items():
                    connection.putheader(key, value)
                connection.endheaders()
                with Path(path).open('rb') as stream:
                    for block in iter(lambda: stream.read(READ_BYTES), b''):
                        require(time.monotonic()-started < 2400, 'Asset upload exceeded its time budget')
                        connection.send(block)
                response = connection.getresponse()
                require(response.status == 201, f'Release upload returned {response.status}; existing assets are never overwritten')
                item = json.loads(response.read(1024**2))
            finally:
                connection.close()
        require(item['name'] == name and item['state'] == 'uploaded' and item['size'] == expected['bytes']
                and item.get('digest') == 'sha256:'+expected['sha256'], 'Release asset bytes or digest mismatch')
        return {key: item[key] for key in ('id', 'name', 'size', 'digest')}


def verify_origin(client, expected):
    run = client.request('GET', f"actions/runs/{expected['run_id']}")
    require(run['head_sha'] == expected['commit'] and run['run_attempt'] == expected['run_attempt']
            and run['path'] == '.github/workflows/triaxial-segments.yml' and run['event'] == 'workflow_dispatch',
            'Producer workflow identity mismatch')
    jobs = client.request('GET', f"actions/runs/{expected['run_id']}/attempts/{expected['run_attempt']}/jobs?per_page=100")['jobs']
    third = [job for job in jobs if job['name'] == 'third / segment']
    require(len(third) == 1 and third[0]['conclusion'] == 'success', 'No successfully sealed final numerical segment')
    return {'run_id': run['id'], 'run_attempt': run['run_attempt'], 'head_sha': run['head_sha'],
        'status': run['status'], 'conclusion': run['conclusion'], 'third_job_id': third[0]['id'],
        'scientific_acceptance': 'Not inferred from job success or raw cache integrity.'}


def verify_tag(client, expected):
    tag = client.request('GET', 'git/ref/tags/'+expected['release_tag'])['object']
    if tag['type'] == 'tag':
        tag = client.request('GET', 'git/tags/'+tag['sha'])['object']
    require(tag['type'] == 'commit' and tag['sha'] == expected['commit'], 'Pre-existing data tag is not pinned to the producer')


def release_marker(expected):
    return f"<!-- raw-cache:{expected['commit']}:{expected['run_id']}:{expected['run_attempt']}:{expected['receipt_sha256']} -->"


def verify_draft(release, expected):
    require(release['draft'] is True and release['tag_name'] == expected['release_tag']
            and release_marker(expected) in release['body'],
            'Existing release is published or belongs to another dataset')


def draft_release(client, expected):
    verify_tag(client, expected)
    release = client.request('GET', 'releases/tags/'+expected['release_tag'], missing_ok=True)
    if release is None:
        release = client.request('POST', 'releases', {'tag_name': expected['release_tag'],
            'name': f"Raw triaxial data, run {expected['run_id']}", 'draft': True, 'prerelease': True,
            'make_latest': 'false', 'generate_release_notes': False,
            'body': 'Lossless saved raw data. Numerical acceptance and public publication are separate.\n\n'+release_marker(expected)})
    verify_draft(release, expected)
    return release


def exporter_identity(root):
    commit = git(root, 'rev-parse', 'HEAD').decode().strip()
    require(commit == os.environ['GITHUB_SHA'], 'Exporter commit differs from workflow dispatch')
    files = ('tools/export_triaxial_cache.py', '.github/workflows/triaxial-cache-export.yml')
    hashes = {}
    for name in files:
        raw = git(root, 'cat-file', 'blob', commit+':'+name)
        require(Path(root, name).read_bytes() == raw, 'Exporter source changed after checkout')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return {'commit': commit, 'run_id': os.environ['GITHUB_RUN_ID'], 'run_attempt': os.environ['GITHUB_RUN_ATTEMPT'],
        'files': hashes, 'role': 'Read-only raw-data verifier and draft asset uploader; not the numerical producer.'}


def export(cache, expected, source_checkout, output, client, exporter):
    inventory = inspect_cache(cache, expected, source_checkout)
    origin = verify_origin(client, expected)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output).free
    extra = sum(item['bytes'] for item in inventory['files'].values() if item['bytes'] > ASSET_LIMIT)
    require(free >= extra+2*1024**3, 'Insufficient free disk for lossless export staging')
    manifest, local = prepare_assets(cache, inventory, output/'parts')
    release = draft_release(client, expected)
    wanted = set(local) | {MANIFEST_NAME}
    existing = client.assets(release['id'])
    require({item['name'] for item in existing} <= wanted and not any(item['name'] == MANIFEST_NAME for item in existing),
            'Draft contains unexpected assets or an already finalized manifest')
    uploaded = {}
    fixed = {part['name']: {key: part[key] for key in ('bytes', 'sha256')}
             for record in manifest['files'].values() for part in record['parts']}
    require(set(fixed) == set(local), 'Prepared assets differ from the manifest')
    for name, path in local.items():
        require(file_record(path) == fixed[name], 'Prepared asset changed before upload: '+name)
        uploaded[name] = client.upload(release['id'], name, path, fixed[name])
        save(output/'upload-progress.json', {'release_id': release['id'], 'assets': uploaded})
    require(inspect_cache(cache, expected, source_checkout) == inventory, 'Cache changed while exporting')
    manifest.update(exporter=exporter, producer_workflow=origin, assets=uploaded, release_id=release['id'])
    save(output/MANIFEST_NAME, manifest)
    manifest_record = file_record(output/MANIFEST_NAME)
    manifest_asset = client.upload(release['id'], MANIFEST_NAME, output/MANIFEST_NAME, manifest_record)
    actual = client.assets(release['id'])
    require({item['name'] for item in actual} == wanted, 'Release asset inventory is incomplete')
    expected_assets = {**uploaded, MANIFEST_NAME: manifest_asset}
    require(all({key: item[key] for key in ('id', 'name', 'size', 'digest')} == expected_assets[item['name']] for item in actual),
            'Release asset changed during final verification')
    final_release = client.request('GET', f"releases/{release['id']}")
    require(final_release['id'] == release['id'], 'Release identity changed during upload')
    verify_draft(final_release, expected)
    verify_tag(client, expected)
    report = {'export_complete': True, 'draft_only': True, 'production_eligible': False, 'producer': expected,
        'exporter': exporter, 'release_id': release['id'], 'manifest_asset': manifest_asset,
        'manifest_sha256': manifest_record['sha256'], 'raw_files': len(manifest['files']), 'raw_bytes': inventory['raw_bytes'],
        'local_backup_verified': False, 'scope': 'Draft assets are verified copies. Local download/reconstruction and independent numerical review remain separate.'}
    save(output/'export-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'export', 'verify-download'))
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--producer-run', required=True, type=int)
    parser.add_argument('--producer-attempt', type=int, default=1)
    parser.add_argument('--receipt-sha', required=True)
    parser.add_argument('--source-checkout', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('.local/triaxial-raw-export'))
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--manifest-sha')
    args = parser.parse_args()
    expected = identity(args.producer_commit, args.producer_run, args.producer_attempt, args.receipt_sha)
    if args.action == 'verify-download':
        require(args.assets is not None and args.manifest_sha is not None, 'Downloaded assets and trusted manifest SHA are required')
        require(digest(args.assets/MANIFEST_NAME) == args.manifest_sha, 'Downloaded manifest digest mismatch')
        manifest = read_json(args.assets/MANIFEST_NAME)
        require(manifest['producer'] == expected, 'Downloaded data producer differs from the requested backup')
        result = reconstruct(manifest, args.assets, args.output)
        inventory = inspect_cache(args.output, expected, args.source_checkout)
        save(args.output.parent/(args.output.name+'-backup-receipt.json'), {'full_raw_backup_verified': True,
            'producer': expected, 'manifest_sha256': args.manifest_sha, **result,
            'source_verification': inventory['source_verification'], 'scientific_acceptance': inventory['scientific_acceptance']})
        print(json.dumps(result))
        return
    require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.name == 'posix', 'Export runs only in the independent Actions job')
    if args.action == 'prepare':
        require(not CACHE_PATH.exists(), 'Restore requires a fresh cache destination')
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as stream:
            stream.write('cache_key='+expected['cache_key']+'\nproducer_commit='+expected['commit']+'\n')
        return
    require(os.environ.get('CACHE_HIT') == 'true' and os.environ.get('CACHE_MATCH') == expected['cache_key'], 'Only an exact cache hit is accepted')
    root = Path(__file__).resolve().parents[1]
    report = export(CACHE_PATH, expected, args.source_checkout, args.output,
                    GitHub(os.environ.get('GH_TOKEN', '')), exporter_identity(root))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
