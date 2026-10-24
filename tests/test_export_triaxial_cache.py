import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tools import export_triaxial_cache as export


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope='module')
def producer(tmp_path_factory):
    root = tmp_path_factory.mktemp('raw-producer')
    sources = {'tools/run_triaxial_segments.py': b'# producer\n',
               'examples/example.py': b'# numerical source\n',
               '.github/workflows/triaxial-segments.yml': b'name: producer\n',
               'pyproject.toml': b'[project]\nname="fixture"\n'}
    for name, raw in sources.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.STDOUT).decode().strip()
    git('init', '--quiet')
    git('config', 'core.autocrlf', 'false')
    git('config', 'user.name', 'Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    git('add', '.')
    git('commit', '--quiet', '-m', 'Fixture producer')
    files = {name: sha(raw) for name, raw in sources.items()}
    source = {'source_kind': 'github-actions', 'repository': export.REPOSITORY,
        'commit': git('rev-parse', 'HEAD'), 'tree': git('rev-parse', 'HEAD^{tree}'),
        'run_id': '12345', 'run_attempt': '1', 'files': files,
        'lf_sha256': files.copy(),
        'git_blob_sha': {name: git('rev-parse', 'HEAD:'+name) for name in sources},
        'fingerprint': sha(json.dumps(files, sort_keys=True).encode())}
    return root, source


def seal(path, receipt):
    for _ in range(10):
        export.save(path, receipt)
        size = sum(record['bytes'] for record in receipt['files'].values())+path.stat().st_size
        if size == receipt['raw_snapshot_bytes']:
            return
        receipt['raw_snapshot_bytes'] = size
        receipt['cumulative_snapshot_bytes'] = size+receipt['prior_snapshot_bytes']
    raise AssertionError('Fixture byte count did not converge')


@pytest.fixture
def cache(tmp_path, producer):
    root, source = producer
    folder = tmp_path/'cache'
    prefix = 'data/formal/theta-00/hertz'
    names = {f'runtime/{name}' for name in export.RUNTIME_FILES}
    names |= {f'{prefix}/{name}' for name in export.RESULT_FILES}
    for name in names:
        path = folder/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture raw '+name.encode())
    request = {'mode': 'formal', 'stops': [20, 40, 60], 'case': {'law': 'hertz', 'theta': 0}}
    plan = {'frame_steps': list(range(61)), 'source': 'fixture'}
    export.save(folder/'runtime/source.json', source)
    export.save(folder/'runtime/request.json', request)
    export.save(folder/'runtime/plan.json', plan)
    export.save(folder/prefix/'plan.json', plan)
    status = {'completed': True, 'frame': 60, 'steps': 60, 'coupling_valid': True,
              'signature': sha(json.dumps(plan, sort_keys=True).encode()), 'phase': 'completed',
              'probe_complete': False, 'segment_complete': False, 'segment_end_frame': 60}
    export.save(folder/prefix/'status.json', status)
    export.save(folder/prefix/'summary.json', status)
    previous = None
    expected = export.identity(source['commit'], 12345, 1, '0'*64)
    for frame in (20, 40, 60):
        path = folder/f'receipts/frame-{frame:02d}.json'
        receipt = {'schema': 'triaxial-segment-cache-v1', 'sealed': True, 'source': source,
            'segment_runner_sha256': source['files']['tools/run_triaxial_segments.py'],
            'request': request, 'end_frame': frame, 'production_eligible': False,
            'cache_key': expected['cache_key'][:-2]+f'{frame:02d}',
            'parent': previous, 'plan_sha256': export.digest(folder/'runtime/plan.json'),
            'checkpoint_sha256': export.digest(folder/prefix/'checkpoint.npz'),
            'files': {name: export.file_record(folder/name) for name in sorted(names)},
            'endpoint': {'completed_frame': frame, 'completed_steps': frame,
                         'continuation_ready': frame != 60, 'full_protocol_completed': frame == 60},
            'raw_snapshot_bytes': 0, 'prior_snapshot_bytes': prior if previous else 0,
            'cumulative_snapshot_bytes': 0, 'workflow_cache_budget_bytes': 8_000_000_000}
        seal(path, receipt)
        prior = receipt['cumulative_snapshot_bytes']
        previous = {'frame': frame, 'receipt_sha256': export.digest(path),
                    'checkpoint_sha256': receipt['checkpoint_sha256'], 'cache_key': receipt['cache_key']}
        names.add(path.relative_to(folder).as_posix())
    expected['receipt_sha256'] = export.digest(folder/'receipts/frame-60.json')
    return folder, expected, root


def test_complete_cache_source_and_parent_chain(cache):
    folder, expected, root = cache
    before = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    proof = export.inspect_cache(folder, expected, root)
    assert proof['raw_bytes'] == sum(map(len, before.values()))
    assert proof['source_verification']['checked_source_files'] == 4
    assert [item['frame'] for item in proof['receipt_chain']] == [20, 40, 60]
    assert proof['earlier_checkpoint_bytes_included'] is False
    assert before == {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['changed', 'missing', 'extra', 'receipt', 'source'])
def test_changed_or_incomplete_raw_cache_is_rejected(cache, change):
    folder, expected, root = cache
    path = folder/'data/formal/theta-00/hertz/trajectory_x.npy'
    if change == 'changed':
        path.write_bytes(path.read_bytes()+b'changed')
    elif change == 'missing':
        path.unlink()
    elif change == 'extra':
        (folder/'private.txt').write_text('not in cache receipt')
    elif change == 'receipt':
        (folder/'receipts/frame-40.json').write_text('{}')
    else:
        (folder/'runtime/source.json').write_text('{}')
    with pytest.raises((ValueError, OSError)):
        export.inspect_cache(folder, expected, root)


@pytest.mark.parametrize('key', ['parent', 'cache_key', 'prior_snapshot_bytes', 'cumulative_snapshot_bytes', 'segment_runner_sha256'])
def test_rehashed_final_receipt_cannot_break_original_chain(cache, key):
    folder, expected, root = cache
    path = folder/'receipts/frame-60.json'
    receipt = export.read_json(path)
    if key == 'parent':
        receipt[key]['receipt_sha256'] = 'f'*64
    elif key.endswith('bytes'):
        receipt[key] += 1
    else:
        receipt[key] = 'wrong'
    if key != 'cumulative_snapshot_bytes':
        receipt['raw_snapshot_bytes'] = 0
        seal(path, receipt)
    else:
        export.save(path, receipt)
    expected['receipt_sha256'] = export.digest(path)
    with pytest.raises(ValueError):
        export.inspect_cache(folder, expected, root)


@pytest.mark.parametrize('field', ['tree', 'fingerprint', 'files', 'lf_sha256', 'git_blob_sha', 'run_attempt'])
def test_producer_git_objects_are_independently_checked(producer, field):
    root, original = producer
    source = copy.deepcopy(original)
    expected = export.identity(source['commit'], 12345, 1, '0'*64)
    if field in ('files', 'lf_sha256', 'git_blob_sha'):
        source[field]['examples/example.py'] = '0'*64
        if field == 'files':
            source['fingerprint'] = sha(json.dumps(source[field], sort_keys=True).encode())
    else:
        source[field] = 'wrong'
    with pytest.raises(ValueError):
        export.verify_source(source, expected, root)


@pytest.mark.parametrize('name', ['', '.', '..', '../file', '/absolute', '//server/file',
                                 'a/../b', 'a//b', 'a/./b', 'C:/file', 'a\\b', 'a:stream'])
def test_unsafe_paths_are_rejected(name):
    with pytest.raises(ValueError):
        export.relative_name(name)


@pytest.fixture
def parts(tmp_path):
    cache = tmp_path/'raw'
    cache.mkdir()
    (cache/'small.bin').write_bytes(b'original')
    (cache/'large.bin').write_bytes(bytes(range(70)))
    files = {path.name: export.file_record(path) for path in cache.iterdir()}
    manifest, local = export.prepare_assets(cache, {'files': files}, tmp_path/'parts', asset_limit=32, part_bytes=16)
    assets = tmp_path/'assets'
    assets.mkdir()
    for name, path in local.items():
        shutil.copyfile(path, assets/name)
    return cache, manifest, assets


def test_whole_files_and_ordered_parts_reconstruct_exact_bytes(parts, tmp_path):
    raw, manifest, assets = parts
    assert len(manifest['files']['large.bin']['parts']) == 5
    result = export.reconstruct(manifest, assets, tmp_path/'backup')
    assert result == {'complete_files': 2, 'raw_bytes': 78}
    assert all((tmp_path/'backup'/p.name).read_bytes() == p.read_bytes() for p in raw.iterdir())


@pytest.mark.parametrize('change', ['missing', 'changed', 'extra', 'offset', 'duplicate', 'unsafe', 'size', 'whole_digest'])
def test_download_damage_or_incomplete_part_coverage_is_rejected(parts, tmp_path, change):
    _, manifest, assets = parts
    part = manifest['files']['large.bin']['parts'][0]
    if change == 'missing':
        (assets/part['name']).unlink()
    elif change == 'changed':
        (assets/part['name']).write_bytes(b'changed')
    elif change == 'extra':
        (assets/'unknown.bin').write_bytes(b'extra')
    elif change == 'offset':
        part['offset'] = 1
    elif change == 'duplicate':
        manifest['files']['small.bin']['parts'][0]['name'] = part['name']
    elif change == 'unsafe':
        manifest['files']['../escape.bin'] = manifest['files'].pop('small.bin')
    elif change == 'size':
        manifest['files']['large.bin']['bytes'] -= 1
    else:
        manifest['files']['large.bin']['sha256'] = '0'*64
    with pytest.raises((ValueError, OSError)):
        export.reconstruct(manifest, assets, tmp_path/'backup')
    assert not (tmp_path/'escape.bin').exists()


def test_no_existing_backup_or_source_staging_is_overwritten(parts, tmp_path):
    raw, manifest, assets = parts
    target = tmp_path/'existing'
    target.mkdir()
    sentinel = target/'sentinel'
    sentinel.write_bytes(b'keep')
    with pytest.raises(ValueError, match='fresh'):
        export.reconstruct(manifest, assets, target)
    with pytest.raises(ValueError, match='fresh'):
        export.prepare_assets(raw, {'files': {}}, target)
    assert sentinel.read_bytes() == b'keep'


def test_symlink_is_rejected_before_reading_payload(cache, parts, tmp_path):
    folder, expected, root = cache
    _, manifest, assets = parts
    link = folder/'link'
    try:
        link.symlink_to(tmp_path/'absent')
    except OSError:
        pytest.skip('The operating system does not permit creating fixture symlinks')
    with pytest.raises(ValueError, match='symlink'):
        export.inspect_cache(folder, expected, root)
    (assets/'link').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError, match='symlink'):
        export.reconstruct(manifest, assets, tmp_path/'backup')


@pytest.mark.parametrize('text', ['{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'])
def test_ambiguous_or_nonfinite_json_is_rejected(tmp_path, text):
    path = tmp_path/'bad.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        export.read_json(path)


class ReleaseFixture:
    def __init__(self, expected):
        self.expected = expected
        self.release = None
        self.uploaded = {}
        self.calls = []
        self.tag_commit = expected['commit']
    def request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, body))
        if path.startswith('git/ref/tags/'):
            return {'object': {'type': 'commit', 'sha': self.tag_commit}}
        if path.startswith('releases/tags/'):
            return self.release
        if method == 'POST' and path == 'releases':
            self.release = {'id': 9, **body}
            return self.release
        if path == 'releases/9':
            return self.release
        if path.startswith('actions/runs/'):
            if '/jobs?' in path:
                return {'jobs': [{'id': 7, 'name': 'third / segment', 'conclusion': 'success'}]}
            return {'id': self.expected['run_id'], 'run_attempt': self.expected['run_attempt'],
                'head_sha': self.expected['commit'], 'path': '.github/workflows/triaxial-segments.yml',
                'event': 'workflow_dispatch', 'status': 'completed', 'conclusion': 'failure'}
        raise AssertionError((method, path))
    def assets(self, release_id):
        return [{**item, 'state': 'uploaded'} for item in self.uploaded.values()]
    def upload(self, release_id, name, path, expected):
        assert export.file_record(path) == expected
        item = {'id': len(self.uploaded)+1, 'name': name, 'size': expected['bytes'], 'digest': 'sha256:'+expected['sha256']}
        self.uploaded[name] = item
        return item


def test_export_is_draft_only_separate_from_numerical_acceptance(cache, tmp_path, monkeypatch):
    folder, expected, root = cache
    client = ReleaseFixture(expected)
    monkeypatch.setattr(export.shutil, 'disk_usage', lambda _: type('Disk', (), {'free': 4*1024**3})())
    report = export.export(folder, expected, root, tmp_path/'export', client, {'commit': '1'*40})
    manifest = export.read_json(tmp_path/'export'/export.MANIFEST_NAME)
    assert report['export_complete'] and report['draft_only']
    assert report['local_backup_verified'] is False and report['production_eligible'] is False
    assert manifest['exporter']['commit'] != manifest['producer']['commit']
    assert manifest['producer_workflow']['conclusion'] == 'failure'
    assert client.release['draft'] is True and client.release['make_latest'] == 'false'
    assert list(client.uploaded)[-1] == export.MANIFEST_NAME
    assert all(method == 'GET' or (method == 'POST' and path == 'releases') for method, path, _ in client.calls)
    assert 'target_commitish' not in client.release


def test_changed_staged_part_cannot_be_certified_as_the_original(cache, tmp_path, monkeypatch):
    folder, expected, root = cache
    output = tmp_path/'export'
    original_prepare = export.prepare_assets
    monkeypatch.setattr(export, 'prepare_assets', lambda *args: original_prepare(*args, asset_limit=1024, part_bytes=1024))
    original_draft = export.draft_release
    def changed_part(client, wanted):
        result = original_draft(client, wanted)
        part = next((output/'parts').iterdir())
        raw = part.read_bytes()
        part.write_bytes(bytes([raw[0] ^ 1])+raw[1:])
        return result
    monkeypatch.setattr(export, 'draft_release', changed_part)
    monkeypatch.setattr(export.shutil, 'disk_usage', lambda _: type('Disk', (), {'free': 4*1024**3})())
    client = ReleaseFixture(expected)
    before = {p.relative_to(folder).as_posix(): export.digest(p) for p in folder.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='Prepared asset changed'):
        export.export(folder, expected, root, output, client, {'commit': '1'*40})
    assert before == {p.relative_to(folder).as_posix(): export.digest(p) for p in folder.rglob('*') if p.is_file()}
    assert export.MANIFEST_NAME not in client.uploaded
    assert not (output/'export-report.json').exists()


def test_split_copy_is_bound_to_original_whole_digest(tmp_path, monkeypatch):
    original = tmp_path/'raw'
    original.mkdir()
    source = original/'large.bin'
    source.write_bytes(bytes(range(70)))
    record = export.file_record(source)
    real_record = export.file_record
    def changed_after_check(path):
        result = real_record(path)
        if Path(path) == source:
            source.write_bytes(b'X'*70)
        return result
    monkeypatch.setattr(export, 'file_record', changed_after_check)
    with pytest.raises(ValueError, match='original whole-file digest'):
        export.prepare_assets(original, {'files': {'large.bin': record}}, tmp_path/'parts', asset_limit=32, part_bytes=16)


@pytest.mark.parametrize('change', ['tag', 'release_tag', 'body', 'draft'])
def test_final_release_and_tag_identity_are_rechecked(cache, tmp_path, monkeypatch, change):
    folder, expected, root = cache
    client = ReleaseFixture(expected)
    upload = client.upload
    def change_after_upload(release_id, name, path, record):
        item = upload(release_id, name, path, record)
        if name == export.MANIFEST_NAME:
            if change == 'tag':
                client.tag_commit = 'f'*40
            elif change == 'release_tag':
                client.release['tag_name'] = 'wrong-dataset'
            elif change == 'body':
                client.release['body'] = 'wrong-dataset'
            else:
                client.release['draft'] = False
        return item
    monkeypatch.setattr(client, 'upload', change_after_upload)
    monkeypatch.setattr(export.shutil, 'disk_usage', lambda _: type('Disk', (), {'free': 4*1024**3})())
    with pytest.raises(ValueError):
        export.export(folder, expected, root, tmp_path/'export', client, {'commit': '1'*40})
    assert not (tmp_path/'export/export-report.json').exists()


def test_existing_tag_must_pin_producer_and_published_release_is_never_modified(cache):
    _, expected, _ = cache
    client = ReleaseFixture(expected)
    client.tag_commit = 'f'*40
    with pytest.raises(ValueError, match='pinned'):
        export.draft_release(client, expected)
    assert all(call[0] == 'GET' for call in client.calls)
    client.tag_commit = expected['commit']
    export.draft_release(client, expected)
    client.release['draft'] = False
    count = len(client.calls)
    with pytest.raises(ValueError, match='published'):
        export.draft_release(client, expected)
    assert all(call[0] == 'GET' for call in client.calls[count:])


@pytest.mark.parametrize('change', ['digest', 'size', 'state'])
def test_existing_asset_mismatch_cannot_be_overwritten(tmp_path, monkeypatch, change):
    path = tmp_path/'raw.bin'
    path.write_bytes(b'original')
    expected = export.file_record(path)
    item = {'id': 1, 'name': 'raw.bin', 'size': expected['bytes'], 'state': 'uploaded', 'digest': 'sha256:'+expected['sha256']}
    item[change] = {'digest': 'sha256:'+'f'*64, 'size': 999, 'state': 'starter'}[change]
    client = export.GitHub('fixture-not-a-real-token')
    monkeypatch.setattr(client, 'assets', lambda _: [item])
    monkeypatch.setattr(export.http.client, 'HTTPSConnection', lambda *a, **kw: pytest.fail('Unexpected network mutation'))
    with pytest.raises(ValueError, match='digest mismatch'):
        client.upload(9, 'raw.bin', path, expected)
