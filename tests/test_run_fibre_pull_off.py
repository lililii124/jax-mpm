import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import run_fibre_pull_off as runner


def source(run='1'):
    return {'repository': 'example/project', 'commit': 'a'*40, 'tree': 'b'*40,
        'run_id': run, 'run_attempt': '1', 'files': {'solver.py': 'c'*64},
        'git_blob_sha': {'solver.py': 'd'*40}, 'lf_sha256': {'solver.py': 'c'*64}, 'fingerprint': 'e'*64}


def fixture(tmp_path, monkeypatch):
    """Byte-chain fixture only; these files are not a mechanical result."""
    cache, logs = tmp_path/'cache', tmp_path/'logs'
    monkeypatch.setattr(runner, 'LOGS', logs)
    monkeypatch.setattr(runner, 'repository_cache_usage', lambda repository: {'existing_bytes': 0, 'repository': repository})
    req = runner.request('hertz')
    context = {'request': req, 'source': source(), 'end_frame': 2, 'parent_key': '',
        'parent_receipt_sha256': '', 'cache_key': runner.cache_key(source(), 'hertz', 2)}
    folder = runner.result_folder(cache, req)
    folder.mkdir(parents=True)
    for name in runner.RESULT_NAMES:
        (folder/name).write_bytes(('synthetic byte-chain fixture '+name).encode())
    runtime = cache/'runtime'
    runtime.mkdir()
    environment = {'python': runner.PYTHON_VERSION, 'packages': {'jax': '0.12.0.1', 'jaxlib': '0.12.0.1', 'numpy': '2.5.4'}}
    runner.save(runtime/'environment.json', environment)
    runner.save(runtime/'source.json', source())
    runner.save(runtime/'request.json', req)
    (runtime/'requirements.txt').write_text(runner.requirements_text(environment), encoding='utf-8')
    runner.save(logs/'job.json', context)
    audit = {'prefix_verified': True, 'production_eligible': False, 'frame': 2, 'steps': 816}
    runner.seal(context, audit, cache)
    child = {**context, 'source': source('2'), 'end_frame': 10,
        'parent_key': context['cache_key'], 'parent_receipt_sha256': runner.digest(cache/'receipt.json'),
        'cache_key': runner.cache_key(source('2'), 'hertz', 10)}
    return cache, logs, context, child


def test_fixed_request_and_runtime_endpoint_do_not_shorten_physical_protocol():
    req = runner.request('hertz')
    assert req['duration_s'] == .03 and req['steps'] == 24495 and req['particles'] == 202240
    context = {'request': req, 'end_frame': 2, 'parent_key': ''}
    command = runner.command_line(context, Path('cache'))
    assert command[command.index('--frames')+1] == '60'
    assert command[command.index('--stop-after-frame')+1] == '2'
    assert '--duration' not in command and '--probe-steps' not in command and '--resume' not in command
    assert '--resume' in runner.command_line({**context, 'parent_key': 'exact'})


def test_prepare_rejects_changed_source_key_and_nonprefix_fresh_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'LOGS', tmp_path)
    monkeypatch.setattr(runner, 'source_record', source)
    monkeypatch.setattr(runner, 'repository_cache_usage', lambda repository: {'existing_bytes': 0, 'repository': repository})
    context = runner.prepare('hertz', 2)
    assert context['end_frame'] == 2
    with pytest.raises(ValueError, match='first full-scale'):
        runner.prepare('hertz', 10)
    with pytest.raises(ValueError, match='both exact'):
        runner.prepare('hertz', 10, 'some-key')
    with pytest.raises(ValueError, match='exact source'):
        runner.prepare('hertz', 10, runner.cache_key({**source(), 'commit': 'b'*40}, 'hertz', 2), 'c'*64)


@pytest.mark.parametrize('name,version', [('jax', 'https://example.invalid/a.whl'), ('--index-url', '1'), ('numpy', '2.2\n-r malicious'), ('bad/name', '1')])
def test_environment_pin_cannot_execute_cached_urls_or_options(name, version):
    with pytest.raises(ValueError):
        runner.requirements_text({'python': runner.PYTHON_VERSION, 'packages': {name: version}})


def test_cache_verifies_every_raw_byte_before_resume(tmp_path, monkeypatch):
    cache, _, _, child = fixture(tmp_path, monkeypatch)
    receipt = runner.verify_restored(cache, child)
    assert receipt['frame'] == 2
    path = runner.result_folder(cache, child['request'])/'checkpoint.npz'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='raw cache bytes'):
        runner.verify_restored(cache, child)


@pytest.mark.parametrize('mutation', ['receipt', 'request', 'source', 'backward', 'unknown_file'])
def test_cache_rejects_wrong_identity_or_payload(tmp_path, monkeypatch, mutation):
    cache, _, _, child = fixture(tmp_path, monkeypatch)
    child = copy.deepcopy(child)
    if mutation == 'receipt':
        child['parent_receipt_sha256'] = '0'*64
    elif mutation == 'request':
        child['request']['model'] = 'md'
    elif mutation == 'source':
        child['source']['commit'] = 'f'*40
    elif mutation == 'backward':
        child['end_frame'] = 2
    else:
        (cache/'cached-script.py').write_text('raise RuntimeError', encoding='utf-8')
    with pytest.raises(ValueError):
        runner.verify_restored(cache, child)


def test_successive_receipts_keep_original_parent_identity(tmp_path, monkeypatch):
    cache, _, _, child = fixture(tmp_path, monkeypatch)
    runner.verify_restored(cache, child)
    parent_bytes = (cache/'receipt.json').read_bytes()
    runner.seal(child, {'prefix_verified': True, 'production_eligible': False, 'frame': 10, 'steps': 4082}, cache)
    assert (cache/'receipts'/f'{child["parent_receipt_sha256"]}.json').read_bytes() == parent_bytes
    next_context = {**child, 'source': source('3'), 'end_frame': 20,
        'parent_key': child['cache_key'], 'parent_receipt_sha256': runner.digest(cache/'receipt.json'),
        'cache_key': runner.cache_key(source('3'), 'hertz', 20)}
    assert runner.verify_restored(cache, next_context)['frame'] == 10


def test_failed_or_partial_audit_cannot_be_sealed(tmp_path, monkeypatch):
    cache, _, context, _ = fixture(tmp_path, monkeypatch)
    before = (cache/'receipt.json').read_bytes()
    with pytest.raises(ValueError):
        runner.seal(context, {'prefix_verified': False, 'production_eligible': False, 'frame': 2}, cache)
    assert (cache/'receipt.json').read_bytes() == before


def test_failed_diagnostic_cache_preserves_invalid_raw_bytes_but_never_resumes(tmp_path, monkeypatch):
    cache, _, context, child = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, 'source_record', source)
    folder = runner.result_folder(cache, context['request'])
    invalid = folder/'status.json'
    invalid.write_text('{"completed": false, "phase": "failed", "invalid": true}', encoding='utf-8')
    temporary = folder/'checkpoint.npz.tmp'
    temporary.write_bytes(b'incomplete original compression stream')
    before = invalid.read_bytes()
    receipt = runner.seal_failure(context, cache)
    assert invalid.read_bytes() == before
    assert receipt['never_resumable'] is True and receipt['numerical_all_checks'] is False
    assert receipt['kind'] == 'failed-diagnostic' and receipt['cache_key'].endswith('-failed')
    assert f'data/hertz/{temporary.name}' in receipt['files']
    child['parent_key'] = receipt['cache_key']
    child['parent_receipt_sha256'] = runner.digest(cache/'receipt.json')
    with pytest.raises(ValueError, match='not a valid continuation'):
        runner.verify_restored(cache, child)


def test_scalar_checkpoint_leaf_shape_is_not_collapsed_into_vector():
    assert runner._array_record(np.asarray(1.))['shape'] == []
    assert runner._array_record(np.asarray([1.]))['shape'] == [1]
    assert runner._array_record(np.asarray(1.)) != runner._array_record(np.asarray([1.]))


def test_shared_deadline_keeps_review_cache_and_failure_evidence_budget():
    context = {'job_deadline_monotonic': 330*60}
    assert runner.remaining_solve_seconds(context, 0) == 265*60
    assert runner.remaining_solve_seconds(context, 30*60) == 240*60
    with pytest.raises(TimeoutError):
        runner.remaining_solve_seconds(context, 270*60)


def test_repository_cache_budget_refuses_eviction_pressure_without_deletion(monkeypatch):
    monkeypatch.setattr(runner, 'repository_cache_usage', lambda repository: {'existing_bytes': 8_500_000_000, 'repository': repository})
    with pytest.raises(ValueError, match='repository budget'):
        runner.check_cache_quota(source(), runner.GIB)
    assert runner.check_cache_quota(source(), 400_000_000)['passed'] is True


def test_packaging_uses_complete_files_and_explicit_omissions(tmp_path, monkeypatch):
    cache, logs, _, _ = fixture(tmp_path, monkeypatch)
    large = runner.result_folder(cache, runner.request('hertz'))/'trajectory_x.npy'
    large.write_bytes(b'X'*(2*1024**2))
    target = tmp_path/'delivery'
    manifest = runner.package(cache=cache, logs=logs, delivery=target, limit=2*1024**2)
    name = 'result/hertz/trajectory_x.npy'
    assert name in manifest['omitted'] and name not in manifest['files']
    assert not (target/name).exists()
    for name, record in manifest['files'].items():
        assert runner.file_record(target/name) == record
    assert sum(path.stat().st_size for path in target.rglob('*') if path.is_file()) <= manifest['budget_bytes']
    assert manifest['production_eligible'] is False and manifest['media_created'] is False
    with pytest.raises(FileExistsError):
        runner.package(cache=cache, logs=logs, delivery=target, limit=2*1024**2)


def test_all_numerical_and_audit_child_commands_are_memory_monitored():
    tree = ast.parse(Path(runner.__file__).read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'engine' and node.func.attr == 'command']
    assert len(calls) == 2
    assert all(any(item.arg == 'monitor' and isinstance(item.value, ast.Constant) and item.value.value is True
        for item in call.keywords) for call in calls)
    assert all(isinstance(call.args[1], ast.List) and call.args[1].elts[0].value == 'timeout' for call in calls)


def test_workflow_has_exact_restore_then_verify_and_never_renders_prefix():
    text = (runner.ROOT/'.github/workflows/fibre-pull-off-checks.yml').read_text(encoding='utf-8')
    assert 'runs-on: ubuntu-24.04' in text and 'persist-credentials: false' in text
    assert text.index('run_fibre_pull_off.py verify') < text.index('pip install -r')
    assert 'restore-keys:' not in text and 'lookup-only: true' in text
    assert 'CACHE_MATCH' in text and 'RECEIPT_SHA' in text
    assert 'paraview' not in text.lower() and '--gif' not in text
    assert 'retention-days: 2' in text and 'include-hidden-files: true' in text
