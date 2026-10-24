"""Pure IO admission tests; these fixtures do not claim a real plate solve."""

import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

from tools import run_woven_large_segments as runner


def records():
    req = runner.request()
    source = {'files': {'kernel.py': 'a' * 64}, 'fingerprint': 'b' * 64}
    plan = {'source': source, 'architecture': 'plain', 'loading': 'sphere-impact',
        'particles': 819200, 'frames': 60, 'probe_steps': 0, 'stop_steps': 14460,
        'full_steps': 14460, 'duration_s': 200e-6, 'grid': {'dx_m': .0005}, 'CFL': .15,
        'frame_steps': list(range(0, 14461, 241)), 'output_steps': list(range(0, 14461, 241))}
    summary = {'source': source, 'particles': [819200], 'full_steps': 14460,
        'target_simulation_time': 200e-6, 'completed': False, 'probe_complete': False,
        'segment_complete': True, 'phase': 'checkpoint-ready', 'segment_start_frame': 0,
        'segment_end_frame': 20, 'runtime_stop_after_frame': 20, 'steps': 4820,
        'simulation_time': 200e-6 / 3}
    status = {**summary, 'frame': 20, 'frames': 60, 'target_steps': 14460,
        'time': summary['simulation_time'], 'coupling_valid': True, 'signature': runner.object_hash(plan)}
    review = {'checks': {'actual_independent_gate': True}, 'producer_source': source,
        'passed': False, 'segment_passed': True, 'probe_passed': False,
        'full_protocol_completed': False, 'full_saved_history_checked': True,
        'segment_end_frame': 20, 'allocated_states': 61, 'accepted_saved_states': 21}
    return req, source, plan, summary, status, review


def validate(values):
    req, source, plan, summary, status, review = values
    return runner.validate_acceptance(plan, summary, status, review, req, 20, 0, source)


def test_first_segment_is_partial_full_plan_not_probe():
    report = validate(records())
    assert report['accepted_prefix'] and report['segment_passed']
    assert not report['full_protocol_completed'] and not report['media_created']
    assert not report['production_eligible']


@pytest.mark.parametrize('target,key,value', [
    (2, 'probe_steps', 128), (2, 'stop_steps', 4820), (2, 'duration_s', 200e-6 / 3),
    (2, 'source', {}), (3, 'particles', [128]), (3, 'completed', True),
    (3, 'segment_start_frame', 19), (3, 'simulation_time', .1),
    (4, 'signature', '0' * 64), (4, 'coupling_valid', False), (4, 'target_steps', 4820),
    (4, 'frame', 19), (4, 'runtime_stop_after_frame', None),
    (5, 'checks', {}), (5, 'checks', {'gate': 1}), (5, 'checks', {'gate': False}),
    (5, 'producer_source', {}), (5, 'full_saved_history_checked', False),
    (5, 'accepted_saved_states', 61), (5, 'allocated_states', 21), (5, 'passed', True),
])
def test_false_or_mixed_prefix_is_rejected(target, key, value):
    values = list(records())
    values[target][key] = value
    with pytest.raises(ValueError):
        validate(values)


def test_full_f60_requires_full_review_but_does_not_approve_public_display():
    req, source, plan, summary, status, review = records()
    for record in (summary, status):
        record.update(completed=True, segment_complete=False, phase='completed', segment_start_frame=40,
                      segment_end_frame=60, runtime_stop_after_frame=60, steps=14460)
    summary['simulation_time'] = status['time'] = 200e-6
    status['frame'] = 60
    review.update(passed=True, segment_passed=False, full_protocol_completed=True,
                  segment_end_frame=60, accepted_saved_states=61)
    result = runner.validate_acceptance(plan, summary, status, review, req, 60, 40, source)
    assert result['full_review_passed'] and not result['production_eligible']
    review['passed'] = False
    with pytest.raises(ValueError):
        runner.validate_acceptance(plan, summary, status, review, req, 60, 40, source)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(runner, 'LOGS', tmp_path / 'logs')
    monkeypatch.setattr(runner, 'CONTEXT', tmp_path / 'logs/job.json')
    monkeypatch.setattr(runner, 'FINAL', tmp_path / 'final')
    runner.LOGS.mkdir()
    source = {'source_kind': 'github-actions', 'repository': 'owner/repo', 'commit': 'c' * 40,
              'tree': 'd' * 40, 'numerical_source': records()[1], 'files': {'test.py': 'e' * 64},
              'review_source_sha256': 'f' * 64}
    execution = {**source, 'run_id': '123', 'run_attempt': '1'}
    monkeypatch.setattr(runner, 'source_record', lambda: source)
    monkeypatch.setattr(runner.engine, 'identity', lambda: execution)
    locked = {'python': runner.PYTHON_VERSION,
              'packages': {name: '1.0' for name in ('numpy', 'scipy', 'jax', 'jaxlib', 'psutil', 'pillow')}}
    monkeypatch.setattr(runner, 'environment', lambda: locked)
    monkeypatch.setattr(runner, 'emit', lambda _: None)
    context = {'stage': '20', 'end_frame': 20, 'previous_frame': 0, 'request': runner.request(),
               'source': source, 'execution': execution, 'parent_key': '', 'parent_receipt_sha256': '',
               'deadline_unix_s': 1e20}
    context['cache_key'] = runner.cache_key(source['commit'], '123', '1', context['request'], 20)
    folder = runner.result_folder(runner.CACHE, context['request'])
    folder.mkdir(parents=True)
    for name in runner.probe.REVIEW_INPUTS:
        path = folder / name
        if name.endswith('.json'):
            runner.save(path, {'fixture': name})
        elif name.endswith('.npz'):
            np.savez(path, state_x=np.zeros((2, 3)), frame=np.array(20))
        elif name.endswith('.npy'):
            np.save(path, np.zeros((2, 2)))
        else:
            path.write_text('time,value\n0,0\n', encoding='utf-8')
    runner.save(folder / 'interpretation.json', {'checks': {'fixture': True}})
    runtime = runner.CACHE / 'runtime'
    runtime.mkdir()
    runner.save(runtime / 'environment.json', locked)
    runner.save(runtime / 'request.json', context['request'])
    runner.save(runtime / 'source.json', source)
    (runtime / 'requirements.txt').write_text(runner.requirements_text(locked), encoding='utf-8')
    return context, folder


def seal_fixture(context):
    acceptance = {'accepted_prefix': True, 'end_frame': context['end_frame'],
        'full_protocol_completed': context['end_frame'] == 60, 'production_eligible': False, 'media_created': False}
    command = {'completed': True, 'memory': {'samples': 10, 'stopped_for_memory': False,
        'peak_tree_uss_bytes': 1000, 'minimum_system_available_bytes': 4 * 1024**3}}
    return runner.seal(context, acceptance, command)


def restore_context(context, frame=20):
    return {**context, 'stage': str(frame + 20), 'previous_frame': frame,
        'parent_key': context['cache_key'],
        'parent_receipt_sha256': runner.digest(runner.CACHE / 'receipts' / f'frame-{frame}.json')}


def test_complete_seal_and_exact_restore_preserve_checkpoint_leaf_bytes(workspace):
    context, folder = workspace
    receipt = seal_fixture(context)
    actual = sum(path.stat().st_size for path in runner.CACHE.rglob('*') if path.is_file())
    assert receipt['snapshot_raw_bytes'] == receipt['cumulative_raw_snapshot_bytes'] == actual
    restored = runner.verify_restore(restore_context(context))
    assert restored['checkpoint_members'] == runner.checkpoint_members(folder / 'checkpoint.npz')
    assert restored['kind'] == 'accepted-prefix' and restored['never_resumable'] is False


@pytest.mark.parametrize('change', ['raw', 'missing', 'extra', 'source', 'receipt'])
def test_restore_rejects_changed_or_unlisted_bytes_before_scientific_import(workspace, change):
    context, folder = workspace
    seal_fixture(context)
    resumed = restore_context(context)
    if change == 'raw':
        (folder / 'trajectory_x.npy').write_bytes(b'corrupted')
    elif change == 'missing':
        (folder / 'reference.npz').unlink()
    elif change == 'extra':
        (runner.CACHE / 'credential.txt').write_text('not allowed')
    elif change == 'source':
        runner.save(runner.CACHE / 'runtime/source.json', {'changed': True})
    else:
        path = runner.CACHE / 'receipts/frame-20.json'
        path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError):
        runner.verify_restore(resumed)


def test_checkpoint_members_reject_duplicate_and_path_traversal(tmp_path):
    for name in ('../state.npy', 'state.json'):
        path = tmp_path / 'bad.npz'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr(name, b'not an admitted member')
        with pytest.raises(ValueError):
            runner.checkpoint_members(path)
    path = tmp_path / 'duplicate.npz'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('state.npy', b'a')
        with pytest.warns(UserWarning):
            archive.writestr('state.npy', b'b')
    with pytest.raises(ValueError):
        runner.checkpoint_members(path)


def test_failed_raw_is_retained_but_diagnostic_cache_cannot_resume(workspace, monkeypatch):
    context, folder = workspace
    runner.save(folder / 'status.json', {'coupling_valid': False, 'phase': 'failed', 'frame': 7})
    before = runner.file_record(folder / 'status.json')
    diagnostic = runner.seal_failure(context)
    assert diagnostic['never_resumable'] is True and diagnostic['numerically_accepted'] is False
    assert diagnostic['original_status']['frame'] == 7
    assert runner.file_record(folder / 'status.json') == before
    with pytest.raises(ValueError):
        runner.validate_receipt(diagnostic, context, 20, diagnostic['cache_key'])
    monkeypatch.setenv('CACHE_HIT', 'true')
    monkeypatch.setenv('CACHE_MATCHED_KEY', diagnostic['cache_key'])
    runner.confirm_cache(context)
    assert runner.read(runner.LOGS / 'cache-confirmation.json')['never_resumable'] is True


def test_incomplete_failed_raw_is_preserved_without_claiming_complete_history(workspace):
    context, folder = workspace
    (folder / 'trajectory_x.npy').unlink()
    diagnostic = runner.seal_failure(context)
    assert diagnostic['missing_result_files'] == ['trajectory_x.npy']
    assert diagnostic['never_resumable'] is True


def test_atomic_write_remainders_and_broken_checkpoint_are_kept_as_failed_bytes(workspace):
    context, folder = workspace
    (folder / 'checkpoint.npz').write_bytes(b'incomplete ZIP bytes')
    (folder / 'checkpoint.npz.tmp').write_bytes(b'pending uncommitted write')
    (folder / 'status.json').write_bytes(b'{"interrupted":')
    diagnostic = runner.seal_failure(context)
    assert diagnostic['checkpoint_members'] is None
    assert 'BadZipFile' in diagnostic['checkpoint_member_read_error']
    assert 'decode_error' in diagnostic['original_status']
    assert any(name.endswith('checkpoint.npz.tmp') for name in diagnostic['files'])
    assert diagnostic['numerically_accepted'] is False


def test_saved_cache_requires_exact_hit_not_prefix_match(workspace, monkeypatch):
    context, _ = workspace
    seal_fixture(context)
    monkeypatch.setenv('CACHE_HIT', 'false')
    monkeypatch.setenv('CACHE_MATCHED_KEY', context['cache_key'])
    with pytest.raises(ValueError):
        runner.confirm_cache(context)
    monkeypatch.setenv('CACHE_HIT', 'true')
    monkeypatch.setenv('CACHE_MATCHED_KEY', context['cache_key'] + '-other')
    with pytest.raises(ValueError):
        runner.confirm_cache(context)


def test_two_segment_chain_binds_parent_receipt_and_original_checkpoint(workspace):
    context, folder = workspace
    first = seal_fixture(context)
    resumed = restore_context(context)
    runner.verify_restore(resumed)
    resumed.update(end_frame=40, stage='40', cache_key=runner.cache_key('c' * 40, '456', '1', context['request'], 40))
    np.savez(folder / 'checkpoint.npz', state_x=np.ones((2, 3)), frame=np.array(40))
    second = seal_fixture(resumed)
    assert second['parent']['checkpoint_sha256'] == first['checkpoint']['sha256']
    assert second['cumulative_raw_snapshot_bytes'] == first['snapshot_raw_bytes'] + second['snapshot_raw_bytes']
    runner.verify_restore(restore_context(resumed, 40))


def test_finalization_cannot_advance_seal_or_save_another_cache(workspace):
    context, _ = workspace
    context['stage'] = 'finalize'
    for function, args in ((runner.worker, (context,)), (runner.run, (context,)),
                           (runner.seal, (context, {}, {})), (runner.confirm_cache, (context,)),
                           (runner.seal_failure, (context,))):
        with pytest.raises(ValueError):
            function(*args)
    assert runner.previous_frame('finalize') == runner.frame_for_stage('finalize') == 60


def test_guard_is_real_and_deadline_reserves_evidence_tail(workspace, monkeypatch):
    context, _ = workspace
    context['deadline_unix_s'] = 1000 + 300 * 60
    monkeypatch.setattr(runner.time, 'time', lambda: 1000)
    calls = []
    monkeypatch.setattr(runner.engine, 'command', lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.guarded(context, 'solve', ['python', 'worker.py'], 265, 40)
    assert calls[0][1]['monitor'] is True
    assert '15600s' in calls[0][0][1]  # 260 minutes after the fixed evidence tail.
    context['deadline_unix_s'] = 1010
    with pytest.raises(TimeoutError):
        runner.guarded(context, 'solve', ['python'], 265, 40)


def test_budget_reserves_complete_raw_snapshots_and_actual_free_disk():
    estimate = runner.cache_estimate()
    assert estimate['history_bytes'] == 61 * 819200 * 36
    assert estimate['estimated_three_snapshot_bytes'] < 8_000_000_000
    with pytest.raises(ValueError):
        runner.disk_admission('finalize', 1)
    accepted = runner.disk_admission('finalize', 100 * 1024**3)
    assert accepted['final_copy_and_vtk_reserve_bytes'] > estimate['estimated_snapshot_bytes']


def test_package_budget_is_whole_files_including_manifest(workspace, tmp_path):
    context, folder = workspace
    (folder / 'checkpoint.npz').write_bytes(b'z' * (2 * 1024**2))
    delivery = tmp_path / 'delivery'
    manifest = runner.package(context, delivery=delivery, limit=2 * 1024**2)
    assert 'result/checkpoint.npz' in manifest['omitted']
    assert not (delivery / 'result/checkpoint.npz').exists()
    assert sum(path.stat().st_size for path in delivery.rglob('*') if path.is_file()) <= 2 * 1024**2
    for name, record in manifest['files'].items():
        assert runner.file_record(delivery / name) == record


def test_cache_confirmation_controls_compact_package_not_solver_completion(workspace, tmp_path):
    context, _ = workspace
    seal_fixture(context)
    runner.save(runner.LOGS / 'cache-confirmation.json', {'verified': True, 'cache_key': context['cache_key']})
    manifest = runner.package(context, delivery=tmp_path / 'compact')
    assert manifest['budget_bytes'] == 8 * 1024**2
    assert manifest['cache_verified'] and not manifest['media_created']
    assert not manifest['production_eligible']
    assert 'result/trajectory_x.npy' in manifest['omitted']


def test_prepare_rejects_probe_parent_and_wrong_receipt_hash_before_io(workspace, monkeypatch):
    context, _ = workspace
    # Preparation is normally a fresh checkout, not a populated fixture cache.
    monkeypatch.setattr(runner, 'CACHE', runner.ROOT / 'empty-cache')
    args = argparse.Namespace(stage='20', architecture='plain', loading='sphere-impact',
                              parent_run='128', parent_attempt='1', parent_receipt_sha='a' * 64)
    with pytest.raises(ValueError):
        runner.prepare(args)
    args.stage, args.parent_receipt_sha = '40', 'short'
    with pytest.raises(ValueError):
        runner.prepare(args)


def test_prepare_finalization_uses_existing_f60_key(workspace, monkeypatch):
    context, _ = workspace
    monkeypatch.setattr(runner, 'CACHE', runner.ROOT / 'empty-cache')
    monkeypatch.setattr(runner, 'repository_cache_usage', lambda _: {'active_caches_size_in_bytes': 800_000_000})
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: argparse.Namespace(free=100 * 1024**3))
    args = argparse.Namespace(stage='finalize', architecture='plain', loading='sphere-impact',
                              parent_run='100', parent_attempt='1', parent_receipt_sha='a' * 64)
    value = runner.prepare(args)
    assert value['parent_key'] == value['cache_key']
    assert value['end_frame'] == value['previous_frame'] == 60


def test_workflow_has_exact_restore_before_install_and_no_automatic_next_stage():
    path = Path(__file__).resolve().parents[1] / '.github/workflows/woven-large-segments.yml'
    text = path.read_text(encoding='utf-8')
    assert text.index('python tools/run_woven_large_segments.py verify') < text.index('python -m pip install')
    assert 'fail-on-cache-miss: true' in text and 'restore-keys:' not in text
    assert 'persist-credentials: false' in text and 'cancel-in-progress: false' in text
    assert 'workflow_dispatch:' in text and 'workflow_run:' not in text
    assert "steps.run.outcome == 'failure'" in text and 'seal-failure' in text
    assert "if: inputs.stage == 'finalize'" in text
    assert 'include-hidden-files: true' in text


def test_source_record_does_not_import_scientific_packages(monkeypatch):
    monkeypatch.delenv('GITHUB_SHA', raising=False)
    monkeypatch.setattr(runner.engine, 'identity', lambda: {'source_kind': 'github-actions', 'repository': 'owner/repo',
        'commit': 'a' * 40, 'tree': 'b' * 40})
    monkeypatch.setattr(runner, 'committed_source', lambda files: {name: 'c' * 40 for name in files})
    before = {name for name in sys.modules if name == 'jax' or name.startswith('jax.')}
    source = runner.source_record()
    after = {name for name in sys.modules if name == 'jax' or name.startswith('jax.')}
    assert before == after
    assert source['numerical_source']['fingerprint'] == runner.object_hash(source['numerical_source']['files'])
    assert source['review_source_sha256'] == source['files']['tools/review_woven_large.py']


def test_cached_requirements_cannot_embed_install_options():
    locked = {'python': runner.PYTHON_VERSION,
              'packages': {name: '1.0' for name in ('numpy', 'scipy', 'jax', 'jaxlib', 'psutil', 'pillow')}}
    locked['packages']['numpy'] = '1.0\n--index-url https://untrusted.invalid/'
    with pytest.raises(ValueError):
        runner.requirements_text(locked)


def test_actual_git_blob_identity_is_checked_before_source_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    path = tmp_path / 'kernel.py'
    path.write_bytes(b'x = 1\n')
    expected = runner.hashlib.sha1(b'blob 6\0x = 1\n').hexdigest()
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *args, **kwargs: expected + '\n')
    assert runner.committed_source({'kernel.py': runner.digest(path)}) == {'kernel.py': expected}
    path.write_bytes(b'x = 2\n')
    with pytest.raises(ValueError):
        runner.committed_source({'kernel.py': runner.digest(path)})


def test_finalization_rechecks_actual_free_space_after_restore(workspace, monkeypatch):
    context, _ = workspace
    context['stage'] = 'finalize'
    monkeypatch.setattr(runner, 'verify_restore', lambda _: None)
    monkeypatch.setattr(runner, 'inspect_result', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: argparse.Namespace(free=1))
    with pytest.raises(ValueError, match='disk space'):
        runner.finalize(context)
    assert not runner.FINAL.exists()


def test_integrated_saved_tool_routes_the_one_sphere_ledger_not_the_whole_definition():
    sphere = {'motion_kind': 'integrated-saved-states', 'state_file': 'observables.csv'}
    summary = {'definition': {'rigid_spheres': [sphere], 'loading': 'sphere-impact'}}
    assert runner.integrated_tool_declaration(summary) is sphere
    summary['definition']['rigid_spheres'] = [sphere, sphere]
    with pytest.raises(ValueError):
        runner.integrated_tool_declaration(summary)


def test_unaccepted_prefix_cannot_be_sealed(workspace):
    context, _ = workspace
    with pytest.raises(ValueError, match='accepted saved prefix'):
        runner.seal(context, {'accepted_prefix': False}, {})
