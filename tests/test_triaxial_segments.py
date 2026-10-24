import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import run_triaxial_segments as segments


def source():
    return {'commit': 'a'*40, 'run_id': '123', 'run_attempt': '1',
            'files': {'tools/run_triaxial_segments.py': 'b'*64}}


def endpoint(cache, req, frame, *, final=False):
    folder = segments.result_folder(cache, req)
    folder.mkdir(parents=True, exist_ok=True)
    runtime = cache/'runtime'
    runtime.mkdir(exist_ok=True)
    plan = {'frame_steps': [0, 3, 6, 10, 13], 'source': {'test': True}}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    for path in (folder/'plan.json', runtime/'plan.json'):
        segments.save(path, plan)
    status = {'signature': signature, 'frame': frame, 'steps': plan['frame_steps'][frame],
        'coupling_valid': True, 'completed': final, 'probe_complete': False,
        'segment_end_frame': frame, 'segment_complete': not final,
        'phase': 'completed' if final else 'checkpoint-ready'}
    segments.save(folder/'summary.json', status)
    segments.save(folder/'status.json', status)
    x, damage, failed = np.zeros((2, 3)), np.zeros(2), np.zeros(2, dtype=bool)
    np.savez(folder/'checkpoint.npz', frame=frame, signature=signature,
        aggregate=np.array([plan['frame_steps'][frame], -1, 0., 0., 1., 0.]),
        state_x=x, history_damage=damage, history_failed=failed,
        state_valid=True, history_valid=np.ones(2, dtype=bool), contact_law_valid=np.ones(1, dtype=bool), contact_valid=True)
    for name, data in [('x', x), ('damage', damage), ('failed', failed)]:
        np.save(folder/f'trajectory_{name}.npy', np.repeat(data[None], 5, axis=0))
    return plan


def sealed_fixture(tmp_path, stage=1):
    cache = tmp_path/'cache'
    req = segments.request()
    runtime = cache/'runtime'
    runtime.mkdir(parents=True)
    src = source()
    segments.save(runtime/'request.json', req)
    segments.save(runtime/'source.json', src)
    segments.save(runtime/'environment.json', {'python': '3.12.13'})
    context = {'request': req, 'source': src, 'stage': 1, 'end_frame': 1,
        'cache_key': segments.cache_key(src, 'tiny', 1)}
    endpoint(cache, req, 1)
    receipt, sha = segments.seal(cache, context)
    if stage == 2:
        context.update(stage=2, end_frame=2, cache_key=segments.cache_key(src, 'tiny', 2),
            parent={'frame': 1, 'receipt_sha256': sha, 'checkpoint_sha256': receipt['checkpoint_sha256'],
                    'cache_key': receipt['cache_key'], 'cumulative_snapshot_bytes': receipt['cumulative_snapshot_bytes']})
        endpoint(cache, req, 2)
        receipt, sha = segments.seal(cache, context)
    return cache, context, receipt, sha


def test_fixed_request_and_same_physical_protocol_between_segments():
    req = segments.request()
    assert req['stops'] == [1, 2, 4]
    assert req['plan_kwargs']['frames'] == 4
    assert req['plan_kwargs']['load_transits'] == .03
    assert req['plan_kwargs']['max_strain'] == 1e-7
    assert 'stop_after_frame' not in req['plan_kwargs']
    formal = segments.request('formal', theta=30, law='jkr', load_transits=12.857142857, size='20x14')
    assert formal['stops'] == [20, 40, 60]
    assert formal['plan_kwargs']['load_transits'] == 12.857142857
    assert formal['plan_kwargs']['grid_storage'] == 'tube'
    assert formal['production_eligible'] is False


@pytest.mark.parametrize('kw', [{'theta': 5}, {'load_transits': 20}, {'size': '20x14'}])
def test_tiny_rejects_silently_ignored_requests(kw):
    with pytest.raises(ValueError):
        segments.request(**kw)


def test_cache_keys_separate_attempts_and_endpoints():
    src = source()
    keys = {segments.cache_key(src, 'tiny', frame) for frame in (1, 2, 4)}
    src['run_attempt'] = '2'
    keys.add(segments.cache_key(src, 'tiny', 1))
    assert len(keys) == 4


def test_geometry_archive_preserves_arrays_metadata_and_candidates(tmp_path):
    geometry = {'x': np.array([[.1, .2, .3]], dtype=np.float64), 'mass': np.array([.4]),
        'fibre_id': np.array([2], dtype=np.int32), 'reference_centrelines': {'offsets': np.array([0, 1])},
        'metadata': {'angle': .3, 'fibres': [{'particle_slice': [0, 1]}]},
        'contact_candidates': [{'normal': [1., 0., 0.], 'active': False}]}
    segments.write_geometry(tmp_path, geometry)
    restored = segments.read_geometry(tmp_path)
    np.testing.assert_array_equal(restored['x'], geometry['x'])
    assert restored['fibre_id'].dtype == np.int32
    assert restored['metadata'] == geometry['metadata']
    assert restored['contact_candidates'] == geometry['contact_candidates']
    assert restored['x'].tobytes() == geometry['x'].tobytes()
    with (tmp_path/'geometry.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        segments.read_geometry(tmp_path)


def test_default_cache_estimate_and_oversize_rejection():
    plan = {'particles': 683232, 'trajectory_estimated_bytes': 61*683232*45}
    assert segments.cache_estimate(plan)['estimated_three_snapshot_bytes'] < 8_000_000_000
    plan = {'particles': 898704, 'trajectory_estimated_bytes': 61*898704*45}
    with pytest.raises(ValueError, match='exceeding'):
        segments.cache_estimate(plan)


def test_final_disk_budget_includes_raw_copy_and_vtk():
    plan = {'particles': 683232, 'saved_states': 61, 'trajectory_estimated_bytes': 61*683232*45}
    record = segments.disk_admission(plan, stage=4, free_bytes=30_000_000_000, raw_already_present=True)
    assert record['final_copy_and_render_reserve_bytes'] > 6_000_000_000
    assert record['raw_growth_bytes'] == 0
    with pytest.raises(ValueError, match='free disk'):
        segments.disk_admission(plan, stage=4, free_bytes=3_000_000_000, raw_already_present=True)


@pytest.mark.parametrize('stage', [1, 2])
def test_sealed_snapshot_hashes_parent_chain_and_exact_byte_budget(tmp_path, stage):
    cache, context, receipt, sha = sealed_fixture(tmp_path, stage)
    checked, _ = segments.verify_cache(cache, context['request'], context['source'], context['end_frame'], sha, context['cache_key'])
    assert checked == receipt
    assert checked['full_review_passed'] is False
    assert checked['endpoint']['continuation_ready'] is True
    assert checked['raw_snapshot_bytes'] == sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    if stage == 2:
        assert checked['parent']['frame'] == 1
        assert checked['prior_snapshot_bytes'] > 0


@pytest.mark.parametrize('mutation', ['bytes', 'extra', 'missing', 'source', 'request', 'key', 'receipt'])
def test_restored_tampering_is_rejected_before_continuation(tmp_path, mutation):
    cache, context, receipt, sha = sealed_fixture(tmp_path)
    req, src, key = copy.deepcopy(context['request']), copy.deepcopy(context['source']), context['cache_key']
    folder = segments.result_folder(cache, req)
    if mutation == 'bytes': (folder/'trajectory_x.npy').write_bytes(b'bad')
    if mutation == 'extra': (cache/'credential.txt').write_text('not allowed')
    if mutation == 'missing': (folder/'checkpoint.npz').unlink()
    if mutation == 'source': src['commit'] = 'c'*40
    if mutation == 'request': req['plan_kwargs']['young'] = 1.
    if mutation == 'key': key += '-fallback'
    if mutation == 'receipt': sha = '0'*64
    with pytest.raises((ValueError, FileNotFoundError)):
        segments.verify_cache(cache, req, src, 1, sha, key)


def test_old_receipt_cannot_be_replaced_in_parent_chain(tmp_path):
    cache, context, receipt, sha = sealed_fixture(tmp_path, 2)
    old = cache/'receipts/frame-01.json'
    value = segments.read_json(old)
    value['checkpoint_sha256'] = '0'*64
    segments.save(old, value)
    with pytest.raises(ValueError):
        segments.verify_cache(cache, context['request'], context['source'], 2, sha, context['cache_key'])


@pytest.mark.parametrize('field,value', [('coupling_valid', False), ('phase', 'failed'), ('completed', True), ('segment_complete', False)])
def test_invalid_partial_never_becomes_a_ready_segment(tmp_path, field, value):
    req = segments.request()
    plan = endpoint(tmp_path, req, 1)
    folder = segments.result_folder(tmp_path, req)
    status = segments.read_json(folder/'status.json')
    status[field] = value
    segments.save(folder/'status.json', status)
    with pytest.raises(ValueError, match='Endpoint'):
        segments.check_endpoint(folder, plan, 1, final=False)


def test_checkpoint_and_saved_mmap_must_match(tmp_path):
    req = segments.request()
    plan = endpoint(tmp_path, req, 1)
    folder = segments.result_folder(tmp_path, req)
    values = np.load(folder/'trajectory_x.npy')
    values[1, 0, 0] = 1.
    np.save(folder/'trajectory_x.npy', values)
    with pytest.raises(AssertionError):
        segments.check_endpoint(folder, plan, 1, final=False)


def test_complete_final_state_is_not_only_partial_ready(tmp_path):
    req = segments.request()
    plan = endpoint(tmp_path, req, 4, final=True)
    result = segments.check_endpoint(segments.result_folder(tmp_path, req), plan, 4, final=True)
    assert result['full_protocol_completed'] is True
    assert result['continuation_ready'] is False
    assert result['full_review_passed'] is False


def test_resume_worker_loads_original_geometry_without_resampling(tmp_path, monkeypatch):
    from examples import triaxial_failure as driver
    cache = tmp_path/'cache'
    runtime = cache/'runtime'
    runtime.mkdir(parents=True)
    req = segments.request()
    geo = {'x': np.array([[.12345678912345678, 0., 0.]]), 'metadata': {'real': True}}
    segments.write_geometry(runtime, geo)
    plan = {'source': {'a': 'b'}, 'sampling_hash': driver.descriptor(geo['x']), 'geometry': geo['metadata']}
    segments.save(runtime/'plan.json', plan)
    segments.save(runtime/'environment.json', {'test': True})
    segments.save(runtime/'request.json', req)
    context = {'stage': 2, 'request': req, 'source': source(), 'end_frame': 2}
    monkeypatch.setattr(segments, 'CACHE', cache)
    monkeypatch.setattr(segments, 'environment', lambda: {'test': True})
    monkeypatch.setattr(segments.study, 'source_record', lambda: source())
    monkeypatch.setattr(driver, 'source_record', lambda: {'a': 'b'})
    monkeypatch.setattr(segments, 'install_layout_checks', lambda *args, **kwargs: None)
    def forbidden(**kwargs):
        raise AssertionError('Resume regenerated geometry')
    monkeypatch.setattr(driver, 'plan_case', forbidden)
    calls = []
    monkeypatch.setattr(driver, 'run_case', lambda geometry, saved_plan, output, **kwargs: calls.append((geometry, saved_plan, output, kwargs)))
    segments.worker(context, 'solve')
    assert calls[0][0]['x'].tobytes() == geo['x'].tobytes()
    assert calls[0][1] == plan
    assert calls[0][3] == {'resume': True, 'stop_after_frame': 2}


def test_layout_and_coverage_drift_rejected_without_replacing_original(tmp_path):
    from dataclasses import dataclass
    @dataclass
    class Layout:
        coverage_pairs: object
        coverage_seed: object
        radius: float
    record = segments.structure_descriptor(Layout(np.array([[1, 2], [3, 4]]), np.array([0, 0]), .0001175))
    path = tmp_path/'layout.json'
    segments.verify_layout_record(path, 'contact', record, allow_create=True)
    original = path.read_bytes()
    changed = segments.structure_descriptor(Layout(np.array([[1, 2], [4, 5]]), np.array([0, 0]), .0001175))
    with pytest.raises(ValueError, match='coverage'):
        segments.verify_layout_record(path, 'contact', changed, allow_create=False)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='missing'):
        segments.verify_layout_record(path, 'grid', {}, allow_create=False)


def test_tiny_wrapper_returns_all_missing_inputs_without_claiming_admission(tmp_path):
    report = segments.tiny_parity(tmp_path/'missing-left', tmp_path/'missing-right')
    assert report['passed'] is False
    assert report['production_eligible'] is False
    assert len(report['errors']) > 1
    assert report['schema'] == 'triaxial-tiny-cross-cpu-parity-v1'


def test_memory_guard_is_explicit_for_every_new_child_command():
    import ast
    tree = ast.parse(Path(segments.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
             and node.func.value.id == 'engine' and node.func.attr == 'command']
    assert len(calls) == 4
    assert all(any(kw.arg == 'monitor' and isinstance(kw.value, ast.Constant) and kw.value.value is True
                   for kw in call.keywords) for call in calls)


def test_segment_workflows_are_serial_exact_and_raw_cache_precedes_render():
    parent = (segments.ROOT/'.github/workflows/triaxial-segments.yml').read_text()
    child = (segments.ROOT/'.github/workflows/triaxial-segment-job.yml').read_text()
    assert 'needs: first' in parent and 'needs: second' in parent and 'needs: third' in parent
    assert 'group: triaxial-segmented-study-cache' in parent
    assert 'restore-keys:' not in child
    assert 'fail-on-cache-miss: true' in child and 'lookup-only: true' in child
    assert 'ref: ${{ github.sha }}' in child and "python-version: '3.12.13'" in child
    assert child.index('actions/cache/save@v5') < child.index('run_triaxial_segments.py finalize')
    assert child.index('run_triaxial_segments.py verify') < child.index('pip install -r')
    assert 'include-hidden-files: true' in child


def package_fixture(tmp_path, monkeypatch, mode='formal'):
    cache, logs, delivery = tmp_path/'cache', tmp_path/'logs', tmp_path/'delivery'
    logs.mkdir()
    req = segments.request(mode)
    folder = segments.result_folder(cache, req)
    folder.mkdir(parents=True)
    segments.save(folder/'status.json', {'completed': False})
    np.savez(folder/'checkpoint.npz', value=np.arange(4))
    np.save(folder/'trajectory_x.npy', np.zeros((5, 2, 3)))
    context = {'request': req, 'stage': 1, 'cache_key': 'test-key', 'sealed_receipt_sha256': 'a'*64}
    for name, value in [('ROOT', tmp_path), ('CACHE', cache), ('LOGS', logs), ('DELIVERY', delivery)]:
        monkeypatch.setattr(segments, name, value)
    return context, folder, logs, delivery


def test_verified_intermediate_formal_archives_only_small_metadata(tmp_path, monkeypatch):
    context, folder, logs, delivery = package_fixture(tmp_path, monkeypatch)
    segments.save(logs/'cache-confirmation.json', {'verified': True, 'cache_key': 'test-key', 'receipt_sha256': 'a'*64})
    report = segments.package(context)
    assert report['total_budget_bytes'] == 8*1024**2
    assert all(not name.endswith(('.npz', '.npy')) for name in report['files'])
    assert report['production_eligible'] is False


def test_tiny_package_keeps_complete_raw_history_within_total_budget(tmp_path, monkeypatch):
    context, folder, logs, delivery = package_fixture(tmp_path, monkeypatch, mode='tiny')
    report = segments.package(context)
    assert report['full_tiny_payload_selected'] is True
    assert any(name.endswith('checkpoint.npz') for name in report['files'])
    assert any(name.endswith('trajectory_x.npy') for name in report['files'])
    assert report['total_budget_bytes'] == 192*1024**2


def test_unsaved_formal_snapshot_uses_full_diagnostic_packaging(tmp_path, monkeypatch):
    context, folder, logs, delivery = package_fixture(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(segments.study, 'package', lambda *args, **kwargs: called.append(kwargs) or {'diagnostics': True})
    assert segments.package(context) == {'diagnostics': True}
    assert len(called) == 1


def test_cache_confirmation_cannot_accept_partial_key_hit(tmp_path, monkeypatch):
    context, folder, logs, delivery = package_fixture(tmp_path, monkeypatch)
    context['end_frame'] = 20
    monkeypatch.setenv('CACHE_HIT', 'false')
    monkeypatch.setenv('CACHE_MATCH', 'test-key-other')
    monkeypatch.setenv('RECEIPT_SHA', 'a'*64)
    with pytest.raises(ValueError, match='exact cache hit'):
        segments.confirm_cache(context)


def test_finalization_job_uses_existing_last_key_and_never_advances_or_seals(tmp_path, monkeypatch):
    import argparse
    req = segments.request()
    monkeypatch.setattr(segments, 'ROOT', tmp_path)
    monkeypatch.setattr(segments, 'CACHE', tmp_path/'cache')
    monkeypatch.setattr(segments, 'LOGS', tmp_path/'logs')
    monkeypatch.setattr(segments, 'CONTEXT', tmp_path/'logs/job.json')
    monkeypatch.setattr(segments.platform, 'python_version', lambda: '3.12.13')
    monkeypatch.setattr(segments.study, 'source_record', source)
    args = argparse.Namespace(mode='tiny', theta=0, law='hertz', load_transits=10., size='24x16', stage=4, parent_receipt='a'*64)
    segments.prepare(args)
    context = segments.read_json(tmp_path/'logs/job.json')
    assert context['previous_frame'] == context['end_frame'] == req['stops'][-1]
    assert context['cache_key'] == context['parent_key'] == segments.cache_key(source(), 'tiny', 4)
    with pytest.raises(ValueError, match='must not advance'):
        segments.run(context)
    with pytest.raises(ValueError, match='three numerical'):
        segments.seal(tmp_path/'cache', context)
    with pytest.raises(ValueError, match='never saves'):
        segments.confirm_cache(context)


def test_fourth_job_workflow_does_not_run_or_save_a_fourth_segment():
    child = (segments.ROOT/'.github/workflows/triaxial-segment-job.yml').read_text()
    assert child.count('if: inputs.stage < 4') == 4
    assert 'if: inputs.stage == 4' in child
    assert 'if: inputs.stage == 3' not in child
    assert child.count('actions/cache/save@v5') == 2  # Accepted raw and separate non-resumable failure evidence.
    assert "steps.run.outcome == 'failure'" in child
    assert 'confirm-diagnostic' in child
