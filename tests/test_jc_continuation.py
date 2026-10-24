import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np
import pytest

from tools import run_jc_continuation as continuation
from tools import run_jc_refinement as frozen


helpers = runpy.run_path(str(Path(__file__).with_name('test_jc_refinement.py')))


@pytest.fixture
def parent(tmp_path, monkeypatch):
    folder, plan = helpers['prefix'].__wrapped__(tmp_path, monkeypatch)
    frozen.audit_prefix(folder, plan['source'])
    cache = tmp_path/'cache'
    frozen.seal_carry(folder, plan['source'], cache)
    request = continuation.request_values(frozen.cache_key(plan['source']),
        continuation.digest(cache/'carry-receipt.json'), 10, 20)
    return folder, plan, cache, request


def extend_prefix(folder, frame=20, *, failed=False, support_cut=False):
    plan, cp = [continuation.read_json(folder/name) for name in ('plan.json', 'checkpoint.json')]
    with np.load(folder/cp['states'][0]) as archive:
        arrays = {key: archive[key] for key in archive.files}
    reference = arrays['x'].copy()
    with np.load(folder/'frames_data/frame_00010.npz') as archive:
        original = {key: archive[key] for key in archive.files}
    dt = frozen.DURATION/(frozen.FRAMES*frozen.SUBSTEPS)
    for index in range(11, frame+1):
        arrays['time'] = np.asarray(index*frozen.SUBSTEPS*dt)
        if failed and index >= 15:
            arrays['history_failed'][0] = True
            arrays['history_damage'][0] = 1.
        if support_cut:
            arrays['x'][2:, 0] = reference[2:, 0]+.0005
        metrics = frozen.final_metrics(arrays, reference, .0016/48)
        values = {key: value.copy() for key, value in original.items()}
        values.update(x=arrays['x'].copy(), time=arrays['time'], damage=arrays['history_damage'].copy(),
            failed=arrays['history_failed'].astype('uint8'))
        np.savez_compressed(folder/'frames_data'/f'frame_{index:05d}.npz', **values)
        row = copy.deepcopy(cp['rows'][-1])
        row['time'] = float(arrays['time'])
        observed = copy.deepcopy(cp['observations'][-1])
        observed.update(time=float(arrays['time']), **{name: metrics[name] for name in (
            'failed_particles', 'max_damage', 'central_active_particles',
            'maximum_equivalent_plastic_strain', 'central_material_section_force_N')})
        cp['rows'].append(row)
        cp['observations'].append(observed)
    cp.update(frame=frame, solver_seconds=cp['solver_seconds']+3.)
    np.savez(folder/cp['states'][0], **arrays)
    continuation.save(folder/'checkpoint.json', cp)
    status = continuation.read_json(folder/'status.json')
    status.update(frame=frame, steps_completed=frame*frozen.SUBSTEPS, time=frame*frozen.SUBSTEPS*dt)
    for name in ('status.json', 'summary.json', 'segment.json'):
        continuation.save(folder/name, status)


def test_original_carry_is_validated_without_relabeling_old_source(parent):
    _, plan, cache, request = parent
    receipt = continuation.verify_carry(cache, request, check_source=False)
    assert receipt['source'] == plan['source']
    assert receipt['frame'] == 10
    assert not receipt['cross_job_resume_verified']


@pytest.mark.parametrize('defect', ['hash', 'extra', 'missing', 'frame', 'key', 'source', 'receipt', 'review'])
def test_parent_carry_refuses_corruption_or_wrong_identity(parent, defect):
    _, _, cache, request = parent
    if defect == 'hash':
        (cache/'result/frames_data/frame_00005.npz').write_bytes(b'changed')
    elif defect == 'extra':
        (cache/'unexpected.py').write_text('pass')
    elif defect == 'missing':
        (cache/'result/frames_data/frame_00005.npz').unlink()
    elif defect == 'frame':
        request['from_frame'] = 20
    elif defect == 'key':
        request['parent_key'] += '-wrong'
    elif defect == 'receipt':
        request['parent_receipt_sha256'] = '0'*64
    else:
        name = 'numerical-source.json' if defect == 'source' else 'prefix-review.json'
        value = continuation.read_json(cache/'result'/name)
        value['fingerprint' if defect == 'source' else 'passed'] = 'wrong' if defect == 'source' else False
        continuation.save(cache/'result'/name, value)
    with pytest.raises(ValueError):
        continuation.verify_carry(cache, request, check_source=False)


@pytest.mark.parametrize('start,end', [(0, 10), (10, 10), (10, 30), (40, 60), (50, 60), (10., 20)])
def test_only_next_ten_frame_partial_endpoints_are_supported(start, end):
    with pytest.raises(ValueError):
        continuation.request_values('jc-refinement-v1-a-1-1-frame10', 'a'*64, start, end)


def test_later_prefix_retains_plan_and_old_bytes_while_allowing_actual_failure(parent):
    folder, plan, _, _ = parent
    original_plan = (folder/'plan.json').read_bytes()
    before = continuation.prefix_snapshot(folder, 10)
    extend_prefix(folder, failed=True)
    continuation.verify_prefix_preserved(folder, before)
    report = continuation.audit_prefix(folder, plan['source'], 20, frozen)
    assert report['passed']
    assert report['frame'] == 20 and report['steps'] == 19300
    assert report['full_target_steps'] == 57900
    assert report['final_checkpoint_metrics']['failed_particles'] == 1
    assert report['first_saved_failed_material']['frame'] == 15
    assert report['checks']['original_25us_prefix_intact']
    assert not report['completed'] and not report['production_eligible']
    assert (folder/'plan.json').read_bytes() == original_plan


def test_axial_support_cut_before_any_material_failure_is_rejected(parent):
    folder, plan, _, _ = parent
    extend_prefix(folder, support_cut=True)
    report = continuation.audit_prefix(folder, plan['source'], 20, frozen)
    assert not report['passed']
    assert not report['checks']['no_support_cut_before_material_failure']
    assert report['first_saved_support_cut']['frame'] == 11
    assert report['first_saved_failed_material'] is None


@pytest.mark.parametrize('defect', ['old_frame', 'plan', 'old_observation', 'solver_clock', 'state_missing', 'invalid', 'mass', 'ep', 'force', 'failure_recovery'])
def test_later_prefix_rejects_lost_history_and_state_corruption(parent, defect):
    folder, plan, _, _ = parent
    before = continuation.prefix_snapshot(folder, 10)
    extend_prefix(folder, failed=(defect == 'failure_recovery'))
    cp = continuation.read_json(folder/'checkpoint.json')
    if defect == 'old_frame':
        (folder/'frames_data/frame_00004.npz').write_bytes(b'changed')
    elif defect == 'plan':
        (folder/'plan.json').write_text((folder/'plan.json').read_text()+' ')
    elif defect in ('old_observation', 'solver_clock', 'ep', 'force'):
        if defect == 'old_observation':
            cp['observations'][2]['central_material_section_force_N'] += 1.
        elif defect == 'solver_clock':
            cp['solver_seconds'] = 0.
        else:
            key = 'maximum_equivalent_plastic_strain' if defect == 'ep' else 'central_material_section_force_N'
            cp['observations'][-1][key] += 1.
        continuation.save(folder/'checkpoint.json', cp)
    elif defect == 'failure_recovery':
        path = folder/'frames_data/frame_00018.npz'
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
        values['failed'][0] = 0
        np.savez(path, **values)
    else:
        path = folder/cp['states'][0]
        with np.load(path) as archive:
            values = {key: archive[key] for key in archive.files}
        if defect == 'state_missing':
            values.pop('history_plastic_tensor')
        elif defect == 'invalid':
            values['history_valid'][0] = False
        else:
            values['mass'][0] *= 1.01
        np.savez(path, **values)
    with pytest.raises((ValueError, AssertionError)):
        continuation.verify_prefix_preserved(folder, before)
        continuation.audit_prefix(folder, plan['source'], 20, frozen)


def test_pinned_versions_are_exact_but_cpu_identity_is_recorded_separately():
    original = {'python': '3.12.13', 'system': 'Linux', 'machine': 'x86_64',
        'packages': {name: '1.0' for name in continuation.PINNED_PACKAGES}, 'cpu': 'cpu-a'}
    current = copy.deepcopy(original)
    current['cpu'] = 'cpu-b'
    assert continuation.environment_matches(original, current)
    current['packages']['numpy'] = '1.1'
    assert not continuation.environment_matches(original, current)
    assert not continuation.environment_matches({**original, 'packages': {}}, current)


def test_original_solver_source_requires_real_fixed_git_tree_and_all_raw_bytes(tmp_path, monkeypatch):
    root = tmp_path/'solver'
    root.mkdir()
    files = {'tools/run_jc_refinement.py': 'X=1\n', 'examples/common.py': 'Y=2\n', 'pyproject.toml': 'name="fixture"\n'}
    for name, content in files.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, newline='\n')
    continuation.git(root, 'init')
    continuation.git(root, 'add', '.')
    continuation.git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture')
    commit = continuation.git(root, 'rev-parse', 'HEAD').decode().strip()
    monkeypatch.setattr(continuation, 'SOLVER_COMMIT', commit)
    hashes = {name: continuation.digest(root/name) for name in files}
    source = {'commit': commit, 'tree': continuation.git(root, 'rev-parse', 'HEAD^{tree}').decode().strip(),
        'repository': continuation.REPOSITORY, 'source_kind': 'github-actions',
        'git_blob_sha': continuation.solver_files(root), 'files': hashes, 'lf_sha256': hashes,
        'fingerprint': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        'runner_source_sha256': hashes['tools/run_jc_refinement.py']}
    continuation.verify_solver_source(source, root)
    (root/'examples/extra.py').write_text('pass\n')
    with pytest.raises(ValueError, match='Untracked'):
        continuation.verify_solver_source(source, root)
    (root/'examples/extra.py').unlink()
    (root/'examples/common.py').write_text('Y=3\n')
    with pytest.raises(ValueError, match='working tree'):
        continuation.verify_solver_source(source, root)


def test_numerical_module_loaded_from_controller_tree_is_rejected(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(continuation, 'SOLVER', tmp_path/'solver')
    monkeypatch.setitem(sys.modules, 'examples._wrong_source', types.SimpleNamespace(__file__=str(tmp_path/'controller/examples/test.py')))
    with pytest.raises(ValueError, match='escaped'):
        continuation.verify_imports()


def test_frozen_import_rejects_preexisting_numerical_namespace():
    with pytest.raises(RuntimeError, match='before selecting'):
        continuation.load_frozen_helper()


def test_deadline_never_changes_physical_target_or_overruns_solve_cap(monkeypatch):
    monkeypatch.setattr(continuation.time, 'time', lambda: 1000.)
    command = continuation.limited({'started_epoch': 1000.}, 'worker', 265)
    assert command[3] == '15900s' and command[-1] == 'worker'
    command = continuation.limited({'started_epoch': 1000.-290*60}, 'worker', 265)
    assert command[3] == '600s'
    with pytest.raises(TimeoutError):
        continuation.limited({'started_epoch': 1000.-320*60}, 'audit-worker', 20)


def test_split_state_is_lossless_and_hashes_entire_file(tmp_path):
    source = tmp_path/'state.npz'
    np.savez(source, x=np.arange(80.))
    target = tmp_path/'parts'
    report = continuation.split_complete_file(source, target, 'state', part_bytes=113)
    data = b''.join((target/item['path']).read_bytes() for item in report['parts'])
    assert data == source.read_bytes()
    assert report['sha256'] == hashlib.sha256(data).hexdigest()
    assert all(item['bytes'] <= 113 for item in report['parts'])
    assert continuation.valid_npz(source)


def test_package_without_cache_uses_only_complete_latest_state_and_never_imports_solver(tmp_path, monkeypatch, parent):
    folder, _, _, _ = parent
    for name, value in {'ROOT': tmp_path, 'OUTPUT': folder, 'LOGS': tmp_path/'logs',
        'CACHE': tmp_path/'no-cache', 'DELIVERY': tmp_path/'delivery'}.items():
        monkeypatch.setattr(continuation, name, value)
    monkeypatch.setattr(continuation, 'load_frozen_helper', lambda: pytest.fail('packaging must be stdlib-only'))
    continuation.package()
    manifest = continuation.read_json(tmp_path/'delivery/delivery-manifest.json')
    assert not manifest['production_eligible'] and not manifest['media_created']
    assert any(name.endswith('checkpoint_0_0.npz') for name in manifest['state_files'])
    assert not any(name.endswith('checkpoint_1_0.npz') for name in manifest['state_files'])
    assert not any('trajectory_x.npy' in name for name in manifest['files'])


def test_importing_controller_alone_does_not_import_jax_or_old_solver():
    command = [sys.executable, '-c', 'import sys; from tools import run_jc_continuation; assert "jax" not in sys.modules; assert "examples.common" not in sys.modules']
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True, timeout=30)


def test_separate_child_routes_numerical_imports_to_frozen_checkout(tmp_path):
    old, wrong = tmp_path/'old', tmp_path/'wrong'
    for root in (old, wrong):
        for name in ('tools', 'examples'):
            (root/name).mkdir(parents=True)
            (root/name/'__init__.py').write_text('')
    (old/'examples/common.py').write_text('ORIGIN="frozen"\n')
    (wrong/'examples/common.py').write_text('raise RuntimeError("wrong numerical tree")\n')
    (old/'tools/run_jc_refinement.py').write_text(
        'from types import SimpleNamespace\nfrom examples import common\nengine=SimpleNamespace(LOGS=None)\n')
    script = tmp_path/'child.py'
    script.write_text('import runpy,sys\nfrom pathlib import Path\n'
        f'ns=runpy.run_path({str(Path(continuation.__file__).resolve())!r})\n'
        f'ns["load_frozen_helper"].__globals__["SOLVER"]=Path({str(old)!r})\n'
        'helper=ns["load_frozen_helper"]()\nns["verify_imports"]()\n'
        'assert helper.common.ORIGIN=="frozen"\nassert "jax" not in sys.modules\n')
    subprocess.run([sys.executable, str(script)], cwd=old,
        env={**os.environ, 'PYTHONPATH': str(wrong)}, check=True, timeout=30)


@pytest.fixture
def continued_carry(parent, tmp_path, monkeypatch):
    folder, plan, cache, request = parent
    controller = {'repository': continuation.REPOSITORY, 'commit': 'c'*40, 'run_id': '777', 'run_attempt': '1',
        'files': {name: continuation.digest(continuation.ROOT/name) for name in continuation.CONTROLLER_FILES}}
    context = {'request': request, 'source': plan['source'], 'controller': controller,
        'cache_key': continuation.next_key(controller, 20)}
    before = continuation.prefix_snapshot(folder, 10)
    extend_prefix(folder, failed=True)
    continuation.verify_prefix_preserved(folder, before)
    report = continuation.audit_prefix(folder, plan['source'], 20, frozen)
    report['controller_source'] = controller
    continuation.save(folder/'prefix-review.json', report)
    logs = tmp_path/'logs'
    for name, record in {'parent-prefix.json': before, 'worker-runtime.json': {}, 'loaded-numerics.json': {},
        'jc-continuation-solve.json': {'completed': True}, 'environment-lock.json': {'packages': {}}}.items():
        continuation.save(logs/name, record)
    for name, value in {'PARENT': cache, 'OUTPUT': folder, 'CACHE': tmp_path/'next', 'LOGS': logs}.items():
        monkeypatch.setattr(continuation, name, value)
    monkeypatch.setattr(continuation, 'verify_solver_source', lambda source: None)
    monkeypatch.setattr(continuation, 'verify_controller_history', lambda source: None)
    continuation.seal_carry(context, report)
    next_request = continuation.request_values(context['cache_key'], continuation.digest(tmp_path/'next/carry-receipt.json'), 20, 30)
    return tmp_path/'next', next_request, context, cache


def test_two_generation_carry_retains_original_source_review_and_prefix(continued_carry):
    cache, request, context, parent = continued_carry
    report = continuation.verify_carry(cache, request)
    assert report['frame'] == 20 and not report['full_protocol_completed']
    assert report['source'] == context['source']
    assert report['controller']['commit'] != report['source']['commit']
    assert continuation.digest(cache/'lineage/frame-10-receipt.json') == context['request']['parent_receipt_sha256']
    assert (cache/'lineage/frame-10-prefix-review.json').read_bytes() == (parent/'result/prefix-review.json').read_bytes()
    assert (cache/'result/frames_data/frame_00000.npz').read_bytes() == (parent/'result/frames_data/frame_00000.npz').read_bytes()
    assert not any('trajectory_' in name or 'checkpoint_1_0.npz' in name for name in report['files'])
    assert report['cross_job_resume_performed']
    assert not report['continuation_equivalence_to_uninterrupted_run_verified']


@pytest.mark.parametrize('defect', ['parent_link', 'parent_frame', 'controller_review', 'parent_review', 'extra_source', 'lock_escape'])
def test_logical_chain_tampering_rejected_even_when_outer_manifest_is_resealed(continued_carry, defect):
    cache, request, _, _ = continued_carry
    receipt = continuation.read_json(cache/'carry-receipt.json')
    if defect == 'parent_link':
        receipt['parent_receipt_sha256'] = 'f'*64
    elif defect == 'parent_frame':
        path = cache/'lineage/frame-10-receipt.json'
        value = continuation.read_json(path)
        value['frame'] = 20
        continuation.save(path, value)
        receipt['parent_receipt_sha256'] = continuation.digest(path)
    elif defect == 'controller_review':
        receipt['controller']['files']['tools/run_jc_continuation.py'] = 'e'*64
    elif defect == 'parent_review':
        path = cache/'lineage/frame-10-prefix-review.json'
        value = continuation.read_json(path)
        value['passed'] = False
        continuation.save(path, value)
    elif defect == 'extra_source':
        path = cache/'result/arbitrary.py'
        path.write_text('pass\n')
        receipt['files']['result/arbitrary.py'] = continuation.file_record(path)
    else:
        receipt['environment_lock_file'] = '../outside.json'
    for name in receipt['files']:
        receipt['files'][name] = continuation.file_record(cache/name)
    receipt['payload_bytes'] = sum(item['bytes'] for item in receipt['files'].values())
    continuation.save(cache/'carry-receipt.json', receipt)
    request['parent_receipt_sha256'] = continuation.digest(cache/'carry-receipt.json')
    with pytest.raises(ValueError):
        continuation.verify_carry(cache, request)


def test_cache_budget_is_checked_before_creating_any_next_payload(parent, tmp_path, monkeypatch):
    folder, plan, cache, request = parent
    extend_prefix(folder)
    report = continuation.audit_prefix(folder, plan['source'], 20, frozen)
    context = {'request': request, 'source': plan['source'], 'controller': {}, 'cache_key': 'unused'}
    for name, value in {'PARENT': cache, 'OUTPUT': folder, 'CACHE': tmp_path/'next-budget', 'LOGS': tmp_path/'logs-budget'}.items():
        monkeypatch.setattr(continuation, name, value)
    for name in ('parent-prefix.json', 'worker-runtime.json', 'loaded-numerics.json', 'jc-continuation-solve.json', 'environment-lock.json'):
        continuation.save(continuation.LOGS/name, {})
    monkeypatch.setattr(continuation, 'CACHE_BUDGET', 1024)
    with pytest.raises(ValueError, match='exceeds budget'):
        continuation.seal_carry(context, report)
    assert not continuation.CACHE.exists()
