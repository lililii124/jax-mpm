import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools import run_jc_refinement as jc


def fake_source():
    runner = jc.digest(Path(jc.__file__))
    return {'commit': 'a'*40, 'run_id': '123', 'run_attempt': '1', 'fingerprint': 'b'*64,
        'files': {'tools/run_jc_refinement.py': runner}, 'runner_source_sha256': runner}


def reference_state(count=4):
    h = .0016/48
    x = np.array([[.005-h/2, .005-h/2, .005], [.005-h/2, .005+h/2, .005],
                  [.005+h/2, .005-h/2, .005], [.005+h/2, .005+h/2, .005]])
    values = {'x': x, 'v': np.zeros((count, 3)), 'F': np.repeat(np.eye(3)[None], count, axis=0),
        'C': np.zeros((count, 3, 3)), 'mass': np.full(count, 7830.*h**3), 'volume': np.full(count, h**3),
        'time': np.array(0.), 'valid': np.array(True)}
    for field in jc.HISTORY_FIELDS:
        if field in ('stress', 'plastic_tensor', 'backstress'):
            value = np.zeros((count, 3, 3))
        elif field in ('failed', 'postpeak', 'valid'):
            value = np.full(count, field == 'valid', dtype=bool)
        else:
            value = np.full(count, 298. if field == 'temperature' else 7830. if field == 'density' else 0.)
        values['history_'+field] = value
    return values


def fixture_plan(arrays):
    boundaries = [dict(lower=lo, upper=hi, velocity=v, components=None, start=0., end=math.inf, ramp_time=20e-6)
        for lo, hi, v in [([0., 0., 0.], [.0022, .01, .01], [-10., 0., 0.]),
                         ([.0078, 0., 0.], [.01, .01, .01], [10., 0., 0.])]]
    config = dict(n_grid=100, domain_size=.01, dim=3, dt=jc.DURATION/(jc.FRAMES*jc.SUBSTEPS),
        boundaries=boundaries, failure_mode='ballistic', material='johnson-cook',
        gravity=[0., 0., 0.], boundary_cells=0)
    material = dict(mu=200e9/(2*1.29), bulk=200e9/(3*(1-2*.29)), A=792e6, B=510e6,
        n=.26, C=.014, m=1.03, reference_rate=1., reference_temperature=298., melt_temperature=1793.,
        heat_capacity=477., heat_fraction=1., density=7830., failure=[.05, 3.44, -2.12, .002, .61], failure_enabled=True)
    definition = dict(bar_diameter_m=.0016, waist_radius_m=.00075, transition_half_length_m=.00075,
        arc_radius_m=(.00075**2+(.0008-.00075)**2)/(4*(.0008-.00075)),
        gauge_length_m=.0056, requested_particles_per_cell_per_axis=3, transverse_lattice_count=48,
        grip_speed_m_per_s=10., ramp_time_s=20e-6, duration_s=jc.DURATION, reference_duration_s=jc.DURATION,
        acoustic_CFL_requested=.15, bar_ends_x_m=[.0018, .0082], grip_inner_faces_x_m=[.0022, .0078], particle_spacing_m=.0016/48)
    payload = dict(configs=[config], materials=[material], definition=definition,
        frames=jc.FRAMES, substeps=jc.SUBSTEPS, label=jc.LABEL)
    return {'signature_data': jc.encode(payload),
        'common_signature': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        'full_frames': jc.FRAMES, 'full_steps': jc.FRAMES*jc.SUBSTEPS, 'full_duration_s': jc.DURATION,
        'segment_steps': jc.END_FRAME*jc.SUBSTEPS, 'segment_time_s': 25e-6, 'stop_after_frame': jc.END_FRAME,
        'particles': len(arrays['x']), 'initial_state_fields': {key: jc.descriptor(value) for key, value in arrays.items()},
        'initial_mass_kg': float(arrays['mass'].sum()), 'initial_reference_volume_m3': float(arrays['volume'].sum()),
        'source': fake_source(), 'completed': False, 'production_eligible': False}


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(jc, 'PARTICLES', 4)
    folder = tmp_path/'result'
    (folder/'frames_data').mkdir(parents=True)
    initial = reference_state()
    plan = fixture_plan(initial)
    jc.save(folder/'plan.json', plan)
    jc.save(folder/'numerical-source.json', fake_source())
    jc.save(folder/'environment.json', {'python': '3.12.13'})
    dt = jc.DURATION/(jc.FRAMES*jc.SUBSTEPS)
    arrays = copy.deepcopy(initial)
    arrays['history_stress'][:, 0, 0] = 10e6
    arrays['history_plastic_strain'][:] = .02
    arrays['history_damage'][:] = .1
    arrays['time'] = np.array(jc.END_FRAME*jc.SUBSTEPS*dt)
    final = jc.final_metrics(arrays, initial['x'], .0016/48)
    rows, observations = [], []
    for frame in range(jc.END_FRAME+1):
        time = frame*jc.SUBSTEPS*dt
        ratio = frame/jc.END_FRAME
        row = {'body': 0, 'time': time, 'min_J': 1., 'max_J': 1., 'mass': plan['initial_mass_kg']}
        observed = {'time': time, 'failed_particles': 0, 'max_damage': .1*ratio,
            'external_work_trapezoidal': 0., 'boundary_impulse_x': 0., 'boundary_impulse_y': 0., 'boundary_impulse_z': 0.,
            'central_active_particles': 4, 'maximum_equivalent_plastic_strain': .02*ratio,
            'central_material_section_force_N': final['central_material_section_force_N']*ratio}
        rows.append(row); observations.append(observed)
        np.savez_compressed(folder/'frames_data'/f'frame_{frame:05d}.npz', x=arrays['x'],
            speed=np.zeros(4), damage=np.full(4, .1*ratio), temperature=np.full(4, 298.),
            failed=np.zeros(4, dtype=np.uint8), time=np.array(time), contact_impulse=np.zeros(3))
    np.savez(folder/'checkpoint_0_0.npz', **arrays)
    np.savez(folder/'checkpoint_1_0.npz', ignored_previous_slot=np.zeros(2))
    np.save(folder/'trajectory_x.npy', np.zeros((61, 4, 3)))
    checkpoint = {'signature': plan['common_signature'], 'frame': jc.END_FRAME, 'states': ['checkpoint_0_0.npz'],
        'rows': rows, 'observations': observations, 'contact_impulse': [0., 0., 0.], 'external_impulse': [0., 0., 0.],
        'external_work': 0., 'solver_seconds': 2.}
    jc.save(folder/'checkpoint.json', checkpoint)
    status = {'completed': False, 'segment_complete': True, 'phase': 'checkpoint-ready',
        'signature': plan['common_signature'], 'frame': jc.END_FRAME, 'frames': jc.FRAMES,
        'steps_completed': jc.END_FRAME*jc.SUBSTEPS, 'target_steps': jc.FRAMES*jc.SUBSTEPS,
        'particles': [4], 'time': jc.END_FRAME*jc.SUBSTEPS*dt, 'target_time': jc.DURATION}
    for name in ('status.json', 'summary.json', 'segment.json'):
        jc.save(folder/name, status)
    return folder, plan


def test_unbounded_grip_marker_preserves_common_signature_exactly():
    payload = {'tuple': (1., math.inf, -math.inf), 'failure_enabled': True}
    safe = jc.encode(payload)
    json.dumps(safe, allow_nan=False)
    assert json.dumps(jc.decode(safe), sort_keys=True) == json.dumps(payload, sort_keys=True)
    with pytest.raises(ValueError):
        jc.encode({'value': math.nan})


def test_first_segment_audit_preserves_full_target_and_does_not_claim_fracture(prefix):
    folder, plan = prefix
    report = jc.audit_prefix(folder, fake_source())
    assert report['passed'] is True
    assert report['completed'] is report['full_protocol_completed'] is report['production_eligible'] is False
    assert report['steps'] == 9650 and report['full_target_steps'] == 57900
    assert math.isclose(report['time_s'], 25e-6, rel_tol=1e-14)
    assert report['full_target_time_s'] == 150e-6
    assert len(report['positive_support_history']) == 11
    assert report['first_saved_support_cut'] is None
    assert report['initial_mass_kg'] == plan['initial_mass_kg']


@pytest.mark.parametrize('field,value', [('full_duration_s', 25e-6), ('full_steps', 9650),
    ('stop_after_frame', 30), ('particles', 8), ('common_signature', 'bad')])
def test_changed_plan_cannot_be_misrepresented_as_first_prefix(prefix, field, value):
    folder, plan = prefix
    plan[field] = value
    with pytest.raises(ValueError):
        jc.validate_plan(plan)


@pytest.mark.parametrize('mutation', ['missing_frame', 'future_frame', 'row_time', 'row_count', 'observer_ep',
    'observer_force', 'source', 'signature', 'complete_claim', 'invalid_history', 'missing_state_field', 'mass', 'nonfinite'])
def test_partial_integrity_defects_are_rejected(prefix, mutation):
    folder, plan = prefix
    if mutation == 'missing_frame': (folder/'frames_data/frame_00003.npz').unlink()
    elif mutation == 'future_frame': (folder/'frames_data/frame_00011.npz').write_bytes((folder/'frames_data/frame_00010.npz').read_bytes())
    elif mutation in ('row_time', 'row_count', 'observer_ep', 'observer_force', 'signature'):
        checkpoint = jc.read_json(folder/'checkpoint.json')
        if mutation == 'row_time': checkpoint['rows'][2]['time'] = 1.
        if mutation == 'row_count': checkpoint['rows'].pop()
        if mutation == 'observer_ep': checkpoint['observations'][-1]['maximum_equivalent_plastic_strain'] = .03
        if mutation == 'observer_force': checkpoint['observations'][-1]['central_material_section_force_N'] *= 2
        if mutation == 'signature': checkpoint['signature'] = 'bad'
        jc.save(folder/'checkpoint.json', checkpoint)
    elif mutation == 'source': jc.save(folder/'numerical-source.json', {'wrong': True})
    elif mutation == 'complete_claim':
        status = jc.read_json(folder/'status.json'); status['completed'] = True
        jc.save(folder/'status.json', status)
    else:
        with np.load(folder/'checkpoint_0_0.npz', allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
        if mutation == 'invalid_history': values['history_valid'][0] = False
        if mutation == 'missing_state_field': values.pop('history_backstress')
        if mutation == 'mass': values['mass'][0] *= 1.1
        if mutation == 'nonfinite': values['C'][0, 0, 0] = np.nan
        np.savez(folder/'checkpoint_0_0.npz', **values)
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        jc.audit_prefix(folder, fake_source())


def test_positive_stencil_uses_real_weight_support_instead_of_three_dx():
    x = np.array([.002449, .002651])
    assert x[1]-x[0] < 3e-4
    _, _, _, cuts = jc.engine.axial_support_intervals(x, .0001, 100)
    assert cuts.tolist() == [0]


def test_carry_excludes_mmap_and_inactive_slot_but_keeps_exact_prefix(prefix, tmp_path):
    folder, plan = prefix
    jc.audit_prefix(folder, fake_source())
    cache = tmp_path/'carry'
    receipt = jc.seal_carry(folder, fake_source(), cache)
    assert len([name for name in receipt['files'] if '/frames_data/' in name]) == 11
    assert 'result/checkpoint_0_0.npz' in receipt['files']
    assert not any('checkpoint_1_' in name or name.endswith('.npy') for name in receipt['files'])
    assert receipt['cross_job_resume_verified'] is False
    assert receipt['completed'] is False and receipt['segment_complete'] is True
    assert (cache/'result/checkpoint_0_0.npz').read_bytes() == (folder/'checkpoint_0_0.npz').read_bytes()
    jc.verify_carry(cache, fake_source(), jc.digest(cache/'carry-receipt.json'))


@pytest.mark.parametrize('defect', ['bytes', 'missing', 'extra', 'source', 'receipt'])
def test_carry_verification_rejects_missing_tampered_or_cross_source_data(prefix, tmp_path, defect):
    folder, plan = prefix
    jc.audit_prefix(folder, fake_source())
    cache = tmp_path/'carry'
    jc.seal_carry(folder, fake_source(), cache)
    receipt_sha = jc.digest(cache/'carry-receipt.json')
    source = fake_source()
    if defect == 'bytes': (cache/'result/checkpoint_0_0.npz').write_bytes(b'bad')
    if defect == 'missing': (cache/'result/frames_data/frame_00002.npz').unlink()
    if defect == 'extra': (cache/'unexpected.py').write_text('no source in cache')
    if defect == 'source': source['commit'] = 'd'*40
    if defect == 'receipt': receipt_sha = '0'*64
    with pytest.raises(ValueError):
        jc.verify_carry(cache, source, receipt_sha)


def test_cache_budget_failure_never_leaves_partial_carry(prefix, tmp_path, monkeypatch):
    folder, plan = prefix
    jc.audit_prefix(folder, fake_source())
    monkeypatch.setattr(jc, 'CACHE_BUDGET', 1024)
    cache = tmp_path/'carry'
    with pytest.raises(ValueError, match='budget'):
        jc.seal_carry(folder, fake_source(), cache)
    assert not cache.exists()


def test_carry_cannot_relabel_another_source_as_its_producer(prefix, tmp_path):
    folder, plan = prefix
    jc.audit_prefix(folder, fake_source())
    other = fake_source()
    other['commit'] = 'f'*40
    with pytest.raises(ValueError, match='fully checked'):
        jc.seal_carry(folder, other, tmp_path/'carry')


def test_new_runner_is_start_only_and_all_children_are_memory_guarded():
    text = Path(jc.__file__).read_text()
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'engine' and node.func.attr == 'command']
    assert len(calls) == 2
    assert all(any(kw.arg == 'monitor' and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords) for call in calls)
    assert "resume=False, stop_after_frame=END_FRAME" in text
    assert "choices=('prepare', 'worker', 'run', 'audit-worker', 'confirm-cache', 'package')" in text
    workflow = (jc.ROOT/'.github/workflows/jc-refinement.yml').read_text()
    assert 'lookup-only: true' in workflow and 'fail-on-cache-miss: true' in workflow
    assert 'restore-keys:' not in workflow and '--end-time' not in workflow
    assert 'ParaView' not in workflow and 'include-hidden-files: true' in workflow


def test_successful_cache_artifact_is_light_and_contains_no_state_duplication(prefix, tmp_path, monkeypatch):
    folder, plan = prefix
    jc.audit_prefix(folder, fake_source())
    logs, cache, delivery = tmp_path/'logs', tmp_path/'carry', tmp_path/'delivery'
    logs.mkdir()
    jc.save(logs/'numerical-source.json', fake_source())
    jc.seal_carry(folder, fake_source(), cache)
    jc.save(logs/'cache-confirmation.json', {'verified': True, 'cache_key': jc.cache_key(fake_source()),
        'receipt_sha256': jc.digest(cache/'carry-receipt.json')})
    for name, value in [('ROOT', tmp_path), ('LOGS', logs), ('OUTPUT', folder), ('CACHE', cache), ('DELIVERY', delivery)]:
        monkeypatch.setattr(jc, name, value)
    manifest = jc.package()
    assert manifest['total_budget_bytes'] == 8*1024**2 and manifest['cache_saved_exactly'] is True
    assert manifest['state_files'] == {}
    assert not any(name.endswith(('.npz', '.npy')) for name in manifest['files'])
    assert manifest['completed'] is False


def test_unsaved_cache_preserves_complete_latest_state_before_frames(prefix, tmp_path, monkeypatch):
    folder, plan = prefix
    logs, delivery = tmp_path/'logs', tmp_path/'delivery'
    logs.mkdir()
    for name, value in [('ROOT', tmp_path), ('LOGS', logs), ('OUTPUT', folder), ('DELIVERY', delivery)]:
        monkeypatch.setattr(jc, name, value)
    manifest = jc.package()
    assert manifest['cache_saved_exactly'] is False
    assert manifest['total_budget_bytes'] == 192*1024**2
    assert 'result/checkpoint_0_0.npz' in manifest['state_files']
    assert not any('checkpoint_1_' in name or name.endswith('.npy') for name in manifest['state_files'])
    state = manifest['state_files']['result/checkpoint_0_0.npz']
    data = b''.join((delivery/part['path']).read_bytes() for part in state['parts'])
    assert data == (folder/'checkpoint_0_0.npz').read_bytes()
    assert hashlib.sha256(data).hexdigest() == state['sha256']
