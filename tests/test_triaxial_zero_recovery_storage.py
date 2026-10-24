"""New storage contract: complete restart, real frame shards, explicit gaps."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'tools/run_triaxial_zero_recovery.py'
    spec = importlib.util.spec_from_file_location('zero_recovery_storage_test', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_complete_capsule_roundtrip_and_cloud_bound(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'verify_controller', lambda: {'commit': 'test-controller'})
    base = tmp_path / 'first'
    monkeypatch.setattr(m, 'BASE', base)
    folder = base / 'work/formal/theta-00/hertz'
    folder.mkdir(parents=True)
    pins = {name: '1.0' for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'psutil', 'pillow')}
    context = {'start_frame': 20, 'end_frame': 21, 'parent_archive_sha256': 'a' * 64,
        'controller': {'commit': 'test-controller'},
        'parent_checkpoint': {'bytes': 1, 'sha256': 'b' * 64}, 'kwargs': {},
        'environment': {'packages': pins}, 'provenance': {'gap': list(range(1, 20))}}
    m.save(base / 'context.json', context)
    m.save(base / 'parent/plan.json', {'frame_steps': [i * 320 for i in range(61)]})
    m.save(folder / 'status.json', {'coupling_valid': True, 'frame': 21, 'phase': 'checkpoint-ready'})
    row = {'time': 0., 'mean_grip_tensile_force_N': 1., 'actual_clamp_work_J': 0.,
        'kinetic_energy_J': 0., 'elastic_energy_J': 0., 'contact_potential_J': 0.,
        'active_fibre_x_support_cuts': 0, 'sampled_section_graph_connects_grips': True,
        'failed_particles': 1}
    np.savez(folder / 'checkpoint.npz', frame=21, aggregate=[6720., 0., 0., 0., 1., 0.],
        state_valid=True, history_valid=[True], contact_valid=True, contact_law_valid=[True],
        tables=json.dumps({'observables': [row]}), state_x=np.arange(9).reshape(3, 3))
    for field in m.FIELDS:
        shape = (61, 3, 3) if field == 'x' else (61, 3)
        values = np.full(shape, np.nan) if field != 'failed' else np.full(shape, 255, np.uint8)
        values[20], values[21] = 20, 21
        np.save(folder / f'trajectory_{field}.npy', values)
    for name in ('summary.json', 'observables.csv', 'diagnostics.csv', 'layout.json',
                 'contact-model.json', 'contact-metrics.json', 'particle_fields.npz'):
        (folder / name).write_bytes(b'original metadata')
    for name in ('history-availability.json', 'memory.json', 'solver.log'):
        (base / name).write_bytes(b'diagnostic')
    m.package()
    capsule = base / 'delivery/capsule.zip'
    with zipfile.ZipFile(capsule) as z:
        manifest = json.loads(z.read('capsule-manifest.json'))
        assert manifest['real_particle_frames'] == [20, 21]
        assert manifest['full_original_particle_history_available'] is False
        for name, record in manifest['files'].items():
            assert len(z.read(name)) == record['bytes']
            assert hashlib.sha256(z.read(name)).hexdigest() == record['sha256']
        assert not any('frame-19/' in name for name in z.namelist())
        checkpoint_raw = z.read('restart/checkpoint.npz')
    monkeypatch.setattr(m, 'BASE', tmp_path / 'next')
    m.prepare(capsule, m.digest(capsule), 22, 0,
              repository_bytes=27_440_000_000, repository_count=262)
    assert (m.BASE / 'parent/checkpoint.npz').read_bytes() == checkpoint_raw
    assert m.read(m.BASE / 'context.json')['start_frame'] == 21
    assert m.read(m.BASE / 'context.json')['repository_artifact_bytes_observed'] == 27_440_000_000
    assert m.read(m.BASE / 'artifact-accounting.json')['repository_total_is_not_a_gate'] is True
    wrapper = tmp_path / 'github-outer.zip'
    with zipfile.ZipFile(wrapper, 'x') as z:
        z.write(capsule, 'capsule.zip')
        z.write(base / 'delivery/receipt.json', 'receipt.json')
    monkeypatch.setattr(m, 'BASE', tmp_path / 'outer-next')
    m.prepare(wrapper, m.digest(wrapper), 22, 0)
    assert (m.BASE / 'parent/checkpoint.npz').read_bytes() == checkpoint_raw
    transport = m.read(m.BASE / 'context.json')['parent_transport']
    assert transport['inner_capsule']['sha256'] == m.digest(capsule)
    assert transport['outer_archive']['sha256'] == m.digest(wrapper)
    bad = tmp_path / 'bad-receipt-outer.zip'
    with zipfile.ZipFile(bad, 'x') as z:
        z.write(capsule, 'capsule.zip')
        receipt = json.loads((base / 'delivery/receipt.json').read_text())
        receipt['capsule']['sha256'] = '0' * 64
        z.writestr('receipt.json', json.dumps(receipt))
    monkeypatch.setattr(m, 'BASE', tmp_path / 'bad-outer-next')
    with pytest.raises(ValueError, match='transport receipt size/SHA'):
        m.prepare(bad, m.digest(bad), 22, 0)
    monkeypatch.setattr(m, 'BASE', tmp_path / 'cloud-rejected')
    with pytest.raises(ValueError, match='500 MiB'):
        m.prepare(capsule, m.digest(capsule), 22, m.CLOUD_LIMIT)
    assert not m.BASE.exists()


def test_rolling_pool_excludes_unique_evidence_in_other_artifacts():
    m = module()
    pages = [{'artifacts': [
        {'id': 1, 'name': 'unique-ceramic-raw', 'size_in_bytes': 27_440_000_000, 'expired': False},
        {'id': 2, 'name': 'theta-zero-capsule-f21-40000000001-1', 'size_in_bytes': 100, 'expired': False},
        {'id': 3, 'name': 'theta-zero-recovery-diagnostic-40000000002-1', 'size_in_bytes': 20, 'expired': False},
        {'id': 4, 'name': 'theta-zero-capsule-f22-40000000003-1', 'size_in_bytes': 999, 'expired': True}]}]
    usage = m.artifact_usage(pages)
    assert usage['repository_artifact_bytes_observed'] == 27_440_000_120
    assert usage['repository_artifact_count_observed'] == 3
    assert usage['rolling_pool_used_bytes'] == 100
    assert usage['rolling_pool_artifact_ids'] == [2]


def test_failed_manifest_is_rejected_inside_authenticated_outer_wrapper(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, 'BASE', tmp_path / 'rejected-parent')
    monkeypatch.setattr(m, 'verify_controller', lambda: {'commit': 'test-controller'})
    inner = tmp_path / 'failed-inner.zip'
    with zipfile.ZipFile(inner, 'x') as z:
        z.writestr('failed-manifest.json', json.dumps({'never_resumable': True}))
    outer = tmp_path / 'failed-outer.zip'
    with zipfile.ZipFile(outer, 'x') as z:
        z.write(inner, 'capsule.zip')
        z.writestr('receipt.json', json.dumps({'schema': 'triaxial-zero-cloud-capsule-v1',
            'capsule': m.record(inner), 'end_frame': 21}))
    with pytest.raises(ValueError, match='Failed or unclassified capsule'):
        m.prepare(outer, m.digest(outer), 23, 0)


def test_first_material_failure_is_not_rupture():
    m = module()
    row = {'time': 0., 'mean_grip_tensile_force_N': 10., 'actual_clamp_work_J': 1.,
        'kinetic_energy_J': .1, 'elastic_energy_J': .2, 'contact_potential_J': 0.,
        'active_fibre_x_support_cuts': 0, 'sampled_section_graph_connects_grips': True,
        'failed_particles': 1}
    assert m.rupture_evidence([row])['sampled_rupture_candidate'] is None
    cut = {**row, 'time': 1., 'mean_grip_tensile_force_N': 4.,
        'active_fibre_x_support_cuts': 1, 'sampled_section_graph_connects_grips': False}
    evidence = m.rupture_evidence([row, cut])
    assert evidence['sampled_rupture_candidate']['force_drop_fraction'] == .6
    assert evidence['fracture_dissipation_verified'] is False
    assert evidence['production_eligible'] is False
