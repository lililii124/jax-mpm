"""Transport failures must not discard a whole new state or clear rejection."""
import importlib.util
import json
from pathlib import Path
import zipfile
import sys

import numpy as np


def load(name, filename):
    path = Path(__file__).resolve().parents[1] / 'tools' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejected_new_state_and_all_real_frames_survive_transport_failure(tmp_path, monkeypatch):
    m = load('failed_zero_main', 'run_triaxial_zero_recovery.py')
    fallback = load('failed_zero_fallback', 'preserve_triaxial_zero_failure.py')
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setattr(m, 'verify_controller', lambda: {'commit': 'original-controller'})
    folder = tmp_path / 'work/formal/theta-00/hertz'
    folder.mkdir(parents=True)
    m.save(tmp_path / 'context.json', {'controller': {'commit': 'original-controller'},
        'start_frame': 20, 'end_frame': 21, 'parent_archive_sha256': 'a' * 64})
    m.save(tmp_path / 'parent/plan.json', {'unchanged': True})
    np.savez(folder / 'checkpoint.npz', frame=21, aggregate=[6401, 0, 0, 0, 0, 1],
        state_valid=True, history_valid=[True], contact_valid=False, contact_law_valid=[False],
        state_x=np.arange(9).reshape(3, 3))
    original = (folder / 'checkpoint.npz').read_bytes()
    for field in m.FIELDS:
        shape = (61, 3, 3) if field == 'x' else (61, 3)
        values = np.zeros(shape, dtype=np.uint8 if field == 'failed' else np.float64)
        np.save(folder / f'trajectory_{field}.npy', values)
    result = fallback.preserve(m)
    assert result['ready'] is True and result['never_resumable'] is True
    assert result['complete_real_frames_archived'] == [20, 21]
    with zipfile.ZipFile(tmp_path / 'failure-delivery/capsule.zip') as z:
        report = json.loads(z.read('failed-manifest.json'))
        assert report['restart_valid'] is False and report['never_resumable'] is True
        assert report['checkpoint']['contact_valid'] is False
        assert report['checkpoint']['aggregate_valid'] is False
        assert z.read('diagnostic/checkpoint.npz') == original
        assert 'capsule-manifest.json' not in z.namelist()
    assert (folder / 'checkpoint.npz').read_bytes() == original


def test_already_sealed_valid_capsule_is_reused_byte_for_byte(tmp_path, monkeypatch):
    m = load('valid_zero_main', 'run_triaxial_zero_recovery.py')
    fallback = load('valid_zero_fallback', 'preserve_triaxial_zero_failure.py')
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setattr(m, 'verify_controller', lambda: {'commit': 'original-controller'})
    m.save(tmp_path / 'context.json', {'controller': {'commit': 'original-controller'}})
    delivery = tmp_path / 'delivery'
    delivery.mkdir()
    with zipfile.ZipFile(delivery / 'capsule.zip', 'x') as z:
        z.writestr('capsule-manifest.json', json.dumps({'controller': {'commit': 'original-controller'}}))
        z.writestr('restart/checkpoint.npz', b'complete-original-state')
    m.save(delivery / 'receipt.json', {'capsule': m.record(delivery / 'capsule.zip')})
    original = (delivery / 'capsule.zip').read_bytes()
    result = fallback.preserve(m)
    assert result['ready'] is True
    assert (tmp_path / 'failure-delivery/capsule.zip').read_bytes() == original


def test_prepare_failure_fallback_does_not_require_numpy(tmp_path, monkeypatch):
    m = load('no_numpy_zero_main', 'run_triaxial_zero_recovery.py')
    fallback = load('no_numpy_zero_fallback', 'preserve_triaxial_zero_failure.py')
    monkeypatch.setattr(m, 'BASE', tmp_path)
    monkeypatch.setitem(sys.modules, 'numpy', None)
    assert fallback.preserve(m)['ready'] is False
    monkeypatch.setattr(m, 'verify_controller', lambda: {'commit': 'original-controller'})
    m.save(tmp_path / 'context.json', {'controller': {'commit': 'original-controller'}})
    assert fallback.preserve(m)['ready'] is False
