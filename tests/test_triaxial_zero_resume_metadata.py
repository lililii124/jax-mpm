"""Resume metadata must be actual parent bytes and agree with the full state."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest


def fixture(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/run_triaxial_zero_recovery.py'
    spec = importlib.util.spec_from_file_location('zero_resume_metadata_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, 'BASE', tmp_path)
    parent, folder = tmp_path / 'parent', tmp_path / 'work'
    parent.mkdir()
    folder.mkdir()
    plan = {'saved_states': 61, 'stop_steps': 19200, 'frame_steps': [320*i for i in range(61)],
        'source': {'files': {'unchanged.py': 'original'}}, 'contact_model': {'planning': True}}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    np.savez(parent / 'checkpoint.npz', frame=20, aggregate=[6400, 0, 0, 0, 1, 0],
        state_time=.25, signature=signature, state_valid=True, history_valid=[True],
        contact_valid=True, contact_law_valid=[True])
    status = {'completed': False, 'probe_complete': False, 'segment_complete': True,
        'coupling_valid': True, 'phase': 'checkpoint-ready', 'frame': 20,
        'segment_end_frame': 20, 'frames': 60, 'steps': 6400, 'target_steps': 19200,
        'signature': signature}
    contact = {'runtime_mapping': 'actual-original-mapping'}
    layout = {'node_map': 'actual-original-storage-mapping'}
    summary = {'completed': False, 'probe_complete': False, 'segment_complete': True,
        'phase': 'checkpoint-ready', 'segment_end_frame': 20, 'steps': 6400,
        'definition': {**plan, 'contact_model': contact}, 'source': plan['source'], 'simulation_time': .25,
        'layout': layout}
    for name, data in [('status.json', status), ('summary.json', summary), ('contact-model.json', contact), ('layout.json', layout)]:
        m.save(parent / name, data)
    shutil.copyfile(parent / 'checkpoint.npz', folder / 'checkpoint.npz')
    context = {'start_frame': 20, 'parent_checkpoint': m.record(parent / 'checkpoint.npz'),
        'parent_metadata': {name: m.record(parent / name) for name in m.RESUME_METADATA}}
    return m, parent, folder, plan, context


def test_actual_parent_metadata_copied_byte_for_byte(tmp_path, monkeypatch):
    m, parent, folder, plan, context = fixture(tmp_path, monkeypatch)
    checkpoint = (folder / 'checkpoint.npz').read_bytes()
    m.restore_resume_metadata(folder, plan, context)
    for name in m.RESUME_METADATA:
        assert (folder / name).read_bytes() == (parent / name).read_bytes()
    assert (folder / 'checkpoint.npz').read_bytes() == checkpoint
    assert m.read(tmp_path / 'resume-metadata-proof.json')['old_particle_history_reconstructed'] is False


@pytest.mark.parametrize('tamper', ['failed_phase', 'signature', 'checkpoint_validity'])
def test_semantic_tamper_rejected_even_after_rehash(tmp_path, monkeypatch, tamper):
    m, parent, folder, plan, context = fixture(tmp_path, monkeypatch)
    if tamper == 'checkpoint_validity':
        with np.load(folder / 'checkpoint.npz') as cp:
            leaves = {key: cp[key] for key in cp.files}
        leaves['aggregate'][4] = 0
        np.savez(folder / 'checkpoint.npz', **leaves)
        context['parent_checkpoint'] = m.record(folder / 'checkpoint.npz')
    else:
        status = m.read(parent / 'status.json')
        status['phase' if tamper == 'failed_phase' else 'signature'] = 'failed' if tamper == 'failed_phase' else '0'*64
        m.save(parent / 'status.json', status)
        context['parent_metadata']['status.json'] = m.record(parent / 'status.json')
    with pytest.raises(ValueError):
        m.restore_resume_metadata(folder, plan, context)
    assert not (folder / 'status.json').exists()
