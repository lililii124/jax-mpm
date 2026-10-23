import csv
import hashlib
import json
import shutil

import numpy as np
import pytest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(case, folder, *, resume=False, stop=None):
    from examples.common import run
    frames = 4
    substeps = round(case.duration/(frames*case.configs[0].dt))
    return run(case.states, case.materials, case.configs, frames=frames,
               substeps=substeps, output=folder, label='JC continuation check',
               observer=case.observer, metadata=case.description,
               render_output=False, resume=resume, stop_after_frame=stop)


def test_jc_actual_numerical_continuation_preserves_target_and_histories(tmp_path):
    import jax
    from examples.jc_necking import make_case

    jax.config.update('jax_enable_x64', True)
    case = make_case(grid=50, ppc=1, frames=4, end_time=20e-9)
    full, split = tmp_path/'full', tmp_path/'split'
    _, full_summary = invoke(case, full)
    _, segment = invoke(case, split, stop=2)
    assert segment['completed'] is False and segment['segment_complete'] is True
    assert segment['phase'] == 'checkpoint-ready'
    assert segment['frame'] == 2 and segment['frames'] == 4
    assert segment['target_time'] == case.duration
    assert not (split/'final_state_0.npz').exists()
    assert json.loads((split/'summary.json').read_text())['completed'] is False
    checkpoint = json.loads((split/'checkpoint.json').read_text())
    signature = checkpoint['signature']
    history_sha = sha(split/'trajectory_x.npy')
    with pytest.raises(ValueError, match='follow the saved'):
        invoke(case, split, resume=True, stop=2)
    assert sha(split/'trajectory_x.npy') == history_sha
    bad = tmp_path/'invalid'
    shutil.copytree(split, bad)
    state_path = bad/checkpoint['states'][0]
    with np.load(state_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    data['history_valid'][0] = False
    np.savez_compressed(state_path, **data)
    with pytest.raises(ValueError, match='invalid numerical state'):
        invoke(case, bad, resume=True)
    assert sha(bad/'trajectory_x.npy') == history_sha
    bad_time = tmp_path/'bad-time'
    shutil.copytree(split, bad_time)
    state_path = bad_time/checkpoint['states'][0]
    with np.load(state_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    data['time'] += case.configs[0].dt
    np.savez_compressed(state_path, **data)
    with pytest.raises(ValueError, match='state time'):
        invoke(case, bad_time, resume=True)
    assert sha(bad_time/'trajectory_x.npy') == history_sha
    missing = tmp_path/'missing-frame'
    shutil.copytree(split, missing)
    (missing/'frames_data/frame_00001.npz').unlink()
    with pytest.raises(FileNotFoundError):
        invoke(case, missing, resume=True)
    assert sha(missing/'trajectory_x.npy') == history_sha
    _, resumed = invoke(case, split, resume=True)
    assert resumed['completed'] is True and full_summary['completed'] is True
    assert resumed['simulation_time'] == case.duration
    assert json.loads((split/'checkpoint.json').read_text())['signature'] == signature
    for filename in ('final_state_0.npz', 'trajectory.npz'):
        with np.load(split/filename, allow_pickle=False) as actual, np.load(full/filename, allow_pickle=False) as expected:
            assert set(actual.files) == set(expected.files)
            for name in expected.files:
                if expected[name].dtype.kind in 'fc':
                    scale = float(np.max(np.abs(expected[name]), initial=0))
                    np.testing.assert_allclose(actual[name], expected[name], rtol=2e-12, atol=2e-14*scale)
                else:
                    np.testing.assert_array_equal(actual[name], expected[name])
    for filename in ('diagnostics.csv', 'observables.csv'):
        with (split/filename).open(newline='') as actual, (full/filename).open(newline='') as expected:
            assert list(csv.DictReader(actual)) == list(csv.DictReader(expected))
    for name in ('x', 'speed', 'damage', 'temperature', 'failed'):
        actual = np.load(split/f'trajectory_{name}.npy', mmap_mode='r')
        expected = np.load(full/f'trajectory_{name}.npy', mmap_mode='r')
        try:
            np.testing.assert_array_equal(actual, expected)
        finally:
            actual._mmap.close(); expected._mmap.close()
    jax.clear_caches()
