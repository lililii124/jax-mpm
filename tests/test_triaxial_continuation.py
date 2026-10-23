import csv
import json

import numpy as np

from examples.triaxial_failure import output_directory, plan_case, run_case


def assert_same_array(actual, expected):
    if expected.dtype.kind in 'fc':
        scale = float(np.max(np.abs(expected), initial=0.))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14*scale)
    else:
        np.testing.assert_array_equal(actual, expected)


def assert_same_archive(actual_path, expected_path, *, excluded=()):
    with np.load(actual_path, allow_pickle=False) as actual, np.load(expected_path, allow_pickle=False) as expected:
        assert actual.files == expected.files
        for name in expected.files:
            if name not in excluded:
                assert_same_array(actual[name], expected[name])


def test_actual_jax_complete_protocol_equals_two_planned_segments(tmp_path):
    import jax

    jax.config.update('jax_enable_x64', True)
    geometry, plan = plan_case(smoke=True, law='jkr', grid_storage='tube',
        load_transits=.03, max_strain=1e-7, frames=4)
    assert len(geometry['x']) == 40416 and plan['steps'] == 13
    full_root, split_root = tmp_path/'full', tmp_path/'split'
    full = run_case(geometry, plan, full_root)
    partial = run_case(geometry, plan, split_root, stop_after_frame=2)
    full_folder, split_folder = output_directory(full_root, plan), output_directory(split_root, plan)
    status = json.loads((split_folder/'status.json').read_text())
    assert partial['segment_complete'] and not partial['completed'] and not partial['probe_complete']
    assert partial['steps'] == plan['frame_steps'][2] and partial['segment_end_frame'] == 2
    assert status['phase'] == 'checkpoint-ready' and status['coupling_valid']
    assert json.loads((split_folder/'plan.json').read_text()) == plan
    prefix = {}
    for name in ('x', 'speed', 'damage', 'failed', 'temperature'):
        mapped = np.load(split_folder/f'trajectory_{name}.npy', mmap_mode='r')
        try:
            assert mapped.shape[0] == plan['saved_states']
            prefix[name] = mapped[:3].copy()
        finally:
            mapped._mmap.close()
    with np.load(split_folder/'checkpoint.npz', allow_pickle=False) as checkpoint:
        assert int(checkpoint['frame']) == 2 and checkpoint['aggregate'][4] == 1
        assert float(checkpoint['aggregate'][0]) == plan['frame_steps'][2]
        saved_signature = str(checkpoint['signature'])
    resumed = run_case(geometry, plan, split_root, resume=True, stop_after_frame=4)
    assert full['completed'] and resumed['completed'] and not resumed['segment_complete']
    assert resumed['steps'] == plan['steps'] and resumed['segment_start_frame'] == 2
    assert resumed['segment_end_frame'] == 4
    status = json.loads((split_folder/'status.json').read_text())
    assert status['phase'] == 'completed' and status['signature'] == saved_signature
    assert_same_archive(split_folder/'final_state_0.npz', full_folder/'final_state_0.npz')
    assert_same_archive(split_folder/'particle_fields.npz', full_folder/'particle_fields.npz')
    assert_same_archive(split_folder/'trajectory.npz', full_folder/'trajectory.npz')
    assert_same_archive(split_folder/'checkpoint.npz', full_folder/'checkpoint.npz',
                        excluded=('solver_seconds', 'tables'))
    with np.load(split_folder/'checkpoint.npz', allow_pickle=False) as resumed_checkpoint, \
            np.load(full_folder/'checkpoint.npz', allow_pickle=False) as full_checkpoint:
        assert json.loads(str(resumed_checkpoint['tables'])) == json.loads(str(full_checkpoint['tables']))
    for name in prefix:
        actual = np.load(split_folder/f'trajectory_{name}.npy', mmap_mode='r')
        expected = np.load(full_folder/f'trajectory_{name}.npy', mmap_mode='r')
        try:
            np.testing.assert_array_equal(actual[:3], prefix[name])
            assert_same_array(actual, expected)
        finally:
            actual._mmap.close(); expected._mmap.close()
    for name in ('observables.csv', 'diagnostics.csv'):
        with (split_folder/name).open(newline='') as left, (full_folder/name).open(newline='') as right:
            assert list(csv.DictReader(left)) == list(csv.DictReader(right))
    for key in ('first_material_failure_time_s', 'peak_dynamic_grip_force_N',
                'peak_dynamic_grip_force_time_s', 'actual_clamp_work_J', 'grid_kinetic_correction_J'):
        assert resumed[key] == full[key]
    jax.clear_caches()
