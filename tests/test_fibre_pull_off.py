import json

import numpy as np
import pytest

from examples.fibre_pull_off import (
    MODELS, _validate_static_fields, fixture_motion, make_geometry, output_directory, plan_case, run_case,
)


def test_uniform_crossed_fibres_have_exact_centroid_symmetry_and_independent_mass():
    geometry = make_geometry(spacing=.001)
    metadata = geometry['metadata']
    n, cross = metadata['fibres'][0]['longitudinal_sections'], metadata['cross_section_particles']
    first, second = geometry['x'].reshape(2, n, cross, 3)
    np.testing.assert_allclose(first.mean(1)[:, 1:], 0., atol=2e-18)
    np.testing.assert_allclose(second.mean(1)[:, [0, 1]], np.tile([0., .00405], (n, 1)), atol=2e-18)
    np.testing.assert_allclose(np.diff(first.mean(1)[:, 0]), .001, rtol=0, atol=2e-17)
    np.testing.assert_array_equal(geometry['mass'], 1000*geometry['volume'])
    np.testing.assert_array_equal(geometry['volume'], np.full(2*n*cross, .001**3))
    assert np.count_nonzero(geometry['grip_masks'][0]) == np.count_nonzero(geometry['grip_masks'][1])
    assert not np.any(geometry['grip_masks'][0] & geometry['grip_masks'][1])
    assert abs(metadata['sampled_cross_section_area_m2']-cross*.001**2) < 1e-20


def test_four_models_share_identical_real_geometry_material_and_protocol():
    plans = [plan_case(model=model, dx=.002, spacing=.001, frames=4) for model in MODELS]
    geometry, original = plans[0]
    for other_geometry, plan in plans[1:]:
        for key in ('x', 'mass', 'volume', 'fibre_id'):
            np.testing.assert_array_equal(geometry[key], other_geometry[key])
        for key in ('geometry', 'material', 'protocol', 'grid', 'dt', 'steps', 'frame_steps', 'contact'):
            assert plan[key] == original[key]
    assert original['geometry_scale'] == 1


def test_formal_plan_uses_narrow_independent_grids_and_resolved_beam_diameter():
    geometry, plan = plan_case()
    assert len(geometry['x']) == 202240
    assert plan['grid']['nodes'] == 380250
    assert plan['geometry']['diameter_m']/plan['grid']['dx_m'] == 10
    assert plan['steps'] == 24495
    assert plan['frame_steps'][0] == 0 and plan['frame_steps'][-1] == plan['steps']
    assert np.all(np.diff(plan['frame_steps']) > 0)
    assert plan['dt'] <= plan['cfl']*plan['grid']['dx_m']/plan['wave_speed_m_per_s']


def test_fixture_displacement_has_smooth_ends_hold_and_exact_discrete_travel():
    _, plan = plan_case(dx=.002, spacing=.001, frames=4)
    protocol = plan['protocol']
    times = np.array([0., *protocol['times_s'], .04])
    displacement, velocity = fixture_motion(times, protocol)
    np.testing.assert_allclose(displacement, [0., -.0003, -.0003, .0015, .0015, .0015], rtol=0, atol=1e-18)
    np.testing.assert_allclose(velocity, 0., rtol=0, atol=1e-16)
    steps = np.arange(1, plan['steps']+1)*plan['dt']
    discrete = (fixture_motion(steps, protocol)[0]-fixture_motion(steps-plan['dt'], protocol)[0])/plan['dt']
    roundoff = 8*np.finfo(float).eps*plan['steps']*protocol['lift_m']
    np.testing.assert_allclose(discrete.sum()*plan['dt'], .0015, rtol=0, atol=roundoff)


@pytest.mark.parametrize('arguments', [dict(spacing=.0003), dict(initial_gap=-1e-6), dict(grip_length=.04)])
def test_geometry_rejects_overlapping_or_inconsistent_reference_sampling(arguments):
    with pytest.raises(ValueError):
        make_geometry(**arguments)


@pytest.mark.parametrize('arguments', [dict(cfl=.5), dict(coverage_skin=.001), dict(model='unknown'), dict(frames=True)])
def test_plan_rejects_unsupported_inputs(arguments):
    with pytest.raises(ValueError):
        plan_case(dx=.002, spacing=.001, **arguments)


def test_mutated_reference_grips_are_rejected_before_jax_runtime(tmp_path):
    geometry, plan = plan_case(dx=.002, spacing=.001)
    geometry['grip_masks'][0][:] = False
    with pytest.raises(ValueError, match='grip masks'):
        run_case(geometry, plan, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('mutation', ['value', 'dtype', 'extra', 'missing'])
def test_resume_static_field_validation_rejects_changed_labels(tmp_path, mutation):
    expected = {'phase': np.ones(8, dtype=np.int8), 'family_id': np.repeat(np.arange(2, dtype=np.int32), 4)}
    path = tmp_path/'particle_fields.npz'
    np.savez(path, **expected)
    _validate_static_fields(path, expected)
    changed = {key: value.copy() for key, value in expected.items()}
    if mutation == 'value':
        changed['family_id'][0] = 1
    elif mutation == 'dtype':
        changed['family_id'] = changed['family_id'].astype(np.int64)
    elif mutation == 'extra':
        changed['extra'] = np.zeros(8)
    else:
        del changed['phase']
    np.savez(path, **changed)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Static particle field'):
        _validate_static_fields(path, expected)
    assert path.read_bytes() == before


@pytest.mark.parametrize('model', MODELS)
def test_small_actual_mpm_protocol_balances_each_fibre_momentum(tmp_path, model):
    import jax
    jax.config.update('jax_enable_x64', True)
    geometry, plan = plan_case(model=model, dx=.002, spacing=.001, frames=4,
        duration=.00003, indentation=2e-7, lift=5e-7, initial_gap=1e-7, rings=16)
    result = run_case(geometry, plan, tmp_path)
    assert result['completed'] and result['steps'] == plan['steps'] == 5
    assert result['max_damage'] == 0 and result['min_J'] > 0
    folder = output_directory(tmp_path, plan)
    rows = np.genfromtxt(folder/'observables.csv', delimiter=',', names=True)
    diagnostics = np.genfromtxt(folder/'diagnostics.csv', delimiter=',', names=True)
    assert np.max(diagnostics['body_momentum_balance_residual_Ns']) < 1e-18
    assert np.max(rows['contact_force_balance_N']) < 1e-14
    assert np.max(rows['contact_torque_balance_Nm']) < 1e-15
    assert rows['grip_position_error_m'].max() < 1e-14
    if model in ('md', 'self-consistent'):
        assert rows['contact_force_N'].min() < 0
        assert abs(rows['upper_contact_impulse_y_Ns'][-1]) > 0
        assert abs(rows['lower_momentum_y_Ns'][-1]) > 0
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as saved:
        assert int(saved['totals_steps']) == plan['steps']
        np.testing.assert_allclose(saved['contact_particle_force'].sum(0), 0., atol=1e-14)
        assert np.all(saved['metrics_valid'])
    jax.clear_caches()


def test_actual_self_consistent_mpm_full_and_segmented_states_are_identical(tmp_path):
    import jax
    jax.config.update('jax_enable_x64', True)
    geometry, plan = plan_case(model='self-consistent', dx=.002, spacing=.001, frames=4,
        duration=.00003, indentation=2e-7, lift=5e-7, initial_gap=1e-7, rings=16)
    run_case(geometry, plan, tmp_path/'full')
    partial = run_case(geometry, plan, tmp_path/'split', stop_after_frame=2)
    assert partial['segment_complete'] and not partial['completed']
    split = output_directory(tmp_path/'split', plan)
    checkpoint_before = (split/'checkpoint.npz').read_bytes()
    with pytest.raises(ValueError):
        run_case(geometry, plan, tmp_path/'split', resume=True, stop_after_frame=2)
    assert (split/'checkpoint.npz').read_bytes() == checkpoint_before
    result = run_case(geometry, plan, tmp_path/'split', resume=True)
    assert result['completed'] and not result['segment_complete']
    full = output_directory(tmp_path/'full', plan)
    with np.load(full/'checkpoint.npz', allow_pickle=False) as a, np.load(split/'checkpoint.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if key != 'solver_seconds':
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for name in ('x', 'speed', 'damage', 'failed', 'temperature'):
        a = np.load(full/f'trajectory_{name}.npy', mmap_mode='r')
        b = np.load(split/f'trajectory_{name}.npy', mmap_mode='r')
        try:
            np.testing.assert_array_equal(a, b)
        finally:
            a._mmap.close()
            b._mmap.close()
    assert (full/'observables.csv').read_bytes() == (split/'observables.csv').read_bytes()
    assert (full/'diagnostics.csv').read_bytes() == (split/'diagnostics.csv').read_bytes()
    assert json.loads((split/'status.json').read_text())['phase'] == 'completed'
    jax.clear_caches()
