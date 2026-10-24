"""Host plan checks and separate low-speed JAX continuation controls."""

import json
from pathlib import Path

import numpy as np
import pytest

from examples import metal_sphere_impact as example


def test_default_plan_preserves_geometry_loading_and_real_mass():
    geometry, plan = example.plan_case()
    definition = plan['definition']
    assert plan['particles'] == 409504
    assert definition['speed_m_s'] == 5000. and plan['duration_s'] == 1e-6
    assert plan['frames'] == 60 and plan['saved_states'] == 61
    np.testing.assert_array_equal(plan['output_times_s'], np.linspace(0, 1e-6, 61))
    assert plan['material'] == example.MATERIAL
    assert definition['density_kg_m3'] == example.MATERIAL['density'] == 2700.
    np.testing.assert_array_equal(np.unique(geometry['body_id']), [0, 1])
    sphere = geometry['body_id'] == 0
    np.testing.assert_array_equal(geometry['v'][sphere, 1], -5000.)
    np.testing.assert_array_equal(geometry['v'][~sphere], 0.)
    mass = geometry['volume']*plan['material']['density']
    np.testing.assert_allclose(mass[sphere].sum(), definition['sphere_mass_kg'], rtol=2e-15)
    np.testing.assert_allclose(mass[~sphere].sum(), definition['plate_mass_kg'], rtol=2e-15)
    assert definition['boundary']['backing'] is False
    assert 'None.' in plan['failure_model']


def test_packed_bounds_include_reference_and_free_flight_without_cubic_air_grid():
    geometry, plan = example.plan_case()
    grid = plan['grid']
    shapes, lower, dx = np.array(grid['shapes']), np.array(grid['lower']), grid['dx_m']
    assert grid['nodes'] == int(np.prod(shapes, axis=1).sum()) == 3952985
    assert grid['nodes'] < grid['max_nodes']
    for b in (0, 1):
        points = geometry['x'][geometry['body_id'] == b]
        base = np.floor(points/dx-.5).astype(int)-lower[b]
        assert (base >= 0).all() and (base+2 < shapes[b]).all()
    sphere = geometry['x'][geometry['body_id'] == 0].copy()
    sphere[:, 1] -= plan['definition']['speed_m_s']*plan['duration_s']
    final_base = np.floor(sphere/dx-.5).astype(int)-lower[0]
    assert (final_base >= 0).all() and (final_base+2 < shapes[0]).all()


def test_tiny_control_is_distinct_from_default_protocol():
    geometry, plan = example.tiny_control_plan()
    assert len(geometry['x']) == 352
    assert plan['definition']['speed_m_s'] == 10.
    assert plan['duration_s'] == 2e-7 and plan['frames'] == 4
    assert plan['grid']['nodes'] == 21018


@pytest.mark.parametrize('density', [1500., 2700.01, float('nan')])
def test_density_cannot_differ_from_actual_material(density):
    with pytest.raises(ValueError, match='density'):
        example.plan_case(density=density)


@pytest.mark.parametrize('endpoint,previous', [(True, -1), (-1, -1), (5, -1), (2, 2), (1, 2), (1.2, -1)])
def test_invalid_runtime_endpoint_rejected(endpoint, previous):
    with pytest.raises(ValueError):
        example.validate_endpoint(endpoint, 4, previous)


def test_runtime_endpoint_is_not_in_physical_plan():
    _, plan = example.tiny_control_plan()
    assert 'stop_after_frame' not in plan
    assert example.validate_endpoint(2, 4) is None
    assert example.validate_endpoint(None, 4, 2) is None


@pytest.mark.parametrize('mutation', ['value', 'dtype', 'extra', 'missing'])
def test_static_labels_have_exact_fields_values_and_dtype(tmp_path, mutation):
    fields = {'family_id': np.array([0, 1], dtype=np.int32), 'physical_body_id': np.array([0, 1], dtype=np.int32)}
    changed = {k:v.copy() for k,v in fields.items()}
    if mutation == 'value': changed['family_id'][1] = 0
    if mutation == 'dtype': changed['family_id'] = changed['family_id'].astype(np.int64)
    if mutation == 'extra': changed['invented'] = np.array([0, 1])
    if mutation == 'missing': del changed['family_id']
    path = tmp_path/'fields.npz'
    example.save_npz(path, **changed)
    with pytest.raises(ValueError): example.check_arrays(path, fields)


def test_run_rejects_modified_geometry_before_importing_runtime(tmp_path):
    geometry, plan = example.tiny_control_plan()
    geometry['volume'][0] *= .5
    with pytest.raises(ValueError, match='geometry'):
        example.run_case(geometry, plan, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_runtime_low_speed_full_and_two_segments_are_identical(tmp_path):
    import jax
    jax.config.update('jax_enable_x64', True)
    geometry, plan = example.tiny_control_plan()
    full, segmented = tmp_path/'full', tmp_path/'segmented'
    result = example.run_case(geometry, plan, full)
    partial = example.run_case(geometry, plan, segmented, stop_after_frame=2)
    assert not partial['completed'] and partial['segment_complete']
    assert partial['simulation_time'] == plan['output_times_s'][2]
    assert partial['target_simulation_time'] == plan['duration_s']
    prefix_sha = example.digest(segmented/'trajectory_x.npy')
    with pytest.raises(ValueError, match='later complete'):
        example.run_case(geometry, plan, segmented, resume=True, stop_after_frame=2)
    assert example.digest(segmented/'trajectory_x.npy') == prefix_sha
    resumed = example.run_case(geometry, plan, segmented, resume=True)
    assert result['completed'] and resumed['completed']
    assert not result['physical_validation_passed'] and not result['production_eligible']
    assert result['simulation_time'] == plan['duration_s']
    with np.load(full/'final_state_0.npz', allow_pickle=False) as a, np.load(segmented/'final_state_0.npz', allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for field in a.files:
            np.testing.assert_array_equal(a[field], b[field], err_msg=field)
        np.testing.assert_array_equal(a['mass'], geometry['volume']*2700.)
        assert bool(a['valid'])
        assert (a['maximum_temperature'] >= 299.99).all()
    for name in example.FIELDS:
        with_arrays = [np.load(path/f'trajectory_{name}.npy', mmap_mode='r', allow_pickle=False) for path in (full,segmented)]
        try: np.testing.assert_array_equal(*with_arrays)
        finally:
            for a in with_arrays: a._mmap.close()
    for name in ('observables.csv', 'diagnostics.csv'):
        assert (full/name).read_bytes() == (segmented/name).read_bytes()
    # This control remains separated and does not exercise a shock/contact event.
    assert result['observables']['final']['sphere_contact_projection_loss_J'] == 0.


def test_runtime_corrupted_reference_or_prefix_rejected_without_writing(tmp_path):
    geometry, plan = example.tiny_control_plan()
    folder = tmp_path/'run'
    example.run_case(geometry, plan, folder, stop_after_frame=1)
    fields = {'family_id':geometry['body_id'].copy(), 'physical_body_id':geometry['body_id'].copy()}
    fields['family_id'][0] = 1
    example.save_npz(folder/'particle_fields.npz', **fields)
    before = example.digest(folder/'trajectory_x.npy')
    with pytest.raises(ValueError, match='bytes changed'):
        example.run_case(geometry, plan, folder, resume=True)
    assert example.digest(folder/'trajectory_x.npy') == before


def test_runtime_budget_failure_preserves_accepted_checkpoint_and_raw_candidate(tmp_path):
    geometry, plan = example.tiny_control_plan()
    folder = tmp_path/'run'
    with pytest.raises(FloatingPointError, match='retained'):
        example.run_case(geometry, plan, folder, max_steps_per_frame=1)
    status = json.loads((folder/'status.json').read_text())
    failure = json.loads((folder/'failure.json').read_text())
    assert status['phase'] == 'failed' and not status['completed'] and status['frame'] == 0
    assert failure['accepted_checkpoint_sha256'] == example.digest(folder/'checkpoint.npz')
    with np.load(folder/'checkpoint.npz', allow_pickle=False) as accepted, np.load(folder/'failed_state.npz', allow_pickle=False) as failed:
        assert accepted['frame'] == 0 and accepted['state_valid']
        assert not failed['valid'] and failed['time'] < plan['output_times_s'][1]
    assert json.loads((folder/'summary.json').read_text())['target_simulation_time'] == plan['duration_s']
    with pytest.raises(ValueError, match='failed'):
        example.run_case(geometry, plan, folder, resume=True)
