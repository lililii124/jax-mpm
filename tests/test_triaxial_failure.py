import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.triaxial_failure import (
    _graph_observation, loading_motion, open_trajectory_arrays, output_directory, packed_grid_plan,
    plan_case, run_case, save_json, validate_stop_after_frame,
)


def test_full_plan_has_requested_size_material_length_and_independent_grid_budget():
    geometry, plan = plan_case()
    assert 100000 <= plan['particles'] < 1000000
    assert plan['particles'] == len(geometry['x'])
    assert plan['geometry']['dimensions_m'] == [.024, .016]
    assert plan['geometry']['gauge_periods'] == 18
    assert plan['geometry']['diameter_m'] == 235e-6
    assert plan['geometry']['clearance_m'] == 2e-6
    assert plan['material']['characteristic_length'] == plan['grid']['dx_m'] == 75e-6
    assert plan['material']['density'] == 1500.
    assert 0 < plan['grid']['nodes'] <= 35_000_000
    assert plan['steps'] == 19200 and plan['saved_states'] == 61
    np.testing.assert_allclose(plan['dt']*plan['wave_speed_m_per_s']/plan['grid']['dx_m'], .15, rtol=1e-15)
    assert not plan['production_eligible']
    assert plan['load_transits'] == 10.
    assert plan['sampling_parameters']['max_particles'] == 1_000_000
    assert plan['fracture_length']['below_upper_bound'] is True


def test_load_relative_angle_is_negative_weave_rotation():
    _, plan = plan_case(15, smoke=True)
    assert plan['theta_deg'] == 15 and plan['geometry_theta_deg'] == -15
    assert plan['geometry']['nominal_family_angles_deg'] == [-15., 45., 105.]
    assert min(plan['grip_particle_counts']) > 0
    assert plan['scope'] == 'smoke'


def test_tube_plan_keeps_physics_and_reports_only_the_dense_storage_bound():
    _,dense=plan_case(smoke=True,probe_steps=8)
    _,tube=plan_case(smoke=True,probe_steps=8,grid_storage='tube')
    for key in ('material','geometry','dt','load_time_s','steps','stop_steps','sampling_hash','total_fixture_extension_m'):
        assert tube[key]==dense[key]
    assert dense['grid']['storage']=='dense' and tube['grid']['storage']=='tube'
    assert tube['grid']['nodes']==dense['grid']['nodes']
    assert tube['grid']['planned_active_nodes'] is None
    assert tube['grid']['dense_to_active_int32_bytes']==4*dense['grid']['nodes']
    with pytest.raises(ValueError,match='grid_storage'):
        plan_case(grid_storage='shared')


def test_probe_keeps_full_loading_physics_but_separate_output_scope():
    _, full = plan_case(smoke=True)
    _, probe = plan_case(smoke=True, probe_steps=8)
    for name in ('material', 'geometry', 'dt', 'load_time_s', 'duration_s', 'steps', 'max_engineering_strain', 'total_fixture_extension_m'):
        assert probe[name] == full[name]
    assert probe['frame_steps'] == [0, 8]
    assert probe['stop_steps'] == 8 and probe['saved_states'] == 2
    assert output_directory(Path('outputs/test'), probe) != output_directory(Path('outputs/test'), full)
    assert probe['scope'] == 'smoke-probe' and not probe['production_eligible']


@pytest.mark.parametrize('transits', [20., 40.])
def test_loading_transit_change_keeps_material_geometry_dt_and_total_strain(transits):
    geometry, original = plan_case(smoke=True)
    slower_geometry, slower = plan_case(smoke=True, load_transits=transits)
    np.testing.assert_array_equal(slower_geometry['x'], geometry['x'])
    for name in ('material', 'geometry', 'dt', 'grid', 'sampling_hash', 'total_fixture_extension_m'):
        assert slower[name] == original[name]
    ratio = transits/10
    assert slower['load_time_s'] == ratio*original['load_time_s']
    assert slower['requested_duration_s'] == ratio*original['requested_duration_s']
    assert slower['requested_hold_transits'] == .2*transits
    assert 0 <= slower['duration_rounding_seconds'] < slower['dt']
    assert slower['steps'] >= ratio*(original['steps']-1)
    original_motion = loading_motion(.35*original['load_time_s'], original['total_fixture_extension_m'], original['load_time_s'])
    slower_motion = loading_motion(.35*slower['load_time_s'], slower['total_fixture_extension_m'], slower['load_time_s'])
    np.testing.assert_allclose(slower_motion[0], original_motion[0], rtol=1e-14)
    np.testing.assert_allclose(slower_motion[1]*ratio, original_motion[1], rtol=1e-14)


def test_explicit_dimensions_grips_geometry_and_material_override_smoke_defaults():
    geometry, plan = plan_case(smoke=True, dimensions=(.008, .005), grip_length=.0015,
        diameter=200e-6, pitch=.001, crimp=140e-6, clearance=25e-6,
        young=12e9, poisson=.2, density=1600., tensile_strength=80e6, fracture_energy=120.)
    assert plan['scope'] == 'smoke'
    assert plan['geometry']['dimensions_m'] == [.008, .005]
    np.testing.assert_allclose(plan['grip_x_intervals_m'], [[0., .0015], [.0065, .008]], rtol=0, atol=1e-18)
    assert plan['geometry']['diameter_m'] == 200e-6
    assert plan['geometry']['clearance_m'] == 25e-6
    assert plan['material'] == dict(young=12e9, poisson=.2, density=1600., tensile_strength=80e6,
                                    fracture_energy=120., characteristic_length=75e-6)
    np.testing.assert_allclose(geometry['mass'], 1600.*geometry['volume'], rtol=0, atol=0)
    expected_wave_speed = np.sqrt(12e9*.8/(1.2*.6*1600.))
    np.testing.assert_allclose(plan['wave_speed_m_per_s'], expected_wave_speed, rtol=1e-15)
    np.testing.assert_allclose(plan['load_time_s'], 10*.005/expected_wave_speed, rtol=1e-15)
    assert plan['physical_parameters']['material'] == plan['material']


def test_sampling_packing_changes_resolution_without_changing_mass_or_fracture_length():
    geometry, original = plan_case(smoke=True)
    fine_geometry, fine = plan_case(smoke=True, packing=2.)
    assert len(fine_geometry['x']) > len(geometry['x'])
    for name in ('material', 'dt', 'load_time_s', 'fracture_length', 'physical_parameters'):
        assert fine[name] == original[name]
    for name in ('reference_volume_m3', 'mass_kg', 'nominal_envelope_solid_fraction'):
        np.testing.assert_allclose(fine['geometry'][name], original['geometry'][name], rtol=2e-15)
    assert fine['sampling_parameters']['packing'] == 2.
    assert fine['geometry']['sampling_density_multiplier'] == 2.
    assert fine['sampling_parameters']['nominal_sampling_spacing_m'] < original['particle_spacing_m']


@pytest.mark.parametrize('arguments', [
    dict(load_transits=0.), dict(load_transits=float('inf')), dict(poisson=.5),
    dict(poisson=-1.), dict(young=-1.), dict(density=0.), dict(tensile_strength=0.),
    dict(fracture_energy=1.), dict(packing=0.), dict(max_particles=0),
    dict(max_nodes=2**31), dict(dimensions=(.006,)), dict(grip_length=.003),
])
def test_parameterized_plan_rejects_invalid_material_protocol_and_budgets(arguments):
    with pytest.raises(ValueError):
        plan_case(smoke=True, **arguments)


def test_particle_budget_is_enforced_without_coarsening_requested_sampling():
    with pytest.raises(ValueError, match='max_particles=100'):
        plan_case(smoke=True, max_particles=100)


def test_tube_still_respects_explicit_dense_addressing_budget():
    _, plan = plan_case(smoke=True, grid_storage='tube')
    with pytest.raises(ValueError, match='max_nodes'):
        plan_case(smoke=True, grid_storage='tube', max_nodes=plan['grid']['nodes']-1)
    assert 'dense AABB' in plan['grid']['budget_scope']


def test_half_cosine_grips_preserve_symmetry_and_hold_without_velocity_jump():
    stroke, duration = .00036, .00006
    for t in (0., duration, 1.2*duration):
        displacement, velocity = loading_motion(t, stroke, duration)
        np.testing.assert_array_equal(velocity, 0)
        np.testing.assert_allclose(displacement.sum(axis=0), 0, atol=1e-20)
    displacement, _ = loading_motion(1.2*duration, stroke, duration)
    np.testing.assert_allclose(displacement[:, 0], [-stroke/2, stroke/2], rtol=1e-15)
    t, h = .37*duration, duration*1e-5
    derivative = (loading_motion(t+h, stroke, duration)[0]-loading_motion(t-h, stroke, duration)[0])/(2*h)
    np.testing.assert_allclose(derivative, loading_motion(t, stroke, duration)[1], rtol=2e-10, atol=1e-12)
    np.testing.assert_array_equal(loading_motion(t, stroke, duration)[1][:, 1:], 0)


def test_packed_plan_counts_only_local_axis_aligned_fields_and_refuses_large_budget():
    geometry = {'metadata': {'fibres': [
        {'reference_tube_bounds_m': [0., 1., 0., 1., 0., 1.]},
        {'reference_tube_bounds_m': [2., 3., 0., 1., 0., 1.]},
    ]}}
    plan = packed_grid_plan(geometry, dx=.25, padding_cells=2, max_nodes=2000)
    assert plan['fibre_shapes'] == [[9, 9, 9], [9, 9, 9]]
    assert plan['nodes'] == 2*9**3
    with pytest.raises(ValueError, match='max_nodes'):
        packed_grid_plan(geometry, dx=.25, padding_cells=2, max_nodes=1000)
    with pytest.raises(ValueError, match='int32 index range'):
        packed_grid_plan(geometry, dx=1e-15, padding_cells=2, max_nodes=1000)
    with pytest.raises(ValueError, match='max_nodes'):
        packed_grid_plan(geometry, dx=1e-8, padding_cells=2, max_nodes=1000)


@pytest.mark.parametrize('arguments', [dict(theta=32), dict(dx=.0003), dict(max_strain=0.), dict(probe_steps=-1)])
def test_invalid_physical_or_protocol_inputs_fail_before_runtime(arguments):
    with pytest.raises(ValueError):
        plan_case(**arguments)


def test_dry_run_subprocess_never_imports_jax_or_allocates_a_grid(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = "import sys,json; from examples.triaxial_failure import plan_case; g,p=plan_case(smoke=True,probe_steps=8); assert 'jax' not in sys.modules; assert 'mpm_jax' not in sys.modules; print(json.dumps({'nodes':p['grid']['nodes'],'points':len(g['x'])}))"
    result = subprocess.run([sys.executable, '-c', script], cwd=root, capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert record['nodes'] > record['points'] > 0


def test_tube_dry_run_does_not_import_jax_scipy_or_construct_the_active_mask():
    root=Path(__file__).resolve().parents[1]
    script="import sys,json; from examples.triaxial_failure import plan_case; g,p=plan_case(smoke=True,probe_steps=8,grid_storage='tube'); assert 'jax' not in sys.modules and 'mpm_jax' not in sys.modules and 'scipy' not in sys.modules; print(json.dumps(p['grid']))"
    result=subprocess.run([sys.executable,'-c',script],cwd=root,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['planned_active_nodes'] is None


def test_parameterized_cli_dry_run_records_resolved_inputs_without_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'unused-output'
    args = [sys.executable, '-m', 'examples.triaxial_failure', '--dry-run', '--smoke',
            '--grid-storage', 'tube', '--dimensions', '.008', '.005', '--grip-length', '.0015',
            '--load-transits', '20', '--diameter', '.0002', '--pitch', '.001', '--crimp', '.00014',
            '--clearance', '.000025', '--young', '12000000000', '--poisson', '.2', '--density', '1600',
            '--tensile-strength', '80000000', '--fracture-energy', '120', '--packing', '1.2',
            '--padding-cells', '7', '--max-particles', '200000', '--max-nodes', '10000000', '--output', str(output)]
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan['load_transits'] == 20.
    assert plan['physical_parameters']['dimensions_m'] == [.008, .005]
    assert plan['material']['young'] == 12e9
    assert plan['sampling_parameters']['packing'] == 1.2
    assert plan['sampling_parameters']['max_particles'] == 200000
    assert plan['grid']['max_nodes'] == 10000000 and plan['grid']['padding_cells'] == 7
    assert plan['grid']['planned_active_nodes'] is None
    assert not output.exists()


def test_completed_resume_is_read_only_and_source_changes_are_refused(tmp_path):
    geometry, plan = plan_case(smoke=True, probe_steps=8)
    folder = output_directory(tmp_path, plan)
    save_json(folder/'plan.json', plan)
    save_json(folder/'status.json', {'completed': False, 'probe_complete': True})
    summary = {'completed': False, 'probe_complete': True, 'scope': 'smoke-probe'}
    save_json(folder/'summary.json', summary)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert run_case(geometry, plan, tmp_path, resume=True) == summary
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError, match='already exists'):
        run_case(geometry, plan, tmp_path)
    altered = {**plan, 'source': {'fingerprint': 'changed-source'}}
    with pytest.raises(ValueError, match='identical'):
        run_case(geometry, altered, tmp_path, resume=True)


def test_summary_json_accepts_numpy_scalars_and_marks_undefined_ratios(tmp_path):
    save_json(tmp_path/'result.json', {'completed': np.bool_(False), 'steps': np.int64(8), 'ratio': np.float64(np.nan)})
    assert json.loads((tmp_path/'result.json').read_text()) == {'completed': False, 'steps': 8, 'ratio': None}


def test_section_contact_graph_is_reported_separately_from_axial_support_cuts():
    x = np.array([[-.3, 0., 0.], [-.2, 0., 0.], [.2, 0., 0.], [.3, 0., 0.]])
    geometry = {'fibre_id': np.array([0, 0, 1, 1]), 'metadata': {'fibres': [{'id': 0}, {'id': 1}]}}
    data = SimpleNamespace(particle_section=np.arange(4), section_count=4, segments=np.array([[0, 1], [2, 3]]))
    masks = (np.array([True, False, False, False]), np.array([False, False, False, True]))
    history = SimpleNamespace(segments=np.array([[0, 1]]), law=SimpleNamespace(active=np.array([False])))
    assert _graph_observation(x, np.zeros(4, bool), geometry, data, history, .1, masks) == (False, 0)
    history.law.active[0] = True
    assert _graph_observation(x, np.zeros(4, bool), geometry, data, history, .1, masks) == (True, 0)
    x[0, 0] = -1.
    # Section adjacency is deliberately labelled approximate, so the sufficient x-cut is separate.
    assert _graph_observation(x, np.zeros(4, bool), geometry, data, history, .1, masks) == (True, 1)


@pytest.mark.parametrize('frame', [-1, 5, True, 1.5])
def test_runtime_segment_stop_rejects_nonframe_values(frame):
    with pytest.raises(ValueError, match='saved frame index'):
        validate_stop_after_frame(frame, {'saved_states': 5})


def test_runtime_segment_stop_accepts_initial_final_and_default_frames():
    plan = {'saved_states': 5}
    for frame in (None, 0, 2, 4):
        validate_stop_after_frame(frame, plan)
    for frame in (0, 2):
        with pytest.raises(ValueError, match='later than'):
            validate_stop_after_frame(frame, plan, checkpoint_frame=2)
    validate_stop_after_frame(4, plan, checkpoint_frame=2)


def segment_fixture(tmp_path, *, phase='checkpoint-ready', valid=True):
    plan = {'saved_states': 5, 'scope': 'smoke', 'theta_deg': 0, 'law': 'jkr', 'source': {}}
    folder = output_directory(tmp_path, plan)
    save_json(folder/'plan.json', plan)
    save_json(folder/'status.json', {'completed': False, 'phase': phase, 'frame': 2})
    np.savez(folder/'checkpoint.npz', frame=2, aggregate=[6., -1., 1., .1, float(valid), 0.])
    return plan, folder


@pytest.mark.parametrize('frame', [0, 2])
def test_runtime_segment_resume_rejects_earlier_stop_without_touching_checkpoint(tmp_path, frame):
    plan, folder = segment_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(ValueError, match='later than'):
        run_case({}, plan, tmp_path, resume=True, stop_after_frame=frame)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize('phase,valid', [('failed', True), ('checkpoint-ready', False)])
def test_runtime_segment_cannot_clear_a_failed_run_or_invalid_checkpoint(tmp_path, phase, valid):
    plan, folder = segment_fixture(tmp_path, phase=phase, valid=valid)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    with pytest.raises(ValueError, match='diagnostic evidence|rejected coupled step'):
        run_case({}, plan, tmp_path, resume=True, stop_after_frame=4)
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


def test_runtime_segment_option_does_not_enter_plan_or_signature(tmp_path, monkeypatch):
    from examples import triaxial_failure as example

    plan = {'saved_states': 5, 'scope': 'smoke', 'theta_deg': 0, 'law': 'jkr', 'source': {}}
    calls = []
    monkeypatch.setattr(example, 'source_record', lambda: {})

    def runtime(geometry, actual_plan, folder, signature, resume, stop_after_frame):
        calls.append((signature, stop_after_frame))
        assert actual_plan == plan
        assert json.loads((folder/'plan.json').read_text()) == plan
        return {'segment_complete': stop_after_frame is not None}

    monkeypatch.setattr(example, '_run_runtime', runtime)
    example.run_case({}, plan, tmp_path/'full')
    example.run_case({}, plan, tmp_path/'segmented', stop_after_frame=2)
    assert calls[0][0] == calls[1][0]
    assert [call[1] for call in calls] == [None, 2]


def test_runtime_segment_completed_resume_remains_read_only(tmp_path):
    plan, folder = segment_fixture(tmp_path)
    save_json(folder/'status.json', {'completed': True, 'phase': 'completed', 'frame': 4})
    summary = {'completed': True, 'segment_complete': False}
    save_json(folder/'summary.json', summary)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert run_case({}, plan, tmp_path, resume=True, stop_after_frame=1) == summary
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


@pytest.mark.parametrize('change', ['shape', 'dtype', 'missing'])
def test_runtime_segment_resume_requires_complete_trajectory_allocations(tmp_path, change):
    shapes, dtypes = {'x': (5, 2, 3), 'damage': (5, 2)}, {'x': np.float64, 'damage': np.float64}
    arrays = open_trajectory_arrays(tmp_path, shapes, dtypes, False)
    for array in arrays.values():
        array.flush(); array._mmap.close()
    path = tmp_path/'trajectory_damage.npy'
    if change == 'missing':
        path.unlink()
    else:
        np.save(path, np.zeros((3, 2) if change == 'shape' else (5, 2),
                              dtype=np.float32 if change == 'dtype' else np.float64))
    with pytest.raises((ValueError, FileNotFoundError)):
        open_trajectory_arrays(tmp_path, shapes, dtypes, True)
    (tmp_path/'trajectory_x.npy').unlink()


def test_runtime_segment_cli_dry_run_keeps_complete_plan(tmp_path):
    root = Path(__file__).resolve().parents[1]
    args = [sys.executable, '-m', 'examples.triaxial_failure', '--dry-run', '--smoke',
            '--frames', '4', '--load-transits', '.03', '--max-strain', '1e-7']
    full = json.loads(subprocess.run(args, cwd=root, capture_output=True, text=True, check=True).stdout)
    segmented = json.loads(subprocess.run([*args, '--stop-after-frame', '2'],
        cwd=root, capture_output=True, text=True, check=True).stdout)
    assert full == segmented
    assert full['saved_states'] == 5 and full['probe_steps'] == 0 and full['stop_steps'] == full['steps']
