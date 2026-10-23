import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.triaxial_failure import (
    _graph_observation, loading_motion, output_directory, packed_grid_plan,
    plan_case, run_case, save_json,
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
