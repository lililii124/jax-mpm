import copy
import json

import numpy as np
import pytest

from tools.jc_grid_separation import audit, inspect_state


def inspect(x, failed=None, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    return inspect_state(x, np.zeros(len(x), dtype=bool) if failed is None else failed,
        dx=1., n_grid=12, split_plane=6.,
        left_grip=np.array([True] + [False] * (len(x)-1)),
        right_grip=np.array([False] * (len(x)-1) + [True]), **kwargs)


def test_unfailed_particles_can_belong_to_decoupled_fragments():
    report = inspect([[3., 6., 6.], [9., 6., 6.]])
    assert report['no_shared_grid_support_at_split']
    assert report['failed_particles'] == 0
    assert report['shared_positive_3d_grid_nodes'] == 0
    assert report['reference_grips_unfailed_and_on_own_sides']
    assert report['reference_grip_sets_grid_decoupled']


def test_shared_node_couples_particles_even_without_overlapping_centres():
    report = inspect([[5.1, 6., 6.], [6.9, 6., 6.]])
    assert not report['no_shared_grid_support_at_split']
    assert report['shared_positive_3d_grid_nodes'] == 9


def test_true_3d_support_distinguishes_transverse_separation_from_x_projection():
    report = inspect([[5.1, 2., 2.], [6.9, 10., 10.]])
    assert report['no_shared_grid_support_at_split']
    assert report['empty_interior_axial_node_planes'] == []


def test_exact_zero_edge_weights_do_not_couple_but_tiny_positive_weights_do():
    touching = inspect([[4.5, 6., 6.], [7.5, 6., 6.]])
    assert touching['no_shared_grid_support_at_split']
    close = inspect([[np.nextafter(4.5, np.inf), 6., 6.],
                     [np.nextafter(7.5, -np.inf), 6., 6.]])
    assert not close['no_shared_grid_support_at_split']


def test_ballistic_debris_cannot_create_a_grid_bridge():
    x = [[3., 6., 6.], [5., 6., 6.], [7., 6., 6.], [9., 6., 6.]]
    assert not inspect(x)['no_shared_grid_support_at_split']
    report = inspect(x, np.array([False, True, True, False]))
    assert report['no_shared_grid_support_at_split']
    assert report['reference_grips_unfailed_and_on_own_sides']


def test_failed_or_crossed_reference_grip_is_reported_separately():
    report = inspect([[3., 6., 6.], [9., 6., 6.]], np.array([True, False]))
    assert not report['both_current_sides_occupied']
    assert not report['no_shared_grid_support_at_split']
    assert not report['reference_grips_unfailed_and_on_own_sides']
    assert not report['reference_grip_sets_grid_decoupled']
    assert not inspect([[9., 6., 6.], [3., 6., 6.]])['reference_grips_unfailed_and_on_own_sides']


def test_active_out_of_domain_support_is_refused_but_failed_debris_can_leave():
    with pytest.raises(ValueError, match='in-domain'):
        inspect([[-1., 6., 6.], [9., 6., 6.]])
    inspect([[-1., 6., 6.], [9., 6., 6.]], np.array([True, False]))


@pytest.mark.parametrize('bad', [np.array([0, 1]), np.array([False]), np.array([[False, False]])])
def test_failure_labels_are_not_coerced_to_boolean(bad):
    with pytest.raises(ValueError, match='Boolean'):
        inspect([[3., 6., 6.], [9., 6., 6.]], bad)


def test_binary32_support_cannot_be_silently_evaluated_as_binary64():
    with pytest.raises(ValueError, match='binary64'):
        inspect_state(np.array([[3., 6., 6.], [9., 6., 6.]], dtype=np.float32),
            np.array([False, False]), dx=1., n_grid=12, split_plane=6.,
            left_grip=np.array([True, False]), right_grip=np.array([False, True]))


@pytest.fixture
def case(tmp_path):
    x0 = np.array([[3., 6., 6.], [9., 6., 6.]])
    x = np.array([[2., 6., 6.], [10., 6., 6.]])
    np.save(tmp_path / 'trajectory_x.npy', np.stack((x0, x)))
    np.save(tmp_path / 'trajectory_failed.npy', np.zeros((2, 2), dtype=np.uint8))
    np.savez(tmp_path / 'trajectory.npz', time=np.array([0., 1.]))
    np.savez(tmp_path / 'final_state_0.npz', x=x, history_failed=np.zeros(2, dtype=bool),
        valid=np.array(True), history_valid=np.ones(2, dtype=bool), time=np.array(1.))
    summary = {'configs': [{'dim': 3, 'material': 'johnson-cook', 'failure_mode': 'ballistic',
                           'domain_size': 12., 'n_grid': 12, 'dt': .1}],
        'completed': True, 'steps': 10, 'simulation_time': 1., 'particles': [2],
        'definition': {'series': 'jc-necking-sphinxsys-pr1127-v1',
                       'grip_inner_faces_x_m': [3., 9.], 'bar_ends_x_m': [2., 10.]}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    (tmp_path / 'status.json').write_text(json.dumps({'frame': 1, 'frames': 1, 'steps_completed': 10,
        'time': 1., 'completed': True, 'particles': 2}), encoding='utf-8')
    return tmp_path, summary


def test_folder_diagnostic_binds_original_raw_inputs_without_fracture_admission(case):
    folder, _ = case
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    report = audit(folder)
    assert report['diagnostic_complete'] and report['state']['no_shared_grid_support_at_split']
    assert not report['production_eligible'] and not report['full_history_checked']
    assert report['original_acceptance_unchanged']
    assert set(report['input_sha256']) == set(before)
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


def test_retained_failed_mass_case_cannot_use_ballistic_inference(case):
    folder, summary = case
    summary = copy.deepcopy(summary)
    summary['configs'][0]['failure_mode'] = 'retain'
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='ballistic'):
        audit(folder)


def test_diagnostic_refuses_mismatched_checkpoint_and_saved_geometry(case):
    folder, _ = case
    np.save(folder / 'trajectory_x.npy', np.zeros((2, 2, 3)))
    with pytest.raises(ValueError, match='differs'):
        audit(folder)


@pytest.mark.parametrize('valid,history_valid', [
    (np.array(1), np.ones(2, dtype=bool)),
    (np.array(True), np.ones(0, dtype=bool)),
    (np.array(True), np.ones(2, dtype=int)),
])
def test_invalid_checkpoint_flag_shapes_and_types_are_refused(case, valid, history_valid):
    folder, _ = case
    x = np.array([[2., 6., 6.], [10., 6., 6.]])
    np.savez(folder / 'final_state_0.npz', x=x, history_failed=np.zeros(2, dtype=bool),
        valid=valid, history_valid=history_valid, time=np.array(1.))
    with pytest.raises(ValueError, match='invalid'):
        audit(folder)


@pytest.mark.parametrize('defect', ['state_clock', 'saved_clock', 'status_clock', 'partial', 'failed',
                                  'failure_labels', 'steps', 'summary_clock'])
def test_stale_state_cannot_be_paired_with_a_different_saved_endpoint(case, defect):
    folder, summary = case
    status = json.loads((folder / 'status.json').read_text())
    if defect == 'state_clock':
        with np.load(folder / 'final_state_0.npz') as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays['time'] = np.array(99.)
        np.savez(folder / 'final_state_0.npz', **arrays)
    elif defect == 'saved_clock':
        np.savez(folder / 'trajectory.npz', time=np.array([0., .2]))
    elif defect == 'status_clock':
        status['time'] = .1
    elif defect == 'partial':
        status['completed'] = False
    elif defect == 'failed':
        status['failed'] = True
    elif defect == 'failure_labels':
        np.save(folder / 'trajectory_failed.npy', np.ones((2, 2), dtype=np.uint8))
    elif defect == 'steps':
        status['steps_completed'] = 9
    else:
        summary['simulation_time'] = .1
    (folder / 'status.json').write_text(json.dumps(status), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError):
        audit(folder)


@pytest.mark.parametrize('defect', ['extra_frame', 'binary32', 'summary_particles', 'status_particles'])
def test_complete_final_geometry_and_particle_counts_have_exact_shapes(case, defect):
    folder, summary = case
    status = json.loads((folder / 'status.json').read_text())
    if defect in ('extra_frame', 'binary32'):
        data = np.load(folder / 'trajectory_x.npy')
        data = np.concatenate((data, data[-1:])) if defect == 'extra_frame' else data.astype(np.float32)
        np.save(folder / 'trajectory_x.npy', data)
    elif defect == 'summary_particles':
        summary['particles'] = [3]
    else:
        status['particles'] = 3
    (folder / 'status.json').write_text(json.dumps(status), encoding='utf-8')
    (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError):
        audit(folder)
