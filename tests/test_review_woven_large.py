from copy import deepcopy
import os

import numpy as np
import pytest

from examples.woven_large_geometry import sample_large_weave
from tools.review_woven_large import (_numpy_elastic, review_clamp_owner, review_reference_geometry,
                                     saved_energy_gate, sphere_and_initial_energy_checks)


def reference_fixture():
    sample = sample_large_weave("plain", size=(.01, .01))
    values = dict(reference=sample.positions, volume=sample.volume,
                  mass=sample.volume*np.where(sample.phase == 1, 1400., 1100.),
                  phase=sample.phase, yarn_id=sample.yarn_id, family_id=sample.family_id,
                  orientation=sample.orientation, geometry=sample.metadata,
                  dx=.0005, tool_diameter=.1, expected_size=(.01, .01))
    return values


def test_reference_reconstruction_keeps_coarse_resolution_explicit():
    report = review_reference_geometry(**reference_fixture())
    assert report["passed"]
    assert not report["resolution_checks"]["three_cells_across_yarn_thickness"]
    assert report["resolution_checks"]["eight_particle_layers_through_plate"]
    assert not report["production_eligible"]


@pytest.mark.parametrize("field", ["reference", "volume", "mass", "phase", "yarn_id", "family_id", "orientation"])
def test_a_tampered_reference_array_is_rejected(field):
    values = reference_fixture()
    values[field] = values[field].copy()
    values[field].flat[0] += 1
    assert not review_reference_geometry(**values)["passed"]


def test_pitch_cannot_replace_the_true_repeating_cell():
    values = reference_fixture()
    values["tool_diameter"] = .025
    report = review_reference_geometry(**values)
    assert not report["checks"]["tool_diameter_at_least_ten_true_cells"]
    values = reference_fixture()
    values["geometry"] = deepcopy(values["geometry"])
    values["geometry"]["unit_cell"]["orthogonal_repeating_cell_lengths_xz_m"] = [.0025, .0025]
    assert not review_reference_geometry(**values)["checks"]["unit_cell_metadata_matches"]


def clamp_fixture():
    dx = .0005
    sample = sample_large_weave("plain", size=(.01, .01))
    lower = np.array((-16, -8, -16))
    shape = np.array((33, 17, 33))
    coordinates = (np.indices(tuple(shape)).reshape(3, -1).T+lower)*dx
    masks = (coordinates[:, 0] <= -.00325, coordinates[:, 0] >= .00325,
             coordinates[:, 2] <= -.00325, coordinates[:, 2] >= .00325)
    owner = np.full(len(coordinates), -1, dtype=np.int8)
    for group, mask in enumerate(masks):
        owner[mask & (owner < 0)] = group
    return dict(owner=owner, reference=sample.positions, dx=dx, grid_origin=(0., 0., 0.),
                lower=lower, shape=shape, size=(.01, .01), band=.001)


def test_symmetric_four_edge_mask_has_full_support_without_backing():
    report = review_clamp_owner(**clamp_fixture())
    assert report["passed"], report


def test_backing_plane_or_missing_edge_cannot_pass_as_four_edge_clamp():
    values = clamp_fixture()
    owner = values["owner"].reshape(tuple(values["shape"]))
    owner[16, 0, 16] = 0
    report = review_clamp_owner(**values)
    assert not report["checks"]["no_clamped_nodes_outside_four_edge_bands"]
    values = clamp_fixture()
    values["owner"][values["owner"] == 3] = -1
    report = review_clamp_owner(**values)
    assert not report["checks"]["four_nonempty_edge_groups"]
    assert not report["checks"]["all_positive_edge_support_clamped"]


def test_asymmetric_extra_node_plane_is_rejected():
    values = clamp_fixture()
    owner = values["owner"].reshape(tuple(values["shape"]))
    owner[10, :, :] = 0
    report = review_clamp_owner(**values)
    assert not report["checks"]["clamp_union_x_reflection_equivalent"]


def energy_row(*, time=0., elastic=0., translation=0., affine=0., ball=2., tool=0., external=0., loss=0., projection=0.):
    return {"time": time, "stored_elastic_energy_J": elastic, "kinetic_translational_J": translation,
            "kinetic_apic_affine_J": affine, "ball_kinetic_energy_J": ball,
            "prescribed_tool_work_J": tool, "particle_external_work_J": external,
            "contact_kinetic_loss_J": loss, "grid_projection_work_J": projection}


def test_one_sided_energy_gate_accounts_for_contact_and_fixed_grid_losses():
    rows = [energy_row(), energy_row(time=1., elastic=.4, translation=.1, affine=.1, ball=1.2, loss=.1, projection=-.1)]
    report = saved_energy_gate(rows, 1., 1.)
    assert report["passed"]
    assert report["history"][-1]["creation_residual_J"] == pytest.approx(0., abs=5e-16)
    rows[-1]["stored_elastic_energy_J"] += .11
    assert not saved_energy_gate(rows, 1., 1.)["passed"]


def test_additional_apic_dissipation_is_not_mislabeled_nonconservation_failure():
    rows = [energy_row(), energy_row(time=1., elastic=.1, translation=.1, ball=.1, loss=.1)]
    report = saved_energy_gate(rows, 1., 1.)
    assert report["passed"]
    assert report["history"][-1]["creation_residual_J"] < -1.


def test_prescribed_tool_energy_does_not_need_a_free_sphere_kinetic_term():
    rows = [energy_row(ball=None), energy_row(time=1., elastic=.8, translation=.2, ball=None, tool=1.)]
    assert saved_energy_gate(rows, 1., 1.)["passed"]
    rows[-1]["stored_elastic_energy_J"] = 1.
    assert not saved_energy_gate(rows, 1., 1.)["passed"]


@pytest.mark.parametrize("key,value", [("contact_kinetic_loss_J", -.1), ("grid_projection_work_J", .1),
                                      ("stored_elastic_energy_J", np.nan), ("kinetic_apic_affine_J", -.1)])
def test_nonphysical_or_nonfinite_energy_records_fail(key, value):
    rows = [energy_row(), energy_row(time=1., ball=1.)]
    rows[-1][key] = value
    assert not saved_energy_gate(rows, 1., 1.)["passed"]


def test_independent_elastic_energy_and_stress_are_objective_and_energy_conjugate():
    from examples.woven_large_plate import WOVEN_MATERIAL
    F = np.array([[[1.002, .003, 0.], [0., .999, .001], [0., 0., 1.]]]*2)
    Q = np.broadcast_to(np.eye(3), F.shape).copy()
    volume, phase = np.array([.7, 1.3]), np.array([0, 1])
    energy, _, tau, _ = _numpy_elastic(F, volume, phase, Q, WOVEN_MATERIAL)
    a = .371
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rotated_energy, _, rotated_tau, _ = _numpy_elastic(rotation@F, volume, phase, Q, WOVEN_MATERIAL)
    assert rotated_energy == pytest.approx(energy, rel=2e-11)
    np.testing.assert_allclose(rotated_tau, rotation@tau@rotation.T, rtol=1e-10, atol=2e-6)
    h = 1e-7
    plus, minus = F.copy(), F.copy()
    plus[1, 0, 0] += h
    minus[1, 0, 0] -= h
    derivative = (_numpy_elastic(plus, volume, phase, Q, WOVEN_MATERIAL)[0]
                  -_numpy_elastic(minus, volume, phase, Q, WOVEN_MATERIAL)[0])/(2*h)
    piola = tau[1]@np.linalg.inv(F[1]).T
    assert derivative == pytest.approx(volume[1]*piola[0, 0], rel=2e-8)


def ball_ledger():
    plan = {"loading": "sphere-impact", "tool": {"mass_kg": .01,
            "initial_center_m": [0., .052, 0.], "initial_velocity_m_s": [0., -20., 0.]}}
    rows = [energy_row(), energy_row(time=1e-6)]
    for i, row in enumerate(rows):
        for axis, position, velocity in zip("xyz", [0., .052-i*20e-6, 0.], [0., -20., 0.]):
            row[f"ball_center_{axis}_m"] = position
            row[f"ball_velocity_{axis}_m_s"] = velocity
    final = {"sphere_center": np.array([0., .052-20e-6, 0.]), "sphere_velocity": np.array([0., -20., 0.])}
    return rows, final, plan


def test_saved_ball_energy_is_bound_to_vectors_and_the_actual_final_state():
    assert all(sphere_and_initial_energy_checks(*ball_ledger()).values())


def test_inflated_initial_energy_cannot_enlarge_the_five_percent_budget():
    rows, final, plan = ball_ledger()
    rows[0]["ball_kinetic_energy_J"] = 2000.
    rows[1]["ball_kinetic_energy_J"] = 2050.
    assert saved_energy_gate(rows, 1., 1.)["passed"]
    checks = sphere_and_initial_energy_checks(rows, final, plan)
    assert not checks["saved_ball_kinetic_energy_matches_vector_velocity"]
    rows, final, plan = ball_ledger()
    rows[0]["stored_elastic_energy_J"] = 2000.
    assert not sphere_and_initial_energy_checks(rows, final, plan)["initial_energy_ledger_matches_quiescent_plate"]


def test_prescribed_tool_cannot_be_given_an_extra_free_ball_energy_supply():
    rows, final, plan = ball_ledger()
    plan["loading"] = "hemisphere-indentation"
    assert not sphere_and_initial_energy_checks(rows, final, plan)["prescribed_or_absent_tool_has_no_free_ball_energy"]


def test_wrong_saved_endpoint_or_initial_velocity_units_are_rejected():
    rows, final, plan = ball_ledger()
    final["sphere_velocity"][1] = -19.
    assert not sphere_and_initial_energy_checks(rows, final, plan)["saved_final_tool_state_matches_checkpoint"]
    rows, final, plan = ball_ledger()
    rows[0]["ball_velocity_y_m_s"] = -20/3.6
    assert not sphere_and_initial_energy_checks(rows, final, plan)["saved_initial_tool_state_matches_plan"]


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="actual JAX driver/reviewer integration is reserved for the cloud CI")
def test_actual_jax_prefix_review_and_render_label_tampering(tmp_path, monkeypatch):
    from examples import woven_large_plate as example
    from tools.review_woven_large import review_folder

    monkeypatch.setattr(example, "sample_large_weave", lambda architecture:
                        sample_large_weave(architecture, size=(.02, .02)))
    sample, plan = example.make_plan("plain", "pressure-bending", probe_steps=3)
    result = example.run_case(sample, plan, tmp_path)
    assert result["probe_complete"]
    folder = tmp_path/"plain"/"pressure-bending"
    report = review_folder(folder)
    failed = {name for name, passed in report["checks"].items() if not passed}
    # This integration fixture deliberately has a smaller footprint. It must
    # remain ineligible, while every time/state/contact/energy check executes.
    assert failed == {"reference_geometry", "rectangular_grid_has_no_cubic_air_extension"}, report
    assert not report["passed"] and not report["probe_passed"]
    with np.load(folder/"particle_fields.npz", allow_pickle=False) as data:
        fields = {key: data[key].copy() for key in data.files}
    fields["family_id"][0] = 7
    example.save_npz(folder/"particle_fields.npz", **fields)
    changed = review_folder(folder)
    assert not changed["checks"]["rendered_static_labels_match_reference"]
    assert not changed["checks"]["static_label_descriptors_match_render_file"]
