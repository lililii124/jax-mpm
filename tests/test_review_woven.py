import json

import numpy as np
import pytest

from tools.review_woven import (
    Comparisons, END_TIME, LOAD_TIME, array_descriptor, elastic_response,
    equivalent_record, expected_boundaries, motion, stiffness_from_constants, support_mask,
    positive_stored_energy, section_layout, section_resultants,
)


def material():
    parameters = {
        "yarn_young": [40e9, 4e9, 4e9], "yarn_poisson": [.25, .25, .30],
        "yarn_shear": [1.5e9, 2e9, 2e9],
    }
    return {"yarn_stiffness": stiffness_from_constants(parameters), "matrix_mu": 2e9 / 2.6, "matrix_lam": 2e9 * .3 / (1.3 * .4)}


def test_independent_yarn_energy_and_stress_for_uniaxial_extension():
    constants = material()
    stretch = 1.02
    F = np.diag([stretch, 1., 1.])[None]
    energy, strain_norm, stress = elastic_response(F, np.eye(3)[None], np.array([1]), constants)
    green = (stretch**2 - 1) / 2
    stiffness = constants["yarn_stiffness"]
    np.testing.assert_allclose(energy, [stiffness[0, 0] * green**2 / 2], rtol=1e-13)
    np.testing.assert_allclose(strain_norm, [green], rtol=1e-13)
    np.testing.assert_allclose(stress[0], np.diag([stretch**2 * stiffness[0, 0] * green, stiffness[1, 0] * green, stiffness[2, 0] * green]), rtol=1e-13)


def test_matrix_volume_dilation_matches_closed_form():
    constants = material()
    stretch = 1.01
    F = (stretch * np.eye(3))[None]
    energy, _, stress = elastic_response(F, np.eye(3)[None], np.array([0]), constants)
    mu, lam = constants["matrix_mu"], constants["matrix_lam"]
    expected_energy = 1.5 * mu * (stretch**2 - 1) - 3 * mu * np.log(stretch) + 4.5 * lam * np.log(stretch)**2
    np.testing.assert_allclose(energy, [expected_energy], rtol=1e-11)
    np.testing.assert_allclose(stress[0], np.eye(3) * (mu * (stretch**2 - 1) + 3 * lam * np.log(stretch)), rtol=1e-12)


def test_energy_is_objective_and_pure_rotation_has_zero_strain():
    constants = material()
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    F = np.broadcast_to(np.array([[1.01, .01, 0], [0, .995, 0], [0, 0, 1.]]), (2, 3, 3))
    axes = np.broadcast_to(np.eye(3), (2, 3, 3))
    phase = np.array([0, 1])
    energy, norm, stress = elastic_response(F, axes, phase, constants)
    rotated_energy, rotated_norm, rotated_stress = elastic_response(rotation @ F, axes, phase, constants)
    np.testing.assert_allclose(rotated_energy, energy, rtol=1e-12)
    np.testing.assert_allclose(rotated_norm, norm, rtol=1e-12)
    np.testing.assert_allclose(rotated_stress, rotation @ stress @ rotation.T, rtol=1e-12, atol=1e-8)
    energy, norm, stress = elastic_response(np.broadcast_to(rotation, (2, 3, 3)), axes, phase, constants)
    np.testing.assert_array_equal(energy, [0, 0])
    np.testing.assert_array_equal(norm, [0, 0])
    np.testing.assert_array_equal(stress, np.zeros((2, 3, 3)))


def test_nonpositive_J_and_invalid_phase_are_rejected():
    with pytest.raises(ValueError, match="determinant"):
        elastic_response(np.diag([-1., 1., 1.])[None], np.eye(3)[None], np.array([1]), material())
    with pytest.raises(ValueError, match="phase"):
        elastic_response(np.eye(3)[None], np.eye(3)[None], np.array([2]), material())


def test_fixed_physical_fixtures_and_motion_are_verified_separately_from_solver():
    for grid in (48, 96, 192):
        boundaries, _, _ = expected_boundaries("transverse-bending", .016 / grid)
        np.testing.assert_allclose([record["upper"][0] - record["lower"][0] for record in boundaries[:3]], .0004, atol=1e-17)
        boundaries, _, _ = expected_boundaries("transverse-compression", .016 / grid)
        np.testing.assert_allclose(boundaries[1]["lower"][1] - boundaries[0]["upper"][1], .00125, atol=1e-17)
        for boundary in boundaries:
            np.testing.assert_allclose(motion(END_TIME, boundary)[0], boundary["displacement"], atol=1e-16)
            np.testing.assert_array_equal(motion(END_TIME, boundary)[1], np.zeros(3))
            np.testing.assert_allclose(motion(LOAD_TIME / 2, boundary)[0], np.array(boundary["displacement"]) / 2, atol=1e-16)


def test_conservative_grip_sample_survives_translated_grid_phases():
    dx = .016 / 96
    boundary = expected_boundaries("warp-tension", dx)[0][0]
    reference = np.column_stack((.003 + (np.arange(12) + .5) * dx / 2, np.full(12, .008), np.full(12, .008)))
    selected = support_mask(reference, dx, boundary)
    assert 0 < selected.sum() < len(reference)
    for instant in np.linspace(0, END_TIME, 101):
        translation = motion(instant, boundary)[0]
        lower_node = np.floor((reference[selected] + translation) / dx - .5) * dx
        assert np.all(lower_node >= np.asarray(boundary["lower"]) + translation - 1e-14)
        assert np.all(lower_node + 2 * dx <= np.asarray(boundary["upper"]) + translation + 1e-14)


def test_raw_field_hash_binds_dtype_shape_and_changed_labels():
    labels = np.array([0, 1, 1, 0], dtype=np.uint8)
    original = array_descriptor(labels)
    assert original != array_descriptor(labels.astype(np.int32))
    assert original != array_descriptor(labels.reshape(2, 2))
    labels[1] = 0
    assert original != array_descriptor(labels)


def test_comparison_never_accepts_missing_values_nan_or_changed_constraints():
    comparison = Comparisons()
    comparison.add("missing", None, 1.0)
    comparison.add("nonfinite", np.nan, 1.0)
    comparison.add("valid_none", None, None)
    assert comparison.matches == {"missing": False, "nonfinite": False, "valid_none": True}
    json.dumps(comparison.errors, allow_nan=False)
    assert not equivalent_record({"components": [True, True, True]}, {"components": [True, False, True]})
    assert not equivalent_record({"target": .00008}, {"target": .0008})


def test_displaced_points_with_identity_F_cannot_pass_stored_energy_gate():
    reference = np.array([[.0075, .008, .008], [.0085, .008, .008]])
    displaced = reference.copy()
    displaced[:, 0] += [-.00008, .00008]
    assert np.linalg.norm(displaced - reference) > 0
    F = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    axes = F.copy()
    phase = np.array([0, 1])
    W, _, _ = elastic_response(F, axes, phase, material())
    assert not positive_stored_energy(float(np.sum(W * 1e-9)))
    F[:, 0, 0] = 1.02
    W, _, _ = elastic_response(F, axes, phase, material())
    assert positive_stored_energy(float(np.sum(W * 1e-9)))
    assert not positive_stored_energy(np.inf)
    assert not positive_stored_energy(np.nan)
    assert not positive_stored_energy(-1.)


@pytest.mark.parametrize("counts", [(4, 6, 8), (3, 5, 7)])
def test_section_resultants_recover_constant_piola_traction_in_chunks(counts):
    spacing = .001
    axes = [.008 + (np.arange(count) - (count - 1) / 2) * spacing for count in counts]
    reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n = len(reference)
    F = np.broadcast_to(np.array([[1.02, .03, 0], [0, .99, .01], [0, 0, 1.01]]), (n, 3, 3))
    second_piola = np.diag([5e6, 3e6, 2e6])
    tau = F @ second_piola @ F.swapaxes(-1, -2)
    masks, widths = section_layout(reference, spacing)
    np.testing.assert_allclose(widths, [2 * spacing] * 3 if counts[0] % 2 == 0 else [spacing] * 3)
    force, moment = np.zeros((3, 3)), 0.
    for start in range(0, n, 7):
        end = min(start + 7, n)
        partial, partial_moment = section_resultants(F[start:end], tau[start:end], reference[start:end],
            np.full(end - start, spacing**3), [mask[start:end] for mask in masks], widths)
        force += partial
        moment += partial_moment
    expected_piola = F[0] @ second_piola
    dimensions = np.asarray(counts) * spacing
    expected_force = np.stack([expected_piola[:, axis] * np.prod(np.delete(dimensions, axis)) for axis in range(3)])
    np.testing.assert_allclose(force, expected_force, rtol=1e-13, atol=1e-10)
    assert abs(moment) < 1e-12


def test_section_resultants_reference_bending_moment_has_correct_sign_and_scale():
    spacing = .001
    counts = (4, 6, 8)
    axes = [.008 + (np.arange(count) - (count - 1) / 2) * spacing for count in counts]
    reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n = len(reference)
    F = np.broadcast_to(np.eye(3), (n, 3, 3))
    tau = np.zeros((n, 3, 3))
    stress_gradient = 2e9
    tau[:, 0, 0] = stress_gradient * (reference[:, 1] - .008)
    masks, widths = section_layout(reference, spacing)
    force, moment = section_resultants(F, tau, reference, np.full(n, spacing**3), masks, widths)
    expected = -stress_gradient * counts[2] * spacing**2 * np.sum((axes[1] - .008)**2)
    np.testing.assert_allclose(force, 0, atol=1e-10)
    np.testing.assert_allclose(moment, expected, rtol=1e-13)
    assert moment < 0
