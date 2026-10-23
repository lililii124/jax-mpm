import numpy as np
import pytest

from examples.triaxial_fibre_geometry import (
    DEFAULT_ANGLES_DEG, centreline, clearance_report, make_geometry, symmetry_report,
)


def small_geometry(**kwargs):
    return make_geometry(**{"dimensions": (.006, .004), "particle_spacing": .00012, "grip_length": .001, **kwargs})


def sorted_points(points):
    quantized = np.rint(points/1e-12).astype(np.int64)
    return quantized[np.lexsort((quantized[:, 2], quantized[:, 1], quantized[:, 0]))]


def test_default_weave_has_conservative_positive_clearance():
    report = clearance_report(.001, .0002, .00014, .000025)
    assert report["passed"]
    assert report["minimum_surface_clearance_lower_bound_m"] > .000035
    assert report["minimum_centreline_distance_lower_bound_m"] <= report["polyline_minimum_distance_m"]
    assert report["maximum_radius_times_curvature"] < .5


def test_intersecting_or_excessively_curved_fibres_are_rejected():
    with pytest.raises(ValueError, match="clearance"):
        small_geometry(diameter=.00026)
    with pytest.raises(ValueError, match="curvature"):
        small_geometry(crimp=.001)


def test_crossings_are_two_fibre_over_under_and_alternate():
    pitch, amplitude = .001, .00014
    heights = []
    for other in (1, 2):
        angle = np.pi*other/3
        n_other = np.array([-np.sin(angle), np.cos(angle)])
        crossing = np.linalg.solve(np.stack(([0., 1.], n_other)), np.array([pitch/2, pitch/2]))
        first = centreline(0, 0, crossing[0], pitch=pitch, crimp=amplitude)[0]
        second = centreline(other, 0, crossing@np.array([np.cos(angle), np.sin(angle)]), pitch=pitch, crimp=amplitude)[0]
        np.testing.assert_allclose(first[[0, 2]], second[[0, 2]], atol=1e-18)
        assert np.isclose(abs(first[1]-second[1]), 2*amplitude, rtol=0, atol=1e-18)
        heights.append(first[1])
    np.testing.assert_allclose(heights, [amplitude, -amplitude], atol=1e-18)


def test_periodic_symmetry_is_conditional_on_thickness_turnover():
    report = symmetry_report(.001, .00014)
    assert report["passed"]
    assert report["rotation_60_max_coordinate_error_m"] < 1e-17
    assert report["in_plane_reflection_with_thickness_turnover_max_coordinate_error_m"] < 1e-17
    assert "One-sided contact" in report["angle_interval_condition"]
    assert DEFAULT_ANGLES_DEG == (0, 5, 10, 15, 20, 25, 30)


def test_finite_sample_is_unchanged_by_a_60_degree_weave_rotation():
    original, rotated = small_geometry(), small_geometry(theta_deg=60)
    np.testing.assert_array_equal(sorted_points(original["x"]), sorted_points(rotated["x"]))
    assert np.isclose(original["volume"].sum(), rotated["volume"].sum(), rtol=1e-14)


def test_offset_angle_mirror_requires_reversing_thickness():
    positive, negative = small_geometry(theta_deg=17), small_geometry(theta_deg=-17)
    points = positive["x"].copy()
    origin = np.array(positive["metadata"]["origin_m"])
    points[:, 1:] = 2*origin[1:]-points[:, 1:]
    np.testing.assert_array_equal(sorted_points(points), sorted_points(negative["x"]))


def test_particle_weights_recover_true_tube_volume_and_mass():
    geometry = small_geometry()
    metadata = geometry["metadata"]
    assert geometry["x"].dtype == geometry["volume"].dtype == geometry["tangent"].dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(geometry["tangent"], axis=1), 1, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(geometry["mass"], metadata["density_kg_m3"]*geometry["volume"])
    np.testing.assert_allclose(geometry["particle_spacing_equivalent_m"]**3, geometry["volume"], rtol=1e-15)
    assert metadata["matrix_present"] is False
    for fibre in metadata["fibres"]:
        selected = geometry["fibre_id"] == fibre["id"]
        expected = np.pi*(metadata["diameter_m"]/2)**2*fibre["reference_arclength_m"]
        assert selected.sum() == fibre["particle_count"]
        assert set(geometry["family_id"][selected]) == {fibre["family_id"]}
        np.testing.assert_allclose(geometry["volume"][selected].sum(), expected, rtol=1e-14)
        assert np.ptp(geometry["volume"][selected]) == 0


def test_sampling_refinement_does_not_change_fibre_volume_or_geometry_length():
    coarse = small_geometry()
    fine = small_geometry(packing=2.)
    assert len(fine["x"]) > len(coarse["x"])
    np.testing.assert_allclose(fine["volume"].sum(), coarse["volume"].sum(), rtol=1e-14)
    assert fine["metadata"]["geometric_reference_length_m"] == coarse["metadata"]["geometric_reference_length_m"] == .0002
    assert "Neither is assigned" in fine["metadata"]["fracture_length_note"]


def test_rectangular_domain_and_grips_leave_an_interior_gauge():
    geometry = small_geometry(theta_deg=30)
    assert np.all(geometry["x"][:, [0, 2]] >= 0)
    assert np.all(geometry["x"][:, [0, 2]] <= [.006, .004])
    assert geometry["metadata"]["gauge_length_m"] == .004
    assert (geometry["x"][:, 0] < .001).any()
    assert (geometry["x"][:, 0] > .005).any()


def test_particle_budget_is_enforced_before_growing_large_arrays():
    with pytest.raises(ValueError, match="max_particles"):
        small_geometry(max_particles=10)


def test_reference_centrelines_and_bounding_boxes_are_auditable():
    geometry = small_geometry(theta_deg=13)
    reference, metadata = geometry['reference_centrelines'], geometry['metadata']
    origin = np.array(metadata['origin_m'])
    assert reference['offsets'][0] == 0
    assert reference['offsets'][-1] == len(reference['x'])
    for fibre in metadata['fibres']:
        start, end = fibre['centreline_sample_slice']
        parameter = reference['parameter_m'][start:end]
        np.testing.assert_array_equal(parameter[[0, -1]], fibre['parameter_range_m'])
        expected = centreline(fibre['family_id'], fibre['periodic_line_index'], parameter,
            pitch=metadata['pitch_m'], crimp=metadata['crimp_amplitude_m'], theta_deg=metadata['theta_deg'])[0]+origin
        np.testing.assert_array_equal(expected, reference['x'][start:end])
        assert np.all(reference['fibre_id'][start:end] == fibre['id'])
        bounds = np.array(fibre['reference_tube_bounds_m']).reshape(3, 2)
        particles = geometry['x'][geometry['fibre_id'] == fibre['id']]
        assert np.all(particles >= bounds[:, 0]) and np.all(particles <= bounds[:, 1])


def test_crossing_candidates_have_physical_radial_surface_points():
    geometry = small_geometry(theta_deg=13)
    metadata = geometry['metadata']
    assert metadata['contact_candidate_count'] == len(geometry['contact_candidates']) > 0
    seen = set()
    for candidate in geometry['contact_candidates']:
        pair = tuple(candidate['fibre_ids'])
        assert pair not in seen and pair[0] != pair[1]
        seen.add(pair)
        centers = np.array(candidate['reference_centreline_points_m'])
        surfaces = np.array(candidate['reference_surface_points_m'])
        normal = np.array(candidate['normal_from_first_to_second'])
        np.testing.assert_allclose(np.linalg.norm(normal), 1, rtol=0, atol=3e-16)
        np.testing.assert_allclose(surfaces-centers, metadata['diameter_m']/2*np.array([normal, -normal]), atol=1e-18)
        np.testing.assert_allclose(surfaces[1]-surfaces[0], candidate['reference_gap_m']*normal, atol=1e-18)
        assert candidate['reference_gap_m'] >= metadata['clearance_validation']['minimum_surface_clearance_lower_bound_m']
        for identifier, parameter in zip(pair, candidate['reference_parameters_m']):
            fibre = metadata['fibres'][identifier]
            _, tangent, _ = centreline(fibre['family_id'], fibre['periodic_line_index'], parameter,
                pitch=metadata['pitch_m'], crimp=metadata['crimp_amplitude_m'], theta_deg=metadata['theta_deg'])
            assert abs(tangent@normal) < 1e-10


@pytest.mark.parametrize("argument", [{"packing": 0}, {"density": -1}, {"theta_deg": np.nan}, {"grip_length": .004}])
def test_invalid_sampling_or_loading_geometry_is_rejected(argument):
    with pytest.raises(ValueError):
        small_geometry(**argument)
