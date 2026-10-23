import numpy as np
import pytest

from examples.woven_geometry import (
    ARCHITECTURES,
    audit_yarn_surfaces,
    build_paths,
    sample_weave,
)


GEOMETRY = dict(
    size=(4.0, 4.0),
    pitch=1.0,
    yarn_width=0.25,
    yarn_height=0.2,
    gap=0.2,
    matrix_cover=0.1,
)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_uniform_disjoint_phase_occupancy_and_reference_frames(architecture):
    sample = sample_weave(architecture, **GEOMETRY, spacing=0.1)
    assert len(sample.positions) == 40 * 8 * 40
    np.testing.assert_allclose(sample.volume, 0.001)
    assert 0 < sample.phase.sum() < len(sample.phase)
    assert sample.metadata["sampled_yarn_intersections"] == 0
    assert sample.metadata["unresolved_normal_projections"] == 0
    assert np.all(sample.yarn_id[sample.phase == 0] == -1)
    assert np.all(sample.family_id[sample.phase == 0] == -1)
    assert np.all(sample.yarn_id[sample.phase == 1] >= 0)
    expected_families = 3 if architecture == "triaxial" else 2
    assert set(sample.family_id[sample.phase == 1]) == set(range(expected_families))
    np.testing.assert_allclose(
        sample.orientation.swapaxes(-1, -2) @ sample.orientation,
        np.broadcast_to(np.eye(3), sample.orientation.shape),
        atol=1e-14,
    )
    np.testing.assert_allclose(np.linalg.det(sample.orientation), 1.0, atol=1e-14)
    assert np.isclose(sample.volume.sum(), 4 * 0.8 * 4)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_surface_clearance_separate_from_particle_sampling(architecture):
    parameters = {
        key: value for key, value in GEOMETRY.items() if key != "matrix_cover"
    }
    audit = audit_yarn_surfaces(architecture, **parameters)
    assert audit["passed"]
    assert audit["surface_samples"] > 10000
    assert audit["minimum_other_yarn_ellipse_radius_squared"] > 1.2
    assert audit["maximum_half_height_times_curvature_bound"] < 0.6


def test_triaxial_paths_exchange_vertical_order_without_fixed_layers():
    paths, crossings, angles = build_paths(
        "triaxial", size=(4, 4), pitch=1, amplitude=0.2
    )
    assert angles == pytest.approx([0, 60, -60])
    indexed = {path.yarn_id: path for path in paths}
    for path in paths:
        assert path.heights.min() < 0 < path.heights.max()
    for crossing in crossings:
        upper = crossing["above_yarn"]
        point = np.asarray(crossing["point_xz"])
        for yarn in crossing["yarns"]:
            path = indexed[yarn]
            height, slope, _ = path.profile(point @ path.direction)
            assert height == pytest.approx(0.2 if yarn == upper else -0.2)
            assert slope == pytest.approx(0, abs=1e-12)


def test_geometry_rejects_overlaps_and_anisotropic_lattice_adjustment():
    with pytest.raises(ValueError, match="sampled yarn overlaps"):
        sample_weave(
            "triaxial",
            size=(4, 4),
            pitch=1,
            yarn_width=0.375,
            yarn_height=0.25,
            gap=0.125,
            matrix_cover=0.1875,
            spacing=0.1,
        )
    with pytest.raises(ValueError, match="spacing must divide"):
        sample_weave("plain", **GEOMETRY, spacing=0.11)


def test_origin_only_translates_the_same_uniform_geometry():
    first = sample_weave("plain", **GEOMETRY, spacing=0.2)
    shift = np.array((2.1, 3.2, -1.8))
    second = sample_weave("plain", **GEOMETRY, spacing=0.2, origin=shift)
    np.testing.assert_allclose(
        second.positions - first.positions,
        np.broadcast_to(shift, first.positions.shape),
    )
    np.testing.assert_array_equal(second.phase, first.phase)
    np.testing.assert_array_equal(second.orientation, first.orientation)
