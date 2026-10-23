import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.woven_composite import (
    ARCHITECTURES, DOMAIN, DURATION, LOADING_TIME, LOADS, MATERIAL,
    SmoothDisplacementBox, _support_inside, loading_boundaries, loading_resolution,
    make_case, rigid_grips, specimen_bounds,
)

jax.config.update("jax_enable_x64", True)


def test_prescribed_motion_derivative_starts_and_ends_at_rest_then_holds():
    boundary = SmoothDisplacementBox((0, 0, 0), (1, 1, 1), (0.2, -0.1, 0.3), 0.6, start=0.1)
    boundary.validate(3, 1)
    for time in (0.0, 0.1, 0.7, 1.0):
        np.testing.assert_array_equal(boundary.motion(time)[1], np.zeros(3))
    for time in (0.2, 0.4, 0.6):
        derivative = (boundary.motion(time + 1e-6)[0] - boundary.motion(time - 1e-6)[0]) / 2e-6
        np.testing.assert_allclose(derivative, boundary.motion(time)[1], rtol=2e-10, atol=1e-11)
    np.testing.assert_allclose(boundary.motion(0.4)[0], [0.1, -0.05, 0.15], atol=1e-15)
    np.testing.assert_allclose(boundary.motion(1.0)[0], [0.2, -0.1, 0.3], atol=1e-15)


def test_compression_constraints_translate_and_preserve_tangential_velocity():
    boundary = SmoothDisplacementBox((0.1, 0.4, 0.1), (0.5, 0.5, 0.5), (0, -0.2, 0), 0.6, (False, True, False))
    positions = jnp.array([[0.3, 0.25, 0.3], [0.3, 0.45, 0.3]])
    velocity = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(boundary.apply(velocity, positions, 1.0), [[1, 0, 3], [4, 5, 6]])


@pytest.mark.parametrize("duration,start", [(0, 0), (-1, 0), (np.inf, 0), (1, -1), (1, np.nan)])
def test_invalid_motion_parameters_are_rejected(duration, start):
    boundary = SmoothDisplacementBox((0, 0, 0), (1, 1, 1), (0, 1, 0), duration, start=start)
    with pytest.raises(ValueError):
        boundary.validate(3, 1)


@pytest.mark.parametrize("loading", LOADS)
def test_fixture_motion_matches_loading_boxes_and_all_loads_hold(loading):
    boundaries, names, _ = loading_boundaries(loading, DOMAIN / 96)
    by_name = dict(zip(names, boundaries))
    for fixture in rigid_grips(loading, boundaries, names, DOMAIN / 96):
        boundary = by_name[fixture["name"]]
        assert np.all(np.asarray(fixture["upper"]) > fixture["lower"])
        assert fixture["motion_kind"] == "half-cosine-displacement-hold"
        assert fixture["loading_duration"] == LOADING_TIME
        displacement, velocity = boundary.motion(DURATION) if hasattr(boundary, "motion") else (np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(fixture["displacement"], displacement, atol=1e-15)
        np.testing.assert_array_equal(velocity, np.zeros(3))
    assert len(set(names)) == len(names)


def test_bending_rollers_allow_axial_slide_and_lateral_anchor_is_local():
    boundaries, names, _ = loading_boundaries("transverse-bending", DOMAIN / 96)
    lower, _ = specimen_bounds()
    velocity = jnp.array([[1.0, 2.0, 3.0]])
    left = jnp.array([[0.004, lower[1], 0.008]])
    right = jnp.array([[0.012, lower[1], 0.008]])
    np.testing.assert_array_equal(boundaries[0].apply(velocity, left, 0), [[0, 0, 3]])
    np.testing.assert_array_equal(boundaries[1].apply(velocity, right, 0), [[1, 0, 3]])
    np.testing.assert_array_equal(boundaries[2].apply(velocity, left, 0), [[1, 2, 0]])
    np.testing.assert_array_equal(boundaries[2].apply(velocity, right, 0), velocity)
    assert names[-1] == "loading_patch"


def test_fixture_width_and_compression_free_height_do_not_change_with_grid():
    lower, upper = specimen_bounds()
    for grid in (48, 96, 192):
        boundaries, _, bending = loading_boundaries("transverse-bending", DOMAIN / grid)
        np.testing.assert_allclose([b.upper[0] - b.lower[0] for b in boundaries[:3]], 0.0004, atol=1e-17)
        assert bending["support_width_m"] == 0.0004
        boundaries, _, compression = loading_boundaries("transverse-compression", DOMAIN / grid)
        np.testing.assert_allclose(boundaries[0].upper[1] - lower[1], 0.000375, atol=1e-17)
        np.testing.assert_allclose(upper[1] - boundaries[1].lower[1], 0.000375, atol=1e-17)
        assert compression["free_height_m"] == pytest.approx(0.00125)
        assert compression["nominal_final_thickness_strain"] == pytest.approx(-0.025)
        assert compression["nominal_final_free_height_strain"] == pytest.approx(-0.04)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_sampled_phases_define_mass_and_orientation_without_damage(architecture):
    case, substeps = make_case(architecture, "warp-tension", grid=48, ppc=1, frames=4)
    state, material, config = case.states[0], case.materials[0], case.configs[0]
    phase = case.point_fields["phase"]
    expected_density = np.where(phase == 1, MATERIAL["yarn_density"], MATERIAL["matrix_density"])
    np.testing.assert_allclose(state.mass, np.asarray(state.volume) * expected_density, rtol=1e-14)
    np.testing.assert_allclose(state.history.density, expected_density, rtol=0, atol=0)
    np.testing.assert_array_equal(material.phase, phase)
    assert set(case.point_fields) == {"phase", "family_id", "yarn_id"}
    assert len(np.unique(case.point_fields["family_id"][phase == 1])) == (3 if architecture == "triaxial" else 2)
    assert not bool(state.history.failed.any())
    assert not bool(state.history.damage.any())
    assert config.dt * substeps * 4 == pytest.approx(DURATION)
    assert case.description["initial_acoustic_CFL"] <= 0.15
    assert case.description["grid_cells_across_yarn_height"] == pytest.approx(1.5)
    assert case.description["nominal_grid_resolution_target_met"] is False
    np.testing.assert_allclose(np.asarray(state.volume).sum(), 0.010 * 0.002 * 0.010, rtol=1e-14)


def test_shear_and_tension_use_distinct_physical_loading_directions():
    warp, _, _ = loading_boundaries("warp-tension", DOMAIN / 96)
    weft, _, _ = loading_boundaries("weft-tension", DOMAIN / 96)
    shear, _, definition = loading_boundaries("in-plane-shear", DOMAIN / 96)
    np.testing.assert_allclose(warp[1].displacement, [0.00008, 0, 0])
    np.testing.assert_allclose(weft[1].displacement, [0, 0, 0.00008])
    np.testing.assert_allclose(shear[1].displacement, [0.00015, 0, 0])
    assert definition["nominal_final_strain_or_shear"] == pytest.approx(0.0375)
    assert shear[1].lower[2] == weft[1].lower[2]
    assert len(ARCHITECTURES) * len(LOADS) == 18


def test_hemisphere_is_local_unilateral_geometry_with_sliding_plane_support():
    boundaries, names, definition = loading_boundaries("hemispherical-compression", DOMAIN / 192)
    wall, tool = boundaries
    assert names == ("lower_plate", "hemisphere")
    assert type(wall).__name__ == "PlaneWall" and wall.friction == 0
    assert wall.axis == 1 and wall.side == "lower"
    assert type(tool).__name__ == "RigidHemisphereBoundary"
    lower, upper = specimen_bounds()
    assert 2 * tool.radius / (upper[0] - lower[0]) == pytest.approx(.25)
    assert tool.center[1] - tool.radius == pytest.approx(upper[1])
    np.testing.assert_allclose(tool.motion(DURATION)[0], np.asarray(tool.center) + [0, -.0001, 0])
    assert definition["nominal_contact_radius_cells"] >= 4
    assert loading_boundaries("hemispherical-compression", DOMAIN / 96)[2]["nominal_contact_radius_cells"] < 4
    for boundary in boundaries:
        boundary.validate(3, DOMAIN)
        assert not _support_inside(np.asarray([[.008, .008, .008]]), DOMAIN / 192, boundary).any()


def test_moving_grip_samples_stay_inside_at_every_translated_grid_phase():
    dx = DOMAIN / 96
    boundary = loading_boundaries("warp-tension", dx)[0][0]
    x = 0.003 + (np.arange(12) + 0.5) * dx / 2
    reference = np.column_stack((x, np.full(len(x), 0.008), np.full(len(x), 0.008)))
    inside = _support_inside(reference, dx, boundary)
    assert inside.any() and not inside.all()
    for time in np.linspace(0, LOADING_TIME, 25):
        displacement = np.asarray(boundary.motion(time)[0])
        positions = reference[inside] + displacement
        lower_node = np.floor(positions / dx - 0.5) * dx
        assert np.all(lower_node >= np.asarray(boundary.lower) + displacement - 1e-14)
        assert np.all(lower_node + 2 * dx <= np.asarray(boundary.upper) + displacement + 1e-14)


def test_invalid_lattice_spacing_cannot_silently_change_specimen_dimensions():
    with pytest.raises(ValueError, match="spacing must divide"):
        make_case("plain", "warp-tension", grid=50, ppc=1, frames=4)
    with pytest.raises(ValueError, match="unknown"):
        make_case("plain", "axial-compression")


def test_cli_resolution_defaults_follow_loading_and_preserve_explicit_values():
    for loading in LOADS:
        assert loading_resolution(loading) == ((192, 1) if loading == "hemispherical-compression" else (96, 2))
        assert loading_resolution(loading, 48, 1) == (48, 1)
    assert loading_resolution("hemispherical-compression", 96, 2) == (96, 2)
    assert loading_resolution("hemispherical-compression", grid=144) == (144, 1)
    assert loading_resolution("warp-tension", ppc=3) == (96, 3)
    with pytest.raises(ValueError, match="unknown"):
        loading_resolution("unknown")
