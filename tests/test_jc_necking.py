import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.jc_necking import (
    ARC_RADIUS,
    CENTER,
    DURATION,
    HALF_LENGTH,
    RADIUS,
    RAMP_TIME,
    SPEED,
    TRANSITION_LENGTH,
    WAIST_RADIUS,
    make_case,
    reference_lattice,
    rod_radius,
    verification_steel,
)
from mpm_jax.initialization import validate_state
from mpm_jax.materials import initial_history
from mpm_jax.materials.johnson_cook import update_johnson_cook

jax.config.update("jax_enable_x64", True)


def test_circular_arc_waist_is_tangent_and_matches_source_dimensions():
    assert ARC_RADIUS == pytest.approx(0.002825)
    np.testing.assert_allclose(
        rod_radius([0, TRANSITION_LENGTH, HALF_LENGTH]), [WAIST_RADIUS, RADIUS, RADIUS]
    )
    offsets = np.linspace(0, HALF_LENGTH, 101)
    np.testing.assert_array_equal(rod_radius(offsets), rod_radius(-offsets))
    increment = 1e-9
    for location in (0.0, TRANSITION_LENGTH / 2, TRANSITION_LENGTH):
        left = (rod_radius(location) - rod_radius(location - increment)) / increment
        right = (rod_radius(location + increment) - rod_radius(location)) / increment
        assert abs(left - right) < 2e-6


@pytest.mark.parametrize("grid,ppc", [(50, 1), (80, 3), (100, 2)])
def test_reference_lattice_is_uniform_symmetric_and_cropped_without_warping(grid, ppc):
    position, spacing, transverse_count = reference_lattice(grid, ppc)
    local = position - CENTER
    assert transverse_count % 2 == 0
    assert spacing**3 > 0
    assert np.all(np.sum(local[:, 1:] ** 2, axis=1) < rod_radius(local[:, 0]) ** 2)
    np.testing.assert_allclose(local.mean(axis=0), 0, atol=1e-16)
    for axis in range(3):
        unique = np.unique(local[:, axis])
        np.testing.assert_allclose(np.diff(unique), spacing, rtol=1e-13, atol=1e-18)
        np.testing.assert_allclose(unique, -unique[::-1], atol=1e-18)
    if grid == 50:
        assert spacing == pytest.approx(0.0002)
        assert len(position) == 1648
    elif grid == 80:
        assert transverse_count == 40
        assert spacing == pytest.approx(0.00004)
        assert len(position) > 100_000
    else:
        assert spacing == pytest.approx(0.00005)
        assert len(position) == 102_320


def test_coarse_case_initial_mass_sections_and_all_component_grips():
    case = make_case(grid=50, ppc=1, frames=4, end_time=80e-6)
    state, config = case.states[0], case.configs[0]
    validate_state(state, config)
    np.testing.assert_array_equal(state.F, np.broadcast_to(np.eye(3), state.F.shape))
    np.testing.assert_allclose(state.volume, (0.0002) ** 3, rtol=1e-15)
    np.testing.assert_allclose(state.mass, 7830 * (0.0002) ** 3, rtol=1e-15)
    assert not np.any(state.history.failed)
    assert config.failure_mode == "ballistic"
    assert case.description["acoustic_CFL"] <= 0.15
    initial = case.observer(case.states, 0.0)
    assert initial["central_geometric_rms_radius_ratio"] == pytest.approx(1.0)
    assert initial["minimum_intact_gauge_rms_radius_ratio"] == pytest.approx(1.0)
    assert initial["minimum_gauge_load_bearing_area_ratio"] == pytest.approx(1.0)
    assert initial["grip_inner_particles"] > 0
    assert initial["central_material_section_force_N"] == 0.0
    expanded = state._replace(x=state.x.at[:, 1:].set(CENTER + 1.1 * (state.x[:, 1:] - CENTER)))
    geometric = case.observer((expanded,), 0.0)
    assert geometric["minimum_intact_gauge_rms_radius_ratio"] == pytest.approx(1.1)
    for side, boundary in zip((-1, 1), config.boundaries):
        displacement, velocity = boundary.motion(DURATION)
        np.testing.assert_allclose(displacement, (side * 0.0014, 0, 0), atol=1e-18)
        np.testing.assert_allclose(velocity, (side * SPEED, 0, 0), atol=1e-14)
        point = jnp.array([[CENTER + side * 0.003 + displacement[0], CENTER, CENTER]])
        constrained = boundary.apply(jnp.array([[1.0, 2.0, 3.0]]), point, DURATION)
        np.testing.assert_allclose(constrained, [[side * SPEED, 0, 0]], atol=1e-14)
        np.testing.assert_allclose(boundary.motion(RAMP_TIME / 2)[1], (side * SPEED / 2, 0, 0))


def test_verification_steel_uses_pr_constants_without_changing_failure_convention():
    material = verification_steel()
    assert material.A == 792e6 and material.B == 510e6
    assert material.n == 0.26 and material.C == 0.014 and material.m == 1.03
    assert material.heat_fraction == 1.0 and material.heat_capacity == 477
    assert material.density == 7830 and material.reference_rate == 1
    np.testing.assert_allclose(material.failure, (0.05, 3.44, -2.12, 0.002, 0.61))
    d1, d2, d3, _, _ = np.asarray(material.failure)
    assert d1 + d2 * np.exp(d3 / 3) < d1 + d2 * np.exp(-d3 / 3)


def test_steel_material_point_implicit_return_heat_and_plastic_rate_failure():
    material = verification_steel()
    dt = 1e-6
    increment = np.diag([0.02, 0.0, 0.0])
    deviator = increment - np.trace(increment) * np.eye(3) / 3
    trial = 2 * material.mu * deviator
    trial_q = np.sqrt(1.5 * np.sum(trial**2))

    def flow(dep):
        return (792e6 + 510e6 * dep**0.26) * (1 + 0.014 * np.log(max(dep / dt, 1)))

    low, high = 0.0, trial_q / (3 * material.mu)
    for _ in range(100):
        midpoint = (low + high) / 2
        if trial_q - 3 * material.mu * midpoint > flow(midpoint):
            low = midpoint
        else:
            high = midpoint
    dep = (low + high) / 2
    q = flow(dep)
    old = initial_history(1, jnp.float64, material, material.density)
    actual = update_johnson_cook(old, jnp.asarray(increment[None]), jnp.ones(1), dt, material)
    np.testing.assert_allclose(actual.plastic_strain, dep, rtol=1e-12)
    np.testing.assert_allclose(actual.temperature, 298 + q * dep / (7830 * 477), rtol=1e-13)
    fracture = (0.05 + 3.44) * (1 + 0.002 * np.log(max(dep / dt, 1)))
    np.testing.assert_allclose(actual.damage, dep / fracture, rtol=1e-12)
    assert bool(actual.valid[0])
    # The PR's first explicit step has zero old rate and zero hardening tangent.
    pr_dep = (trial_q - 792e6) / (3 * material.mu)
    pr_failure_rate = np.sqrt(2 / 3 * np.sum(deviator**2)) / dt
    assert pr_dep > dep
    assert q > 792e6
    assert pr_failure_rate > dep / dt


def test_necking_rejects_unsupported_duration_and_grid():
    with pytest.raises(ValueError, match="grid"):
        reference_lattice(20, 3)
    with pytest.raises(ValueError, match="150"):
        make_case(grid=50, ppc=1, end_time=151e-6)
    with pytest.raises(ValueError, match="cfl"):
        make_case(grid=50, ppc=1, cfl=0.31)
