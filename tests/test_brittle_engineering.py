import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.brittle_engineering import LoadingBox, _damage_observables, setup
from mpm_jax import step
from mpm_jax.contact_laws import boundary_reaction

jax.config.update("jax_enable_x64", True)


def test_bending_fixtures_allow_axial_slide_and_reach_prescribed_displacement():
    case = setup("concrete-bending", grid=16, ppc=1, frames=4)
    left, right, lateral, loading = case.configs[0].boundaries
    velocity = jnp.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(left.apply(velocity, jnp.array([[0.075, 0.15, 0.2]]), 0), [[0, 0, 3]])
    np.testing.assert_allclose(right.apply(velocity, jnp.array([[0.325, 0.15, 0.2]]), 0), [[1, 0, 3]])
    np.testing.assert_allclose(lateral.apply(velocity, jnp.array([[0.075, 0.15, 0.2]]), 0), [[1, 2, 0]])
    displacement, speed = loading.motion(case.duration)
    np.testing.assert_allclose(displacement, [0, -0.002, 0], atol=1e-15)
    np.testing.assert_allclose(loading.motion(0.0)[1], 0.0)
    assert speed[1] < 0
    material = case.materials[0]
    np.testing.assert_allclose(case.states[0].mass.sum(), material.density * 0.3 * 0.06**2)


@pytest.mark.parametrize("scene", ["concrete-bending", "concrete-bending-large", "ceramic-indentation"])
def test_boundary_reactions_balance_specimen_momentum(scene):
    case = setup(scene, grid=16, ppc=1, frames=4)
    state, material, config = case.states[0], case.materials[0], case.configs[0]
    state = state._replace(time=jnp.asarray(0.2 * case.duration))
    forces = jnp.stack(
        [boundary_reaction(state, material, config, index) for index in range(len(config.boundaries))]
    )
    final = step(state, material, config)
    momentum_change = jnp.sum(state.mass[:, None] * (final.v - state.v), axis=0)
    np.testing.assert_allclose(momentum_change, -config.dt * forces.sum(0), rtol=2e-12, atol=1e-12)
    assert bool(final.valid)
    assert float(forces[-1, 1]) > 0.0


def test_damage_region_does_not_imply_material_failure_or_separation():
    case = setup("ceramic-indentation", grid=16, ppc=1, frames=4)
    state = case.states[0]
    count = len(state.x) // 3
    damage = jnp.zeros_like(state.history.damage).at[:count].set(1.0)
    damaged = state._replace(history=state.history._replace(damage=damage))
    observed = _damage_observables(damaged)
    expected = float(state.volume[:count].sum() / state.volume.sum())
    np.testing.assert_allclose(observed["damaged_reference_volume_fraction"], expected)
    assert observed["material_failed_reference_volume_fraction"] == 0.0
    assert case.configs[0].failure_mode == "retain"
    for axis, name in enumerate("xyz"):
        assert observed[f"damaged_region_min_{name}_m"] == float(state.x[:count, axis].min())
        assert observed[f"damaged_region_max_{name}_m"] == float(state.x[:count, axis].max())


def test_rht_uses_same_concrete_geometry_and_conservative_time_step():
    case = setup("concrete-bending", grid=16, ppc=1, model="rht", frames=4)
    state, material, config = case.states[0], case.materials[0], case.configs[0]
    np.testing.assert_allclose(state.mass.sum(), material.density * 0.3 * 0.06**2)
    assert case.description["acoustic_CFL"] <= 0.15
    assert len(config.boundaries) == 4
    assert bool(step(state, material, config).valid)


def test_loading_box_velocity_is_displacement_derivative_and_stops_smoothly():
    boundary = LoadingBox((.1, .2, .1), (.2, .25, .2), (0, -40 / 3, 0),
                          components=(False, True, False), ramp_time=.0003, loading_duration=.0012)
    boundary.validate(3, .4)
    for time in (.00015, .0006, .00105):
        derivative = (boundary.motion(time + 1e-9)[0] - boundary.motion(time - 1e-9)[0]) / 2e-9
        np.testing.assert_allclose(derivative, boundary.motion(time)[1], rtol=2e-9, atol=1e-9)
    for time in (0., .0012, .0015):
        np.testing.assert_array_equal(boundary.motion(time)[1], np.zeros(3))
    for time in (.0012, .0015):
        np.testing.assert_allclose(boundary.motion(time)[0], [0, -.012, 0], atol=1e-15)
    for time in (.0003, .0009, .0012):
        np.testing.assert_allclose(boundary.motion(time - 1e-10)[1], boundary.motion(time + 1e-10)[1], atol=1e-10)


def test_loading_box_holds_the_moved_patch_without_clamping_free_components():
    boundary = LoadingBox((.1, .2, .1), (.2, .25, .2), (0, -40 / 3, 0),
                          components=(False, True, False), ramp_time=.0003, loading_duration=.0012)
    velocity = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    coordinates = jnp.array([[.15, .195, .15], [.15, .245, .15]], dtype=jnp.float32)
    observed = jax.jit(boundary.apply)(velocity, coordinates, jnp.asarray(.0015, dtype=jnp.float64))
    np.testing.assert_array_equal(observed, [[1, 0, 3], [4, 5, 6]])
    assert observed.dtype == velocity.dtype


def test_large_bending_keeps_geometry_and_uses_explicit_strain_floor_profile():
    original = setup("concrete-bending", grid=16, ppc=1, frames=4)
    large = setup("concrete-bending-large", grid=16, ppc=1, frames=20)
    np.testing.assert_array_equal(large.states[0].x, original.states[0].x)
    np.testing.assert_array_equal(large.states[0].mass, original.states[0].mass)
    assert original.materials[0].legacy_v3 is True
    assert large.materials[0].legacy_v3 is False
    assert large.description["material_profile"] == "strain-floor"
    assert large.description["hold_duration_s"] == .0003
    np.testing.assert_allclose(large.duration, .0015, rtol=0, atol=1e-18)
    assert large.configs[0].failure_mode == "retain"
    loading = large.configs[0].boundaries[-1]
    np.testing.assert_allclose(loading.motion(large.duration)[0], [0, -.012, 0], atol=1e-15)
    np.testing.assert_array_equal(loading.motion(large.duration)[1], np.zeros(3))
