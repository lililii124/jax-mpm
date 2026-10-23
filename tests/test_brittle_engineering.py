import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.brittle_engineering import _damage_observables, setup
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


@pytest.mark.parametrize("scene", ["concrete-bending", "ceramic-indentation"])
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
