import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, advance, block, elastic, fluid, step, validate_state
from mpm_jax.core import grid_to_particle, grid_update, kirchhoff_stress, particle_to_grid, stencil


def test_quadratic_partition_and_moments():
    config = Config(n_grid=32)
    x = jnp.asarray(np.random.default_rng(3).uniform(0.1, 0.9, (40, 2)))
    s = stencil(x, config)
    np.testing.assert_allclose(s.weights.sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(jnp.einsum("pk,pki->pi", s.weights, s.dpos), 0, atol=1e-14)
    moment = jnp.einsum("pk,pki,pkj->pij", s.weights, s.dpos, s.dpos)
    expected = np.broadcast_to(config.dx**2 / 4 * np.eye(2), moment.shape)
    np.testing.assert_allclose(moment, expected, atol=1e-14)


def test_p2g_conserves_mass_linear_and_apic_angular_momentum():
    config = Config(n_grid=24, gravity=(0.0, 0.0), boundary_cells=0)
    state = block(config, dtype=jnp.float64)
    rng = np.random.default_rng(1)
    state = state._replace(
        v=jnp.asarray(rng.normal(size=state.v.shape)),
        C=jnp.asarray(rng.normal(size=state.C.shape)),
        F=state.F.at[:, 0, 0].set(1.02),
    )
    mass, momentum = particle_to_grid(state, elastic(), config)
    np.testing.assert_allclose(mass.sum(), state.mass.sum(), rtol=1e-14)
    np.testing.assert_allclose(
        momentum.sum(axis=0), (state.mass[:, None] * state.v).sum(axis=0), atol=1e-12
    )
    node = np.arange((config.n_grid + 1) ** 2)
    x = np.stack((node // (config.n_grid + 1), node % (config.n_grid + 1)), axis=-1) * config.dx
    grid_angular = np.sum(x[:, 0] * momentum[:, 1] - x[:, 1] * momentum[:, 0])
    particle_angular = jnp.sum(
        state.mass * (state.x[:, 0] * state.v[:, 1] - state.x[:, 1] * state.v[:, 0])
    )
    affine_angular = (
        config.dx**2 / 4 * jnp.sum(state.mass * (state.C[:, 1, 0] - state.C[:, 0, 1]))
    )
    np.testing.assert_allclose(grid_angular, particle_angular + affine_angular, atol=1e-12)


def test_apic_reproduces_affine_velocity():
    config = Config(n_grid=24, gravity=(0.0, 0.0), boundary_cells=0)
    state = block(config, dtype=jnp.float64)
    A = jnp.array([[0.1, -0.3], [0.4, 0.2]])
    state = state._replace(
        v=state.x @ A.T + jnp.array([0.2, -0.1]), C=jnp.broadcast_to(A, state.C.shape)
    )
    support = stencil(state.x, config)
    gm, gp = particle_to_grid(state, elastic(), config, support)
    v, C = grid_to_particle(grid_update(gm, gp, config), support, config)
    np.testing.assert_allclose(v, state.v, atol=1e-13)
    np.testing.assert_allclose(C, state.C, atol=1e-12)


def test_rigid_rotation_is_stress_free():
    angle = 0.7
    F = jnp.array([[[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]])
    np.testing.assert_allclose(kirchhoff_stress(F, elastic(), "elastic"), 0, atol=1e-10)


def test_elastic_stress_is_energy_derivative():
    material = elastic()
    F = jnp.array([[1.04, 0.03], [-0.01, 0.96]])

    def energy(F):
        logJ = jnp.log(jnp.linalg.det(F))
        return (
            material.mu / 2 * (jnp.sum(F**2) - 2)
            - material.mu * logJ
            + material.lam / 2 * logJ**2
        )

    expected = jax.grad(energy)(F) @ F.T
    np.testing.assert_allclose(
        kirchhoff_stress(F[None], material, "elastic")[0], expected, rtol=1e-12
    )


def test_fluid_hydrostatic_stress():
    F = jnp.array([[[0.9, 0.2], [0.0, 1.0]]])
    tau = kirchhoff_stress(F, fluid(1000), "fluid")[0]
    np.testing.assert_allclose(tau, -90 * np.eye(2), atol=1e-12)


def test_free_fall_matches_semi_implicit_solution():
    config = Config(n_grid=24, dt=1e-4, boundary_cells=0)
    initial = block(config, velocity=(0.2, 0.1), dtype=jnp.float64)
    n = 100
    final = advance(initial, elastic(), config, n)
    validate_state(final, config)
    g = np.asarray(config.gravity)
    expected_x = initial.x + n * config.dt * initial.v + (n * (n + 1) / 2) * config.dt**2 * g
    np.testing.assert_allclose(final.x, expected_x, atol=1e-12)
    np.testing.assert_allclose(final.v, initial.v + n * config.dt * g, atol=1e-11)
    np.testing.assert_allclose(final.F, initial.F, atol=1e-11)


def test_unloaded_block_remains_at_rest():
    config = Config(n_grid=16, gravity=(0, 0))
    initial = block(config, dtype=jnp.float64)
    final = advance(initial, elastic(), config, 50)
    np.testing.assert_allclose(final.x, initial.x, atol=1e-14)
    np.testing.assert_allclose(final.v, 0, atol=1e-14)


def test_grid_walls_are_unilateral_and_free_slip():
    config = Config(n_grid=16, gravity=(0, 0))
    size = 17**2
    mass = jnp.ones(size)
    momentum = jnp.broadcast_to(jnp.array([-2.0, 1.0]), (size, 2))
    velocity = grid_update(mass, momentum, config).reshape(17, 17, 2)
    np.testing.assert_allclose(velocity[0, 8], [0, 1])
    np.testing.assert_allclose(velocity[16, 8], [-2, 1])
    np.testing.assert_allclose(velocity[8, 16], [-2, 0])


def test_jit_matches_eager_on_deformed_block():
    config = Config(n_grid=16)
    initial = block(config, dtype=jnp.float64)
    initial = initial._replace(F=initial.F.at[:, 0, 0].set(1.02))
    compiled = step(initial, elastic(), config)
    with jax.disable_jit():
        eager = step(initial, elastic(), config)
    for actual, expected in zip(compiled[:-1], eager[:-1]):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
    assert bool(compiled.valid) == bool(eager.valid)


@pytest.mark.parametrize("kind", ["elastic", "fluid"])
def test_long_run_survives_floor_impact(kind):
    config = Config(n_grid=24, dt=1e-3, material=kind)
    state = block(config, particles_per_cell=1, dtype=jnp.float64)
    initial_mass = float(state.mass.sum())
    initial_y = float(jnp.mean(state.x[:, 1]))
    material = elastic() if kind == "elastic" else fluid()
    final = advance(state, material, config, 600)
    validate_state(final, config)
    np.testing.assert_allclose(final.mass.sum(), initial_mass, atol=1e-12)
    assert float(jnp.mean(final.x[:, 1])) < initial_y
    assert float(jnp.min(final.x[:, 1])) > config.dx
    assert float(jnp.min(jnp.linalg.det(final.F))) > 0.1


def test_material_gradient_matches_finite_difference():
    config = Config(n_grid=24, dt=2e-4, gravity=(0, 0), boundary_cells=0)
    state = block(config, lower=(0.4, 0.4), upper=(0.6, 0.6), dtype=jnp.float64)
    state = state._replace(F=state.F.at[:, 0, 0].set(1.04).at[:, 1, 1].set(0.98))

    def response(young):
        result = advance(state, elastic(young), config, 60)
        return jnp.mean((result.x[:, 0] - 0.5) ** 2)

    actual = jax.jit(jax.grad(response))(20000.0)
    expected = (response(20002.0) - response(19998.0)) / 4
    assert abs(float(actual)) > 1e-10
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-13)


def test_outside_particle_is_flagged_instead_of_wrapped():
    config = Config(n_grid=16)
    state = block(config)
    state = state._replace(x=state.x.at[0, 0].set(-0.01))
    final = step(state, elastic(), config)
    assert not bool(final.valid)
    with pytest.raises(FloatingPointError):
        validate_state(final, config)


def test_inversion_is_flagged_instead_of_silently_repaired():
    config = Config(n_grid=16, material="fluid")
    state = block(config)
    state = state._replace(F=state.F.at[0, 0, 0].set(-1))
    final = step(state, fluid(), config)
    assert not bool(final.valid)
    with pytest.raises(FloatingPointError):
        validate_state(final, config)


@pytest.mark.parametrize(
    "kwargs",
    [{"dt": 0}, {"dt": float("nan")}, {"n_grid": 4}, {"material": "snow"}, {"boundary_cells": -1}],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_initial_volume_is_geometric_area():
    config = Config(n_grid=24)
    state = block(config, (0.31, 0.42), (0.67, 0.78), density=1234, dtype=jnp.float64)
    np.testing.assert_allclose(state.volume.sum(), 0.36**2, atol=1e-14)
    np.testing.assert_allclose(state.mass.sum(), 1234 * 0.36**2, atol=1e-12)
