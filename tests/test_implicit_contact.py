import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mpm_jax.core import Config
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment
from mpm_jax.domain_mpm import step as domain_step
from mpm_jax.implicit_contact import (
    advect,
    discrete_pressure,
    line_interface,
    opening,
    solve,
    stored_energy,
    trace,
)


@pytest.mark.parametrize(
    "old,new,pressure_factor,derivative_factor",
    [
        (-0.4, -0.2, -0.3, 0.5),       # compression throughout
        (0.2, 0.4, 0.0, 0.0),           # separated throughout
        (-0.2, 0.3, -0.04, 0.08),      # opening through zero
        (0.3, -0.2, -0.04, 0.32),      # closing through zero
        (-0.2, 0.0, -0.1, 0.5),        # arrives at zero from compression
        (0.0, 0.2, 0.0, 0.0),          # leaves zero into separation
        (0.0, 0.0, 0.0, 0.5),          # selected generalized derivative
    ],
)
def test_discrete_pressure_is_secant_of_contact_potential_and_has_analytic_tangent(
    old, new, pressure_factor, derivative_factor
):
    stiffness = 7.3
    pressure, derivative = discrete_pressure(jnp.asarray(old), jnp.asarray(new), stiffness)
    expected_energy_change = 0.5 * stiffness * (
        min(new, 0.0) ** 2 - min(old, 0.0) ** 2
    )
    assert float(pressure) == pytest.approx(pressure_factor * stiffness, abs=1e-14)
    assert float(derivative) == pytest.approx(derivative_factor * stiffness, abs=1e-14)
    assert float(pressure) * (new - old) == pytest.approx(expected_energy_change, abs=1e-14)
    assert float(derivative) >= 0.0


@pytest.mark.parametrize(
    "initial_gap,relative_velocity,expected_sign",
    [
        (0.001, -0.30, 1),   # stiff closing contact
        (-0.001, 0.30, -1),  # opening from compression, including a zero crossing
    ],
)
def test_implicit_contact_closes_energy_gap_and_preserves_force_and_moment(
    initial_gap, relative_velocity, expected_sign
):
    config = Config(n_grid=16, dt=0.01, domain_size=1.0, boundary_cells=0,
                    material="elastic", dim=2, gravity=(0.0, 0.0))
    interface = line_interface((0.5, 0.3), (0.5, 0.7), (1.0, 0.0), config,
                               gap=initial_gap)
    size = (config.n_grid + 1) ** 2
    rng = np.random.default_rng(321)
    mass = jnp.asarray(0.02 + rng.random(2 * size))
    free = jnp.ones((2 * size, 2))
    support = trace(interface, mass, free, config)
    assert bool(support.valid)

    initial = jnp.zeros((2 * size, 2))
    initial = initial.at[:size, 0].set(0.5 * relative_velocity)
    initial = initial.at[size:, 0].set(-0.5 * relative_velocity)
    dt = 0.01
    stiffness = 5.0e5
    result = solve(interface, support, initial, initial, stiffness, dt, config.dx)
    assert bool(result.valid)
    assert int(result.iterations) > 0
    assert float(result.residual) <= 1e-10
    if expected_sign > 0:
        root_area = jnp.sqrt(interface.area)
        scaled_compliance = root_area[:, None] * support.compliance * root_area[None, :]
        explicit_number = stiffness * dt**2 * jnp.max(jnp.linalg.eigvalsh(scaled_compliance))
        assert float(explicit_number) > 4.0

    midpoint = 0.5 * (initial + result.velocity)
    moved = advect(interface, support, midpoint, dt)
    np.testing.assert_allclose(np.asarray(opening(moved)), np.asarray(result.gap), rtol=2e-11, atol=2e-13)

    kinetic_before = 0.5 * jnp.sum(mass[:, None] * initial**2)
    kinetic_after = 0.5 * jnp.sum(mass[:, None] * result.velocity**2)
    potential_change = stored_energy(moved, stiffness) - stored_energy(interface, stiffness)
    contact_work = dt * jnp.sum(result.force * midpoint)
    assert float(contact_work + potential_change) == pytest.approx(0.0, abs=2e-11)
    assert float(kinetic_after - kinetic_before + potential_change) == pytest.approx(0.0, abs=2e-10)

    plus_force = result.force[:size]
    minus_force = result.force[size:]
    np.testing.assert_allclose(np.asarray(plus_force.sum(axis=0) + minus_force.sum(axis=0)),
                               np.zeros(2), rtol=0.0, atol=2e-11)
    node = jnp.arange(size)
    coordinates = config.dx * jnp.stack((node // (config.n_grid + 1),
                                         node % (config.n_grid + 1)), axis=-1)
    all_force = plus_force + minus_force
    torque = jnp.sum(coordinates[:, 0] * all_force[:, 1]
                     - coordinates[:, 1] * all_force[:, 0])
    assert float(torque) == pytest.approx(0.0, abs=2e-11)


def test_prescribed_normal_velocity_reaction_work_closes_contact_energy():
    config = Config(n_grid=16, dt=0.01, domain_size=1.0, boundary_cells=0,
                    material="elastic", dim=2, gravity=(0.0, 0.0))
    interface = line_interface((0.5, 0.3), (0.5, 0.7), (1.0, 0.0), config,
                               gap=-0.001)
    size = (config.n_grid + 1) ** 2
    mass = jnp.linspace(0.3, 1.7, 2 * size)
    free = jnp.ones((2 * size, 2))
    initial = jnp.zeros((2 * size, 2))
    support_full = trace(interface, mass, free, config)
    constrained_node = int(support_full.plus_indices[0, 4])
    free = free.at[constrained_node, 0].set(0.0)
    prescribed = initial.at[constrained_node, 0].set(0.1)
    support = trace(interface, mass, free, config)
    assert bool(support.valid)

    dt = 0.01
    result = solve(interface, support, initial, prescribed, 2.0e4, dt, config.dx)
    assert bool(result.valid)
    assert float(result.velocity[constrained_node, 0]) == pytest.approx(0.1)
    moved = advect(interface, support, 0.5 * (initial + result.velocity), dt)
    np.testing.assert_allclose(np.asarray(opening(moved)), np.asarray(result.gap), rtol=2e-11, atol=2e-13)

    reaction_impulse = mass[:, None] * (result.velocity - initial) - dt * result.force
    midpoint = 0.5 * (initial + result.velocity)
    boundary_work = jnp.sum(reaction_impulse * midpoint)
    kinetic_change = 0.5 * jnp.sum(mass[:, None] * (result.velocity**2 - initial**2))
    potential_change = stored_energy(moved, 2.0e4) - stored_energy(interface, 2.0e4)
    assert float(kinetic_change + potential_change - boundary_work) == pytest.approx(0.0, abs=2e-10)
    assert float(jnp.max(jnp.abs(reaction_impulse[free == 1.0]))) < 2e-11


def test_domain_midpoint_step_couples_two_fragment_fields_through_contact():
    from mpm_jax import neo_hookean

    config = Config(n_grid=16, dt=1.0e-3, domain_size=1.0, boundary_cells=0,
                    material="neo-hookean", dim=2, gravity=(0.0, 0.0))
    material = neo_hookean(1.0e3, 0.2)
    reference = polygon(jnp.asarray([[-0.1, -0.2], [0.1, -0.2],
                                     [0.1, 0.2], [-0.1, 0.2]]))
    centers = jnp.asarray([[0.6, 0.5], [0.4, 0.5]])
    velocities = jnp.asarray([[-0.2, 0.0], [0.2, 0.0]])
    state = jax.vmap(lambda x, v: fragment(reference, 10.0, x=x, v=v))(centers, velocities)
    fields = jnp.asarray([0, 1])
    contact = line_interface((0.5, 0.3), (0.5, 0.7), (1.0, 0.0), config)

    following, new_contact, work, contact_work = domain_step(
        state, material, config, jnp.asarray(0.0), fields=fields, n_fields=2,
        tile_radius=3, contact=contact, contact_stiffness=2.0e4, dt=config.dt,
    )

    assert bool(jnp.all(following.valid))
    assert bool(new_contact.valid)
    assert int(contact_work.iterations) > 0
    assert float(jnp.max(jnp.abs(contact_work.power_error))) < 2e-12
    assert float(jnp.max(jnp.abs(contact_work.net_moment))) < 2e-11
    assert abs(float(work.ledger_error)) < 2e-10
    assert float(jnp.max(opening(new_contact))) < 0.0
