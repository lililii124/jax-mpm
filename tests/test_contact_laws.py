from dataclasses import asdict
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, block, elastic, step
from mpm_jax.contact_laws import (
    RigidSphereBoundary,
    boundary_reaction,
    hertz_contact,
    jkr_contact,
    jkr_detachment,
    jkr_force,
    jkr_history,
    jkr_indentation,
    reduced_modulus,
    spherical_surface_load,
)

jax.config.update("jax_enable_x64", True)


def test_hertz_scaling_stiffness_and_reduced_modulus():
    modulus = reduced_modulus(2e6, 0.3, 4e6, 0.2)
    np.testing.assert_allclose(modulus, 1 / (0.91 / 2e6 + 0.96 / 4e6))
    radius, delta = 0.2, 0.001
    contact = jax.jit(hertz_contact)(delta, radius, modulus)
    np.testing.assert_allclose(contact.radius, np.sqrt(radius * delta))
    np.testing.assert_allclose(contact.force, 4 / 3 * modulus * np.sqrt(radius) * delta**1.5)
    tangent = jax.grad(lambda d: hertz_contact(d, radius, modulus).force)(delta)
    np.testing.assert_allclose(tangent, 2 * modulus * np.sqrt(radius * delta))
    separated = hertz_contact(jnp.array([-0.1, 0.0]), radius, modulus)
    np.testing.assert_array_equal(separated.force, 0.0)
    np.testing.assert_allclose(
        jax.grad(lambda d: hertz_contact(d, radius, modulus).force)(-0.1), 0.0
    )


@pytest.mark.parametrize(
    "radius,modulus,work", [(0.3, 2e4, 3.0), (1e-3, 1e6, 0.05), (1.0, 0.75, 1 / np.pi)]
)
def test_jkr_radius_against_independent_polynomial_roots(radius, modulus, work):
    critical, detach = jkr_detachment(radius, modulus, work)
    delta = np.linspace(float(detach) * 0.98, float(critical**2 / radius) * 20, 17)
    expected = []
    for d in delta:
        roots = np.roots([1 / radius, 0, 0, -np.sqrt(2 * np.pi * work / modulus), -d])
        positive = roots.real[(np.abs(roots.imag) < 1e-10) & (roots.real > 0)]
        expected.append(positive.max() ** 2)
    result = jax.jit(jax.vmap(lambda d: jkr_contact(d, radius, modulus, work)))(jnp.asarray(delta))
    np.testing.assert_allclose(result.radius, expected, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(
        jkr_indentation(result.radius, radius, modulus, work), delta, rtol=2e-12, atol=1e-15
    )


def test_jkr_load_and_displacement_instabilities_are_distinct():
    radius, modulus, work = 0.3, 2e4, 3.0
    load_critical = (9 * np.pi * work * radius**2 / (8 * modulus)) ** (1 / 3)
    np.testing.assert_allclose(
        jkr_force(load_critical, radius, modulus, work), -1.5 * np.pi * work * radius
    )
    a, delta = jkr_detachment(radius, modulus, work)
    np.testing.assert_allclose(jkr_force(a, radius, modulus, work), -5 / 6 * np.pi * work * radius)
    assert not bool(jkr_contact(delta, radius, modulus, work).active)
    zero_load_radius = (9 * np.pi * work * radius**2 / (2 * modulus)) ** (1 / 3)
    zero_load_delta = jkr_indentation(zero_load_radius, radius, modulus, work)
    np.testing.assert_allclose(
        jkr_contact(zero_load_delta, radius, modulus, work).force, 0.0, atol=5e-13
    )


def test_jkr_history_separation_reattachment_and_zero_adhesion():
    radius, modulus, work = 0.3, 2e4, 3.0
    _, detach = jkr_detachment(radius, modulus, work)
    cycle = jnp.array(
        [1.1 * detach, 0.5 * detach, 0.0, 0.02, 0.5 * detach, 1.1 * detach, 0.5 * detach, 0.0]
    )
    result = jkr_history(cycle, radius, modulus, work)
    np.testing.assert_array_equal(
        result.active, [False, False, True, True, True, False, False, True]
    )
    hertz = hertz_contact(cycle, radius, modulus)
    no_adhesion = jax.vmap(lambda d: jkr_contact(d, radius, modulus, 0.0))(cycle)
    for a, b in zip(hertz, no_adhesion):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(
        jax.grad(lambda d: jkr_contact(d, radius, modulus, work).force)(2 * detach), 0.0
    )


def test_jkr_implicit_derivatives_inside_stable_branch():
    parameters = jnp.array([0.005, 0.3, 2e4, 3.0])

    def radius(p):
        return jkr_contact(*p).radius

    derivative = jax.jacfwd(radius)(parameters)
    difference = []
    for i, value in enumerate(parameters):
        change = float(abs(value) * 1e-5)
        plus, minus = np.asarray(parameters).copy(), np.asarray(parameters).copy()
        plus[i] += change
        minus[i] -= change
        difference.append((float(radius(plus)) - float(radius(minus))) / (2 * change))
    np.testing.assert_allclose(derivative, difference, rtol=2e-8)


def test_surface_load_conserves_resultant_and_has_tensile_ring():
    axis = np.linspace(-0.2, 0.2, 81)
    x, z = np.meshgrid(axis, axis, indexing="ij")
    points = jnp.asarray(np.column_stack((x.ravel(), np.zeros(x.size), z.ravel())))
    radius, modulus, work = 0.3, 2e4, 3.0
    a = (9 * np.pi * work * radius**2 / (2 * modulus)) ** (1 / 3)
    contact = jkr_contact(jkr_indentation(a, radius, modulus, work), radius, modulus, work)
    load = spherical_surface_load(
        points, 0.005**2, jnp.ones(len(points), bool), jnp.zeros(3), (0, 1, 0), contact, 0.005
    )
    np.testing.assert_allclose(load.forces.sum(0), [0, -float(contact.force), 0], atol=3e-12)
    np.testing.assert_allclose(jnp.cross(points, load.forces).sum(0), 0, atol=1e-12)
    assert bool(load.resolved)
    assert load.forces[:, 1].min() < 0 < load.forces[:, 1].max()
    tiny = hertz_contact(1e-10, radius, modulus)
    unresolved = spherical_surface_load(
        points, 0.005**2, jnp.ones(len(points), bool), jnp.zeros(3), (0, 1, 0), tiny, 0.005
    )
    assert not bool(unresolved.resolved)
    np.testing.assert_allclose(unresolved.reaction[1], tiny.force)


def test_rigid_sphere_normal_constraint_motion_and_serialization():
    sphere = RigidSphereBoundary((0.5, 0.7, 0.5), 0.3, (0.0, -0.05, 0.0), 1.0)
    sphere.validate(3, 1.0)
    json.dumps(asdict(sphere))
    center, velocity = sphere.motion(0.25)
    derivative = jax.jacfwd(lambda t: sphere.motion(t)[0])(0.25)
    np.testing.assert_allclose(derivative, velocity, atol=1e-15)
    positions = jnp.array(
        [center + jnp.array([0.0, -0.29, 0.0]), center + jnp.array([0.0, -0.31, 0.0])]
    )
    incoming = jnp.array([[2.0, 0.0, 3.0], [2.0, 0.0, 3.0]])
    result = sphere.apply(incoming, positions, 0.25)
    np.testing.assert_allclose(result[0], [2.0, float(velocity[1]), 3.0])
    np.testing.assert_allclose(result[1], incoming[1])
    separating = incoming.at[:, 1].set(-1.0)
    np.testing.assert_allclose(sphere.apply(separating, positions, 0.25), separating)
    np.testing.assert_allclose(sphere.motion(2.0)[0], sphere.center)
    np.testing.assert_allclose(sphere.motion(2.0)[1], 0.0)


def test_rigid_sphere_reaction_matches_particle_momentum_change():
    sphere = RigidSphereBoundary((0.5, 0.7, 0.5), 0.3, (0.0, -0.05, 0.0), 1.0)
    cfg = Config(
        dim=3, n_grid=12, dt=1e-4, gravity=(0, 0, 0), boundary_cells=0, boundaries=(sphere,)
    )
    state = block(
        cfg,
        (0.3, 0.3, 0.3),
        (0.7, 0.5, 0.7),
        particles_per_cell=1,
        velocity=(0, 1, 0),
        dtype=jnp.float64,
    )
    force = boundary_reaction(state, elastic(), cfg)
    final = step(state, elastic(), cfg)
    change = jnp.sum(state.mass[:, None] * (final.v - state.v), axis=0)
    np.testing.assert_allclose(change, -cfg.dt * force, atol=5e-14)
    assert force[1] > 0
