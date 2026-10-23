from dataclasses import asdict
import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, block, elastic, fixed_box, step
from mpm_jax.adhesion import (
    DugdaleLaw,
    SpherePath,
    penalty_step_number,
    sphere_interaction,
)
from mpm_jax.contact_laws import boundary_reaction

jax.config.update("jax_enable_x64", True)


def test_dugdale_traction_potential_and_separation_work():
    law = DugdaleLaw(0.6, 0.02, 20000.0)
    gaps = jnp.array([-0.01, -0.001, 0.001, 0.019, 0.021])
    expected = jnp.array([200.0, 20.0, -30.0, -30.0, 0.0])
    np.testing.assert_allclose(law.traction(gaps), expected)
    np.testing.assert_allclose(-jax.vmap(jax.grad(law.potential))(gaps), expected)
    np.testing.assert_allclose(law.potential(0.02) - law.potential(0.0), 0.6)
    midpoint_gaps = (jnp.arange(1000) + 0.5) * 0.02 / 1000
    np.testing.assert_allclose(-law.traction(midpoint_gaps).sum() * 0.02 / 1000, 0.6)
    assert float(law.potential(0.03)) == 0.0


def test_sphere_force_is_full_discrete_energy_gradient_and_balances_moment():
    law = DugdaleLaw(0.8, 0.03, 1e5)
    points = jnp.array([[0.0, 0.79, 0.02], [0.11, 0.79, 0.03], [-0.12, 0.86, 0.01]])
    center = jnp.array([0.0, 1.0, 0.0])
    areas = jnp.array([0.001, 0.002, 0.0015])
    result = sphere_interaction(points, areas, center, 0.2, law)
    particle_gradient = jax.grad(
        lambda x: sphere_interaction(x, areas, center, 0.2, law).potential
    )(points)
    center_gradient = jax.grad(
        lambda c: sphere_interaction(points, areas, c, 0.2, law).potential
    )(center)
    np.testing.assert_allclose(result.forces, -particle_gradient, atol=1e-13)
    np.testing.assert_allclose(result.reaction, -center_gradient, atol=1e-13)
    np.testing.assert_allclose(result.reaction + result.forces.sum(0), 0.0, atol=1e-14)
    np.testing.assert_allclose(result.reaction_moment, 0.0, atol=1e-14)
    total_moment = jnp.cross(points, result.forces).sum(0) + jnp.cross(
        center, result.reaction
    )
    np.testing.assert_allclose(total_moment, 0.0, atol=1e-14)


def test_moving_sphere_power_matches_potential_rate():
    law = DugdaleLaw(0.8, 0.03, 1e5)
    points = jnp.array([[0.03, 0.79, 0.02], [-0.07, 0.81, 0.01]])
    center = jnp.array([0.0, 1.0, 0.0])
    point_velocity = jnp.array([[0.01, 0.03, -0.02], [0.04, -0.02, 0.01]])
    sphere_velocity = jnp.array([0.05, -0.03, 0.0])
    result = sphere_interaction(points, 0.001, center, 0.2, law)
    rate = jax.jvp(
        lambda x, c: sphere_interaction(x, 0.001, c, 0.2, law).potential,
        (points, center),
        (point_velocity, sphere_velocity),
    )[1]
    power = jnp.sum(result.forces * point_velocity) + jnp.dot(
        result.reaction, sphere_velocity
    )
    np.testing.assert_allclose(rate + power, 0.0, atol=1e-14)


def test_sphere_separates_and_reference_area_is_integrated_once():
    law = DugdaleLaw(0.6, 0.02, 20000.0)
    points = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    attached = sphere_interaction(
        points, jnp.array([0.001, 0.002]), jnp.array([0.0, 0.21, 0.0]), 0.2, law
    )
    np.testing.assert_allclose(attached.reaction, [0.0, -0.09, 0.0])
    np.testing.assert_allclose(attached.adhesive_reference_area, 0.003)
    assert int(attached.adhesive_points) == 2
    detached = sphere_interaction(points, 0.001, jnp.array([0.0, 0.23, 0.0]), 0.2, law)
    np.testing.assert_array_equal(detached.forces, 0.0)
    assert int(detached.adhesive_points) == 0
    np.testing.assert_allclose(detached.potential, 0.0, atol=1e-16)


def test_path_velocity_bounds_and_serialization():
    path = SpherePath(
        0.2,
        (0.0, 0.1, 0.3, 0.5),
        ((0.3, 0.7, 0.4), (0.3, 0.6, 0.4), (0.4, 0.6, 0.4), (0.4, 0.8, 0.4)),
    )
    json.dumps(asdict(path))
    for time in (-0.1, 0.03, 0.17, 0.37, 0.6):
        position, velocity = path.motion(time)
        np.testing.assert_allclose(
            jax.jacfwd(lambda t: path.motion(t)[0])(float(time)), velocity, atol=1e-14
        )
        assert np.all(np.asarray(position) >= np.min(path.centers, axis=0) - 1e-14)
        assert np.all(np.asarray(position) <= np.max(path.centers, axis=0) + 1e-14)
    np.testing.assert_array_equal(path.motion(-0.1)[1], 0.0)
    np.testing.assert_array_equal(path.motion(0.6)[1], 0.0)
    tree = ast.parse(
        (Path(__file__).parents[1] / "mpm_jax" / "paraview_scene.py").read_text()
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "waypoint_sphere_motion"
    )
    namespace = {"np": np}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "paraview_motion", "exec"
        ),
        namespace,
    )
    times = np.linspace(-0.1, 0.6, 31)
    expected = jax.vmap(lambda t: path.motion(t)[0])(jnp.asarray(times))
    np.testing.assert_allclose(
        namespace["waypoint_sphere_motion"](asdict(path), times), expected, atol=2e-15
    )


def test_undeformed_plane_annulus_against_exact_radial_integral():
    law = DugdaleLaw(0.6, 0.02, 20000.0)
    radius, gap0 = 0.2, 0.005
    height = radius + gap0
    outer = np.sqrt((radius + law.interaction_range) ** 2 - height**2)
    nodes, weights = np.polynomial.legendre.leggauss(48)
    radial = 0.5 * outer * (nodes + 1)
    areas = 2 * np.pi * radial * (0.5 * outer * weights)
    points = jnp.asarray(np.column_stack((radial, np.zeros(48), np.zeros(48))))
    contact = sphere_interaction(points, areas, jnp.array([0, height, 0]), radius, law)
    expected = (
        -2 * np.pi * law.cohesive_stress * height * (law.interaction_range - gap0)
    )
    np.testing.assert_allclose(contact.reaction[1], expected, rtol=2e-13)
    np.testing.assert_allclose(
        contact.adhesive_reference_area, np.pi * outer**2, rtol=2e-13
    )
    assert int(contact.compressed_points) == 0


def test_particle_force_survives_mpm_transfer_and_support_reaction_includes_it():
    config = Config(dim=3, n_grid=12, dt=1e-4, gravity=(0, 0, 0), boundary_cells=0)
    material = elastic(1e4, 0.25)
    state = block(
        config, (0.35, 0.35, 0.35), (0.65, 0.65, 0.65), density=1000, dtype=jnp.float64
    )
    law = DugdaleLaw(0.6, 0.04, 20000.0)
    result = sphere_interaction(state.x, 0.001, jnp.array([0.5, 0.8, 0.5]), 0.2, law)
    new = step(state, material, config, result.forces)
    change = jnp.sum(state.mass[:, None] * (new.v - state.v), axis=0)
    np.testing.assert_allclose(change, -config.dt * result.reaction, atol=1e-14)
    fixed = Config(
        dim=3,
        n_grid=12,
        dt=config.dt,
        gravity=(0, 0, 0),
        boundary_cells=0,
        boundaries=(fixed_box((0, 0, 0), (1, 1, 1)),),
    )
    reaction = boundary_reaction(state, material, fixed, force=result.forces)
    np.testing.assert_allclose(reaction, result.forces.sum(0), atol=1e-12)
    np.testing.assert_array_equal(boundary_reaction(state, material, fixed), 0.0)
    np.testing.assert_allclose(
        penalty_step_number(state.mass, 0.001, config.dt, law),
        law.penalty_stiffness * 0.001 * config.dt**2 / state.mass,
    )


@pytest.mark.parametrize(
    "values", [(-1, 1, 1), (1, 0, 1), (1, 1, 0), (float("nan"), 1, 1)]
)
def test_invalid_law_parameters(values):
    with pytest.raises(ValueError):
        DugdaleLaw(*values)
