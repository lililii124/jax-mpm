import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.brittle_engineering import (
    ceramic_indentation,
    ceramic_indentation_wide,
    setup,
)
from examples.material_engineering import SCENES
from mpm_jax import step
from mpm_jax.contact_laws import boundary_reaction

jax.config.update("jax_enable_x64", True)


def test_wider_tile_smaller_sphere_preserve_loading_and_material():
    old = ceramic_indentation(grid=24, ppc=2, frames=4)
    wide = ceramic_indentation_wide(grid=24, ppc=2, frames=4)
    old_sphere, sphere = old.configs[0].boundaries[-1], wide.configs[0].boundaries[-1]
    assert sphere.radius == old_sphere.radius / 2
    assert sphere.duration == old_sphere.duration == 3e-4
    assert sphere.displacement == old_sphere.displacement == (0, -0.004, 0)
    assert old.description["lower"] == (0.075, 0.075, 0.075)
    assert old.description["upper"] == (0.175, 0.1, 0.175)
    np.testing.assert_allclose(wide.description["lower"], [0.035, 0.075, 0.035])
    np.testing.assert_allclose(wide.description["upper"], [0.215, 0.1, 0.215])
    assert wide.materials[0] == old.materials[0]
    assert wide.configs[0].boundaries[0] == old.configs[0].boundaries[0]
    assert wide.configs[0].failure_mode == "retain"
    np.testing.assert_allclose(wide.states[0].volume.sum(), 0.18 * 0.025 * 0.18)
    np.testing.assert_allclose(
        wide.states[0].mass.sum(), wide.materials[0].density * 0.18 * 0.025 * 0.18
    )
    assert wide.description["sphere_diameter_to_plate_width_ratio"] == pytest.approx(
        5 / 12
    )


def test_sphere_loads_then_returns_to_initial_tangent_plane_at_rest():
    case = ceramic_indentation_wide(grid=24, ppc=2, frames=4)
    sphere = case.configs[0].boundaries[-1]
    for time, indentation in ((0, 0), (case.duration / 2, 0.004), (case.duration, 0)):
        center, speed = sphere.motion(time)
        assert 0.1 - (float(center[1]) - sphere.radius) == pytest.approx(
            indentation, abs=1e-15
        )
        np.testing.assert_allclose(speed, 0, atol=1e-12)
    assert float(sphere.motion(case.duration / 4)[1][1]) < 0
    assert float(sphere.motion(3 * case.duration / 4)[1][1]) > 0


def test_wide_initial_observer_has_fixed_regions_and_no_geometric_contact():
    case = setup("ceramic-indentation-wide", grid=24, ppc=2, frames=4)
    observed = case.observer(case.states, 0.0)
    for name in (
        "central_surface_displacement_y_m",
        "central_surface_max_damage",
        "local_region_max_damage",
        "local_region_mean_damage_reference_volume",
        "indenter_particle_overlap_max_m",
        "particle_centers_inside_indenter",
    ):
        assert observed[name] == 0.0
    assert case.description["central_surface_radius_m"] == 0.004
    assert case.description["local_region_radius_m"] == 0.03
    assert case.description["central_surface_particles"] > 0
    assert "ceramic-indentation-wide" in SCENES


def test_spherical_contact_reaction_balances_one_step_momentum():
    case = ceramic_indentation_wide(grid=24, ppc=2, frames=4)
    state, material, config = case.states[0], case.materials[0], case.configs[0]
    state = state._replace(time=jnp.asarray(case.duration / 4))
    forces = jnp.stack(
        [boundary_reaction(state, material, config, i) for i in range(2)]
    )
    next_state = step(state, material, config)
    delta = jnp.sum(state.mass[:, None] * (next_state.v - state.v), axis=0)
    np.testing.assert_allclose(
        delta, -config.dt * forces.sum(axis=0), rtol=2e-12, atol=1e-11
    )
    assert bool(next_state.valid)
    assert float(forces[1, 1]) > 0


def test_wide_requires_the_ceramic_material():
    with pytest.raises(ValueError, match="requires model='jh2'"):
        ceramic_indentation_wide(model="hjc")
