from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.materials.woven import woven_elastic
from mpm_jax.packed_woven import (
    PackedWovenConfig, kinetic_energy, loading_fraction, make_plate_grid, make_state,
    step_with_diagnostics,
)
from mpm_jax.rigid_sphere_contact import SphereState

jax.config.update('jax_enable_x64', True)


def fixture(mode='pressure-bending', *, velocity=None):
    axes = (np.arange(-8, 8)+.5)*.001
    x = np.stack(np.meshgrid(axes, [-.00025, .00025], axes, indexing='ij'), axis=-1).reshape(-1, 3)
    material = woven_elastic(matrix_young=1000., matrix_poisson=.25, matrix_density=1000.,
        yarn_young=(1000., 1000., 1000.), yarn_poisson=(.25, .25, .25), yarn_shear=(400., 400., 400.),
        yarn_density=1000., orientation=np.broadcast_to(np.eye(3), (len(x), 3, 3)), phase=np.zeros(len(x), dtype=np.uint8))
    volume = np.full(len(x), .001*.0005*.001)
    grid = make_plate_grid(x, size=(.016, .016), dx=.001, band=.002,
        y_bounds=(-.004, .004), padding_cells=2, contact_half_width=.006)
    state = make_state(x, volume, volume*1000., material, v=velocity)
    config = PackedWovenConfig(1e-6, mode, contact_half_width=.006, contact_capacity=256)
    sphere = SphereState(jnp.array([0., .0025, 0.]), jnp.array([0., -.2, 0.]), jnp.array(.01), jnp.array(.003))
    return state, material, grid, config, sphere


def test_packed_plate_preserves_constant_free_translation():
    state, material, grid, config, sphere = fixture()
    initial_velocity = jnp.broadcast_to(jnp.array([.02, .01, -.03]), state.x.shape)
    state = state._replace(v=initial_velocity)
    grid = grid._replace(clamp_owner=jnp.full_like(grid.clamp_owner, -1))
    current, _, diagnostic = step_with_diagnostics(state, material, grid, config, sphere)
    assert bool(current.valid) and bool(diagnostic.valid)
    np.testing.assert_allclose(current.v, initial_velocity, atol=1e-15)
    np.testing.assert_allclose(current.x, state.x+config.dt*initial_velocity, atol=1e-17)
    np.testing.assert_allclose(current.F, state.F, atol=1e-14)


def test_four_edge_clamp_preserves_tagged_particles_and_counts_corner_impulses_once():
    state, material, grid, config, sphere = fixture()
    force = jnp.broadcast_to(jnp.array([0., -.01, 0.]), state.x.shape)
    current, _, diagnostic = step_with_diagnostics(state, material, grid, config, sphere, force)
    assert bool(current.valid)
    boundary = (np.abs(np.asarray(state.x[:, 0])) >= .006) | (np.abs(np.asarray(state.x[:, 2])) >= .006)
    np.testing.assert_allclose(current.x[boundary], state.x[boundary], atol=1e-17)
    np.testing.assert_allclose(current.v[boundary], 0., atol=1e-15)
    expected = config.dt*np.asarray(force).sum(0)*float(loading_fraction(config.dt, config.loading_duration))
    np.testing.assert_allclose(diagnostic.external_impulse, expected, rtol=1e-14, atol=1e-20)
    momentum = np.sum(np.asarray(state.mass)[:, None]*np.asarray(current.v), axis=0)
    np.testing.assert_allclose(momentum, diagnostic.clamp_impulse+diagnostic.external_impulse, atol=1e-18)
    np.testing.assert_allclose(diagnostic.group_impulses.sum(0), diagnostic.clamp_impulse, atol=1e-20)
    assert float(diagnostic.grid_projection_work) <= 0 and float(current.tool_work) == 0


def test_finite_ball_and_plate_balance_actual_contact_momentum():
    state, material, grid, config, sphere = fixture('sphere-impact')
    current, ball, diagnostic = step_with_diagnostics(state, material, grid, config, sphere)
    assert bool(current.valid) and int(diagnostic.contact_nodes) > 0
    change = (np.sum(np.asarray(state.mass)[:, None]*(np.asarray(current.v)-np.asarray(state.v)), axis=0)
              +float(sphere.mass)*(np.asarray(ball.velocity)-np.asarray(sphere.velocity)))
    np.testing.assert_allclose(change, diagnostic.clamp_impulse+diagnostic.external_impulse, atol=1e-16)
    np.testing.assert_allclose(float(sphere.mass)*(ball.velocity-sphere.velocity), -diagnostic.contact_impulse, atol=1e-16)
    assert float(current.contact_loss) >= 0


def test_contact_capacity_overflow_is_rejected_instead_of_ignoring_nodes():
    state, material, grid, config, sphere = fixture('sphere-impact')
    current, _, diagnostic = step_with_diagnostics(state, material, grid, replace(config, contact_capacity=1), sphere)
    assert bool(diagnostic.contact_overflow) and not bool(current.valid)


def test_a_sphere_leaving_the_proven_contact_window_is_rejected():
    state, material, grid, config, sphere = fixture('sphere-impact')
    sphere = sphere._replace(center=sphere.center.at[0].set(.02))
    current, _, diagnostic = step_with_diagnostics(state, material, grid, config, sphere)
    assert not bool(diagnostic.contact_window_valid) and not bool(current.valid)


def test_apic_affine_kinetic_energy_is_reported_separately():
    state, _, grid, _, _ = fixture()
    state = state._replace(C=jnp.broadcast_to(jnp.eye(3)*2., state.C.shape))
    translational, affine = kinetic_energy(state, grid.layout.dx)
    assert float(translational) == 0
    np.testing.assert_allclose(affine, grid.layout.dx**2/8*float(state.mass.sum())*12., rtol=1e-14)
