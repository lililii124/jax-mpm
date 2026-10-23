import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.fibre_mpm import (
    FibreConfig, advance, advance_with_diagnostics, build_layout, grid_coordinates,
    grid_to_particle, make_state, particle_to_grid, reference_grips, stencil, step,
    step_with_diagnostics, stored_energy, validate_state,
)
from mpm_jax.materials.isotropic_brittle import isotropic_brittle

jax.config.update('jax_enable_x64', True)


def material(dx=.01):
    return isotropic_brittle(young=1000., poisson=.25, density=1000., tensile_strength=100.,
                             fracture_energy=10., characteristic_length=dx)


def fixture(x=None, ids=None, **initial):
    x = np.array([[.025, .025, .025]]) if x is None else np.asarray(x)
    ids = np.zeros(len(x), dtype=np.int32) if ids is None else np.asarray(ids, dtype=np.int32)
    layout = build_layout(x, ids, dx=.01)
    constitutive = material()
    state = make_state(x, np.full(len(x), 1e-6), ids, constitutive, layout, **initial)
    return state, constitutive, layout


def total_momentum(state):
    return np.sum(np.asarray(state.mass)[:, None]*np.asarray(state.v), axis=0)


def angular_momentum(state, dx):
    C = np.asarray(state.C)
    intrinsic = np.column_stack((C[:, 2, 1]-C[:, 1, 2], C[:, 0, 2]-C[:, 2, 0], C[:, 1, 0]-C[:, 0, 1]))
    return np.sum(np.asarray(state.mass)[:, None]*(np.cross(state.x, state.v)+dx**2/4*intrinsic), axis=0)


def test_coincident_fibres_never_share_nodes_or_bond_velocities():
    state, constitutive, layout = fixture([[.025]*3, [.025]*3], [0, 1], v=[[1., 0, 0], [-1., 0, 0]])
    support = stencil(state.x, state.fibre_id, layout)
    assert not np.intersect1d(np.asarray(support.indices[0]), np.asarray(support.indices[1])).size
    coordinates, fibres = grid_coordinates(layout, support.indices)
    np.testing.assert_array_equal(coordinates[0], coordinates[1])
    np.testing.assert_array_equal(fibres, np.repeat([[0], [1]], 27, axis=1))
    current = step(state, constitutive, layout, FibreConfig(.001))
    np.testing.assert_allclose(current.v, state.v, rtol=0, atol=1e-14)
    np.testing.assert_allclose(current.x, state.x+.001*state.v, rtol=0, atol=1e-15)
    np.testing.assert_allclose(total_momentum(current), 0, atol=1e-15)
    assert validate_state(current, layout)


def test_affine_velocity_and_apic_gradient_are_reproduced():
    x = np.stack(np.meshgrid(*([np.array([.025, .03, .035])]*3), indexing='ij'), axis=-1).reshape(-1, 3)
    A = np.array([[.2, -.3, .1], [.4, .1, -.2], [-.1, .2, -.1]])
    v = x@A.T+np.array([.1, -.2, .3])
    state, constitutive, layout = fixture(x, v=v, C=np.broadcast_to(A, (len(x), 3, 3)))
    current = step(state, constitutive, layout, FibreConfig(1e-4))
    np.testing.assert_allclose(current.v, v, rtol=0, atol=1e-14)
    np.testing.assert_allclose(current.C, np.broadcast_to(A, current.C.shape), rtol=0, atol=3e-13)
    np.testing.assert_allclose(current.F, np.broadcast_to(np.eye(3)+1e-4*A, current.F.shape), rtol=0, atol=1e-15)
    assert validate_state(current, layout)


def test_rigid_rotation_is_stress_free_and_translates_without_deformation():
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    state, constitutive, layout = fixture(v=[[.1, .2, -.1]], F=rotation[None])
    assert abs(float(stored_energy(state, constitutive))) < 1e-25
    current = advance(state, constitutive, layout, FibreConfig(.001), 5)
    np.testing.assert_allclose(current.F[0], rotation, atol=1e-14)
    np.testing.assert_allclose(current.x, state.x+.005*state.v, atol=1e-14)
    np.testing.assert_allclose(current.history.damage, 0, atol=0)


def test_free_apic_rotation_conserves_orbital_and_intrinsic_angular_momentum():
    x = np.array([[.025, .025, .025], [.035, .025, .025], [.025, .035, .025]])
    spin = np.array([[0., -2., 0.], [2., 0., 0.], [0., 0., 0.]])
    state, constitutive, layout = fixture(x, v=x@spin.T, C=np.broadcast_to(spin, (len(x), 3, 3)))
    current = step(state, constitutive, layout, FibreConfig(1e-4))
    np.testing.assert_allclose(angular_momentum(current, layout.dx), angular_momentum(state, layout.dx), rtol=2e-13, atol=1e-18)
    assert validate_state(current, layout)


def test_scatter_mass_and_external_impulse_are_conservative():
    state, constitutive, layout = fixture([[.025]*3, [.027, .03, .032], [.025]*3], [0, 0, 1])
    config = FibreConfig(.001, gravity=(0., -9.81, 0.))
    force = jnp.array([[.2, .1, .3], [-.1, .2, .1], [.4, -.3, -.2]])
    mass, _ = particle_to_grid(state, layout, config, force=force)
    np.testing.assert_allclose(mass.sum(), state.mass.sum(), rtol=2e-15)
    current, report = step_with_diagnostics(state, constitutive, layout, config, force)
    expected = config.dt*(np.array(force).sum(0)+float(state.mass.sum())*np.array(config.gravity))
    np.testing.assert_allclose(total_momentum(current)-total_momentum(state), expected, atol=1e-16)
    np.testing.assert_allclose(report.external_impulse, expected, atol=1e-16)
    np.testing.assert_allclose(report.boundary_impulse, 0, atol=0)
    np.testing.assert_allclose(current.external_impulse, expected, atol=1e-16)


def test_reference_grip_follows_particles_across_grid_cells():
    state, constitutive, layout = fixture(v=[[0., 0., 0.]])
    boundary = reference_grips([np.array([True])], [lambda time: jnp.array([.4, .1, 0.])])
    config = FibreConfig(.01)
    current, report = advance_with_diagnostics(state, constitutive, layout, config, 8, boundary=boundary)
    np.testing.assert_allclose(current.x, state.x+.08*np.array([[.4, .1, 0.]]), rtol=0, atol=2e-15)
    np.testing.assert_allclose(current.v, [[.4, .1, 0.]], atol=1e-14)
    np.testing.assert_allclose(total_momentum(current)-total_momentum(state), report.boundary_impulse, atol=1e-16)
    np.testing.assert_allclose(report.group_impulses.sum(0), report.boundary_impulse, atol=1e-16)
    np.testing.assert_allclose(report.group_work.sum(), report.boundary_work, atol=1e-16)
    assert validate_state(current, layout)


def test_two_grip_reactions_equal_the_actual_momentum_correction():
    state, constitutive, layout = fixture([[.025]*3, [.105, .025, .025]])
    boundary = reference_grips([[True, False], [False, True]],
                               [lambda time: jnp.array([-.2, 0., 0.]), lambda time: jnp.array([.2, 0., 0.])])
    current, report = step_with_diagnostics(state, constitutive, layout, FibreConfig(.001), boundary=boundary)
    np.testing.assert_allclose(report.group_impulses, [[-.0002, 0, 0], [.0002, 0, 0]], atol=1e-16)
    np.testing.assert_allclose(total_momentum(current)-total_momentum(state), report.boundary_impulse, atol=1e-16)
    assert bool(current.valid)


def test_overlapping_reference_grips_are_explicitly_invalid():
    state, constitutive, layout = fixture()
    boundary = reference_grips([[True], [True]], [lambda time: jnp.zeros(3), lambda time: jnp.ones(3)])
    current = advance(state, constitutive, layout, FibreConfig(.001), 5, boundary=boundary)
    assert not bool(current.valid)
    assert float(current.time) == .001
    with pytest.raises(FloatingPointError, match='boundary overlap'):
        validate_state(current, layout)


def test_advection_outside_local_grid_is_not_clamped_and_stops_advance():
    state, constitutive, layout = fixture(v=[[10., 0., 0.]])
    current = advance(state, constitutive, layout, FibreConfig(.1), 3)
    np.testing.assert_allclose(current.x, [[1.025, .025, .025]], atol=1e-14)
    assert not bool(current.valid)
    assert float(current.time) == .1
    with pytest.raises(FloatingPointError, match='outside its packed grid'):
        validate_state(current, layout)


def test_incomplete_stencil_is_flagged_and_cannot_wrap_to_another_fibre():
    state, _, layout = fixture([[.025]*3, [.025]*3], [0, 1])
    moved = state.x.at[0, 0].set(10.)
    support = stencil(moved, state.fibre_id, layout)
    assert not bool(support.valid)
    assert np.all(np.asarray(support.indices[0]) == layout.node_count)
    assert np.all(np.asarray(support.indices[1]) < layout.node_count)


def test_histories_are_independent_even_at_coincident_points():
    C = np.zeros((2, 3, 3)); C[0, 0, 0] = 20
    state, constitutive, layout = fixture([[.025]*3, [.025]*3], [0, 1], C=C)
    current = step(state, constitutive, layout, FibreConfig(.01))
    assert float(current.history.damage[0]) > 0
    assert float(current.history.damage[1]) == 0
    np.testing.assert_allclose(current.F[1], np.eye(3), atol=1e-15)


def test_state_dependent_force_is_recomputed_inside_advance():
    state, constitutive, layout = fixture()
    force_fn = lambda current, time: -.1*current.mass[:, None]*current.x
    config = FibreConfig(.001)
    batched = advance(state, constitutive, layout, config, 3, force_fn=force_fn)
    manual = state
    for _ in range(3):
        manual = step(manual, constitutive, layout, config, force=force_fn(manual, manual.time))
    np.testing.assert_allclose(batched.x, manual.x, atol=1e-15)
    np.testing.assert_allclose(batched.external_impulse, manual.external_impulse, atol=1e-17)


def test_crack_band_length_cannot_follow_particle_spacing():
    state, _, layout = fixture()
    with pytest.raises(ValueError, match='equal to grid dx'):
        make_state(state.x, state.volume, state.fibre_id, material(.005), layout)


def test_layout_is_locally_bounded_and_budgeted():
    x = np.array([[.01, .01, .01], [10., 10., 10.]])
    layout = build_layout(x, np.array([0, 1]), dx=.01, padding_cells=3)
    assert layout.node_count == 2*7**3
    assert np.array_equal(np.asarray(layout.offsets), [0, 7**3])
    with pytest.raises(ValueError, match='node budget'):
        build_layout(x, np.array([0, 0]), dx=.01, max_nodes=10_000)
