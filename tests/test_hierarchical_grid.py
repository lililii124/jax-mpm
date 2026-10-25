import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.hierarchical_grid import (
    hierarchical_grid, hierarchical_stencil, moment_matrix, affine_velocity,
    apic_to_grid, apic_from_grid, internal_force,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def grid():
    return hierarchical_grid(cells=16, domain_size=1.,
                             refined_lower=(3.5 / 16, 4.5 / 16, 3.5 / 16),
                             refined_upper=(12.5 / 16, 11.5 / 16, 12.5 / 16))


def points(n=512):
    return jnp.asarray(np.random.default_rng(321).uniform(.08, .92, (n, 3)))


def spin(B):
    return np.stack((B[:, 2, 1] - B[:, 1, 2], B[:, 0, 2] - B[:, 2, 0], B[:, 1, 0] - B[:, 0, 1]), -1)


def test_partition_linear_precision_and_derivatives_across_all_refinement_faces(grid):
    x = points()
    support = hierarchical_stencil(x, grid)
    assert bool(support.valid)
    w, grad, dpos = map(np.asarray, (support.weights, support.gradients, support.displacements))
    assert np.min(w) >= 0
    np.testing.assert_allclose(w.sum(1), 1., rtol=0, atol=2e-14)
    np.testing.assert_allclose(np.einsum("pk,pki->pi", w, dpos), 0., rtol=0, atol=2e-15)
    np.testing.assert_allclose(grad.sum(1), 0., rtol=0, atol=2e-12)
    np.testing.assert_allclose(np.einsum("pki,pkj->pij", dpos, grad),
                               np.broadcast_to(np.eye(3), (len(x), 3, 3)), rtol=0, atol=3e-14)
    for axis in range(3):
        delta = np.zeros((1, 3))
        delta[0, axis] = 1e-7
        plus, minus = hierarchical_stencil(x + delta, grid), hierarchical_stencil(x - delta, grid)
        # A smooth spatial polynomial checks the assembled derivative independently of local slot order.
        values = np.prod(np.asarray(grid.positions)**2 + 1., axis=1)
        numerical = (np.sum(values[np.asarray(plus.indices)] * np.asarray(plus.weights), axis=1)
                     - np.sum(values[np.asarray(minus.indices)] * np.asarray(minus.weights), axis=1)) / (2e-7)
        exact = np.sum(values[np.asarray(support.indices)] * grad[..., axis], axis=1)
        np.testing.assert_allclose(numerical, exact, rtol=3e-7, atol=5e-8)
    assert not bool(hierarchical_stencil(jnp.array([[.001, .5, .5]]), grid).valid)


def test_uniform_limits_and_variable_transition_moment(grid):
    x = jnp.array([[.125, .125, .125], [.5, .5, .5], [.25, .5, .5]])
    support = hierarchical_stencil(x, grid)
    D = np.asarray(moment_matrix(support))
    np.testing.assert_allclose(D[0], np.eye(3) / (4 * 16**2), rtol=0, atol=1e-16)
    np.testing.assert_allclose(D[1], np.eye(3) / (4 * 32**2), rtol=0, atol=1e-16)
    assert np.linalg.eigvalsh(D).min() > 0
    assert not np.allclose(D[2], D[0], rtol=1e-8, atol=1e-14)
    assert not np.allclose(D[2], D[1], rtol=1e-8, atol=1e-14)


def test_affine_velocity_preservation_across_both_transfers(grid):
    x = points(257)
    support = hierarchical_stencil(x, grid)
    A = jnp.array([[.3, -.4, .8], [1.2, -.7, .1], [-.6, .9, .2]])
    b = jnp.array([.2, -.1, .5])
    v = x @ A.T + b
    B = A @ moment_matrix(support)
    mass = jnp.linspace(.2, 1.3, len(x))
    gm, gp = apic_to_grid(mass, v, B, support, grid)
    gv = gp / jnp.where(gm > 0, gm, 1)[:, None]
    occupied = np.asarray(gm) > 0
    np.testing.assert_allclose(np.asarray(gv)[occupied], np.asarray(grid.positions @ A.T + b)[occupied], rtol=0, atol=3e-14)
    result_v, result_B, L = apic_from_grid(gv, support)
    np.testing.assert_allclose(result_v, v, rtol=0, atol=2e-14)
    np.testing.assert_allclose(affine_velocity(result_B, moment_matrix(support)),
                               np.broadcast_to(A, (len(x), 3, 3)), rtol=0, atol=2e-12)
    np.testing.assert_allclose(L, np.broadcast_to(A, (len(x), 3, 3)), rtol=0, atol=3e-12)


def test_transfer_mass_linear_angular_momentum_and_energy(grid):
    x = points(321)
    support = hierarchical_stencil(x, grid)
    rng = np.random.default_rng(554)
    mass = jnp.asarray(rng.uniform(.1, 2., len(x)))
    v = jnp.asarray(rng.normal(size=x.shape))
    C = jnp.asarray(rng.normal(size=(len(x), 3, 3)))
    D = moment_matrix(support)
    B = C @ D
    gm, gp = apic_to_grid(mass, v, B, support, grid)
    gv = gp / jnp.where(gm > 0, gm, 1)[:, None]
    out_v, out_B, _ = apic_from_grid(gv, support)
    np.testing.assert_allclose(gm.sum(), mass.sum(), rtol=2e-15)
    momentum = np.sum(np.asarray(mass)[:, None] * np.asarray(v), axis=0)
    np.testing.assert_allclose(gp.sum(0), momentum, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose((mass[:, None] * out_v).sum(0), momentum, rtol=2e-13, atol=2e-13)
    angular = np.sum(np.asarray(mass)[:, None] * (np.cross(x, v) + spin(np.asarray(B))), axis=0)
    grid_angular = np.sum(np.cross(grid.positions, gp), axis=0)
    out_angular = np.sum(np.asarray(mass)[:, None] * (np.cross(x, out_v) + spin(np.asarray(out_B))), axis=0)
    np.testing.assert_allclose(grid_angular, angular, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(out_angular, angular, rtol=2e-13, atol=2e-13)
    kinetic = lambda velocity, B: float(.5 * jnp.sum(mass * (
        jnp.sum(velocity**2, axis=1) + jnp.einsum("pij,pij->p", B, affine_velocity(B, D)))))
    grid_ke = float(.5 * jnp.sum(gm[:, None] * gv**2))
    assert kinetic(out_v, out_B) <= grid_ke + 1e-12 <= kinetic(v, B) + 2e-12


def test_internal_force_and_physical_gradient_are_work_conjugate(grid):
    x = points(177)
    support = hierarchical_stencil(x, grid)
    rng = np.random.default_rng(12)
    stress = rng.normal(size=(len(x), 3, 3))
    stress += stress.swapaxes(-1, -2)
    volume = jnp.asarray(rng.uniform(.1, 1., len(x)))
    gv = jnp.asarray(rng.normal(size=grid.positions.shape))
    force = internal_force(volume, jnp.asarray(stress), support, grid)
    _, _, L = apic_from_grid(gv, support)
    grid_power = float(jnp.sum(force * gv))
    material_power = -float(jnp.sum(volume[:, None, None] * stress * L))
    np.testing.assert_allclose(grid_power, material_power, rtol=3e-14, atol=1e-10)
    np.testing.assert_allclose(force.sum(0), 0., rtol=0, atol=2e-11)
    np.testing.assert_allclose(np.cross(grid.positions, force).sum(0), 0., rtol=0, atol=2e-11)


def test_refinement_bounds_cannot_silently_change_the_nested_knot_grid():
    with pytest.raises(ValueError, match="knots"):
        hierarchical_grid(cells=16, domain_size=1., refined_lower=(.2, .2, .2), refined_upper=(.8, .8, .8))
