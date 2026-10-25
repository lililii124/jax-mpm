import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.hierarchical_grid import (
    apic_to_grid, grid_stencil, hierarchical_grid, internal_force, moment_matrix, uniform_grid,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("kind", ["coarse", "fine", "hierarchical"])
def test_cropping_empty_nodes_preserves_particle_and_grid_operators(kind):
    cells = 32 if kind == "fine" else 16
    make = hierarchical_grid if kind == "hierarchical" else uniform_grid
    args = dict(cells=cells, domain_size=1.)
    if kind == "hierarchical":
        args.update(refined_lower=(3.5/16,)*3, refined_upper=(12.5/16,)*3)
    else:
        args.update(offset=.5 if kind == "fine" else 0.)
    full = make(**args)
    crop = make(**args, node_bounds=((1, 1, 1), (cells-1, cells-1, cells-1)))
    rng = np.random.default_rng(15)
    x = jnp.asarray(rng.uniform(.2, .8, (193, 3)))
    a, b = grid_stencil(x, full), grid_stencil(x, crop)
    assert bool(a.valid) and bool(b.valid)
    active = (np.asarray(a.weights) != 0) | np.any(np.asarray(a.gradients) != 0, axis=-1)
    np.testing.assert_allclose(np.asarray(full.positions[a.indices])[active],
                               np.asarray(crop.positions[b.indices])[active], rtol=0, atol=1e-16)
    np.testing.assert_allclose(a.weights, b.weights, rtol=0, atol=3e-15)
    np.testing.assert_allclose(a.gradients, b.gradients, rtol=0, atol=3e-13)
    mass = jnp.asarray(rng.uniform(.1, 2., len(x)))
    volume = mass / 10.
    velocity = jnp.asarray(rng.normal(size=x.shape))
    C = jnp.asarray(rng.normal(size=(len(x), 3, 3)))
    stress = jnp.asarray(rng.normal(size=(len(x), 3, 3)))
    stress = stress + stress.swapaxes(-1, -2)
    B = C @ moment_matrix(a)
    full_mass, full_momentum = apic_to_grid(mass, velocity, B, a, full)
    crop_mass, crop_momentum = apic_to_grid(mass, velocity, B, b, crop)
    full_force, crop_force = internal_force(volume, stress, a, full), internal_force(volume, stress, b, crop)
    lookup = {tuple(np.rint(p * 128).astype(int)): i for i, p in enumerate(np.asarray(full.positions))}
    indices = np.array([lookup[tuple(np.rint(p * 128).astype(int))] for p in np.asarray(crop.positions)])
    np.testing.assert_allclose(full_mass[indices], crop_mass, rtol=0, atol=1e-13)
    np.testing.assert_allclose(full_momentum[indices], crop_momentum, rtol=0, atol=1e-13)
    np.testing.assert_allclose(full_force[indices], crop_force, rtol=0, atol=1e-11)
    excluded = np.ones(len(full.positions), dtype=bool)
    excluded[indices] = False
    np.testing.assert_array_equal(np.asarray(full_mass)[excluded], 0.)
    assert not bool(grid_stencil(jnp.array([[.04, .5, .5]]), crop).valid)


def test_node_bounds_reject_fractional_or_inverted_indices():
    for bounds in (((0., 0., 0.), (16., 16., 16.)), ((4, 4, 4), (3, 12, 12))):
        with pytest.raises(ValueError, match="Node bounds"):
            uniform_grid(cells=16, domain_size=1., node_bounds=bounds)
