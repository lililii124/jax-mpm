"""Two-level quadratic THB-spline grid and APIC transfer operators."""

from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class HierarchicalGrid(NamedTuple):
    positions: jax.Array
    levels: jax.Array
    coarse_ids: jax.Array
    fine_lower: jax.Array
    fine_upper: jax.Array
    fine_shape: jax.Array
    fine_start: jax.Array
    spacing: jax.Array


class HierarchicalStencil(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    gradients: jax.Array
    displacements: jax.Array
    valid: jax.Array


def hierarchical_grid(*, cells, domain_size, refined_lower, refined_upper, dtype=jnp.float64):
    """Create a fixed interior refinement box whose faces lie on coarse knots."""
    if type(cells) is not int or cells < 8 or not np.isfinite(domain_size) or domain_size <= 0:
        raise ValueError("A positive domain and at least eight coarse cells are required")
    lower, upper = np.asarray(refined_lower, float), np.asarray(refined_upper, float)
    h = domain_size / cells
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all():
        raise ValueError("Refinement bounds must be finite three-vectors")
    if np.any(lower < 1.5 * h) or np.any(upper > domain_size - 1.5 * h) or np.any(upper <= lower):
        raise ValueError("Refinement must be interior with a coarse-support margin")
    knots = np.concatenate((lower, upper)) / h + .5
    if not np.allclose(knots, np.rint(knots), rtol=0, atol=1e-10):
        raise ValueError("Refinement faces must coincide with coarse spline knots")
    fine_h = h / 2
    fine_lower = np.rint(lower / fine_h + 1).astype(np.int32)
    fine_upper = np.rint(upper / fine_h - 2).astype(np.int32)
    fine_shape = fine_upper - fine_lower + 1
    if np.any(fine_shape < 1):
        raise ValueError("Refinement region contains no complete fine basis support")
    coarse = np.stack(np.meshgrid(*[np.arange(cells + 1)] * 3, indexing="ij"), -1)
    removed = np.all((2 * coarse - 2 >= fine_lower) & (2 * coarse + 1 <= fine_upper), axis=-1)
    coarse_ids = np.full(removed.shape, -1, dtype=np.int32)
    coarse_ids[~removed] = np.arange(np.count_nonzero(~removed), dtype=np.int32)
    fine = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(fine_lower, fine_upper)],
                               indexing="ij"), -1).reshape(-1, 3)
    # Quadratic refinement shifts the fine Greville lattice by half a fine cell.
    coarse_positions = coarse[~removed] * h
    fine_positions = (fine + .5) * fine_h
    positions = np.concatenate((coarse_positions, fine_positions))
    levels = np.concatenate((np.zeros(len(coarse_positions), np.int8), np.ones(len(fine_positions), np.int8)))
    return HierarchicalGrid(jnp.asarray(positions, dtype), jnp.asarray(levels), jnp.asarray(coarse_ids),
                            jnp.asarray(fine_lower), jnp.asarray(fine_upper), jnp.asarray(fine_shape),
                            jnp.asarray(len(coarse_positions), jnp.int32), jnp.asarray(h, dtype))


def _quadratic(r):
    absolute = jnp.abs(r)
    value = jnp.where(absolute < .5, .75 - r * r,
                      jnp.where(absolute < 1.5, .5 * (1.5 - absolute)**2, 0.))
    derivative = jnp.where(absolute < .5, -2 * r,
                           jnp.where(absolute < 1.5, jnp.sign(r) * (absolute - 1.5), 0.))
    return value, derivative


def _tensor_product(values, derivatives, offsets):
    factors = values[:, offsets, jnp.arange(3)]
    slopes = derivatives[:, offsets, jnp.arange(3)]
    gradient = jnp.stack((slopes[..., 0] * factors[..., 1] * factors[..., 2],
                          factors[..., 0] * slopes[..., 1] * factors[..., 2],
                          factors[..., 0] * factors[..., 1] * slopes[..., 2]), -1)
    return jnp.prod(factors, axis=-1), gradient


@jax.jit
def hierarchical_stencil(x, grid):
    """Return up to 27 coarse and 27 fine contributions per material point."""
    h, fine_h = grid.spacing, grid.spacing / 2
    offsets = jnp.array(list(product(range(3), repeat=3)), dtype=jnp.int32)
    coarse_base = jnp.floor(x / h - .5).astype(jnp.int32)
    coarse_axis = coarse_base[:, None, :] + jnp.arange(3)[None, :, None]
    children = 2 * coarse_axis[..., None] + jnp.arange(4) - 2
    child_position = (children + .5) * fine_h
    child_weight, child_gradient = _quadratic((x[:, None, :, None] - child_position) / fine_h)
    mask = jnp.array([.25, .75, .75, .25], dtype=x.dtype)
    child_weight = child_weight * mask
    child_gradient = child_gradient * mask / fine_h
    selected = (children >= grid.fine_lower[None, None, :, None]) & (
        children <= grid.fine_upper[None, None, :, None])
    retained = jnp.sum(jnp.where(selected, 0., child_weight), axis=-1)
    retained_gradient = jnp.sum(jnp.where(selected, 0., child_gradient), axis=-1)
    refined = jnp.sum(jnp.where(selected, child_weight, 0.), axis=-1)
    refined_gradient = jnp.sum(jnp.where(selected, child_gradient, 0.), axis=-1)
    whole, whole_gradient = retained + refined, retained_gradient + refined_gradient
    coarse_weight = jnp.zeros((len(x), 27), dtype=x.dtype)
    coarse_gradient = jnp.zeros((len(x), 27, 3), dtype=x.dtype)
    # Positive tensor expansion of Bx*By*Bz - Sx*Sy*Sz avoids cancellation.
    for split in range(3):
        value = jnp.stack([refined[..., axis] if axis < split else
                           retained[..., axis] if axis == split else whole[..., axis] for axis in range(3)], -1)
        slope = jnp.stack([refined_gradient[..., axis] if axis < split else
                           retained_gradient[..., axis] if axis == split else whole_gradient[..., axis]
                          for axis in range(3)], -1)
        weight, gradient = _tensor_product(value, slope, offsets)
        coarse_weight += weight
        coarse_gradient += gradient
    coarse_nodes = coarse_base[:, None, :] + offsets
    cells = grid.coarse_ids.shape[0] - 1
    inside = jnp.all((coarse_nodes >= 0) & (coarse_nodes <= cells), axis=-1)
    safe = jnp.clip(coarse_nodes, 0, cells)
    coarse_ids = grid.coarse_ids[safe[..., 0], safe[..., 1], safe[..., 2]]
    keep = inside & (coarse_ids >= 0)
    coarse_weight = jnp.where(keep, coarse_weight, 0.)
    coarse_gradient = jnp.where(keep[..., None], coarse_gradient, 0.)
    coarse_ids = jnp.maximum(coarse_ids, 0)

    fine_base = jnp.floor(x / fine_h - 1.).astype(jnp.int32)
    fine_axis = fine_base[:, None, :] + jnp.arange(3)[None, :, None]
    fine_value, fine_slope = _quadratic((x[:, None, :] - (fine_axis + .5) * fine_h) / fine_h)
    fine_weight, fine_gradient = _tensor_product(fine_value, fine_slope / fine_h, offsets)
    fine_nodes = fine_base[:, None, :] + offsets
    active = jnp.all((fine_nodes >= grid.fine_lower) & (fine_nodes <= grid.fine_upper), axis=-1)
    local = jnp.clip(fine_nodes - grid.fine_lower, 0, grid.fine_shape - 1)
    fine_ids = grid.fine_start + ((local[..., 0] * grid.fine_shape[1] + local[..., 1]) * grid.fine_shape[2] + local[..., 2])
    fine_weight = jnp.where(active, fine_weight, 0.)
    fine_gradient = jnp.where(active[..., None], fine_gradient, 0.)
    indices = jnp.concatenate((coarse_ids, fine_ids), axis=1)
    weights = jnp.concatenate((coarse_weight, fine_weight), axis=1)
    gradients = jnp.concatenate((coarse_gradient, fine_gradient), axis=1)
    displacements = grid.positions[indices] - x[:, None, :]
    return HierarchicalStencil(indices, weights, gradients, displacements,
                               jnp.all(inside) & jnp.all(jnp.isfinite(x)))


@jax.jit
def moment_matrix(support):
    return jnp.einsum("pk,pki,pkj->pij", support.weights, support.displacements, support.displacements)


@jax.jit
def affine_velocity(B, moment):
    return jnp.swapaxes(jnp.linalg.solve(moment, jnp.swapaxes(B, -1, -2)), -1, -2)


@jax.jit
def apic_to_grid(mass, velocity, B, support, grid):
    C = affine_velocity(B, moment_matrix(support))
    local_velocity = velocity[:, None, :] + jnp.einsum("pij,pkj->pki", C, support.displacements)
    weighted_mass = mass[:, None] * support.weights
    grid_mass = jnp.zeros(len(grid.positions), mass.dtype).at[support.indices].add(weighted_mass)
    grid_momentum = jnp.zeros_like(grid.positions).at[support.indices].add(weighted_mass[..., None] * local_velocity)
    return grid_mass, grid_momentum


@jax.jit
def apic_from_grid(grid_velocity, support):
    gathered = grid_velocity[support.indices]
    velocity = jnp.einsum("pk,pki->pi", support.weights, gathered)
    B = jnp.einsum("pk,pki,pkj->pij", support.weights, gathered, support.displacements)
    gradient = jnp.einsum("pki,pkj->pij", gathered, support.gradients)
    return velocity, B, gradient


@jax.jit
def internal_force(reference_volume, kirchhoff_stress, support, grid):
    contribution = -reference_volume[:, None, None] * jnp.einsum(
        "pij,pkj->pki", kirchhoff_stress, support.gradients)
    return jnp.zeros_like(grid.positions).at[support.indices].add(contribution)
