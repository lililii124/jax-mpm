"""Connected contact regions in a finite-segment coverage graph."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PatchSelection(NamedTuple):
    representatives: jax.Array
    cells: jax.Array
    labels: jax.Array
    active_count: jax.Array
    patch_count: jax.Array
    iterations: jax.Array
    valid: jax.Array


def patch_neighbours(segment_pairs, segment_owner):
    """Eight-neighbour parameter-space graph, confined to each fibre pair.

    Entries index the supplied coverage pairs, not material points. Missing
    neighbours are -1. Adjacent segments from a different fibre never connect.
    """
    pairs, owner = np.asarray(segment_pairs), np.asarray(segment_owner)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or owner.ndim != 1 or owner.dtype.kind not in 'iu' or np.any(owner < 0)
            or len(pairs) >= 2**31 or len(owner) >= 2**31
            or np.any(pairs < 0) or np.any(pairs >= len(owner))):
        raise ValueError('Integer coverage pairs and matching segment ownership are required')
    n, width = len(pairs), len(owner)
    if n == 0:
        return np.empty((0, 8), dtype=np.int32)
    encoded = pairs[:, 0].astype(np.int64)*width+pairs[:, 1]
    order = np.argsort(encoded)
    keys = encoded[order]
    if np.any(np.diff(keys) == 0):
        raise ValueError('Coverage pairs must be unique')
    graph = np.full((n, 8), -1, dtype=np.int32)
    offsets = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    for first in range(0, n, 65536):
        cells = pairs[first:first+65536].astype(np.int64)
        for column, offset in enumerate(offsets):
            neighbour = cells+offset
            inside = np.all((neighbour >= 0)&(neighbour < width), axis=1)
            safe = np.clip(neighbour, 0, width-1)
            same = np.all(owner[safe] == owner[cells], axis=1)
            candidate = neighbour[:, 0]*width+neighbour[:, 1]
            at = np.searchsorted(keys, candidate)
            bounded = np.minimum(at, n-1)
            found = inside&same&(at < n)&(keys[bounded] == candidate)
            graph[first:first+len(cells), column] = np.where(found, order[bounded], -1)
    return graph


@partial(jax.jit, static_argnames=('capacity', 'max_patches', 'max_iterations'))
def contact_patches(gap, active, neighbours, *, capacity=16384, max_patches=4096,
                    max_iterations=128):
    """Select the deepest cell of every connected active region.

    This is topology only: callers must apply a constitutive law, check the
    local contact approximation, and retain histories if using adhesion.
    Overflow or incomplete component convergence returns valid=False.
    Padding representatives/cells equal len(gap); padding labels equal capacity.
    neighbours must be the fixed undirected graph from patch_neighbours.
    """
    if any(type(v) is not int or v < 1 for v in (capacity, max_patches, max_iterations)):
        raise ValueError('Positive integer component budgets are required')
    gap, active, neighbours = jnp.asarray(gap), jnp.asarray(active), jnp.asarray(neighbours)
    if gap.ndim != 1:
        raise ValueError('Gap must be a one-dimensional array')
    n = gap.shape[0]
    if (gap.dtype.kind != 'f' or active.shape != gap.shape or active.dtype != jnp.bool_
            or neighbours.shape != (n, 8) or neighbours.dtype.kind != 'i'):
        raise ValueError('Gap, boolean active mask and an integer (N,8) graph are required')
    if n == 0:
        return PatchSelection(jnp.zeros(max_patches, dtype=jnp.int32),
            jnp.zeros(capacity, dtype=jnp.int32), jnp.full(capacity, capacity, dtype=jnp.int32),
            jnp.asarray(0), jnp.asarray(0), jnp.asarray(0), jnp.asarray(True))
    count = jnp.sum(active)
    cells = jnp.nonzero(active, size=capacity, fill_value=n)[0]
    live = cells < n
    slot = jnp.full(n+1, capacity, dtype=jnp.int32).at[cells].set(jnp.arange(capacity, dtype=jnp.int32))
    adjacent = jnp.take(neighbours, cells, axis=0, mode='fill', fill_value=-1)
    inside = (adjacent >= 0)&(adjacent < n)
    adjacent_slots = jnp.where(inside, slot[jnp.clip(adjacent, 0, n)], capacity)
    initial = jnp.where(live, jnp.arange(capacity, dtype=jnp.int32), capacity)

    def propagate(carry):
        labels, iteration, _ = carry
        padded = jnp.concatenate((labels, jnp.array([capacity], dtype=labels.dtype)))
        minimum = jnp.min(padded[adjacent_slots], axis=1)
        updated = jnp.minimum(labels, minimum)
        return updated, iteration+1, jnp.any(updated != labels)

    labels, iterations, unfinished = jax.lax.while_loop(
        lambda c: (c[1] < max_iterations)&c[2], propagate,
        (initial, jnp.asarray(0), jnp.asarray(True)))
    roots = live&(labels == jnp.arange(capacity))
    patch_count = jnp.sum(roots)
    values = jnp.concatenate((gap, jnp.array([jnp.inf], dtype=gap.dtype)))[cells]
    deepest = jnp.full(capacity+1, jnp.inf, dtype=gap.dtype).at[labels].min(values)
    winner = jnp.where(live&(values == deepest[labels]), cells, n)
    representative = jnp.full(capacity+1, n, dtype=cells.dtype).at[labels].min(winner)
    root_ids = jnp.nonzero(roots, size=max_patches, fill_value=capacity)[0]
    representatives = representative[root_ids]
    valid = ((count <= capacity)&(patch_count <= max_patches)&~unfinished
             &jnp.all(jnp.isfinite(gap))&jnp.all((neighbours >= -1)&(neighbours < n)))
    return PatchSelection(representatives, cells, labels, count, patch_count, iterations, valid)
