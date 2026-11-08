from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.compact_tl import initial_state, point_material, reference_grid, rectilinear_support, step
from mpm_jax.multifield_support import split_disconnected_node_support
from mpm_jax.total_lagrangian import sample_nodal


def response(gap, split):
    gauss = np.asarray(list(product((-1/(2*np.sqrt(3)), 1/(2*np.sqrt(3))), repeat=3)))
    centers = np.array([[.5, .5, .5], [1.5+gap, .5, .5]])
    positions = (centers[:, None, :]+gauss[None, :, :]).reshape(-1, 3)
    support = rectilinear_support(positions, 1/8, np.ones(3))
    if split:
        support = split_disconnected_node_support(support).support
    reference = reference_grid(support, 1.)
    material = point_material(jnp.full(len(positions), 10.), .25)
    load = jnp.zeros_like(reference.nodes).at[reference.indices[:8].reshape(-1), 0].add(
        (reference.weights[:8]/8).reshape(-1))
    state = initial_state(reference, material, load)
    fixed = jnp.zeros_like(load, dtype=bool)
    for _ in range(20):
        state = step(state, reference, material, .001, load, load, fixed)[0]
    assert bool(state.valid)
    displacement = np.asarray(sample_nodal(state.displacement, reference))
    return float(np.linalg.norm(displacement[8:], axis=1).max())


def test_disconnected_fragments_do_not_exchange_grid_momentum():
    jax.config.update('jax_enable_x64', True)
    assert response(gap=1, split=False) > 1e-9
    assert response(gap=1, split=True) < 1e-15
    assert response(gap=0, split=True) > 1e-9
