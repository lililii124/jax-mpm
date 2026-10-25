"""Unadmitted affine FLIP transfer candidate; the frozen solver is unchanged.

The old grid is the APIC P2G velocity BEFORE stress, contact and boundary
increments. All those increments must occur exactly once in final-old.
"""
import jax.numpy as jnp
from .fragment_multifield_v6_mpm import transfer_to_particles


def transfer_delta(state, old_velocity, final_velocity, support, grid):
    """Return v,C and signed specific G2P correction, never a variance loss.

    Per-particle correction sums to K_grid(final)-K_particle(new).
    Together with P2G K_particle(old)-K_grid(old), the two corrections
    equal grid midpoint work minus particle kinetic change. It may be negative.
    Physical stress/contact/boundary work must retain their existing signs;
    this correction belongs in the numerical reservoir, never physical heat.
    """
    dv, dC, _ = transfer_to_particles(final_velocity-old_velocity, support, grid)
    v, C = state.v+dv, state.C+dC
    gathered = jnp.take(final_velocity, support.indices, axis=0, mode='fill', fill_value=0.)
    grid_specific = .5*jnp.sum(support.weights*jnp.sum(gathered**2, axis=-1), axis=-1)
    particle_specific = .5*jnp.sum(v*v, axis=-1)+grid.layout.dx**2/8*jnp.sum(C*C, axis=(-2,-1))
    return v, C, grid_specific-particle_specific
