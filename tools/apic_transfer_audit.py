"""APIC-compatible grid traces and their force adjoints."""

import jax.numpy as jnp

from mpm_jax.core import grid_to_particle


def velocity_projection(velocity, support, particle_mass, grid_mass, config):
    """One G2P/P2G round trip on a fixed stencil, including affine velocities.

    This is a mass-self-adjoint contraction, not an idempotent projection.
    Fields can be split by remapping support.indices before calling it.
    """
    v,C=grid_to_particle(velocity,support,config)
    affine=jnp.einsum('pij,pkj->pki',C,support.dpos)
    momentum=jnp.zeros_like(velocity).at[support.indices].add(
        support.weights[...,None]*particle_mass[:,None,None]*(v[:,None,:]+affine))
    return momentum/jnp.where(grid_mass>0,grid_mass,1.)[:,None]


def force_adjoint(force, support, particle_mass, grid_mass, config):
    """Euclidean force transpose P.T = M P M^-1 on occupied nodes."""
    acceleration=force/jnp.where(grid_mass>0,grid_mass,1.)[:,None]
    return grid_mass[:,None]*velocity_projection(acceleration,support,particle_mass,grid_mass,config)
