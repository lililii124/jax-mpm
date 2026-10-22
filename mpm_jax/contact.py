"""Separate-grid two-body contact with balanced normal and Coulomb impulses."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp

from .core import grid_boundaries, grid_update, particle_to_grid, particle_update, stencil


def contact_impulse(mass_a, mass_b, velocity_a, velocity_b, normal, friction=0.0):
    """normal points from body A to B. Returns updated velocities and impulse on B."""
    norm = jnp.linalg.norm(normal, axis=-1)
    normal = normal / jnp.maximum(norm, 1e-30)[..., None]
    active = (mass_a > 0) & (mass_b > 0) & (norm > 0)
    reduced_mass = mass_a * mass_b / jnp.maximum(mass_a + mass_b, 1e-30)
    relative = velocity_a - velocity_b
    closing = jnp.sum(relative * normal, axis=-1)
    normal_impulse = reduced_mass * jnp.maximum(closing, 0.0)
    tangent = relative - closing[..., None] * normal
    tangential_speed = jnp.linalg.norm(tangent, axis=-1)
    friction_impulse = jnp.minimum(reduced_mass * tangential_speed, friction * normal_impulse)
    impulse = (
        normal_impulse[..., None] * normal
        + (friction_impulse / jnp.maximum(tangential_speed, 1e-30))[..., None] * tangent
    )
    impulse = jnp.where(active[..., None], impulse, 0.0)
    va = velocity_a - impulse / jnp.where(mass_a > 0, mass_a, 1.0)[..., None]
    vb = velocity_b + impulse / jnp.where(mass_b > 0, mass_b, 1.0)[..., None]
    return va, vb, impulse


def _mass_gradient(state, support, size):
    return (
        jnp.zeros((size, state.x.shape[-1]), dtype=state.x.dtype)
        .at[support.indices]
        .add(-state.mass[:, None, None] * support.gradients)
    )


@partial(jax.jit, static_argnames=("configs",))
def coupled_step(states, materials, configs, friction=0.0, forces=(None, None)):
    """Two deformable bodies, including inviscid fluid/solid pairs; no tensile adhesion."""
    ca, cb = configs
    if (ca.n_grid, ca.domain_size, ca.dim, ca.dt) != (cb.n_grid, cb.domain_size, cb.dim, cb.dt):
        raise ValueError("coupled bodies must share grid, dimension, domain and time step")
    supports = tuple(stencil(s.x, c) for s, c in zip(states, configs))
    grids = tuple(
        particle_to_grid(s, m, c, support, force)
        for s, m, c, support, force in zip(states, materials, configs, supports, forces)
    )
    velocities = tuple(
        grid_update(mass, momentum, replace(c, boundary_cells=0, boundaries=()))
        for (mass, momentum), c in zip(grids, configs)
    )
    ma, mb = grids[0][0], grids[1][0]
    ga, gb = tuple(_mass_gradient(s, support, ma.shape[0]) for s, support in zip(states, supports))
    normal = gb / jnp.where(mb > 0, mb, 1.0)[:, None] - ga / jnp.where(ma > 0, ma, 1.0)[:, None]
    center = tuple(jnp.sum(s.mass[:, None] * s.x, axis=0) / jnp.sum(s.mass) for s in states)
    normal = jnp.where(
        (jnp.linalg.norm(normal, axis=-1) > 1e-12)[:, None], normal, center[1] - center[0]
    )
    va, vb, impulse = contact_impulse(ma, mb, *velocities, normal, jnp.maximum(friction, 0.0))
    result = tuple(
        particle_update(s, m, c, support, grid_boundaries(v, c, s.time + c.dt))
        for s, m, c, support, v in zip(states, materials, configs, supports, (va, vb))
    )
    return result, jnp.sum(impulse, axis=0)


@partial(jax.jit, static_argnames=("configs", "steps"))
def coupled_advance(states, materials, configs, steps, friction=0.0, forces=(None, None)):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    def body(carry, _):
        states, impulse = carry
        states, change = coupled_step(states, materials, configs, friction, forces)
        return (states, impulse + change), None

    zero = jnp.zeros(configs[0].dim, dtype=states[0].x.dtype)
    return jax.lax.scan(body, (states, zero), None, length=steps)[0]
