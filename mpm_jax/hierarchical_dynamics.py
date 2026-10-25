"""Explicit neo-Hookean dynamics on uniform or fixed hierarchical grids."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .hierarchical_grid import (
    UniformGrid, affine_velocity, apic_from_grid, apic_to_grid, grid_stencil,
    internal_force, moment_matrix,
)
from .materials.elastic import kirchhoff_stress


class ElasticState(NamedTuple):
    x: jax.Array
    v: jax.Array
    F: jax.Array
    B: jax.Array
    mass: jax.Array
    volume: jax.Array
    time: jax.Array
    valid: jax.Array


def initial_state(x, volume, density, grid, *, velocity=None, deformation=None, affine=None):
    x = jnp.asarray(x, dtype=grid.positions.dtype)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0:
        raise ValueError("Particle coordinates must have shape (n, 3)")
    volume = jnp.broadcast_to(jnp.asarray(volume, x.dtype), (len(x),))
    density = jnp.broadcast_to(jnp.asarray(density, x.dtype), (len(x),))
    v = jnp.broadcast_to(jnp.zeros(3, x.dtype) if velocity is None else jnp.asarray(velocity, x.dtype), x.shape)
    F = jnp.broadcast_to(jnp.eye(3, dtype=x.dtype) if deformation is None else jnp.asarray(deformation, x.dtype),
                         (len(x), 3, 3))
    C = jnp.broadcast_to(jnp.zeros((3, 3), x.dtype) if affine is None else jnp.asarray(affine, x.dtype), F.shape)
    if not all(np.isfinite(a).all() for a in (x, volume, density, v, F, C)):
        raise ValueError("Initial state must be finite")
    if np.any(np.asarray(volume) <= 0) or np.any(np.asarray(density) <= 0) or np.any(np.linalg.det(F) <= 0):
        raise ValueError("Initial volume, density and deformation determinant must be positive")
    support = grid_stencil(x, grid)
    if not bool(support.valid):
        raise ValueError("Initial particles need complete grid support")
    return ElasticState(x, v, F, C @ moment_matrix(support), density * volume, volume,
                        jnp.asarray(0., x.dtype), jnp.asarray(True))


def kinetic_energy(state, moment):
    C = affine_velocity(state.B, moment)
    return .5 * jnp.sum(state.mass * (jnp.sum(state.v**2, axis=-1)
                                      + jnp.einsum("pij,pij->p", state.B, C)))


def strain_energy(state, material):
    logJ = jnp.log(jnp.linalg.det(state.F))
    density = .5 * material.mu * (jnp.sum(state.F**2, axis=(-2, -1)) - 3.)
    density += -material.mu * logJ + .5 * material.lam * logJ**2
    return jnp.sum(state.volume * density)


def _spin(B):
    return jnp.stack((B[:, 2, 1] - B[:, 1, 2], B[:, 0, 2] - B[:, 2, 0],
                      B[:, 1, 0] - B[:, 0, 1]), -1)


def angular_momentum(state):
    return jnp.sum(state.mass[:, None] * (jnp.cross(state.x, state.v) + _spin(state.B)), axis=0)


@partial(jax.jit, static_argnames=("boundaries",))
def elastic_step(state, material, grid, dt, *, gravity=(0., 0., 0.), boundaries=()):
    support = grid_stencil(state.x, grid)
    D0 = moment_matrix(support)
    mass, momentum = apic_to_grid(state.mass, state.v, state.B, support, grid)
    inverse_mass = 1. / jnp.where(mass > 0, mass, 1.)
    v0 = momentum * inverse_mass[:, None]
    tau = kirchhoff_stress(state.F, material, "neo-hookean")
    force = internal_force(state.volume, tau, support, grid)
    body_force = mass[:, None] * jnp.asarray(gravity, state.x.dtype)
    trial = v0 + dt * (force + body_force) * inverse_mass[:, None]
    trial = jnp.where((mass > 0)[:, None], trial, 0.)
    constrained = trial
    for boundary in boundaries:
        constrained = boundary.apply(constrained, grid.positions, state.time + dt)
    # Empty nodes do not acquire prescribed motion or affect a particle gradient.
    constrained = jnp.where((mass > 0)[:, None], constrained, 0.)
    velocity, B, L = apic_from_grid(constrained, support)
    F = (jnp.eye(3, dtype=state.x.dtype) + dt * L) @ state.F
    x = state.x + dt * velocity
    next_support = grid_stencil(x, grid)
    next_D = moment_matrix(next_support)
    next_state = ElasticState(x, velocity, F, B, state.mass, state.volume, state.time + dt, state.valid)
    valid = state.valid & support.valid & next_support.valid & (dt > 0)
    valid &= jnp.all(jnp.linalg.det(F) > 0)
    valid &= jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in (x, velocity, F, B)]))
    next_state = next_state._replace(valid=valid)

    grid_ke = lambda v: .5 * jnp.sum(mass[:, None] * v**2)
    K0, U0 = kinetic_energy(state, D0), strain_energy(state, material)
    K1, U1 = kinetic_energy(next_state, next_D), strain_energy(next_state, material)
    K_old_moment = kinetic_energy(next_state, D0)
    change = constrained - trial
    boundary_work = jnp.sum(mass[:, None] * change * constrained)
    boundary_loss = .5 * jnp.sum(mass[:, None] * change**2)
    midpoint = .5 * (v0 + trial)
    body_work = dt * jnp.sum(body_force * midpoint)
    stress_work = dt * jnp.sum(state.volume[:, None, None] * tau * L)
    material_remainder = U1 - U0 - stress_work
    force_time_remainder = dt * jnp.sum(force * (midpoint - constrained))
    force_gradient_remainder = dt * jnp.sum(force * constrained) + stress_work
    p2g_change = grid_ke(v0) - K0
    g2p_change = K_old_moment - grid_ke(constrained)
    moment_advection = K1 - K_old_moment
    accounted = (body_work + boundary_work - boundary_loss + material_remainder
                 + force_time_remainder + force_gradient_remainder
                 + p2g_change + g2p_change + moment_advection)
    J = jnp.linalg.det(state.F)
    # A conservative neo-Hookean acoustic bound uses ||F||_F² >= lambda_max(F F.T).
    stiffness_bound = material.mu * jnp.sum(state.F**2, axis=(-2, -1))
    stiffness_bound += jnp.maximum(material.lam * (1. - jnp.log(J)) + material.mu, 0.)
    speed_bound = jnp.sqrt(stiffness_bound * state.volume / state.mass)
    h_min = grid.spacing if isinstance(grid, UniformGrid) else grid.spacing / 2
    cfl = dt * jnp.max(speed_bound + jnp.linalg.norm(state.v, axis=-1)) / h_min
    record = {
        "valid": valid, "cfl_bound": cfl, "kinetic_energy": K1, "strain_energy": U1,
        "p2g_change": p2g_change, "g2p_change": g2p_change, "moment_advection": moment_advection,
        "body_work": body_work, "boundary_work": boundary_work, "boundary_loss": boundary_loss,
        "material_remainder": material_remainder, "force_time_remainder": force_time_remainder,
        "force_gradient_remainder": force_gradient_remainder,
        "ledger_error": K1 + U1 - K0 - U0 - accounted,
        "linear_momentum": jnp.sum(state.mass[:, None] * velocity, axis=0),
        "angular_momentum": angular_momentum(next_state), "min_J": jnp.min(jnp.linalg.det(F)),
    }
    return next_state, record


@partial(jax.jit, static_argnames=("steps", "boundaries"))
def advance(state, material, grid, dt, *, steps, gravity=(0., 0., 0.), boundaries=()):
    return jax.lax.scan(lambda carry, _: elastic_step(carry, material, grid, dt,
                        gravity=gravity, boundaries=boundaries), state, None, length=steps)
