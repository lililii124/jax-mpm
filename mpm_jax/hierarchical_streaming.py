"""Bound particle-stencil storage while retaining the full background grid."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .hierarchical_dynamics import ElasticState, angular_momentum, kinetic_energy, strain_energy
from .hierarchical_grid import (
    UniformGrid, affine_velocity, apic_from_grid, grid_stencil, moment_matrix,
)
from .materials.elastic import kirchhoff_stress


def _blocks(state, batch_size):
    if batch_size < 1:
        raise ValueError("Particle batch size must be positive")
    count = len(state.x)
    padding = (-count) % batch_size
    arrays = []
    for name in ("x", "v", "F", "B", "mass", "volume"):
        value = getattr(state, name)
        widths = ((0, padding),) + ((0, 0),) * (value.ndim - 1)
        padded = jnp.pad(value, widths, mode="edge")
        if name in ("mass", "volume"):
            padded = jnp.where(jnp.arange(count + padding) < count, padded, 0.)
        arrays.append(padded.reshape((-1, batch_size) + value.shape[1:]))
    return tuple(arrays)


def _state(arrays, parent):
    return ElasticState(*arrays, parent.time, parent.valid)


def initial_state_streamed(x, volume, density, grid, *, velocity=None, deformation=None,
                           affine=None, batch_size=4096):
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
    temporary = ElasticState(x, v, F, C, density*volume, volume, jnp.asarray(0., x.dtype), jnp.asarray(True))
    blocks = _blocks(temporary, batch_size)

    @jax.jit
    def initialize(arrays, current_grid):
        def one(block):
            support = grid_stencil(block[0], current_grid)
            return block[3] @ moment_matrix(support), support.valid
        return jax.lax.map(one, arrays)

    B, valid = initialize(blocks, grid)
    if not bool(jnp.all(valid)):
        raise ValueError("Initial particles need complete grid support")
    return temporary._replace(B=B.reshape((-1, 3, 3))[:len(x)])


@partial(jax.jit, static_argnames=("batch_size",))
def streamed_energy(state, material, grid, *, batch_size=4096):
    def one(total, arrays):
        block = _state(arrays, state)
        D = moment_matrix(grid_stencil(block.x, grid))
        return total + kinetic_energy(block, D) + strain_energy(block, material), None
    return jax.lax.scan(one, jnp.asarray(0., state.x.dtype), _blocks(state, batch_size))[0]


@partial(jax.jit, static_argnames=("batch_size", "boundaries"))
def streamed_elastic_step(state, material, grid, dt, *, batch_size=4096,
                          gravity=(0., 0., 0.), boundaries=()):
    blocks = _blocks(state, batch_size)
    scalar = jnp.asarray(0., state.x.dtype)
    initial = (jnp.zeros(len(grid.positions), state.x.dtype), jnp.zeros_like(grid.positions),
               jnp.zeros_like(grid.positions), scalar, scalar, state.valid)

    def to_grid(carry, arrays):
        mass, momentum, force, K, U, valid = carry
        block = _state(arrays, state)
        support = grid_stencil(block.x, grid)
        D = moment_matrix(support)
        C = affine_velocity(block.B, D)
        local_v = block.v[:, None, :] + jnp.einsum("pij,pkj->pki", C, support.displacements)
        weighted_mass = block.mass[:, None] * support.weights
        tau = kirchhoff_stress(block.F, material, "neo-hookean")
        contribution = -block.volume[:, None, None] * jnp.einsum("pij,pkj->pki", tau, support.gradients)
        return (mass.at[support.indices].add(weighted_mass),
                momentum.at[support.indices].add(weighted_mass[..., None] * local_v),
                force.at[support.indices].add(contribution),
                K + kinetic_energy(block, D), U + strain_energy(block, material),
                valid & support.valid), None

    (mass, momentum, force, K0, U0, valid), _ = jax.lax.scan(to_grid, initial, blocks)
    inverse_mass = 1. / jnp.where(mass > 0, mass, 1.)
    v0 = momentum * inverse_mass[:, None]
    body_force = mass[:, None] * jnp.asarray(gravity, state.x.dtype)
    trial = v0 + dt * (force + body_force) * inverse_mass[:, None]
    trial = jnp.where((mass > 0)[:, None], trial, 0.)
    constrained = trial
    for boundary in boundaries:
        constrained = boundary.apply(constrained, grid.positions, state.time + dt)
    constrained = jnp.where((mass > 0)[:, None], constrained, 0.)

    def from_grid(_, arrays):
        block = _state(arrays, state)
        support = grid_stencil(block.x, grid)
        D = moment_matrix(support)
        velocity, B, L = apic_from_grid(constrained, support)
        F = (jnp.eye(3, dtype=state.x.dtype) + dt*L) @ block.F
        x = block.x + dt*velocity
        following = grid_stencil(x, grid)
        next_D = moment_matrix(following)
        updated = block._replace(x=x, v=velocity, F=F, B=B, time=state.time+dt)
        block_valid = following.valid & jnp.all(jnp.linalg.det(F) > 0)
        block_valid &= jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in (x, velocity, F, B)]))
        tau = kirchhoff_stress(block.F, material, "neo-hookean")
        stress_work = dt*jnp.sum(block.volume[:, None, None]*tau*L)
        J = jnp.linalg.det(block.F)
        stiffness_bound = material.mu*jnp.sum(block.F**2, axis=(-2, -1))
        stiffness_bound += jnp.maximum(material.lam*(1.-jnp.log(J))+material.mu, 0.)
        speed_bound = jnp.sqrt(stiffness_bound*block.volume/jnp.where(block.mass > 0, block.mass, 1.))
        h_min = grid.spacing if isinstance(grid, UniformGrid) else grid.spacing/2
        cfl = dt*jnp.max(jnp.where(block.mass > 0, speed_bound+jnp.linalg.norm(block.v, axis=-1), 0.))/h_min
        energy = jnp.stack((kinetic_energy(updated, next_D), strain_energy(updated, material),
                            kinetic_energy(updated, D), stress_work))
        measures = (energy, block_valid, cfl, jnp.min(jnp.linalg.det(F)))
        return None, ((x, velocity, F, B), measures)

    _, (fields, measures) = jax.lax.scan(from_grid, None, blocks)
    arrays = [field.reshape((-1,) + field.shape[2:])[:len(state.x)] for field in fields]
    K1, U1, K_old_moment, stress_work = jnp.sum(measures[0], axis=0)
    valid &= jnp.all(measures[1]) & (dt > 0)
    next_state = state._replace(x=arrays[0], v=arrays[1], F=arrays[2], B=arrays[3],
                                time=state.time+dt, valid=valid)
    grid_ke = lambda v: .5*jnp.sum(mass[:, None]*v**2)
    change = constrained-trial
    boundary_work = jnp.sum(mass[:, None]*change*constrained)
    boundary_loss = .5*jnp.sum(mass[:, None]*change**2)
    midpoint = .5*(v0+trial)
    body_work = dt*jnp.sum(body_force*midpoint)
    material_remainder = U1-U0-stress_work
    force_time_remainder = dt*jnp.sum(force*(midpoint-constrained))
    force_gradient_remainder = dt*jnp.sum(force*constrained)+stress_work
    p2g_change, g2p_change = grid_ke(v0)-K0, K_old_moment-grid_ke(constrained)
    moment_advection = K1-K_old_moment
    accounted = (body_work+boundary_work-boundary_loss+material_remainder+force_time_remainder
                 + force_gradient_remainder+p2g_change+g2p_change+moment_advection)
    record = {
        "valid": valid, "cfl_bound": jnp.max(measures[2]), "kinetic_energy": K1, "strain_energy": U1,
        "p2g_change": p2g_change, "g2p_change": g2p_change, "moment_advection": moment_advection,
        "body_work": body_work, "boundary_work": boundary_work, "boundary_loss": boundary_loss,
        "material_remainder": material_remainder, "force_time_remainder": force_time_remainder,
        "force_gradient_remainder": force_gradient_remainder,
        "ledger_error": K1+U1-K0-U0-accounted,
        "linear_momentum": jnp.sum(state.mass[:, None]*next_state.v, axis=0),
        "angular_momentum": angular_momentum(next_state), "min_J": jnp.min(measures[3]),
    }
    return next_state, record


@partial(jax.jit, static_argnames=("steps", "batch_size", "boundaries"))
def advance_streamed(state, material, grid, dt, *, steps, batch_size=4096,
                     gravity=(0., 0., 0.), boundaries=()):
    return jax.lax.scan(lambda carry, _: streamed_elastic_step(carry, material, grid, dt,
        batch_size=batch_size, gravity=gravity, boundaries=boundaries), state, None, length=steps)
