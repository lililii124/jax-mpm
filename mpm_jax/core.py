"""Two- and three-dimensional MLS-MPM with quadratic B-splines and APIC transfers.

Adapted from Taichi's mpm128.py (Apache-2.0). The JAX implementation uses
Neo-Hookean stress and updates deformation after G2P. See NOTICE.
"""

from dataclasses import dataclass
from functools import partial
from itertools import product
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .materials import (
    History,
    INCREMENTAL_MODELS,
    Material,
    elastic,
    fluid,
    kirchhoff_stress,
    update_history,
)


@dataclass(frozen=True)
class Config:
    """Static configuration; n_grid cells per axis, dim = 2 or 3."""

    n_grid: int = 48
    dt: float = 2e-4
    domain_size: float = 1.0
    gravity: tuple[float, ...] | None = None
    boundary_cells: int = 3
    material: str = "elastic"
    dim: int = 2
    boundaries: tuple = ()
    failure_mode: str = "retain"

    def __post_init__(self):
        if not isinstance(self.n_grid, int) or self.n_grid < 8:
            raise ValueError("n_grid must be an integer >= 8")
        if not isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not isfinite(self.domain_size) or self.domain_size <= 0:
            raise ValueError("domain_size must be finite and positive")
        if (
            not isinstance(self.boundary_cells, int)
            or not 0 <= self.boundary_cells < self.n_grid // 2
        ):
            raise ValueError("boundary_cells must be an integer in [0, n_grid//2)")
        if (
            self.material
            not in (
                "elastic",
                "fluid",
                "neo-hookean",
                "linear",
                "orthotropic",
                "transverse-isotropic",
                "blatz-ko",
            )
            + INCREMENTAL_MODELS
        ):
            raise ValueError(f"unknown material: {self.material}")
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if self.failure_mode not in ("retain", "ballistic"):
            raise ValueError("failure_mode must be 'retain' or 'ballistic'")
        if self.material in INCREMENTAL_MODELS and self.dim != 3:
            raise ValueError("incremental plasticity models require dim=3")
        gravity = (
            self.gravity if self.gravity is not None else (0.0, -9.81) + (0.0,) * (self.dim - 2)
        )
        if len(gravity) != self.dim or not all(isfinite(g) for g in gravity):
            raise ValueError("gravity must contain dim finite values")
        object.__setattr__(self, "gravity", tuple(gravity))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        for boundary in self.boundaries:
            boundary.validate(self.dim, self.domain_size)

    @property
    def dx(self):
        return self.domain_size / self.n_grid


class State(NamedTuple):
    x: jax.Array  # (particles, dim)
    v: jax.Array
    F: jax.Array  # (particles, dim, dim)
    C: jax.Array
    mass: jax.Array
    volume: jax.Array  # reference volume; unit thickness in 2D
    history: History
    time: jax.Array
    valid: jax.Array


class Stencil(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    dpos: jax.Array
    valid: jax.Array
    gradients: jax.Array


def particle_activity(state, config):
    """Particles participating in grid transfers and grid contact."""
    if config.failure_mode == "ballistic":
        return ~state.history.failed
    return jnp.ones(state.mass.shape, dtype=bool)


def particle_stencil(state, config):
    active = particle_activity(state, config) if config.failure_mode == "ballistic" else None
    return stencil(state.x, config, active)


def stencil(x, config, active=None):
    """3**dim node support; mask invalid indices to prevent wrapping."""
    if active is not None:
        x = jnp.where(active[:, None], x, 0.5 * config.domain_size)
    q = x / config.dx
    base = jnp.floor(q - 0.5).astype(jnp.int32)
    f = q - base
    w = jnp.stack((0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2), axis=1)
    offsets = jnp.array(list(product(range(3), repeat=config.dim)), dtype=jnp.int32)
    nodes = base[:, None, :] + offsets[None, :, :]
    inside = jnp.all((nodes >= 0) & (nodes <= config.n_grid), axis=-1)
    weights = jnp.prod(w[:, offsets, jnp.arange(config.dim)], axis=-1)
    weights = jnp.where(inside, weights, 0)
    safe = jnp.clip(nodes, 0, config.n_grid)
    strides = (config.n_grid + 1) ** jnp.arange(config.dim - 1, -1, -1)
    indices = jnp.sum(safe * strides, axis=-1)
    dw = jnp.stack((f - 1.5, 2 * (1 - f), f - 0.5), axis=1) / config.dx
    axis_weights = w[:, offsets, jnp.arange(config.dim)]
    gradients = []
    for d in range(config.dim):
        factors = axis_weights.at[..., d].set(dw[:, offsets[:, d], d])
        gradients.append(jnp.prod(factors, axis=-1))
    gradients = jnp.where(inside[..., None], jnp.stack(gradients, axis=-1), 0.0)
    dpos = (offsets[None] - f[:, None]) * config.dx
    if active is not None:
        weights = jnp.where(active[:, None], weights, 0.0)
        gradients = jnp.where(active[:, None, None], gradients, 0.0)
        dpos = jnp.where(active[:, None, None], dpos, 0.0)
    return Stencil(indices, weights, dpos, jnp.all(inside), gradients)


def particle_to_grid(state, material, config, support=None, force=None):
    support = particle_stencil(state, config) if support is None else support
    if config.material in INCREMENTAL_MODELS:
        tau = jnp.linalg.det(state.F)[:, None, None] * state.history.stress
    else:
        tau = kirchhoff_stress(state.F, material, config.material)
    affine = (
        state.mass[:, None, None] * state.C
        - (4 * config.dt / config.dx**2) * state.volume[:, None, None] * tau
    )
    momentum = state.mass[:, None, None] * state.v[:, None, :] + jnp.einsum(
        "pij,pkj->pki", affine, support.dpos
    )
    if force is not None:
        momentum = momentum + config.dt * force[:, None, :]
    mass_contribution = support.weights * state.mass[:, None]
    momentum_contribution = support.weights[..., None] * momentum
    if config.failure_mode == "ballistic":
        active = particle_activity(state, config)
        mass_contribution = jnp.where(active[:, None], mass_contribution, 0.0)
        momentum_contribution = jnp.where(active[:, None, None], momentum_contribution, 0.0)
    size = (config.n_grid + 1) ** config.dim
    grid_mass = jnp.zeros(size, dtype=state.x.dtype).at[support.indices].add(mass_contribution)
    grid_momentum = (
        jnp.zeros((size, config.dim), dtype=state.x.dtype)
        .at[support.indices]
        .add(momentum_contribution)
    )
    return grid_mass, grid_momentum


def grid_update(mass, momentum, config, time=0.0):
    occupied = mass > 0
    velocity = momentum / jnp.where(occupied, mass, 1)[:, None]
    velocity = jnp.where(
        occupied[:, None],
        velocity + config.dt * jnp.asarray(config.gravity, dtype=velocity.dtype),
        0,
    )
    return grid_boundaries(velocity, config, time)


def grid_boundaries(velocity, config, time=0.0):
    if config.boundary_cells or config.boundaries:
        node = jnp.arange((config.n_grid + 1) ** config.dim)
        strides = (config.n_grid + 1) ** jnp.arange(config.dim - 1, -1, -1)
        coordinates = (node[:, None] // strides) % (config.n_grid + 1)
    if config.boundary_cells:
        outward = ((coordinates < config.boundary_cells) & (velocity < 0)) | (
            (coordinates > config.n_grid - config.boundary_cells) & (velocity > 0)
        )
        velocity = jnp.where(outward, 0, velocity)
    for boundary in config.boundaries:
        velocity = boundary.apply(velocity, coordinates * config.dx, time)
    return velocity


def grid_to_particle(grid_velocity, support, config):
    gathered = grid_velocity[support.indices]
    velocity = jnp.sum(support.weights[..., None] * gathered, axis=1)
    affine = (4 / config.dx**2) * jnp.einsum(
        "pk,pki,pkj->pij", support.weights, gathered, support.dpos
    )
    return velocity, affine


@partial(jax.jit, static_argnames=("config",))
def step(state, material, config, force=None):
    """One explicit MLS-MPM step."""
    support = particle_stencil(state, config)
    gm, gp = particle_to_grid(state, material, config, support, force)
    gv = grid_update(gm, gp, config, state.time + config.dt)
    return particle_update(state, material, config, support, gv, force)


def particle_update(state, material, config, support, gv, force=None):
    v, C = grid_to_particle(gv, support, config)
    F = (jnp.eye(config.dim, dtype=state.x.dtype) + config.dt * C) @ state.F
    if config.failure_mode == "ballistic":
        active = particle_activity(state, config)
        acceleration = jnp.asarray(config.gravity, dtype=state.v.dtype)
        if force is not None:
            acceleration = acceleration + force / state.mass[:, None]
        v = jnp.where(active[:, None], v, state.v + config.dt * acceleration)
        C = jnp.where(active[:, None, None], C, 0.0)
        F = jnp.where(active[:, None, None], F, state.F)
    J = jnp.linalg.det(F)
    valid = state.valid & support.valid & jnp.all(J > 0) & jnp.all(jnp.linalg.det(state.F) > 0)
    if config.material == "fluid":
        # Store volume change only; an inviscid fluid carries no shear memory.
        F = jnp.maximum(J, jnp.finfo(F.dtype).tiny)[:, None, None] ** (1 / config.dim) * jnp.eye(
            config.dim, dtype=F.dtype
        )
        if config.failure_mode == "ballistic":
            F = jnp.where(active[:, None, None], F, state.F)
    x = state.x + config.dt * v
    # Require the full quadratic support to remain inside the grid.
    inside = jnp.all((x >= 0.5 * config.dx) & (x < config.domain_size - 0.5 * config.dx), axis=-1)
    if config.failure_mode == "ballistic":
        inside = inside | ~active
    valid = valid & jnp.all(inside)
    valid = (
        valid
        & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(F))
        & jnp.all(jnp.isfinite(C))
    )
    history = state.history
    if config.material in INCREMENTAL_MODELS:
        history = update_history(history, F, state.F, config.dt, material, config.material)
        if config.failure_mode == "ballistic":
            history = jax.tree.map(
                lambda before, after: jnp.where(
                    active.reshape((-1,) + (1,) * (after.ndim - 1)), after, before
                ),
                state.history,
                history,
            )
        valid = valid & jnp.all(history.valid)
    if config.failure_mode == "ballistic":
        C = jnp.where(history.failed[:, None, None], 0.0, C)
    return State(x, v, F, C, state.mass, state.volume, history, state.time + config.dt, valid)


@partial(jax.jit, static_argnames=("config", "steps"))
def advance(state, material, config, steps, force=None):
    """Advance without storing a trajectory. Elastic branches support reverse-mode AD."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    def body(carry, _):
        return step(carry, material, config, force), None

    return jax.lax.scan(body, state, None, length=steps)[0]
