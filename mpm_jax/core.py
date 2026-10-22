"""Two-dimensional MLS-MPM with quadratic B-splines and APIC transfers.

Adapted from Taichi's mpm128.py (Apache-2.0). The JAX implementation uses
Neo-Hookean stress and updates deformation after G2P. See NOTICE.
"""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    """Static configuration; n_grid cells give (n_grid + 1)^2 nodes."""

    n_grid: int = 48
    dt: float = 2e-4
    domain_size: float = 1.0
    gravity: tuple[float, float] = (0.0, -9.81)
    boundary_cells: int = 3
    material: str = "elastic"

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
        if self.material not in ("elastic", "fluid"):
            raise ValueError("material must be 'elastic' or 'fluid'")
        if len(self.gravity) != 2 or not all(isfinite(g) for g in self.gravity):
            raise ValueError("gravity must contain two finite values")
        object.__setattr__(self, "gravity", tuple(self.gravity))

    @property
    def dx(self):
        return self.domain_size / self.n_grid


class Material(NamedTuple):
    """Differentiable material parameters, shared by all particles."""

    mu: jax.Array | float
    lam: jax.Array | float
    bulk: jax.Array | float


def elastic(young=2e4, poisson=0.2):
    """Plane-strain Lame parameters for young > 0 and -1 < poisson < 0.5."""
    return Material(
        young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson)), 0.0
    )


def fluid(bulk=2e5):
    """Compressible inviscid fluid, W(J)=bulk/2*(J-1)^2."""
    return Material(0.0, 0.0, bulk)


class State(NamedTuple):
    x: jax.Array  # (particles, 2), current coordinates
    v: jax.Array  # (particles, 2)
    F: jax.Array  # (particles, 2, 2), deformation gradient
    C: jax.Array  # (particles, 2, 2), APIC affine velocity
    mass: jax.Array  # (particles,), fixed mass
    volume: jax.Array  # (particles,), reference area, unit thickness
    valid: jax.Array  # scalar bool; false stays false across time steps


class Stencil(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    dpos: jax.Array
    valid: jax.Array


def stencil(x, config):
    """Nine-node support; mask invalid indices to prevent wrapping."""
    q = x / config.dx
    base = jnp.floor(q - 0.5).astype(jnp.int32)
    f = q - base
    w = jnp.stack((0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2), axis=1)
    offsets = jnp.array([(i, j) for i in range(3) for j in range(3)], dtype=jnp.int32)
    nodes = base[:, None, :] + offsets[None, :, :]
    inside = jnp.all((nodes >= 0) & (nodes <= config.n_grid), axis=-1)
    weights = w[:, offsets[:, 0], 0] * w[:, offsets[:, 1], 1]
    weights = jnp.where(inside, weights, 0)
    safe = jnp.clip(nodes, 0, config.n_grid)
    indices = safe[..., 0] * (config.n_grid + 1) + safe[..., 1]
    return Stencil(indices, weights, (offsets[None] - f[:, None]) * config.dx, jnp.all(inside))


def kirchhoff_stress(F, material, kind):
    """Return tau = P F^T = J sigma (reference-volume stress weighting)."""
    J = jnp.linalg.det(F)
    eye = jnp.eye(2, dtype=F.dtype)
    if kind == "fluid":
        return (material.bulk * J * (J - 1))[:, None, None] * eye
    B = F @ jnp.swapaxes(F, -1, -2)
    # step flags inverted F; the floor keeps log defined until validation.
    logJ = jnp.log(jnp.maximum(J, jnp.finfo(F.dtype).tiny))
    return material.mu * (B - eye) + material.lam * logJ[:, None, None] * eye


def particle_to_grid(state, material, config, support=None):
    support = stencil(state.x, config) if support is None else support
    tau = kirchhoff_stress(state.F, material, config.material)
    affine = (
        state.mass[:, None, None] * state.C
        - (4 * config.dt / config.dx**2) * state.volume[:, None, None] * tau
    )
    momentum = state.mass[:, None, None] * state.v[:, None, :] + jnp.einsum(
        "pij,pkj->pki", affine, support.dpos
    )
    size = (config.n_grid + 1) ** 2
    grid_mass = (
        jnp.zeros(size, dtype=state.x.dtype)
        .at[support.indices]
        .add(support.weights * state.mass[:, None])
    )
    grid_momentum = (
        jnp.zeros((size, 2), dtype=state.x.dtype)
        .at[support.indices]
        .add(support.weights[..., None] * momentum)
    )
    return grid_mass, grid_momentum


def grid_update(mass, momentum, config):
    occupied = mass > 0
    velocity = momentum / jnp.where(occupied, mass, 1)[:, None]
    velocity = jnp.where(
        occupied[:, None],
        velocity + config.dt * jnp.asarray(config.gravity, dtype=velocity.dtype),
        0,
    )
    if config.boundary_cells:
        node = jnp.arange((config.n_grid + 1) ** 2)
        coordinates = jnp.stack((node // (config.n_grid + 1), node % (config.n_grid + 1)), axis=-1)
        outward = ((coordinates < config.boundary_cells) & (velocity < 0)) | (
            (coordinates > config.n_grid - config.boundary_cells) & (velocity > 0)
        )
        velocity = jnp.where(outward, 0, velocity)
    return velocity


def grid_to_particle(grid_velocity, support, config):
    gathered = grid_velocity[support.indices]
    velocity = jnp.sum(support.weights[..., None] * gathered, axis=1)
    affine = (4 / config.dx**2) * jnp.einsum(
        "pk,pki,pkj->pij", support.weights, gathered, support.dpos
    )
    return velocity, affine


@partial(jax.jit, static_argnames=("config",))
def step(state, material, config):
    """One explicit MLS-MPM step."""
    support = stencil(state.x, config)
    gm, gp = particle_to_grid(state, material, config, support)
    gv = grid_update(gm, gp, config)
    v, C = grid_to_particle(gv, support, config)
    F = (jnp.eye(2, dtype=state.x.dtype) + config.dt * C) @ state.F
    J = jnp.linalg.det(F)
    valid = state.valid & support.valid & jnp.all(J > 0) & jnp.all(jnp.linalg.det(state.F) > 0)
    if config.material == "fluid":
        # Store volume change only; an inviscid fluid carries no shear memory.
        F = jnp.sqrt(jnp.maximum(J, jnp.finfo(F.dtype).tiny))[:, None, None] * jnp.eye(
            2, dtype=F.dtype
        )
    x = state.x + config.dt * v
    # Require the full quadratic support to remain inside the grid.
    valid = valid & jnp.all((x >= 0.5 * config.dx) & (x < config.domain_size - 0.5 * config.dx))
    valid = (
        valid
        & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(v))
        & jnp.all(jnp.isfinite(F))
        & jnp.all(jnp.isfinite(C))
    )
    return State(x, v, F, C, state.mass, state.volume, valid)


@partial(jax.jit, static_argnames=("config", "steps"))
def advance(state, material, config, steps):
    """Advance without storing a trajectory; reverse-mode autodiff supported."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    def body(carry, _):
        return step(carry, material, config), None

    return jax.lax.scan(body, state, None, length=steps)[0]
