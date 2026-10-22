"""Particle initialization and diagnostics."""

import jax.numpy as jnp
import numpy as np

from .core import Config, State


def block(
    config: Config,
    lower=(0.35, 0.55),
    upper=(0.65, 0.85),
    *,
    particles_per_cell=2,
    density=1000.0,
    velocity=(0.0, 0.0),
    dtype=jnp.float32,
):
    """Seed a rectangular block at cell-centered regular sample positions.

    particles_per_cell is the count per axis (2 means 4 particles/cell).
    Reference volumes sum exactly to the prescribed rectangle's area.
    """
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        lo.shape != (2,)
        or hi.shape != (2,)
        or not np.all(np.isfinite([lo, hi]))
        or np.any(hi <= lo)
    ):
        raise ValueError("lower and upper must define a finite, positive 2D rectangle")
    if np.any(lo < config.dx) or np.any(hi > config.domain_size - config.dx):
        raise ValueError("block must lie at least one grid cell inside the domain")
    if not isinstance(particles_per_cell, int) or particles_per_cell < 1:
        raise ValueError("particles_per_cell must be a positive integer")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")
    vel = np.asarray(velocity, dtype=float)
    if vel.shape != (2,) or not np.all(np.isfinite(vel)):
        raise ValueError("velocity must contain two finite values")
    counts = np.maximum(1, np.ceil((hi - lo) / config.dx * particles_per_cell).astype(int))
    axes = [lo[d] + (np.arange(counts[d]) + 0.5) * (hi[d] - lo[d]) / counts[d] for d in range(2)]
    x = jnp.asarray(
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2), dtype=dtype
    )
    n = x.shape[0]
    volume = jnp.full(n, float(np.prod(hi - lo)) / n, dtype=dtype)
    return State(
        x,
        jnp.broadcast_to(jnp.asarray(vel, dtype=dtype), (n, 2)),
        jnp.broadcast_to(jnp.eye(2, dtype=dtype), (n, 2, 2)),
        jnp.zeros((n, 2, 2), dtype=dtype),
        density * volume,
        volume,
        jnp.asarray(True),
    )


def validate_state(state, config):
    """Synchronize and reject invalid states outside JIT; call per output frame."""
    if not bool(state.valid):
        raise FloatingPointError(
            "MPM became invalid: inverted deformation, nonfinite values or particles outside the grid. Reduce dt and inspect the loading."
        )
    arrays = [np.asarray(a) for a in state[:-1]]
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError("MPM state contains NaN or infinity")
    x, _, F, _, mass, volume = arrays
    if np.any(mass <= 0) or np.any(volume <= 0):
        raise ValueError("particle mass and reference volume must be positive")
    if np.any(np.linalg.det(F) <= 0):
        raise FloatingPointError("nonpositive deformation determinant")
    if np.any(x < 0.5 * config.dx) or np.any(x >= config.domain_size - 0.5 * config.dx):
        raise FloatingPointError("particle stencil extends outside the grid")


def diagnostics(state, config):
    validate_state(state, config)
    x, v, F, _, mass, volume, _ = (np.asarray(a) for a in state)
    J = np.linalg.det(F)
    momentum = np.sum(mass[:, None] * v, axis=0)
    return {
        "mass": float(mass.sum()),
        "momentum_x": float(momentum[0]),
        "momentum_y": float(momentum[1]),
        "kinetic_energy": float(0.5 * np.sum(mass[:, None] * v**2)),
        "com_x": float(np.sum(mass * x[:, 0]) / mass.sum()),
        "com_y": float(np.sum(mass * x[:, 1]) / mass.sum()),
        "min_J": float(J.min()),
        "max_J": float(J.max()),
        "current_volume": float(np.sum(volume * J)),
        "max_speed": float(np.linalg.norm(v, axis=1).max()),
    }
