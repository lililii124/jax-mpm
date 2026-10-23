"""Particle initialization and diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np

from .core import Config, State, particle_activity
from .materials import initial_history, INCREMENTAL_MODELS


def block(
    config: Config,
    lower=None,
    upper=None,
    *,
    particles_per_cell=2,
    density=1000.0,
    velocity=None,
    dtype=jnp.float32,
    material=None,
    temperature=None,
):
    """Seed a rectangular block at regularly spaced particle centers.

    particles_per_cell is the count per axis (2 means 8 particles/cell in 3D).
    Reference volumes sum to the prescribed volume (area in 2D).
    """
    dim = config.dim
    lower = (0.35, 0.55) + (0.35,) * (dim - 2) if lower is None else lower
    upper = (0.65, 0.85) + (0.65,) * (dim - 2) if upper is None else upper
    velocity = (0.0,) * dim if velocity is None else velocity
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (
        lo.shape != (dim,)
        or hi.shape != (dim,)
        or not np.all(np.isfinite([lo, hi]))
        or np.any(hi <= lo)
    ):
        raise ValueError("lower and upper must define a finite, positive box matching config.dim")
    if np.any(lo < config.dx) or np.any(hi > config.domain_size - config.dx):
        raise ValueError("block must lie at least one grid cell inside the domain")
    if not isinstance(particles_per_cell, int) or particles_per_cell < 1:
        raise ValueError("particles_per_cell must be a positive integer")
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive")
    vel = np.asarray(velocity, dtype=float)
    if vel.shape != (dim,) or not np.all(np.isfinite(vel)):
        raise ValueError("velocity must contain dim finite values")
    counts = np.maximum(1, np.ceil((hi - lo) / config.dx * particles_per_cell).astype(int))
    axes = [lo[d] + (np.arange(counts[d]) + 0.5) * (hi[d] - lo[d]) / counts[d] for d in range(dim)]
    x = jnp.asarray(
        np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim), dtype=dtype
    )
    n = x.shape[0]
    if config.material in INCREMENTAL_MODELS and material is None:
        raise ValueError("pass material=... when initializing a history-dependent material")
    if (
        material is not None
        and hasattr(material, "density")
        and not np.isclose(density, material.density)
    ):
        raise ValueError("particle density must equal the constitutive reference density")
    volume = jnp.full(n, float(np.prod(hi - lo)) / n, dtype=dtype)
    return State(
        x,
        jnp.broadcast_to(jnp.asarray(vel, dtype=dtype), (n, dim)),
        jnp.broadcast_to(jnp.eye(dim, dtype=dtype), (n, dim, dim)),
        jnp.zeros((n, dim, dim), dtype=dtype),
        density * volume,
        volume,
        initial_history(n, dtype, material, density, temperature),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(True),
    )


def validate_state(state, config):
    """Synchronize and reject invalid states outside JIT; call per output frame."""
    if not bool(state.valid):
        raise FloatingPointError(
            "MPM became invalid: inverted deformation, nonfinite values or particles outside the grid. Reduce dt and inspect the loading."
        )
    arrays = [np.asarray(a) for a in jax.tree.leaves(state)]
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError("MPM state contains NaN or infinity")
    x, F, mass, volume = (np.asarray(a) for a in (state.x, state.F, state.mass, state.volume))
    if np.any(mass <= 0) or np.any(volume <= 0):
        raise ValueError("particle mass and reference volume must be positive")
    if np.any(np.linalg.det(F) <= 0):
        raise FloatingPointError("nonpositive deformation determinant")
    active_x = x[np.asarray(particle_activity(state, config))]
    if np.any(active_x < 0.5 * config.dx) or np.any(
        active_x >= config.domain_size - 0.5 * config.dx
    ):
        raise FloatingPointError("particle stencil extends outside the grid")


def diagnostics(state, config, material=None):
    validate_state(state, config)
    x, v, F, mass, volume = (
        np.asarray(a) for a in (state.x, state.v, state.F, state.mass, state.volume)
    )
    J = np.linalg.det(F)
    momentum = np.sum(mass[:, None] * v, axis=0)
    result = {
        "mass": float(mass.sum()),
        **{f"momentum_{axis}": float(momentum[d]) for d, axis in enumerate("xyz"[: config.dim])},
        "kinetic_energy": float(0.5 * np.sum(mass[:, None] * v**2)),
        **{
            f"com_{axis}": float(np.sum(mass * x[:, d]) / mass.sum())
            for d, axis in enumerate("xyz"[: config.dim])
        },
        "min_J": float(J.min()),
        "max_J": float(J.max()),
        "current_volume": float(np.sum(volume * J)),
        "max_speed": float(np.linalg.norm(v, axis=1).max()),
        "max_plastic_strain": float(jnp.max(state.history.plastic_strain)),
        "max_damage": float(jnp.max(state.history.damage)),
        "max_temperature": float(jnp.max(state.history.temperature)),
    }
    result["gravitational_energy"] = float(-np.sum(mass[:, None] * x * config.gravity))
    result["constitutive_work"] = float(np.sum(mass * np.asarray(state.history.work)))
    if config.failure_mode == "ballistic":
        active = np.asarray(particle_activity(state, config))
        result.update(
            active_particles=int(active.sum()),
            debris_particles=int((~active).sum()),
            active_mass=float(mass[active].sum()),
            debris_mass=float(mass[~active].sum()),
            active_current_volume=float(np.sum((volume * J)[active])),
        )
    if material is not None and config.material not in INCREMENTAL_MODELS:
        from .materials import kirchhoff_stress

        kind = config.material
        if kind in ("elastic", "neo-hookean"):
            W = 0.5 * material.mu * (np.sum(F * F, axis=(-2, -1)) - config.dim)
            W -= material.mu * np.log(J)
            W += 0.5 * material.lam * np.log(J) ** 2
        elif kind == "fluid":
            W = 0.5 * material.bulk * (J - 1) ** 2
        elif kind == "blatz-ko":
            power = 1 - 1 / (1 - 2 * material.poisson)
            W = 0.5 * material.shear * (np.sum(F * F, axis=(-2, -1)) - config.dim)
            W -= material.shear / power * (J**power - 1)
        else:
            strain = 0.5 * (F + F.swapaxes(-2, -1)) - np.eye(config.dim)
            stress = np.asarray(kirchhoff_stress(state.F, material, kind)) / J[:, None, None]
            W = 0.5 * np.sum(strain * stress, axis=(-2, -1))
        result["strain_energy"] = float(np.sum(volume * W))
    else:
        result["strain_energy"] = None
    return result
