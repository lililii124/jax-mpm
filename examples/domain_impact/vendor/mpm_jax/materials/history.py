"""History and kinematics shared by incremental three-dimensional models."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class History(NamedTuple):
    stress: jax.Array
    plastic_tensor: jax.Array
    backstress: jax.Array
    plastic_strain: jax.Array
    plastic_rate: jax.Array
    damage: jax.Array
    temperature: jax.Array
    plastic_volume: jax.Array
    max_compression: jax.Array
    compaction: jax.Array
    alpha: jax.Array
    hardening: jax.Array
    postpeak: jax.Array
    bulking: jax.Array
    energy: jax.Array
    work: jax.Array
    pressure: jax.Array
    density: jax.Array
    failed: jax.Array
    valid: jax.Array


def initial_history(n, dtype, material=None, density=1000.0, temperature=None):
    zero = jnp.zeros(n, dtype=dtype)
    tensor = jnp.zeros((n, 3, 3), dtype=dtype)
    temperature = (
        getattr(material, "reference_temperature", 293.15) if temperature is None else temperature
    )
    alpha = getattr(material, "alpha0", 1.0)
    damage = getattr(material, "initial_damage", 0.0)
    return History(
        tensor,
        tensor,
        tensor,
        zero,
        zero,
        zero + damage,
        zero + temperature,
        zero,
        zero,
        zero,
        zero + alpha,
        zero,
        jnp.zeros(n, dtype=bool),
        zero,
        zero,
        zero,
        zero,
        zero + density,
        jnp.zeros(n, dtype=bool),
        jnp.ones(n, dtype=bool),
    )


def dev(tensor):
    return tensor - jnp.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(3) / 3


def equivalent(stress):
    return jnp.sqrt(jnp.maximum(1.5 * jnp.sum(dev(stress) ** 2, axis=(-2, -1)), 0))


def equivalent_strain(strain):
    return jnp.sqrt(jnp.maximum(2 / 3 * jnp.sum(dev(strain) ** 2, axis=(-2, -1)), 0))


def corotated_increment(F, previous_F):
    """Incremental right Hencky strain and polar rotation, before constitutive integration."""
    delta = F @ jnp.linalg.inv(previous_F)
    values, vectors = jnp.linalg.eigh(jnp.swapaxes(delta, -1, -2) @ delta)
    safe = jnp.maximum(values, jnp.finfo(F.dtype).tiny)
    transpose = jnp.swapaxes(vectors, -1, -2)
    strain = (vectors * (0.5 * jnp.log(safe))[..., None, :]) @ transpose
    rotation = delta @ ((vectors * (1 / jnp.sqrt(safe))[..., None, :]) @ transpose)
    valid = jnp.all(values > 0, axis=-1) & (jnp.linalg.det(delta) > 0)
    return strain, rotation, valid


def finish_update(old, new, strain, density):
    work = 0.5 * jnp.sum((old.stress + new.stress) * strain, axis=(-2, -1)) / density
    new = new._replace(work=old.work + work, density=density)
    finite = jnp.ones_like(new.valid)
    for leaf in new:
        finite = finite & jnp.all(jnp.isfinite(leaf).reshape(leaf.shape[0], -1), axis=-1)
    valid = (
        old.valid
        & new.valid
        & finite
        & (density > 0)
        & (new.plastic_strain >= 0)
        & (new.damage >= old.damage)
        & (new.damage <= 1)
    )
    return new._replace(valid=valid)


def rotate_history(history, rotation):
    transpose = jnp.swapaxes(rotation, -1, -2)
    return history._replace(
        stress=rotation @ history.stress @ transpose,
        plastic_tensor=rotation @ history.plastic_tensor @ transpose,
        backstress=rotation @ history.backstress @ transpose,
    )
