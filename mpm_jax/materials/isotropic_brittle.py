"""Isotropic tensile damage with linear crack-band softening."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class IsotropicBrittle(NamedTuple):
    young: float
    poisson: float
    density: float
    tensile_strength: float
    fracture_energy: float
    characteristic_length: jax.Array


class BrittleHistory(NamedTuple):
    kappa: jax.Array
    damage: jax.Array
    failed: jax.Array
    valid: jax.Array


class BrittleScales(NamedTuple):
    onset_strain: jax.Array
    onset_displacement: jax.Array
    failure_displacement: jax.Array
    failure_strain: jax.Array
    maximum_characteristic_length: jax.Array


def isotropic_brittle(
    *,
    young=10e9,
    poisson=0.25,
    density=1500.0,
    tensile_strength=100e6,
    fracture_energy=100.0,
    characteristic_length,
):
    """Build a material; strength and energy refer to the reference cross section."""
    scalars = (young, poisson, density, tensile_strength, fracture_energy)
    if not all(np.ndim(value) == 0 and np.isfinite(value) for value in scalars):
        raise ValueError("elastic and fracture parameters must be finite scalars")
    if min(young, density, tensile_strength, fracture_energy) <= 0:
        raise ValueError("young, density, tensile_strength and fracture_energy must be positive")
    if not -1 < poisson < 0.5:
        raise ValueError("poisson must lie strictly between -1 and 0.5")
    length = np.asarray(characteristic_length)
    if length.ndim > 1 or length.size == 0 or not np.all(np.isfinite(length) & (length > 0)):
        raise ValueError("characteristic_length must be a positive scalar or particle array")
    delta0 = length * tensile_strength / young
    deltaf = 2 * fracture_energy / tensile_strength
    if np.any(deltaf <= delta0):
        raise ValueError("failure displacement must exceed onset displacement: h < 2*E*Gf/ft**2")
    return IsotropicBrittle(
        float(young), float(poisson), float(density), float(tensile_strength),
        float(fracture_energy), jnp.asarray(length),
    )


def brittle_scales(material):
    """Crack-band lengths in metres and dimensionless strain thresholds."""
    onset = material.tensile_strength / material.young
    final = 2 * material.fracture_energy / material.tensile_strength
    return BrittleScales(
        onset,
        material.characteristic_length * onset,
        final,
        final / material.characteristic_length,
        2 * material.young * material.fracture_energy / material.tensile_strength**2,
    )


def initial_brittle_history(count, dtype=jnp.float64):
    """Independent particle history; kappa is maximum equivalent tensile strain."""
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("count must be a positive integer")
    zero = jnp.zeros(count, dtype=dtype)
    return BrittleHistory(zero, zero, jnp.zeros(count, dtype=bool), jnp.ones(count, dtype=bool))


def _spectrum(F):
    if F.ndim < 2 or F.shape[-2:] != (3, 3):
        raise ValueError("F must have trailing shape (3,3)")
    identity = jnp.eye(3, dtype=F.dtype)
    finite = jnp.all(jnp.isfinite(F), axis=(-2, -1))
    clean = jnp.where(finite[..., None, None], F, identity)
    valid = finite & (jnp.linalg.det(clean) > 0)
    clean = jnp.where(valid[..., None, None], clean, identity)
    values, vectors = jnp.linalg.eigh(jnp.swapaxes(clean, -1, -2) @ clean)
    valid = valid & jnp.all(values > 0, axis=-1)
    stretches = jnp.sqrt(jnp.maximum(values, jnp.finfo(F.dtype).tiny))
    return clean, stretches, vectors, valid


def _elastic_response(F, material):
    clean, stretches, vectors, valid = _spectrum(F)
    strain = stretches - 1
    mu = material.young / (2 * (1 + material.poisson))
    lam = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
    trace = jnp.sum(strain, axis=-1)
    principal = lam * trace[..., None] + 2 * mu * strain
    inverse_stretch = (vectors * (1 / stretches)[..., None, :]) @ jnp.swapaxes(vectors, -1, -2)
    rotation = clean @ inverse_stretch
    first_piola = 2 * mu * clean + (lam * trace - 2 * mu)[..., None, None] * rotation
    energy = 0.5 * lam * trace**2 + mu * jnp.sum(strain**2, axis=-1)
    equivalent = jnp.maximum(jnp.max(principal, axis=-1), 0) / material.young
    return first_piola, energy, equivalent, valid


def damage_from_kappa(kappa, material):
    """Monotonic uniaxial envelope expressed in maximum equivalent tensile strain."""
    scales = brittle_scales(material)
    delta = material.characteristic_length * jnp.maximum(kappa, 0)
    denominator = jnp.maximum(delta, scales.onset_displacement) * (
        scales.failure_displacement - scales.onset_displacement
    )
    return jnp.clip(
        scales.failure_displacement * (delta - scales.onset_displacement) / denominator,
        0, 1,
    )


def update_brittle(history, F, material):
    """Return (new independent history, Kirchhoff stress), without a rate parameter."""
    first_piola, _, equivalent, valid_F = _elastic_response(F, material)
    kappa = jnp.maximum(history.kappa, jnp.where(valid_F, equivalent, history.kappa))
    damage = jnp.maximum(history.damage, damage_from_kappa(kappa, material))
    damage = jnp.where(history.failed, 1, damage)
    valid = (
        history.valid & valid_F & jnp.isfinite(kappa) & jnp.isfinite(damage)
        & (history.kappa >= 0) & (history.damage >= 0) & (history.damage <= 1)
        & (damage >= history.damage)
    )
    failed = history.failed | (damage >= 1)
    new = BrittleHistory(kappa, damage, failed, valid)
    tau = (1 - damage)[..., None, None] * first_piola @ jnp.swapaxes(F, -1, -2)
    tau = jnp.where(valid[..., None, None], tau, 0)
    return new, tau


def brittle_first_piola(F, history, material):
    """First Piola stress at fixed damage; no history advancement."""
    first_piola, _, _, valid = _elastic_response(F, material)
    return jnp.where(
        (valid & history.valid)[..., None, None],
        (1 - history.damage)[..., None, None] * first_piola,
        0,
    )


def brittle_kirchhoff_stress(F, history, material):
    """Kirchhoff stress tau=P F^T at fixed damage."""
    return brittle_first_piola(F, history, material) @ jnp.swapaxes(F, -1, -2)


def brittle_cauchy_stress(F, history, material):
    """True stress; it differs from the reference-area tensile strength."""
    return brittle_kirchhoff_stress(F, history, material) / jnp.linalg.det(F)[..., None, None]


def brittle_energy(F, history, material):
    """Stored energy per reference volume, differentiating F at fixed damage."""
    strain = jnp.sqrt(jnp.linalg.eigvalsh(jnp.swapaxes(F, -1, -2) @ F)) - 1
    mu = material.young / (2 * (1 + material.poisson))
    lam = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
    energy = 0.5 * lam * jnp.sum(strain, axis=-1)**2 + mu * jnp.sum(strain**2, axis=-1)
    return (1 - history.damage) * energy


def brittle_opening(history, material):
    """Maximum equivalent opening h*D*kappa; a scalar diagnostic, not a crack surface."""
    return material.characteristic_length * history.damage * history.kappa
