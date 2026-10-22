"""Bilinear isotropic/kinematic plasticity and prescribed-temperature elasticity."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive, neo_hookean
from .history import dev, equivalent, finish_update


class Bilinear(NamedTuple):
    mu: float
    bulk: float
    yield_stress: float
    hardening: float
    isotropic_fraction: float
    density: float


def bilinear_plastic(
    young=200e9,
    poisson=0.3,
    *,
    yield_stress=250e6,
    tangent=1e9,
    isotropic_fraction=1.0,
    density=7800.0,
):
    """MAT003 rate-independent branch; isotropic_fraction=1 gives MAT012."""
    elastic = neo_hookean(young, poisson)
    _positive(yield_stress=yield_stress, density=density)
    if _concrete(tangent, young, isotropic_fraction):
        if not np.isfinite(tangent) or not 0 <= tangent < young or not 0 <= isotropic_fraction <= 1:
            raise ValueError("require 0 <= tangent < young and 0 <= isotropic_fraction <= 1")
    H = young * tangent / (young - tangent)
    return Bilinear(elastic.mu, elastic.bulk, yield_stress, H, isotropic_fraction, density)


def update_bilinear(old, strain, J, dt, p):
    trial = dev(old.stress) + 2 * p.mu * dev(strain)
    shifted = trial - old.backstress
    q = equivalent(shifted)
    radius = p.yield_stress + p.isotropic_fraction * p.hardening * old.plastic_strain
    dep = jnp.maximum((q - radius) / (3 * p.mu + p.hardening), 0.0)
    increment = (1.5 * dep / jnp.maximum(q, 1e-30))[..., None, None] * shifted
    pressure = old.pressure - p.bulk * jnp.trace(strain, axis1=-2, axis2=-1)
    stress = trial - 2 * p.mu * increment - pressure[..., None, None] * jnp.eye(3)
    backstress = old.backstress + (2 / 3) * (1 - p.isotropic_fraction) * p.hardening * increment
    new = old._replace(
        stress=stress,
        backstress=backstress,
        plastic_tensor=old.plastic_tensor + increment,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        pressure=pressure,
    )
    return finish_update(old, new, strain, p.density / J)


class ThermalPlastic(NamedTuple):
    temperatures: jax.Array
    young: jax.Array
    poisson: jax.Array
    expansion: jax.Array
    yield_stress: jax.Array
    hardening: jax.Array
    reference_temperature: float
    density: float


def thermal_plastic(
    *,
    temperatures=(293.0, 600.0),
    young=(200e9, 150e9),
    poisson=(0.3, 0.3),
    expansion=(12e-6, 14e-6),
    yield_stress=(250e6, 150e6),
    hardening=(1e9, 0.5e9),
    reference_temperature=293.0,
    density=7800.0,
):
    """Two-point thermal property table; temperature is prescribed by the caller."""
    arrays = tuple(
        jnp.asarray(a) for a in (temperatures, young, poisson, expansion, yield_stress, hardening)
    )
    if any(a.shape != (2,) for a in arrays):
        raise ValueError("thermal properties require two-point tables")
    if _concrete(arrays, reference_temperature, density):
        if not all(np.all(np.isfinite(a)) for a in arrays) or temperatures[0] >= temperatures[1]:
            raise ValueError("finite properties and increasing temperatures required")
        if not temperatures[0] <= reference_temperature <= temperatures[1] or density <= 0:
            raise ValueError("invalid reference temperature or density")
        if (
            min(young) <= 0
            or min(poisson) <= -1
            or max(poisson) >= 0.5
            or min(yield_stress) <= 0
            or min(hardening) < 0
        ):
            raise ValueError("invalid elastic/plastic property table")
    return ThermalPlastic(*arrays, reference_temperature, density)


def update_thermal_plastic(old, strain, J, dt, p, temperature=None):
    temperature = (
        old.temperature
        if temperature is None
        else jnp.broadcast_to(temperature, old.temperature.shape)
    )
    at = lambda values: jnp.interp(temperature, p.temperatures, values)
    E, nu, alpha = at(p.young), at(p.poisson), at(p.expansion)
    mu, bulk = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    mechanical = strain - (alpha * (temperature - old.temperature))[..., None, None] * jnp.eye(3)
    trial = dev(old.stress) + 2 * mu[..., None, None] * dev(mechanical)
    q = equivalent(trial)
    H = at(p.hardening)
    dep = jnp.maximum((q - at(p.yield_stress) - H * old.plastic_strain) / (3 * mu + H), 0.0)
    increment = (1.5 * dep / jnp.maximum(q, 1e-30))[..., None, None] * trial
    pressure = old.pressure - bulk * jnp.trace(mechanical, axis1=-2, axis2=-1)
    stress = trial - 2 * mu[..., None, None] * increment - pressure[..., None, None] * jnp.eye(3)
    valid = (temperature >= p.temperatures[0]) & (temperature <= p.temperatures[-1])
    new = old._replace(
        stress=stress,
        plastic_tensor=old.plastic_tensor + increment,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        pressure=pressure,
        temperature=temperature,
        valid=old.valid & valid,
    )
    return finish_update(old, new, strain, p.density / J)
