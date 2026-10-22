"""Steinberg-Guinan solid branch: pressure/temperature strength and linear EOS."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive
from .history import dev, equivalent, finish_update


class SteinbergGuinan(NamedTuple):
    shear: float
    bulk: float
    yield_stress: float
    hardening_coefficient: float
    hardening_exponent: float
    maximum_yield: float
    shear_pressure: float
    yield_pressure: float
    thermal_coefficient: float
    heat_capacity: float
    reference_temperature: float
    melt_temperature: float
    gruneisen: float
    volume_coefficient: float
    density: float


def steinberg_guinan(
    *,
    shear=26e9,
    bulk=100e9,
    yield_stress=300e6,
    hardening_coefficient=2.0,
    hardening_exponent=0.5,
    maximum_yield=600e6,
    shear_pressure=1e-11,
    yield_pressure=1e-11,
    thermal_coefficient=0.001,
    heat_capacity=3e6 / 7800,
    reference_temperature=300.0,
    melt_temperature=1800.0,
    gruneisen=2.0,
    volume_coefficient=1.0,
    density=7800.0,
):
    """Selected MAT011 GAMA=F=0 branch; melting and spall are excluded."""
    _positive(
        shear=shear,
        bulk=bulk,
        yield_stress=yield_stress,
        maximum_yield=maximum_yield,
        heat_capacity=heat_capacity,
        melt_temperature=melt_temperature,
        density=density,
    )
    values = (
        hardening_coefficient,
        hardening_exponent,
        shear_pressure,
        yield_pressure,
        thermal_coefficient,
        reference_temperature,
        gruneisen,
        volume_coefficient,
    )
    if _concrete(values, maximum_yield, yield_stress):
        if (
            not np.all(np.isfinite(values))
            or min(values) < 0
            or maximum_yield < yield_stress
            or reference_temperature >= melt_temperature
        ):
            raise ValueError("invalid Steinberg-Guinan parameters")
    return SteinbergGuinan(
        shear,
        bulk,
        yield_stress,
        hardening_coefficient,
        hardening_exponent,
        maximum_yield,
        shear_pressure,
        yield_pressure,
        thermal_coefficient,
        heat_capacity,
        reference_temperature,
        melt_temperature,
        gruneisen,
        volume_coefficient,
        density,
    )


def steinberg_strength(ep, pressure, temperature, J, p):
    compression = pressure * J ** (1 / 3)
    thermal = p.thermal_coefficient * (temperature - 300.0)
    shear = p.shear * jnp.maximum(0.5, 1 + p.shear_pressure * compression - thermal)
    hardening = jnp.minimum(
        p.yield_stress * (1 + p.hardening_coefficient * ep) ** p.hardening_exponent, p.maximum_yield
    )
    strength = jnp.maximum(
        0.5 * p.yield_stress, hardening * (1 + p.yield_pressure * compression - thermal)
    )
    return shear, strength


def update_steinberg(old, strain, J, dt, p):
    temperature = p.reference_temperature + old.energy / p.heat_capacity
    shear, strength = steinberg_strength(old.plastic_strain, old.pressure, temperature, J, p)
    trial = dev(old.stress) + 2 * shear[..., None, None] * dev(strain)
    qtrial = equivalent(trial)
    q = jnp.minimum(qtrial, strength)
    dep = (qtrial - q) / (3 * shear)
    pressure = p.bulk * (1 / J - 1)
    stress = trial * (q / jnp.maximum(qtrial, 1e-30))[..., None, None] - pressure[
        ..., None, None
    ] * jnp.eye(3)
    density = p.density / J
    work = (
        jnp.sum((old.stress + stress) * strain, axis=(-2, -1)) * (1 / old.density + 1 / density) / 4
    )
    energy = old.energy + work
    new_temperature = p.reference_temperature + energy / p.heat_capacity
    volume = jnp.minimum(J, 1.0)
    melt = (
        p.melt_temperature
        * jnp.exp(2 * p.volume_coefficient * (1 - volume))
        / volume ** (2 * (p.gruneisen - p.volume_coefficient - 1 / 3))
    )
    valid = (
        (temperature >= 0)
        & (temperature < melt)
        & (new_temperature >= 0)
        & (new_temperature < melt)
    )
    result = old._replace(
        stress=stress,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        plastic_tensor=old.plastic_tensor
        + (1.5 * dep / jnp.maximum(qtrial, 1e-30))[..., None, None] * trial,
        pressure=pressure,
        temperature=new_temperature,
        energy=energy,
        valid=old.valid & valid,
    )
    return finish_update(old, result, strain, density)
