"""Johnson-Cook J2 plasticity with adiabatic heating and optional damage."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive, neo_hookean
from .history import dev, equivalent, finish_update


class JohnsonCook(NamedTuple):
    mu: float
    bulk: float
    A: float
    B: float
    n: float
    C: float
    m: float
    reference_rate: float
    reference_temperature: float
    melt_temperature: float
    heat_capacity: float
    heat_fraction: float
    density: float
    failure: jax.Array
    failure_enabled: bool


def johnson_cook(
    young=110e9,
    poisson=0.34,
    *,
    A=90e6,
    B=292e6,
    n=0.31,
    C=0.025,
    m=1.09,
    reference_rate=1.0,
    reference_temperature=293.15,
    melt_temperature=1356.0,
    heat_capacity=383.0,
    heat_fraction=0.9,
    density=8930.0,
    failure=None,
):
    """SI demonstration parameters; failure=(D1,D2,D3,D4,D5) enables JC damage."""
    p = neo_hookean(young, poisson)
    _positive(
        A=A, n=n, m=m, reference_rate=reference_rate, heat_capacity=heat_capacity, density=density
    )
    if _concrete(B, C, reference_temperature, melt_temperature, heat_fraction):
        if not np.all(np.isfinite([B, C, reference_temperature, melt_temperature, heat_fraction])):
            raise ValueError("JC parameters must be finite")
        if (
            min(B, C) < 0
            or melt_temperature <= reference_temperature
            or not 0 <= heat_fraction <= 1
        ):
            raise ValueError("invalid JC hardening, rate or temperature parameters")
    failure_enabled = failure is not None
    failure = jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.0) if failure is None else failure)
    if failure.shape != (5,):
        raise ValueError("failure must contain D1 through D5")
    if _concrete(failure) and not np.all(np.isfinite(failure)):
        raise ValueError("failure parameters must be finite")
    return JohnsonCook(
        p.mu,
        p.bulk,
        A,
        B,
        n,
        C,
        m,
        reference_rate,
        reference_temperature,
        melt_temperature,
        heat_capacity,
        heat_fraction,
        density,
        failure,
        failure_enabled,
    )


def normalized_temperature(T, p):
    return jnp.clip(
        (T - p.reference_temperature) / (p.melt_temperature - p.reference_temperature), 0, 1
    )


def flow_stress(ep, rate, temperature, p):
    hardening = p.A + p.B * jnp.maximum(ep, 0) ** p.n
    rate_factor = 1 + p.C * jnp.log(jnp.maximum(rate / p.reference_rate, 1))
    return hardening * rate_factor * (1 - normalized_temperature(temperature, p) ** p.m)


def update_johnson_cook(old, strain, J, dt, p):
    """Backward-Euler radial return at old temperature, then explicit plastic heating."""
    trial = dev(old.stress) + 2 * p.mu * dev(strain)
    qtrial = equivalent(trial)
    tolerance = jnp.maximum(1e-10, 128 * jnp.finfo(strain.dtype).eps) * jnp.maximum(p.A, qtrial)
    active = qtrial > flow_stress(old.plastic_strain, 0.0, old.temperature, p) + tolerance
    thermal = 1 - normalized_temperature(old.temperature, p) ** p.m
    hardening_bound = (
        jnp.maximum((qtrial / jnp.maximum(thermal, 1e-30) - p.A) / jnp.maximum(p.B, 1e-30), 0.0)
        ** (1 / p.n)
        - old.plastic_strain
    )
    upper = qtrial / (3 * p.mu)
    upper = jnp.where(
        (p.B > 0) & (thermal > 0), jnp.minimum(upper, jnp.maximum(hardening_bound, 0.0)), upper
    )

    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        residual = (
            qtrial
            - 3 * p.mu * mid
            - flow_stress(old.plastic_strain + mid, mid / dt, old.temperature, p)
        )
        return jnp.where(residual > 0, mid, lo), jnp.where(residual > 0, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 64, bisect, (jnp.zeros_like(qtrial), upper))
    dep = jnp.where(active & ~old.failed, 0.5 * (lo + hi), 0)
    q = jnp.maximum(qtrial - 3 * p.mu * dep, 0)
    pressure = p.bulk * (1 / J - 1)
    stress = trial * (q / jnp.maximum(qtrial, 1e-30))[..., None, None] - pressure[
        ..., None, None
    ] * jnp.eye(3)
    plastic_work = q * dep
    density = p.density / J
    temperature = old.temperature + p.heat_fraction * plastic_work / (density * p.heat_capacity)
    f = p.failure
    triaxiality = -pressure / jnp.maximum(q, 1e-20)
    fracture = (
        (f[0] + f[1] * jnp.exp(jnp.clip(f[2] * triaxiality, -80, 80)))
        * (1 + f[3] * jnp.log(jnp.maximum(dep / (dt * p.reference_rate), 1)))
        * (1 + f[4] * normalized_temperature(old.temperature, p))
    )
    damage = jnp.where(
        p.failure_enabled & (dep > 0),
        jnp.minimum(1.0, old.damage + dep / jnp.maximum(fracture, 1e-12)),
        old.damage,
    )
    failed = old.failed | (p.failure_enabled & (damage >= 1))
    stress = jnp.where(failed[..., None, None], 0.0, stress)
    plastic_tensor = (
        old.plastic_tensor + (1.5 * dep / jnp.maximum(qtrial, 1e-30))[..., None, None] * trial
    )
    residual = q - flow_stress(old.plastic_strain + dep, dep / dt, old.temperature, p)
    valid = old.valid & (~active | old.failed | (jnp.abs(residual) <= tolerance))
    valid = valid & (jnp.logical_not(p.failure_enabled) | (dep == 0) | (fracture > 0))
    new = old._replace(
        stress=stress,
        plastic_tensor=plastic_tensor,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        damage=damage,
        temperature=temperature,
        energy=old.energy + plastic_work / density,
        pressure=jnp.where(failed, 0.0, pressure),
        failed=failed,
        valid=valid,
    )
    return finish_update(old, new, strain, density)
