"""JH-2 ceramics: intact/fractured strength, damage, EOS and delayed bulking."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive
from .history import dev, equivalent, equivalent_strain, finish_update


class JH2(NamedTuple):
    shear: float
    A: float
    B: float
    C: float
    M: float
    N: float
    reference_rate: float
    tensile_strength: float
    maximum_fractured_strength: float
    hel: float
    pressure_hel: float
    bulking_fraction: float
    D1: float
    D2: float
    K1: float
    K2: float
    K3: float
    failure_strain: float
    density: float


def jh2(
    *,
    shear=20e9,
    A=0.79,
    B=0.3,
    C=0.007,
    M=0.6,
    N=0.6,
    reference_rate=1.0,
    tensile_strength=100e6,
    maximum_fractured_strength=1.0,
    hel=2.6e9,
    pressure_hel=1.6e9,
    bulking_fraction=0.5,
    D1=0.01,
    D2=0.5,
    K1=130e9,
    K2=0.0,
    K3=0.0,
    failure_strain=0.0,
    density=2500.0,
):
    _positive(shear=shear, reference_rate=reference_rate, hel=hel, K1=K1, density=density)
    if _concrete(
        A,
        B,
        C,
        M,
        N,
        tensile_strength,
        maximum_fractured_strength,
        pressure_hel,
        bulking_fraction,
        D1,
        D2,
        K2,
        K3,
        failure_strain,
    ):
        values = [
            A,
            B,
            C,
            M,
            N,
            tensile_strength,
            maximum_fractured_strength,
            pressure_hel,
            bulking_fraction,
            D1,
            D2,
            K2,
            K3,
            failure_strain,
        ]
        if (
            not np.all(np.isfinite(values))
            or min(values) < 0
            or bulking_fraction > 1
            or pressure_hel >= hel
        ):
            raise ValueError("invalid JH-2 parameters; tensile erosion (FS<0) is not supported")
    if _concrete(pressure_hel) and pressure_hel == 0:

        def bracket(_, pair):
            lo, hi = pair
            mu = 0.5 * (lo + hi)
            residual = ((K3 * mu + K2) * mu + K1) * mu + 4 * shear / 3 * mu / (1 + mu) - hel
            return jnp.where(residual > 0, lo, mu), jnp.where(residual > 0, mu, hi)

        lo, hi = jax.lax.fori_loop(0, 64, bracket, (jnp.asarray(0.0), jnp.asarray(hel / K1)))
        mu = 0.5 * (lo + hi)
        pressure_hel = ((K3 * mu + K2) * mu + K1) * mu
    return JH2(
        shear,
        A,
        B,
        C,
        M,
        N,
        reference_rate,
        tensile_strength,
        maximum_fractured_strength,
        hel,
        pressure_hel,
        bulking_fraction,
        D1,
        D2,
        K1,
        K2,
        K3,
        failure_strain,
        density,
    )


def update_jh2(old, strain, J, dt, p):
    mu = 1 / J - 1
    sigma_hel = 1.5 * (p.hel - p.pressure_hel)
    pressure = jnp.where(mu > 0, ((p.K3 * mu + p.K2) * mu + p.K1) * mu + old.bulking, p.K1 * mu)
    cutoff = (mu <= 0) & (
        pressure / p.pressure_hel + p.tensile_strength / p.pressure_hel + 1e-10 < 0
    )
    damage_old = jnp.where(cutoff, 1.0, old.damage)
    pressure = jnp.where((mu <= 0) & ((damage_old == 1) | cutoff), 0.0, pressure)
    trial = dev(old.stress) + 2 * p.shear * dev(strain)
    qtrial = equivalent(trial)
    rate = equivalent_strain(strain) / dt
    factor = 1 + p.C * jnp.log(jnp.maximum(rate / p.reference_rate, 1.0))
    base = jnp.maximum((pressure + p.tensile_strength) / p.pressure_hel + 1e-10, 0.0)
    intact = p.A * base**p.N * factor * sigma_hel
    cap = jnp.where(p.maximum_fractured_strength == 0, 1e20, p.maximum_fractured_strength)
    fractured = (
        jnp.minimum(cap, p.B * jnp.maximum(pressure / p.pressure_hel, 0.0) ** p.M * factor)
        * sigma_hel
    )
    fractured = jnp.where(pressure > 0, fractured, 0.0)
    strength = (1 - damage_old) * intact + damage_old * fractured
    dep = jnp.maximum((qtrial - strength) / (3 * p.shear), 0.0)
    fracture_strain = p.D1 * base**p.D2
    progressive = (dep > 0) & (mu > 0) & (fracture_strain > 1e-10)
    damage = jnp.where(
        progressive,
        jnp.minimum(1.0, damage_old + dep / jnp.maximum(fracture_strain, 1e-30)),
        damage_old,
    )
    instant = (p.D1 == 0) & (qtrial >= strength) & (damage_old < 1)
    damage = jnp.where(instant, 1.0, damage)
    strength_new = (1 - damage) * intact + damage * fractured
    q = jnp.where(qtrial >= strength, strength_new, qtrial)
    released = jnp.maximum((qtrial**2 - q**2) / (6 * p.shear), 0.0)
    bulk_update = instant | ((dep > 0) & (damage < 1) & (mu > 0))
    released = jnp.where(bulk_update, released, 0.0)
    base_pressure = p.K1 * mu
    factor_bulk = jnp.where(instant, 2.0, 1.0) * p.bulking_fraction * p.K1
    bulking = -base_pressure + jnp.sqrt(
        jnp.maximum((old.bulking + base_pressure) ** 2 + factor_bulk * released, 0.0)
    )
    bulking = jnp.where(bulk_update, bulking, old.bulking)
    stress = trial * (q / jnp.maximum(qtrial, 1e-30))[..., None, None] - pressure[
        ..., None, None
    ] * jnp.eye(3)
    ep = old.plastic_strain + dep
    failed = old.failed | ((p.failure_strain > 0) & (ep >= p.failure_strain))
    new = old._replace(
        stress=jnp.where(failed[..., None, None], 0.0, stress),
        plastic_strain=ep,
        plastic_rate=dep / dt,
        plastic_tensor=old.plastic_tensor
        + (1.5 * dep / jnp.maximum(qtrial, 1e-30))[..., None, None] * trial,
        damage=damage,
        bulking=bulking,
        energy=old.energy + released / (p.density / J),
        pressure=jnp.where(failed, 0.0, pressure),
        failed=failed,
    )
    new = jax.tree.map(
        lambda before, after: jnp.where(
            old.failed.reshape((-1,) + (1,) * (after.ndim - 1)), before, after
        ),
        old,
        new,
    )
    return finish_update(old, new, strain, p.density / J)
