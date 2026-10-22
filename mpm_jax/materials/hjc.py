"""HJC concrete with irreversible compaction and pressure-dependent damage."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive
from .history import dev, equivalent, equivalent_strain, finish_update


class HJC(NamedTuple):
    shear: float
    A: float
    B: float
    C: float
    N: float
    compressive_strength: float
    tensile_strength: float
    reference_rate: float
    minimum_fracture_strain: float
    maximum_strength: float
    crush_pressure: float
    crush_strain: float
    lock_pressure: float
    lock_strain: float
    D1: float
    D2: float
    K1: float
    K2: float
    K3: float
    density: float
    initial_damage: float


def hjc(
    *,
    shear=14.86e9,
    A=0.79,
    B=1.6,
    C=0.007,
    N=0.61,
    compressive_strength=48e6,
    tensile_strength=4e6,
    reference_rate=1.0,
    minimum_fracture_strain=0.01,
    maximum_strength=7.0,
    crush_pressure=16e6,
    crush_strain=0.001,
    lock_pressure=0.8e9,
    lock_strain=0.1,
    D1=0.04,
    D2=1.0,
    K1=85e9,
    K2=-171e9,
    K3=208e9,
    density=2440.0,
    cold_start_damage=False,
):
    """The supplied HJC v3 EOS/damage profile; cold-start seeding is opt-in."""
    _positive(
        shear=shear,
        compressive_strength=compressive_strength,
        tensile_strength=tensile_strength,
        reference_rate=reference_rate,
        minimum_fracture_strain=minimum_fracture_strain,
        maximum_strength=maximum_strength,
        crush_pressure=crush_pressure,
        crush_strain=crush_strain,
        lock_pressure=lock_pressure,
        lock_strain=lock_strain,
        D1=D1,
        K1=K1,
        density=density,
    )
    if _concrete(A, B, C, N, D2, K2, K3, lock_pressure, crush_pressure, lock_strain, crush_strain):
        if not np.all(np.isfinite([A, B, C, N, D2, K2, K3])) or min(A, B, C, N, D2) < 0:
            raise ValueError("invalid HJC strength or EOS parameters")
        if lock_pressure <= crush_pressure or lock_strain + lock_pressure / K1 <= crush_strain:
            raise ValueError("HJC locking must follow crushing")
    raw = D1 * (tensile_strength / compressive_strength) ** D2
    initial_damage = jnp.where(
        cold_start_damage & (raw > minimum_fracture_strain), jnp.minimum(1.0, 1e-4 / raw), 0.0
    )
    return HJC(
        shear,
        A,
        B,
        C,
        N,
        compressive_strength,
        tensile_strength,
        reference_rate,
        minimum_fracture_strain,
        maximum_strength,
        crush_pressure,
        crush_strain,
        lock_pressure,
        lock_strain,
        D1,
        D2,
        K1,
        K2,
        K3,
        density,
        initial_damage,
    )


def update_hjc(old, strain, J, dt, p):
    mu = 1 / J - 1
    initial_bulk = p.crush_pressure / p.crush_strain
    unloading_bulk = initial_bulk + (p.K1 - initial_bulk) * old.compaction
    hardening = (p.lock_pressure - p.crush_pressure) / p.lock_strain
    trial_pressure = unloading_bulk * (mu - old.plastic_volume)
    depv = jnp.maximum(
        (trial_pressure - p.crush_pressure - hardening * old.plastic_volume)
        / (unloading_bulk + hardening),
        0.0,
    )
    depv = jnp.where(old.compaction < 1, depv, 0.0)
    epv = old.plastic_volume + depv
    dense_mu = (mu - p.lock_strain) / (1 + p.lock_strain)
    pressure = jnp.where(
        old.compaction < 1,
        unloading_bulk * (mu - epv),
        ((p.K3 * dense_mu + p.K2) * dense_mu + p.K1) * dense_mu,
    )
    pressure = jnp.maximum(pressure, -p.tensile_strength * (1 - old.damage))
    maximum = jnp.maximum(old.max_compression, mu)
    compaction = jnp.clip(
        (maximum - p.crush_strain) / (p.lock_strain + p.lock_pressure / p.K1 - p.crush_strain),
        0.0,
        1.0,
    )
    trial = dev(old.stress) + 2 * p.shear * dev(strain)
    qtrial = equivalent(trial)
    strength = jnp.where(
        pressure < 0,
        p.A * (1 - old.damage - pressure / p.tensile_strength),
        p.A * (1 - old.damage) + p.B * jnp.maximum(pressure / p.compressive_strength, 0.0) ** p.N,
    )
    rate = jnp.maximum(equivalent_strain(strain) / dt, 1e-8)
    strength = jnp.minimum(
        strength * (1 + p.C * jnp.log(rate / p.reference_rate)), p.maximum_strength
    )
    strength = jnp.maximum(strength, 0.0) * p.compressive_strength
    q = jnp.minimum(qtrial, strength)
    dep = jnp.where(strength > 0, (qtrial - q) / (3 * p.shear), 0.0)
    raw_fracture = (
        p.D1 * jnp.maximum((pressure + p.tensile_strength) / p.compressive_strength, 0.0) ** p.D2
    )
    damage = jnp.where(
        raw_fracture > p.minimum_fracture_strain,
        jnp.minimum(1.0, old.damage + (dep + depv) / jnp.maximum(raw_fracture, 1e-30)),
        old.damage,
    )
    stress = trial * (q / jnp.maximum(qtrial, 1e-30))[..., None, None] - pressure[
        ..., None, None
    ] * jnp.eye(3)
    new = old._replace(
        stress=stress,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        plastic_tensor=old.plastic_tensor
        + (1.5 * dep / jnp.maximum(qtrial, 1e-30))[..., None, None] * trial,
        damage=damage,
        plastic_volume=epv,
        max_compression=maximum,
        compaction=compaction,
        pressure=pressure,
    )
    return finish_update(old, new, strain, p.density / J)
