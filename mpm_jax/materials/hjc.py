"""HJC concrete with irreversible compaction and pressure-dependent damage."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .elastic import _concrete, _positive
from .history import dev, equivalent, equivalent_strain, finish_update

MAT111_SOURCE_SHA256 = '90e7542f33a46cbc9aedad49207679af05511e8033c8f2f2ce56008105c3425a'
MAT111_PROFILE_SCOPE = ('Finite fixed-axis material-point translation of current MAT111 hjc_point; '
    'not host ABI, erosion, beam validation or transfer of its D1=.04 impact finalcheck.')


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
    legacy_v3: bool = True
    mat111_compacted_tension: bool = False
    mat111_compatible: bool = False


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
    profile="legacy-v3",
    compacted_tension="legacy-polynomial",
):
    """Select legacy-v3, strain-floor, or the supplied MAT111 point profile."""
    if profile not in ("legacy-v3", "strain-floor", "mat111-compatible"):
        raise ValueError("Unknown HJC profile")
    if profile == 'mat111-compatible' and cold_start_damage:
        raise ValueError('MAT111-compatible requires zero cold-start damage')
    if compacted_tension not in ("legacy-polynomial", "mat111-linear"):
        raise ValueError("compacted_tension must be 'legacy-polynomial' or 'mat111-linear'")
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
        profile != "strain-floor",
        compacted_tension == "mat111-linear" or profile == 'mat111-compatible',
        profile == 'mat111-compatible',
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
    nonlinear_mu = jnp.where(p.mat111_compacted_tension, jnp.maximum(dense_mu, 0.), dense_mu)
    dense_pressure = p.K1*dense_mu + (p.K2+p.K3*dense_mu)*nonlinear_mu**2
    legacy_dense_pressure = ((p.K3*dense_mu+p.K2)*dense_mu+p.K1)*dense_mu
    pressure = jnp.where(
        old.compaction < 1,
        unloading_bulk * (mu - epv),
        jnp.where(p.mat111_compacted_tension, dense_pressure, legacy_dense_pressure),
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
    compressive_strength = (
        p.A * (1 - old.damage)
        + p.B * jnp.maximum(pressure / p.compressive_strength, 0.0) ** p.N
    )
    legacy_strength = jnp.where(
        pressure < 0,
        p.A * (1 - old.damage - pressure / p.tensile_strength),
        compressive_strength,
    )
    rate = jnp.maximum(equivalent_strain(strain) / dt, 1e-8)
    log_rate = jnp.log(rate / p.reference_rate)
    legacy_strength = jnp.minimum(
        legacy_strength * (1 + p.C * log_rate), p.maximum_strength
    )
    rate_factor = 1 + p.C * jnp.maximum(log_rate, 0.0)
    floor_strength = jnp.where(
        pressure > 0,
        jnp.minimum(compressive_strength * rate_factor, p.maximum_strength),
        p.A * (1 + pressure / p.tensile_strength) * (1 - old.damage) * rate_factor,
    )
    strength = jnp.where(p.legacy_v3, legacy_strength, floor_strength)
    strength = jnp.maximum(strength, 0.0) * p.compressive_strength
    # MAT111 caps only the positive-pressure strength branch.
    source_strength = jnp.where(pressure > 0,
        jnp.minimum(compressive_strength*(1+p.C*log_rate), p.maximum_strength),
        jnp.maximum((p.A*(1-old.damage)-p.A*pressure/p.tensile_strength)*(1+p.C*log_rate), 0.))
    strength = jnp.where(p.mat111_compatible, source_strength*p.compressive_strength, strength)
    q = jnp.minimum(qtrial, strength)
    source_active = qtrial**2 >= strength**2
    q = jnp.where(p.mat111_compatible,
        jnp.where(source_active & (qtrial > 0), strength, qtrial), q)
    plastic_increment = (qtrial - q) / (3 * p.shear)
    dep = jnp.where(
        p.legacy_v3, jnp.where(strength > 0, plastic_increment, 0.0), plastic_increment
    )
    source_dep = jnp.where(qtrial > 0,
        jnp.where(source_active & (strength >= 1e-8), plastic_increment, 0.), 1e-4)
    ep_increment = jnp.where(p.mat111_compatible,
        jnp.where(qtrial > 0, source_dep, 0.), dep)
    dep = jnp.where(p.mat111_compatible, source_dep, dep)
    raw_fracture = (
        p.D1 * jnp.maximum((pressure + p.tensile_strength) / p.compressive_strength, 0.0) ** p.D2
    )
    legacy_damage = jnp.where(
        raw_fracture > p.minimum_fracture_strain,
        jnp.minimum(1.0, old.damage + (dep + depv) / jnp.maximum(raw_fracture, 1e-30)),
        old.damage,
    )
    floor_damage = jnp.minimum(
        1.0, old.damage + (dep + depv) / jnp.maximum(raw_fracture, p.minimum_fracture_strain)
    )
    damage = jnp.where(p.legacy_v3, legacy_damage, floor_damage)
    source_denominator = jnp.maximum(raw_fracture, 1e-8)
    source_damage = jnp.where(source_denominator > p.minimum_fracture_strain,
        jnp.minimum(1., old.damage+jnp.maximum((dep+depv)/source_denominator, 0.)), old.damage)
    damage = jnp.where(p.mat111_compatible, source_damage, damage)
    stress = trial * (q / jnp.maximum(qtrial, 1e-30))[..., None, None] - pressure[
        ..., None, None
    ] * jnp.eye(3)
    new = old._replace(
        stress=stress,
        plastic_strain=old.plastic_strain + ep_increment,
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
