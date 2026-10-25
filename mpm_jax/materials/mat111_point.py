"""Finite MAT111 scalar-point path with explicit arithmetic boundaries."""

import jax
import jax.numpy as jnp

from .history import finish_update


def _add(a, b):
    return jax.lax.optimization_barrier(a + b)


def _sub(a, b):
    return jax.lax.optimization_barrier(a - b)


def _mul(a, b):
    return jax.lax.optimization_barrier(a * b)


def _div(a, b):
    return jax.lax.optimization_barrier(a / b)


def _sum3(values):
    return _add(_add(values[..., 0], values[..., 1]), values[..., 2])


def update_mat111(old, strain, J, dt, p):
    # The source branches on q == 0; fused/reassociated operations can change D.
    p = jax.tree.map(jax.lax.optimization_barrier, p)
    dt = jax.lax.optimization_barrier(dt)
    one, two, three, half, three_halves = (
        jax.lax.optimization_barrier(jnp.asarray(value, dtype=strain.dtype))
        for value in (1., 2., 3., .5, 1.5)
    )
    mu = _sub(_div(one, J), one)
    maximum = jnp.maximum(old.max_compression, mu)
    modified = _div(_sub(mu, p.lock_strain), _add(one, p.lock_strain))
    hardening = _div(_sub(p.lock_pressure, p.crush_pressure), p.lock_strain)
    span = _sub(_add(_div(p.lock_pressure, p.K1), p.lock_strain), p.crush_strain)
    bulk = _add(_div(_mul(_sub(one, old.compaction), p.crush_pressure), p.crush_strain),
                _mul(old.compaction, p.K1))
    elastic_volume = _sub(mu, old.plastic_volume)
    elastic_pressure = _mul(bulk, elastic_volume)
    crushing = _add(_mul(hardening, old.plastic_volume), p.crush_pressure)
    depv = jnp.where((old.compaction < one) & (elastic_pressure > crushing),
        _div(_sub(elastic_pressure, crushing), _add(hardening, bulk)), 0.)
    positive = jnp.maximum(modified, 0.)
    dense_pressure = _add(_mul(_mul(_add(_mul(modified, p.K3), p.K2), positive), positive),
                          _mul(p.K1, modified))
    pressure = jnp.where(old.compaction < one,
        _mul(_sub(elastic_volume, depv), bulk), dense_pressure)
    pressure = jnp.maximum(pressure, _mul(_sub(one, old.damage), -p.tensile_strength))
    compaction = jnp.where(old.compaction < one,
        jnp.clip(_div(_sub(maximum, p.crush_strain), span), 0., one), old.compaction)

    normal_rate = _div(jnp.diagonal(strain, axis1=-2, axis2=-1), dt)
    negative_mean = _div(-_add(_add(normal_rate[..., 1], normal_rate[..., 2]),
                              normal_rate[..., 0]), three)
    normal_rate = _add(normal_rate, negative_mean[..., None])
    shear = jnp.stack((strain[..., 0, 1], strain[..., 1, 2], strain[..., 2, 0]), axis=-1)
    shear_rate = _div(_mul(two, shear), dt)
    old_normal = jnp.diagonal(old.stress, axis1=-2, axis2=-1)
    normal_addition = _div(-_sum3(old_normal), three)
    dtg = _mul(dt, p.shear)
    trial_normal = _add(_add(_mul(normal_rate, _add(dtg, dtg)), old_normal),
                        normal_addition[..., None])
    old_shear = jnp.stack((old.stress[..., 0, 1], old.stress[..., 1, 2],
                          old.stress[..., 2, 0]), axis=-1)
    trial_shear = _add(_mul(shear_rate, dtg), old_shear)
    normal_square = _sum3(_mul(trial_normal, trial_normal))
    shear_square = _sum3(_mul(trial_shear, trial_shear))
    q_squared = jnp.abs(_add(_mul(normal_square, three_halves), _mul(shear_square, three)))
    q = jnp.sqrt(q_squared)
    rate_square = _add(_sum3(_mul(normal_rate, normal_rate)),
                       _mul(_sum3(_mul(shear_rate, shear_rate)), half))
    rate = jnp.sqrt(_div(_mul(rate_square, two), three))
    rate_ratio = _mul(jnp.maximum(rate, 1e-8), _div(one, p.reference_rate))
    factor = _add(_mul(jnp.log(rate_ratio), p.C), one)
    intact = _mul(_sub(one, old.damage), p.A)
    pressure_star = _div(pressure, p.compressive_strength)
    tension_star = _div(p.tensile_strength, p.compressive_strength)
    positive_strength = _mul(jnp.minimum(p.maximum_strength,
        _mul(factor, _add(intact, _mul(jnp.maximum(pressure_star, 0.)**p.N, p.B)))),
        p.compressive_strength)
    tensile_strength = _mul(jnp.maximum(0., _mul(_sub(intact,
        _div(_mul(p.A, pressure_star), tension_star)), factor)), p.compressive_strength)
    strength = jnp.where(pressure_star > 0., positive_strength, tensile_strength)
    active = _sub(q_squared, _mul(strength, strength)) >= 0.
    returned = active & (q > 0.)
    ratio = jnp.where(q == 0., 0., jnp.where(returned,
        _div(strength, jnp.where(q > 0., q, one)), one))
    plastic = _mul(_sub(q, strength), _div(one, _mul(three, p.shear)))
    plastic = jnp.where(returned & (strength >= 1e-8), plastic, 0.)
    dep = jnp.where(q == 0., 1e-4, plastic)
    shifted = jnp.maximum(_add(pressure_star, tension_star), 0.)
    denominator = jnp.maximum(_mul(shifted**p.D2, p.D1), 1e-8)
    damage = jnp.where(denominator > p.minimum_fracture_strain,
        jnp.minimum(_add(jnp.maximum(_div(_add(dep, depv), denominator), 0.), old.damage), one),
        old.damage)
    normal = _sub(_mul(trial_normal, ratio[..., None]), pressure[..., None])
    shear = _mul(trial_shear, ratio[..., None])
    stress = jnp.stack((
        jnp.stack((normal[..., 0], shear[..., 0], shear[..., 2]), axis=-1),
        jnp.stack((shear[..., 0], normal[..., 1], shear[..., 1]), axis=-1),
        jnp.stack((shear[..., 2], shear[..., 1], normal[..., 2]), axis=-1),
    ), axis=-2)
    trial = jnp.stack((
        jnp.stack((trial_normal[..., 0], trial_shear[..., 0], trial_shear[..., 2]), axis=-1),
        jnp.stack((trial_shear[..., 0], trial_normal[..., 1], trial_shear[..., 1]), axis=-1),
        jnp.stack((trial_shear[..., 2], trial_shear[..., 1], trial_normal[..., 2]), axis=-1),
    ), axis=-2)
    new = old._replace(stress=stress, plastic_strain=_add(old.plastic_strain, plastic),
        plastic_rate=dep / dt,
        plastic_tensor=old.plastic_tensor + (1.5 * dep / jnp.maximum(q, 1e-30))[..., None, None] * trial,
        damage=damage, plastic_volume=_add(old.plastic_volume, depv),
        max_compression=maximum, compaction=compaction, pressure=pressure)
    return finish_update(old, new, strain, p.density / J)
