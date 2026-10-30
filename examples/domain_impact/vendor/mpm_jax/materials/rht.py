# SPDX-License-Identifier: AGPL-3.0-or-later
"""RHT three-surface concrete with p-alpha compaction and energy-dependent EOS.

Adapted from rht_material_mod.F90 (2026-09-24). See NOTICE and REFERENCES.md.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _concrete
from .history import dev, equivalent, finish_update


class RHT(NamedTuple):
    density: float = 2314.0
    shear: float = 1.67e10
    onempa: float = 1e6
    erosion_strain: float = 2.0
    B0: float = 1.22
    B1: float = 1.22
    T1: float = 3.527e10
    A: float = 1.6
    N: float = 0.61
    fc: float = 3.5e7
    fs: float = 0.18
    ft: float = 0.1
    Q0: float = 0.6805
    lode_slope: float = 0.0105
    T2: float = 0.0
    reference_rate_c: float = 3e-5
    reference_rate_t: float = 3e-6
    break_rate_c: float = 3e25
    break_rate_t: float = 3e25
    beta_c: float = 0.032
    beta_t: float = 0.036
    tensile_flow: float = 0.001
    elastic_fraction_c: float = 0.53
    elastic_fraction_t: float = 0.70
    hardening_slope: float = 0.50
    D1: float = 0.04
    D2: float = 1.0
    minimum_failure_strain: float = 0.01
    residual_A: float = 1.6
    residual_N: float = 0.61
    gamma: float = 0.0
    A1: float = 3.527e10
    A2: float = 3.958e10
    A3: float = 9.04e9
    crush_pressure: float = 2.33e7
    lock_pressure: float = 6e9
    crush_exponent: float = 3.0
    alpha0: float = 1.1884


def rht(**parameters):
    p = RHT(**parameters)
    if _concrete(p):
        if not np.all(np.isfinite(p)):
            raise ValueError("RHT parameters must be finite")
        positive = (
            p.density,
            p.shear,
            p.onempa,
            p.T1,
            p.A,
            p.N,
            p.fc,
            p.fs,
            p.ft,
            p.reference_rate_c,
            p.reference_rate_t,
            p.elastic_fraction_c,
            p.elastic_fraction_t,
            p.hardening_slope,
            p.D1,
            p.D2,
            p.minimum_failure_strain,
            p.residual_N,
            p.A1,
        )
        if (
            min(positive) <= 0
            or min(
                p.erosion_strain,
                p.B0,
                p.B1,
                p.lode_slope,
                p.beta_c,
                p.beta_t,
                p.residual_A,
                p.gamma,
                p.A2,
                p.A3,
                p.crush_pressure,
            )
            < 0
        ):
            raise ValueError("invalid RHT moduli or strengths")
        if (
            not 0.5 < p.Q0 <= 1
            or p.N > 1
            or max(p.beta_c, p.beta_t) > 1 / 3
            or not 0 < p.tensile_flow <= 1
            or max(p.elastic_fraction_c, p.elastic_fraction_t, p.hardening_slope) > 1
            or p.break_rate_c < p.reference_rate_c
            or p.break_rate_t < p.reference_rate_t
            or p.lock_pressure <= p.crush_pressure
            or p.crush_exponent < 1
            or p.alpha0 < 1
        ):
            raise ValueError("invalid RHT rate, cap or compaction parameters")
        if p.Q0 * p.fs <= float(_lode_factor(jnp.sqrt(3.0) / 2, p.Q0)) * p.ft:
            raise ValueError("RHT tensile meridian requires Q0*fs > R3(pi/6)*ft")
    return p


def _tolerance(dtype):
    return jnp.maximum(2e-10, 32 * jnp.finfo(dtype).eps)


def _safeguarded_newton(function, lower, upper, active, tolerance):
    """Monotone scalar roots, independently bracketed for each material point."""

    def pending(carry):
        iteration, _, _, _, done = carry
        return (iteration < 64) & jnp.any(active & ~done)

    def iteration(carry):
        count, lo, hi, x, done = carry
        value, derivative = jax.jvp(function, (x,), (jnp.ones_like(x),))
        done = done | (jnp.abs(value) <= tolerance)
        lo = jnp.where((value < 0) & ~done, x, lo)
        hi = jnp.where((value > 0) & ~done, x, hi)
        candidate = x - value / jnp.where(derivative > 0, derivative, 1.0)
        admissible = (
            (derivative > 0) & jnp.isfinite(candidate) & (candidate > lo) & (candidate < hi)
        )
        candidate = jnp.where(admissible, candidate, 0.5 * (lo + hi))
        return count + 1, lo, hi, jnp.where(done | ~active, x, candidate), done

    initial = (0, lower, upper, 0.5 * (lower + upper), ~active)
    return jax.lax.while_loop(pending, iteration, initial)[3]


def _lode_factor(cos_theta, Q):
    a, b = 1 - Q**2, 2 * Q - 1
    return (
        2 * a * cos_theta
        + b * jnp.sqrt(jnp.maximum(4 * a * cos_theta**2 + 5 * Q**2 - 4 * Q, 0))
    ) / (4 * a * cos_theta**2 + b**2)


def rht_lode(pressure_star, stress, p):
    scale = jnp.max(jnp.abs(stress), axis=(-2, -1))
    s = stress / jnp.maximum(scale, 1e-30)[..., None, None]
    q = equivalent(s)
    cos3 = jnp.clip(13.5 * jnp.linalg.det(s) / jnp.maximum(q**3, 1e-30), -1, 1)
    Q = jnp.clip(p.Q0 + p.lode_slope * pressure_star, 0.5 + jnp.finfo(stress.dtype).eps, 1.0)
    return jnp.where(scale > 0, _lode_factor(jnp.cos(jnp.arccos(cos3) / 3), Q), 1.0)


def _rate_branch(rate, reference, transition, exponent):
    return jnp.where(
        rate <= transition,
        (jnp.maximum(rate, reference) / reference) ** exponent,
        (transition / reference) ** exponent
        * (jnp.maximum(rate, reference) / transition) ** (1 / 3),
    )


def rht_rates(rate, pressure_star, p):
    fc = _rate_branch(rate, p.reference_rate_c, p.break_rate_c, p.beta_c)
    ft = _rate_branch(rate, p.reference_rate_t, p.break_rate_t, p.beta_t)
    mix = jnp.clip((fc - 3 * pressure_star) / (fc + ft * p.ft), 0, 1)
    rate_factor = fc + mix * (ft - fc)
    mix = jnp.clip(
        (fc * p.elastic_fraction_c - 3 * pressure_star)
        / (fc * p.elastic_fraction_c + ft * p.elastic_fraction_t * p.ft),
        0,
        1,
    )
    elastic_fraction = p.elastic_fraction_c + mix * (p.elastic_fraction_t - p.elastic_fraction_c)
    return fc, rate_factor, elastic_fraction


def rht_failure(pressure_star, rate_factor, p):
    q1 = _lode_factor(jnp.sqrt(3.0) / 2, p.Q0)
    q2 = jnp.clip(
        p.Q0 + p.lode_slope * pressure_star,
        0.5 + jnp.finfo(jnp.asarray(pressure_star).dtype).eps,
        1.0,
    )
    slope = 3 * (p.fs / (q1 * p.ft) - 1 / q2)
    apex = -rate_factor * p.fs / (q1 * slope)
    high = (
        p.A
        * jnp.maximum(pressure_star - rate_factor / 3 + (rate_factor / p.A) ** (1 / p.N), 0.0)
        ** p.N
    )
    middle = rate_factor * p.fs / q1 + 3 * pressure_star * (1 - p.fs / q1)
    low = jnp.maximum(rate_factor * p.fs / q1 + slope * pressure_star, 0.0)
    return (
        jnp.where(
            3 * pressure_star >= rate_factor, high, jnp.where(pressure_star >= 0, middle, low)
        ),
        apex,
    )


def _crush_pressure(alpha, p):
    fraction = jnp.maximum(alpha - 1, 0.0) / jnp.maximum(p.alpha0 - 1, 1e-30)
    return p.lock_pressure - (p.lock_pressure - p.crush_pressure) * fraction ** (
        1 / p.crush_exponent
    )


def solid_eos(density, energy, alpha, p, thermal_density=None):
    thermal_density = density if thermal_density is None else thermal_density
    eta = alpha * density / (p.alpha0 * p.density) - 1
    poly = eta * (p.A1 + eta * (p.A2 + eta * p.A3))
    slope = p.A1 + eta * (2 * p.A2 + 3 * p.A3 * eta)
    tension_poly = eta * (p.T1 + p.T2 * eta)
    tension_slope = p.T1 + 2 * p.T2 * eta
    poly = jnp.where((p.B0 > 0) & (eta < 0), tension_poly, poly)
    slope = jnp.where((p.B0 > 0) & (eta < 0), tension_slope, slope)
    factor = jnp.where(p.B0 > 0, 1.0, 1 - p.gamma * eta / 2)
    thermal = jnp.where(p.B0 > 0, jnp.where(eta >= 0, p.B0 + p.B1 * eta, p.B0), p.gamma)
    pressure = poly * factor / alpha + thermal * thermal_density * energy
    bulk = (1 + eta) * (slope * factor - jnp.where(p.B0 > 0, 0.0, p.gamma * poly / 2)) / alpha
    bulk = bulk + thermal_density * energy * (
        thermal + jnp.where((p.B0 > 0) & (eta >= 0), p.B1 * (1 + eta), 0.0)
    )
    bulk = bulk + thermal * jnp.maximum(pressure, 0.0)
    return pressure, bulk


def rht_eos(density, energy, oldalpha, p):
    alpha = jnp.clip(oldalpha, 1.0, p.alpha0)
    pressure, bulk = solid_eos(density, energy, alpha, p)
    crush = (pressure > _crush_pressure(alpha, p)) & (alpha > 1)
    dense_pressure, dense_bulk = solid_eos(density, energy, jnp.ones_like(alpha), p)
    partial = crush & (dense_pressure < p.lock_pressure)

    def solve(_):
        def body(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            a = jnp.minimum(
                alpha,
                1
                + (p.alpha0 - 1)
                * jnp.maximum((p.lock_pressure - mid) / (p.lock_pressure - p.crush_pressure), 0.0)
                ** p.crush_exponent,
            )
            peos, _ = solid_eos(density, energy, a, p)
            return jnp.where(mid > peos, lo, mid), jnp.where(mid > peos, mid, hi)

        lo, hi = jax.lax.fori_loop(
            0, 52, body, (_crush_pressure(alpha, p), jnp.full_like(alpha, p.lock_pressure))
        )
        mid = 0.5 * (lo + hi)
        return jnp.minimum(
            alpha,
            1
            + (p.alpha0 - 1)
            * jnp.maximum((p.lock_pressure - mid) / (p.lock_pressure - p.crush_pressure), 0.0)
            ** p.crush_exponent,
        )

    part_alpha = jax.lax.cond(jnp.any(partial), solve, lambda _: alpha, None)
    alpha = jnp.where(crush, jnp.where(partial, part_alpha, 1.0), alpha)
    pressure, bulk = solid_eos(density, energy, alpha, p)
    valid = jnp.isfinite(pressure) & jnp.isfinite(bulk) & (density > 0) & (bulk > 0)
    return pressure, alpha, bulk, valid, partial


def rht_surfaces(pressure, trial, alpha, ep, rate, old, p, stage=0, closed_cap=False):
    ps = pressure / p.fc
    frc, fr, fe = rht_rates(rate, ps, p)
    r3 = rht_lode(ps, trial, p)
    failure, apex = rht_failure(ps, fr, p)
    yf = p.fc * failure * r3
    pc = _crush_pressure(alpha, p) / p.fc
    pu = frc * p.elastic_fraction_c / 3 + p.hardening_slope * p.shear * ep / p.fc
    ratio = (ps - pu) / jnp.maximum(pc - pu, 1e-30)
    cap = jnp.where(
        ps >= pc,
        0.0,
        jnp.where((ps > pu) & (pc > pu), jnp.sqrt(jnp.maximum(1 - ratio**2, 0.0)), 1.0),
    )
    cap = jnp.where(closed_cap, 0.0, cap)
    base = fe * cap
    epsh = yf * (1 - base) / (3 * p.hardening_slope * p.shear)
    ep_tolerance = _tolerance(trial.dtype) * p.fc / (3 * p.shear)
    peak = ((ep >= epsh) & (ep > ep_tolerance)) | old.postpeak | (stage == 2)
    dep_damage = jnp.maximum(
        ep - jnp.where(old.postpeak, old.plastic_strain, jnp.maximum(old.plastic_strain, epsh)), 0.0
    )
    postpeak = old.postpeak | (peak & ((dep_damage > 0) | (stage == 2)))
    fracture = jnp.maximum(
        p.minimum_failure_strain, p.D1 * jnp.maximum(ps - (1 - old.damage) * apex, 0.0) ** p.D2
    )
    damage = jnp.where(peak, jnp.minimum(1.0, old.damage + dep_damage / fracture), old.damage)
    residual = p.fc * p.residual_A * jnp.maximum(ps, 0.0) ** p.residual_N
    softened = jnp.where(
        pressure >= 0,
        (1 - damage) * yf + damage * residual,
        yf * jnp.maximum(1 - damage - ps / apex, 0.0),
    )
    y = jnp.where(postpeak & (stage != 1), softened, yf)
    pcut = jnp.where(postpeak & (stage != 1), 1 - damage, 1.0) * apex * p.fc
    solve_hardening = ~peak & (ep > ep_tolerance)

    def hardening_solve(_):
        def residual(gamma):
            gamma = jnp.maximum(gamma, 1e-30)
            fm, _ = rht_failure(ps / gamma, fr, p)
            return p.fc * fm * r3 * (gamma - base) - 3 * p.hardening_slope * p.shear * ep

        tolerance = jnp.maximum(1e-13, 4 * jnp.finfo(trial.dtype).eps) * p.fc
        return _safeguarded_newton(residual, base, jnp.ones_like(base), solve_hardening, tolerance)

    gamma = jax.lax.cond(jnp.any(solve_hardening), hardening_solve, lambda _: base, None)
    gamma = jnp.where(ep <= ep_tolerance, base, gamma)
    fm, pt = rht_failure(ps / jnp.maximum(gamma, 1e-30), fr, p)
    y_pre = jnp.where(gamma > jnp.finfo(trial.dtype).eps, p.fc * fm * r3 * gamma, 0.0)
    cut_pre = jnp.where(gamma > jnp.finfo(trial.dtype).eps, gamma * pt * p.fc, 0.0)
    hard = jnp.where(peak, 1.0, (gamma - base) / jnp.maximum(1 - base, jnp.finfo(trial.dtype).eps))
    return jnp.where(peak, y, y_pre), hard, damage, postpeak, jnp.where(peak, pcut, cut_pre), fr


def _pressure_trial(density, energy, old, p):
    raw, alpha, bulk, valid, crushing = rht_eos(density, energy, old.alpha, p)
    previous, _ = solid_eos(old.density, old.energy, old.alpha, p)
    pressure = jnp.minimum(raw + jnp.maximum(old.pressure - previous, 0.0), jnp.maximum(raw, 0.0))
    return pressure, alpha, bulk, valid, crushing, pressure - raw


def _at_multiplier(z, density, energy, dt, trial, old, p, context, stage):
    pressure0, alpha, bulk0, valid, crushing, offset = context
    multiplier = z / (2 * p.shear)
    tensile = (pressure0 < 0) & (z > 0)

    def pressure_solve(_):
        limit = -jnp.maximum(
            jnp.log(p.alpha0 * p.density / (old.alpha * density)), 0.0
        ) / jnp.maximum(3 * p.tensile_flow * multiplier, 1e-30)
        lower = jnp.where(energy >= 0, jnp.maximum(pressure0, limit), pressure0)

        def residual(mid):
            dpv = -3 * p.tensile_flow * multiplier * mid
            peos, _ = solid_eos(
                density * jnp.exp(jnp.clip(dpv, -100, 100)), energy, old.alpha, p, density
            )
            return mid - peos - offset

        tolerance = jnp.maximum(1e-13, 4 * jnp.finfo(trial.dtype).eps) * jnp.maximum(
            p.fc, jnp.abs(pressure0)
        )
        return _safeguarded_newton(residual, lower, jnp.zeros_like(lower), tensile, tolerance)

    pressure = jax.lax.cond(jnp.any(tensile), pressure_solve, lambda _: pressure0, None)
    pressure = jnp.where(tensile, pressure, pressure0)
    dpv = jnp.where(tensile, -3 * p.tensile_flow * multiplier * pressure, 0.0)
    alpha = jnp.where(tensile, old.alpha, alpha)
    deviator = trial / (1 + z)[..., None, None]
    q = equivalent(deviator)
    dep = 2 * multiplier * q / 3
    y, hard, damage, postpeak, cutoff, fr = rht_surfaces(
        pressure, trial, alpha, old.plastic_strain + dep, dep / dt, old, p, stage, crushing
    )
    residual = jnp.maximum(q - y, cutoff - pressure)
    new = old._replace(
        stress=deviator - pressure[..., None, None] * jnp.eye(3),
        alpha=alpha,
        plastic_strain=old.plastic_strain + dep,
        plastic_rate=dep / dt,
        damage=damage,
        plastic_volume=dpv,
        hardening=hard,
        postpeak=postpeak,
        density=density,
        pressure=pressure,
        energy=energy,
        valid=old.valid & valid,
    )
    return new, residual


def _return_branch(density, energy, dt, trial, old, p, context, stage):
    zero = jnp.zeros_like(density)
    scale = jnp.maximum(p.fc, jnp.max(jnp.abs(trial), axis=(-2, -1)))
    tol = _tolerance(trial.dtype) * scale
    initial, f0 = _at_multiplier(zero, density, energy, dt, trial, old, p, context, stage)
    yielding = f0 > tol

    def solve(_):
        def probe(z):
            return _at_multiplier(z, density, energy, dt, trial, old, p, context, stage)[1]

        def grow_cond(carry):
            i, hi, f = carry
            return (i < 60) & jnp.any(yielding & (f > 0))

        def grow(carry):
            i, hi, f = carry
            hi = jnp.where(yielding & (f > 0), hi * 2, hi)
            return i + 1, hi, probe(hi)

        hi = jnp.full_like(zero, 0.01)
        _, hi, fhi = jax.lax.while_loop(grow_cond, grow, (0, hi, probe(hi)))

        def body(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            f = probe(mid)
            return jnp.where(f > 0, mid, lo), jnp.where(f > 0, hi, mid)

        lo, hi = jax.lax.fori_loop(0, 56, body, (zero, hi))
        result, f = _at_multiplier(
            0.5 * (lo + hi), density, energy, dt, trial, old, p, context, stage
        )
        result = result._replace(valid=result.valid & (fhi <= tol) & (jnp.abs(f) <= 10 * tol))
        return result

    solved = jax.lax.cond(jnp.any(yielding), solve, lambda _: initial, None)
    return jax.tree.map(
        lambda a, b: jnp.where(yielding.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), initial, solved
    )


def _single_step(old, strain, density, energy, dt, p):
    trial = dev(old.stress) + 2 * p.shear * dev(strain)
    context = _pressure_trial(density, energy, old, p)
    first = _return_branch(density, energy, dt, trial, old, p, context, 1)
    needs_post = first.postpeak | old.postpeak
    second = jax.lax.cond(
        jnp.any(needs_post),
        lambda _: _return_branch(density, energy, dt, trial, old, p, context, 2),
        lambda _: first,
        None,
    )
    result = jax.tree.map(
        lambda a, b: jnp.where(needs_post.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), first, second
    )
    qtrial = equivalent(trial)
    dep = jnp.maximum(result.plastic_strain - old.plastic_strain, 0.0)
    result = result._replace(
        plastic_tensor=old.plastic_tensor
        + (1.5 * dep / jnp.maximum(qtrial, 1e-30))[..., None, None] * trial
    )
    # The fully damaged tensile endpoint has finite plastic strain and zero stress.
    endpoint_dep = qtrial / (3 * p.shear)
    _, _, endpoint_damage, _, _, _ = rht_surfaces(
        jnp.zeros_like(density),
        trial,
        old.alpha,
        old.plastic_strain + endpoint_dep,
        endpoint_dep / dt,
        old,
        p,
        2,
    )
    endpoint = (context[0] <= 0) & (endpoint_damage >= 1)
    result = result._replace(
        stress=jnp.where(endpoint[..., None, None], 0.0, result.stress),
        plastic_strain=jnp.where(
            endpoint, old.plastic_strain + endpoint_dep, result.plastic_strain
        ),
        plastic_rate=jnp.where(endpoint, endpoint_dep / dt, result.plastic_rate),
        plastic_tensor=jnp.where(
            endpoint[..., None, None],
            old.plastic_tensor + trial / (2 * p.shear),
            result.plastic_tensor,
        ),
        damage=jnp.where(endpoint, endpoint_damage, result.damage),
        postpeak=result.postpeak | endpoint,
        pressure=jnp.where(endpoint, 0.0, result.pressure),
        valid=jnp.where(endpoint, old.valid & context[3], result.valid),
    )
    result = result._replace(valid=result.valid & (result.damage - old.damage <= 0.025 + 1e-12))
    failed = old.failed | ((p.erosion_strain > 0) & (result.plastic_strain >= p.erosion_strain))
    return result._replace(
        failed=failed, stress=jnp.where(failed[..., None, None], 0.0, result.stress)
    )


def update_rht_prescribed_energy(old, strain, J, dt, p, energy):
    """Constitutive update at caller-supplied specific EOS energy (J/kg)."""
    density = p.density / J
    largest = jnp.maximum(
        jnp.max(jnp.abs(strain)), jnp.max(jnp.abs(jnp.log(density / old.density)))
    )
    initial_substeps = jnp.maximum(1, jnp.ceil(largest / 0.001).astype(jnp.int32))

    def attempt(carry):
        count, subdivisions, _ = carry

        def substep(i, h):
            fraction = (i + 1) / subdivisions
            rho = old.density * jnp.exp(fraction * jnp.log(density / old.density))
            e = old.energy + fraction * (energy - old.energy)
            result = _single_step(h, strain / subdivisions, rho, e, dt / subdivisions, p)
            return jax.tree.map(
                lambda a, b: jnp.where(h.failed.reshape((-1,) + (1,) * (a.ndim - 1)), a, b),
                h,
                result,
            )

        result = jax.lax.fori_loop(0, subdivisions, substep, old)
        return count + 1, subdivisions * 2, result

    def retry(carry):
        count, subdivisions, result = carry
        return (count < 8) & (subdivisions <= 4096) & ((count == 0) | ~jnp.all(result.valid))

    _, _, result = jax.lax.while_loop(
        retry, attempt, (0, initial_substeps, old._replace(valid=jnp.zeros_like(old.valid)))
    )
    return result


def update_rht(old, strain, J, dt, p):
    """Trapezoidal pressure/deviatoric work with iterated EOS energy coupling."""
    density = p.density / J
    v0, v1 = 1 / old.density, 1 / density
    p0, _ = solid_eos(old.density, old.energy, old.alpha, p)
    initial_energy = (
        old.energy
        - p0 * (v1 - v0)
        + jnp.sum(dev(old.stress) * strain, axis=(-2, -1)) * (v0 + v1) / 2
    )
    tolerance = _tolerance(strain.dtype)

    def iteration(carry):
        count, energy, _, _ = carry
        result = update_rht_prescribed_energy(old, strain, J, dt, p, energy)
        p1, _ = solid_eos(density, energy, result.alpha, p)
        work_dev = jnp.sum((dev(old.stress) + dev(result.stress)) * strain, axis=(-2, -1))
        next_energy = old.energy - 0.5 * (p0 + p1) * (v1 - v0) + (v0 + v1) * work_dev / 4
        converged = jnp.abs(next_energy - energy) <= tolerance * jnp.maximum(
            jnp.maximum(p.fc / density, jnp.abs(next_energy)), 1.0
        )
        return count + 1, next_energy, result, converged

    def pending(carry):
        count, _, result, converged = carry
        return (count < 20) & ((count == 0) | (jnp.all(result.valid) & ~jnp.all(converged)))

    _, _, result, converged = jax.lax.while_loop(
        pending, iteration, (0, initial_energy, old, jnp.zeros_like(old.valid))
    )
    return finish_update(old, result._replace(valid=result.valid & converged), strain, density)
