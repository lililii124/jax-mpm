"""Circular equivalent-curvature Hertz/JKR contact with attachment history."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class FibreContactParameters(NamedTuple):
    effective_modulus: jax.Array
    effective_radius: jax.Array
    work_of_adhesion: jax.Array
    adhesive: bool


class FibreContactHistory(NamedTuple):
    active: jax.Array
    valid: jax.Array


class FibreContactResponse(NamedTuple):
    radius: jax.Array
    normal_force: jax.Array
    area: jax.Array
    potential: jax.Array
    active: jax.Array
    snap_in: jax.Array
    snap_out: jax.Array
    critical_radius: jax.Array
    critical_indentation: jax.Array
    valid: jax.Array


class NormalPairForces(NamedTuple):
    first: jax.Array
    second: jax.Array
    valid: jax.Array


def effective_modulus(young_first, poisson_first, young_second, poisson_second):
    """Reduced Young modulus E*, for two isotropic elastic bodies."""
    return 1 / (
        (1 - jnp.asarray(poisson_first)**2) / jnp.asarray(young_first)
        + (1 - jnp.asarray(poisson_second)**2) / jnp.asarray(young_second)
    )


def fibre_contact_parameters(
    *, effective_modulus, effective_radius, work_of_adhesion=0.1, model="jkr",
):
    """Host-side validation; R_eff is supplied by the geometry approximation."""
    if model not in ("hertz", "jkr"):
        raise ValueError("model must be 'hertz' or 'jkr'")
    modulus, radius, work = np.broadcast_arrays(
        np.asarray(effective_modulus, dtype=float),
        np.asarray(effective_radius, dtype=float),
        np.asarray(work_of_adhesion, dtype=float),
    )
    if not all(value.size and np.all(np.isfinite(value)) for value in (modulus, radius, work)):
        raise ValueError("contact parameters must be nonempty and finite")
    if np.any(modulus <= 0) or np.any(radius <= 0) or np.any(work < 0):
        raise ValueError("effective modulus/radius must be positive and adhesion nonnegative")
    return FibreContactParameters(
        jnp.asarray(modulus), jnp.asarray(radius), jnp.asarray(work), model == "jkr",
    )


def initial_fibre_contact_history(shape=(), *, active=False):
    """Independent attachment state, scalar by default or batched with shape."""
    return FibreContactHistory(jnp.full(shape, active, dtype=bool), jnp.ones(shape, dtype=bool))


def jkr_critical_state(parameters):
    """Displacement-control instability, not the load-control pull-off point."""
    work = jnp.where(parameters.adhesive, parameters.work_of_adhesion, 0)
    a = jnp.cbrt(jnp.pi * work * parameters.effective_radius**2 / (8 * parameters.effective_modulus))
    return a, -3 * a**2 / parameters.effective_radius


def contact_potential(contact_radius, indentation, parameters):
    """Potential in joules at independently specified a and delta."""
    a, delta = jnp.asarray(contact_radius), jnp.asarray(indentation)
    E, R = parameters.effective_modulus, parameters.effective_radius
    work = jnp.where(parameters.adhesive, parameters.work_of_adhesion, 0)
    return E * a * delta**2 - 2 * E * a**3 * delta / (3 * R) + E * a**5 / (5 * R**2) - jnp.pi * work * a**2


@jax.custom_jvp
def _stable_radius(delta, radius, modulus, work):
    critical = jnp.cbrt(jnp.pi * work * radius**2 / (8 * modulus))
    lower = critical
    upper = 4 * critical + 2 * jnp.sqrt(radius * jnp.maximum(delta, 0))

    def bisect(_, bounds):
        lo, hi = bounds
        middle = (lo + hi) / 2
        displacement = middle**2 / radius - jnp.sqrt(2 * jnp.pi * work * middle / modulus)
        below = displacement < delta
        return jnp.where(below, middle, lo), jnp.where(below, hi, middle)

    lower, upper = jax.lax.fori_loop(0, 64, bisect, (lower, upper))
    return (lower + upper) / 2


@_stable_radius.defjvp
def _stable_radius_jvp(primals, tangents):
    delta, radius, modulus, work = primals
    ddelta, dradius, dmodulus, dwork = tangents
    a = _stable_radius(delta, radius, modulus, work)
    correction = jnp.sqrt(2 * jnp.pi * work * a / modulus)
    slope = 2 * a / radius - correction / (2 * a)
    numerator = (
        ddelta + a**2 / radius**2 * dradius
        - correction / (2 * modulus) * dmodulus + correction / (2 * work) * dwork
    )
    return a, numerator / slope


def update_fibre_contact(history, indentation, parameters):
    """Advance attachment history; positive indentation and force mean compression."""
    delta = jnp.asarray(indentation)
    E, R = parameters.effective_modulus, parameters.effective_radius
    W = parameters.work_of_adhesion
    valid = (
        history.valid & jnp.isfinite(delta) & jnp.isfinite(E) & jnp.isfinite(R)
        & jnp.isfinite(W) & (E > 0) & (R > 0) & (W >= 0)
    )
    E = jnp.where(valid, E, 1)
    R = jnp.where(valid, R, 1)
    delta = jnp.where(valid, delta, 0)
    use_jkr = parameters.adhesive & (W > 0)
    safe_work = jnp.where(valid & use_jkr, W, 1)
    critical_a = jnp.cbrt(jnp.pi * safe_work * R**2 / (8 * E))
    critical_delta = -3 * critical_a**2 / R
    attached = valid & (delta > critical_delta) & (history.active | (delta >= 0))
    # Evaluate discarded roots at a regular attached point, including W=0 AD.
    target = jnp.where(attached & use_jkr, delta, 0)
    jkr_a = _stable_radius(target, R, E, safe_work)
    hertz_active = valid & (delta > 0)
    hertz_a = jnp.where(hertz_active, jnp.sqrt(R * jnp.where(hertz_active, delta, 1)), 0)
    active = jnp.where(use_jkr, attached, hertz_active)
    a = jnp.where(use_jkr, jnp.where(attached, jkr_a, 0), hertz_a)
    compression = 4 * E * a**3 / (3 * R)
    attraction = jnp.where(
        use_jkr & active,
        jnp.sqrt(8 * jnp.pi * safe_work * E * jnp.where(active, a, 1)**3),
        0,
    )
    force = jnp.where(valid, compression - attraction, 0)
    potential = contact_potential(a, delta, parameters)
    response = FibreContactResponse(
        a, force, jnp.pi * a**2, jnp.where(valid & active, potential, 0), active,
        valid & active & ~history.active, valid & ~active & history.active,
        jnp.where(use_jkr, critical_a, 0), jnp.where(use_jkr, critical_delta, 0), valid,
    )
    new = FibreContactHistory(jnp.where(valid, active, history.active), valid)
    return new, response


def normal_pair_forces(normal, response):
    """Normal points from first body to second; return equal opposite forces."""
    normal = jnp.asarray(normal)
    if normal.ndim < 1 or normal.shape[-1] != 3:
        raise ValueError("normal must have trailing shape (3,)")
    length = jnp.linalg.norm(normal, axis=-1)
    valid = response.valid & jnp.all(jnp.isfinite(normal), axis=-1) & (length > 0)
    unit = jnp.where(valid[..., None], normal, 0) / jnp.where(valid, length, 1)[..., None]
    second = jnp.where(valid[..., None], response.normal_force[..., None] * unit, 0)
    return NormalPairForces(-second, second, valid)
