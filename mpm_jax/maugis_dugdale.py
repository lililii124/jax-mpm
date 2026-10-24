"""Displacement-controlled circular Maugis--Dugdale contact, including a=0."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MaugisParameters(NamedTuple):
    effective_modulus: jax.Array
    effective_radius: jax.Array
    work_of_adhesion: jax.Array
    interaction_range: jax.Array


class MaugisHistory(NamedTuple):
    active: jax.Array
    valid: jax.Array


class MaugisResponse(NamedTuple):
    radius: jax.Array
    adhesive_radius: jax.Array
    normal_force: jax.Array
    area: jax.Array
    adhesive_area: jax.Array
    potential: jax.Array
    phase: jax.Array
    active: jax.Array
    snap_in: jax.Array
    snap_out: jax.Array
    critical_indentation: jax.Array
    attachment_indentation: jax.Array
    central_gap: jax.Array
    maugis_parameter: jax.Array
    displacement_residual: jax.Array
    energy_condition_residual: jax.Array
    stable: jax.Array
    valid: jax.Array


def maugis_parameters(*, effective_modulus, effective_radius,
                      work_of_adhesion=0.1, interaction_range):
    """Host validation; all arguments may be broadcast-compatible arrays."""
    values = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (
        effective_modulus, effective_radius, work_of_adhesion, interaction_range,
    )])
    if any(not x.size or not np.all(np.isfinite(x)) for x in values):
        raise ValueError("MD parameters must be nonempty and finite")
    E, R, W, h = values
    if np.any(E <= 0) or np.any(R <= 0) or np.any(W < 0) or np.any(h <= 0):
        raise ValueError("E*, R and range must be positive; W must be nonnegative")
    return MaugisParameters(*(jnp.asarray(x) for x in values))


def initial_maugis_history(shape=(), *, active=False):
    return MaugisHistory(jnp.full(shape, active, dtype=bool), jnp.ones(shape, dtype=bool))


def _coefficients(q, lam):
    theta = jnp.arctan2(jnp.ones_like(q), q)
    H = jnp.hypot(1, q)
    t = 1 / jnp.maximum(q, 1)
    t2 = t*t
    first = jnp.where(
        q > 100, t*(4/3+t2*(-8/15+t2*(12/35-t2*16/63))),
        q+(1-q*q)*theta,
    )
    derivative = jnp.where(
        q > 100, t**3*(-2/3+t2*(4/5+t2*(-6/7+t2*8/9))),
        q/(1+q*q)-theta,
    )
    alpha = .5*lam*first
    beta = (4/3)*lam*lam*(theta-1/(H+q))
    return alpha, beta, lam*derivative, -(4/3)*lam*lam/(H*H*(H+q)), theta, H


def maugis_branch(q, maugis_parameter):
    """Contact branch (A,C,D,P,dD/dq,(dD/dq)/q), in Maugis units.

    q=a/sqrt(c*c-a*a); this parameterization also represents the unstable
    segment. Selection of the displacement-stable branch is separate.
    """
    q, lam = jnp.broadcast_arrays(jnp.asarray(q), jnp.asarray(maugis_parameter))
    alpha, beta, alpha_pq, beta_pq, theta, H = _coefficients(q, lam)
    B = 2 / (beta+jnp.sqrt(beta*beta+4*alpha))
    A, C = q*B, H*B
    B_pq = -(alpha_pq*B*B+beta_pq*B)/(2*alpha*B+beta)
    D_pq = 2*B*B+(2*q*q*B-(4/3)*lam)*B_pq
    D = A*A-(4/3)*lam*B
    P = A**3-lam*(A*B+(A*A+B*B)*theta)
    return A, C, D, P, q*D_pq, D_pq


def _critical_q(lam):
    folded = maugis_branch(jnp.zeros_like(lam), lam)[-1] < 0
    lo, hi = jnp.zeros_like(lam), jnp.maximum(1, 4*lam)

    def iteration(_, bounds):
        lo, hi = bounds
        mid = (lo+hi)/2
        lower = maugis_branch(mid, lam)[-1] < 0
        return jnp.where(lower, mid, lo), jnp.where(lower, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 64, iteration, (lo, hi))
    return jnp.where(folded, (lo+hi)/2, 0)


@jax.custom_jvp
def _stable_q(D, lam):
    lo = _critical_q(lam)
    hi = 1+4*lam*(1+jnp.sqrt(jnp.sqrt(1+jnp.abs(D))))+jnp.abs(D)

    def expand(_, upper):
        return jnp.where(maugis_branch(upper, lam)[2] < D, 2*upper, upper)

    hi = jax.lax.fori_loop(0, 16, expand, hi)

    def bisect(_, bounds):
        lo, hi = bounds
        mid = (lo+hi)/2
        lower = maugis_branch(mid, lam)[2] < D
        return jnp.where(lower, mid, lo), jnp.where(lower, hi, mid)

    lo, hi = jax.lax.fori_loop(0, 72, bisect, (lo, hi))
    return (lo+hi)/2


@_stable_q.defjvp
def _stable_q_jvp(primals, tangents):
    D, lam = primals
    dD, dlam = tangents
    q = _stable_q(D, lam)
    slope = maugis_branch(q, lam)[4]
    _, partial_lam = jax.jvp(lambda x: maugis_branch(q, x)[2], (lam,), (dlam,))
    return q, (dD-partial_lam)/slope


def _noncontact_potential(C, D, lam):
    return jnp.pi*lam*C**4/8-jnp.pi*lam*D*C*C/2-C*C-(8/9)*lam*lam*C**3


_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(64)


def _contact_potential(q, lam):
    _, C0, D0, *_ = maugis_branch(jnp.zeros_like(q), lam)
    nodes = (jnp.asarray(_GAUSS_X, dtype=q.dtype)+1)*q[..., None]/2
    _, _, _, P, slope, _ = maugis_branch(nodes, lam[..., None])
    integral = q/2*jnp.sum(jnp.asarray(_GAUSS_W, dtype=q.dtype)*P*slope, axis=-1)
    return _noncontact_potential(C0, D0, lam)+integral


def update_maugis(history, indentation, parameters):
    """Return (history, response); positive delta/force denote compression.

    Phase 0 is detached, 1 a finite cohesive disk without intimate contact,
    and 2 intimate contact surrounded by a cohesive annulus. Derivatives are
    defined only away from attachment, detachment and a=0 transitions.
    """
    dtype = jnp.result_type(indentation, *parameters, 0.)
    delta, E, R, W, h = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=dtype) for x in (indentation, *parameters)
    ))
    valid = (history.valid & jnp.isfinite(delta) & jnp.isfinite(E) & jnp.isfinite(R)
             & jnp.isfinite(W) & jnp.isfinite(h) & (E > 0) & (R > 0) & (W >= 0) & (h > 0))
    E, R, h = (jnp.where(valid, x, 1) for x in (E, R, h))
    delta = jnp.where(valid, delta, 0)
    adhesive = valid & (W > 0)
    work = jnp.where(adhesive, W, 1)
    K = (4/3)*E
    L = jnp.cbrt(jnp.pi*work*R*R/K)
    dscale = L*L/R
    lam = 2*(work/h)*R/(K*L)
    D = delta/dscale
    qc = _critical_q(lam)
    _, C0, D0, *_ = maugis_branch(jnp.zeros_like(lam), lam)
    b = 8*lam/(3*jnp.pi)
    hbar = 2/(jnp.pi*lam)
    critical = jnp.where(qc > 0, maugis_branch(qc, lam)[2], -hbar-b*b/2)
    eps = jnp.finfo(delta.dtype).eps
    endpoint_roundoff = 8*eps*jnp.maximum(jnp.abs(critical), hbar)
    attached = valid & (D > critical+endpoint_roundoff) & (history.active | (D >= -hbar))
    contact = attached & ((qc > 0) | (D > D0))
    target = jnp.where(adhesive & contact, D, jnp.maximum(0, D0+1))
    q = _stable_q(target, lam)
    A, C, calculated_D, P, slope, _ = maugis_branch(q, lam)
    alpha, beta, *_ = _coefficients(q, lam)
    B = C/jnp.hypot(1, q)
    er = alpha*B*B+beta*B-1
    nc_C = b+jnp.sqrt(jnp.maximum(0, b*b+2*(hbar+D)))
    nc_P = -jnp.pi*lam*nc_C*nc_C/2
    nc_D = nc_C*nc_C/2-b*nc_C-hbar
    A = jnp.where(contact, A, 0)
    C = jnp.where(contact, C, nc_C)
    P = jnp.where(contact, P, nc_P)
    U = jnp.where(contact, _contact_potential(q, lam), _noncontact_potential(nc_C, D, lam))
    residual = jnp.where(contact, calculated_D-D, nc_D-D)
    er = jnp.where(contact, er, 0)
    central_gap = jnp.where(contact, 0, (-D-(4/3)*lam*nc_C)*dscale)
    stable = jnp.where(contact, slope > 0, nc_C > b)
    tol = 512*eps
    solved = (~attached | (jnp.isfinite(P) & jnp.isfinite(U) & stable
              & (jnp.abs(residual) <= tol*(1+jnp.abs(D))) & (jnp.abs(er) <= tol)
              & (central_gap >= -tol*dscale)))
    valid = valid & (~adhesive | solved)
    active = jnp.where(adhesive, attached, delta > 0) & valid
    ha = jnp.sqrt(R*jnp.where(delta > 0, delta, 1))
    hf = (4/3)*E*ha**3/R
    a = jnp.where(adhesive, A*L, ha)
    c = jnp.where(adhesive, C*L, ha)
    force = jnp.where(adhesive, jnp.pi*work*R*P, hf)
    potential = jnp.where(adhesive, jnp.pi*work*L*L*U, .4*hf*delta)
    a, c, force, potential = (jnp.where(active, x, 0) for x in (a, c, force, potential))
    phase = jnp.where(active, jnp.where(adhesive & ~contact, 1, 2), 0).astype(jnp.int32)
    response = MaugisResponse(
        a, c, force, jnp.pi*a*a, jnp.pi*jnp.maximum(0, c*c-a*a), potential,
        phase, active, valid & active & ~history.active, valid & ~active & history.active,
        jnp.where(adhesive, critical*dscale, 0), jnp.where(adhesive, -h, 0),
        jnp.where(active & adhesive, jnp.maximum(0, central_gap), jnp.maximum(0, -delta)),
        jnp.where(adhesive, lam, 0), jnp.where(active & adhesive, residual*dscale, 0),
        jnp.where(active & adhesive, er, 0), jnp.where(active & adhesive, stable, True), valid,
    )
    return MaugisHistory(jnp.where(valid, active, history.active), valid), response


def maugis_pressure(radial_distance, response, parameters):
    """Analytic reference pressure, not an MPM boundary condition."""
    r = jnp.asarray(radial_distance)
    a, c = response.radius, response.adhesive_radius
    inside = r < a
    root = jnp.sqrt(jnp.maximum(0, a*a-r*r))
    annulus = jnp.sqrt(jnp.maximum(0, c*c-a*a))
    sigma = parameters.work_of_adhesion/parameters.interaction_range
    inner = 2*parameters.effective_modulus/(jnp.pi*parameters.effective_radius)*root
    inner -= 2*sigma/jnp.pi*jnp.arctan2(annulus, jnp.where(inside, root, 1))
    return jnp.where(response.active & response.valid & (r >= 0) & (r < c),
                     jnp.where(inside, inner, -sigma), 0)
