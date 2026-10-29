"""Separate joint Mohr-Coulomb/Rankine extension; native MAT173 is untouched.

Stress is tension-positive. This branch solves the selected adapter's six MC
planes and a maximum-principal tensile cutoff in one active-set return. It
intentionally does not apply MAT173 NOVOID's hydrostatic-tension exception.
That difference is part of this explicitly separate constitutive assumption.
"""
from itertools import combinations
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .history import finish_update
from .mat173 import _surface_matrices
from .mat173_softening import softening_fraction, softening_properties


class TensileCutoff(NamedTuple):
    base: object
    tensile_strength: object
    separate_shear_softening: bool = False


def with_tensile_cutoff(base, tensile_strength=0.0, *, separate_shear_softening=False):
    """Wrap a MAT173 adapter or softening material with tensile strength T>=0."""
    values = jnp.asarray(tensile_strength)
    if bool(jnp.any(~jnp.isfinite(values))) or bool(jnp.any(values < 0)):
        raise ValueError("tensile_strength must be finite and >= 0 Pa")
    if separate_shear_softening:
        if bool(jnp.any(values != 0)):
            raise ValueError("Separate shear softening requires T0=0")
        if (
            not hasattr(base, "cohesion_residual")
            or bool(jnp.any(jnp.asarray(base.cohesion_residual) != 0))
            or bool(jnp.any(jnp.asarray(base.dilation_residual) != 0))
        ):
            raise ValueError("Separated option requires zero residual cohesion and dilation")
    return TensileCutoff(base, tensile_strength, bool(separate_shear_softening))


def separated_principal_increments(returned, flow, psi, cohesion, tolerance, G, already_residual):
    """Invariant MC multiplier sum for T0=0 before saturation, not tensor partition.

    All active MC planes connect maximum/minimum returned principal groups.
    A strictly compressive minimum group cannot carry Rankine flow. Its total
    plastic flow is -(1-sin(psi))*sum(lambda_MC), including degenerate minima.
    Individual MC/Rankine tensors can be nonunique at corners. The scalar
    per-face equivalent sum is the explicit hardening definition here.
    """
    sp = jnp.sin(psi)
    b = 1 - sp
    group_tol = 4 * tolerance
    minimum = jnp.min(returned, axis=-1)
    minimum_group = returned <= minimum[:, None] + group_tol[:, None]
    compressive = minimum < -group_tol
    flow_tol = 16 * tolerance / jnp.maximum(2 * G, jnp.finfo(flow.dtype).tiny)
    norm = jnp.linalg.norm(flow, axis=-1)
    raw_sum = -jnp.sum(jnp.where(minimum_group, flow, 0.0), axis=-1) / b
    mc_sum = jnp.where(compressive, raw_sum, 0.0)
    # Positive cohesion and allzero returned stress admit only Rankine flow.
    near_apex = (~compressive) & (2 * cohesion <= 8 * tolerance) & (norm > flow_tol)
    ambiguous = near_apex & already_residual
    pre_saturation_ambiguous = near_apex & (~already_residual)
    # At residual c=psi=0, allzero-stress MC/tension partition is not unique.
    # It has no remaining strength effect: freeze driving history; tensile
    # multiplier SUM remains trace(flow) because MC trace is zero.
    mc_sum = jnp.where(already_residual & (~compressive), 0.0, mc_sum)
    tension_sum = jnp.sum(flow, axis=-1) - 2 * sp * mc_sum
    good = (~pre_saturation_ambiguous) & (mc_sum >= -flow_tol) & (tension_sum >= -flow_tol)
    good = good & jnp.all(jnp.isfinite(returned), axis=-1) & jnp.all(jnp.isfinite(flow), axis=-1)
    mc_sum = jnp.maximum(mc_sum, 0.0)
    tension_sum = jnp.maximum(tension_sum, 0.0)
    gamma = (2 / jnp.sqrt(jnp.asarray(3.0, dtype=flow.dtype))) * jnp.sqrt(1 + sp * sp / 3)
    shear_increment = jnp.where(already_residual, 0.0, gamma * mc_sum)
    return shear_increment, tension_sum, good, ambiguous, mc_sum


_ACTIVE_SETS = tuple(c for n in (1, 2, 3) for c in combinations(range(9), n))
_ACTIVE_SET_IDS = tuple(tuple(c) + (0,) * (3 - len(c)) for c in _ACTIVE_SETS)
_ACTIVE_SET_MASK = tuple(tuple(i < len(c) for i in range(3)) for c in _ACTIVE_SETS)
_ACTIVE_SURFACE_MASK = tuple(tuple(j in c for j in range(9)) for c in _ACTIVE_SETS)


def _joint_return_full(s0, A, B, Ce, hard, tol):
    """Exhaustive feasible 1/2/3-face return with full nine-face KKT checks.

    The 129 candidate sets are constant gather indices. This keeps the exact
    exhaustive algorithm while avoiding hundreds of traced scatter updates.
    """
    dtype = s0.dtype
    ids = jnp.asarray(_ACTIVE_SET_IDS, dtype=jnp.int32)
    active = jnp.asarray(_ACTIVE_SET_MASK, dtype=bool)
    active_surface = jnp.asarray(_ACTIVE_SURFACE_MASK, dtype=bool)
    rows_a, rows_b, rows_hard = A[ids], B[ids], hard[ids]
    m = jnp.einsum("kri,ij,ksj->krs", rows_a, Ce, rows_b)
    m = jnp.where(active[:, :, None] & active[:, None, :], m, 0)
    m = m + jnp.eye(3, dtype=dtype)[None] * (~active)[:, :, None]
    scale = jnp.max(jnp.abs(m), axis=(-2, -1))
    count = jnp.sum(active, axis=-1)
    nonsingular = jnp.abs(jnp.linalg.det(m)) > (1e-13 * jnp.maximum(scale, 1e-300) ** count)
    safe = jnp.where(nonsingular[:, None, None], m, jnp.eye(3, dtype=dtype)[None])
    rhs = jnp.where(active, jnp.einsum("kri,i->kr", rows_a, s0) - rows_hard, 0)
    multipliers = jnp.linalg.solve(safe, rhs[..., None])[..., 0]
    flow = jnp.einsum("kr,kri->ki", jnp.maximum(multipliers, 0), rows_b)
    candidates = s0[None, :] - jnp.einsum("ij,kj->ki", Ce, flow)
    yields = candidates @ A.T - hard[None, :]
    lambda_tol = tol / jnp.maximum(jnp.max(jnp.abs(Ce)), 1e-300)
    feasible = (
        nonsingular
        & jnp.all(jnp.where(active, multipliers >= -lambda_tol, True), axis=-1)
        & (jnp.max(yields, axis=-1) <= 4 * tol)
        & (jnp.max(jnp.where(active_surface, jnp.abs(yields), 0), axis=-1) <= 4 * tol)
    )
    idx = jnp.argmax(feasible.astype(jnp.int32))
    any_yield = jnp.max(A @ s0 - hard) > tol
    returned = jnp.where(any_yield, candidates[idx], s0)
    flow_vector = jnp.where(any_yield, flow[idx], jnp.zeros(3, dtype=dtype))
    chosen_faces = jnp.where(any_yield, active_surface[idx], jnp.zeros((9,), dtype=bool))
    return returned, flow_vector, (~any_yield) | jnp.any(feasible), chosen_faces


_JOINT_RETURN_PACK = 4096


def _joint_return_batch_with_count(s0, A, B, Ce, hard, tol):
    """Order-preserving elastic/single-face fast path, compact full fallback.

    The exact 129-set implementation remains the fallback. Candidate order is
    unchanged: a feasible single face precedes every pair/triple, while among
    singles the first feasible original face wins.
    """
    n = s0.shape[0]
    a1, b1, h1 = A[:, :9, :], B[:, :9, :], hard[:, :9]
    # Same padded 3x3 solve as the exhaustive path, with the inactive rows
    # carrying identity pivots. This preserves determinant and solve behavior.
    m00 = jnp.einsum("nfi,nij,nfj->nf", a1, Ce, b1)
    mat = jnp.zeros((n, 9, 3, 3), dtype=s0.dtype)
    mat = mat.at[:, :, 0, 0].set(m00)
    mat = mat.at[:, :, 1, 1].set(1.0)
    mat = mat.at[:, :, 2, 2].set(1.0)
    scale = jnp.maximum(jnp.max(jnp.abs(mat), axis=(-2, -1)), 1e-300)
    nonsingular = jnp.abs(m00) > (1e-13 * scale)
    safe = jnp.where(
        nonsingular[:, :, None, None], mat, jnp.broadcast_to(jnp.eye(3, dtype=s0.dtype), mat.shape)
    )
    rhs = jnp.zeros((n, 9, 3), dtype=s0.dtype)
    rhs = rhs.at[:, :, 0].set(jnp.einsum("nfi,ni->nf", a1, s0) - h1)
    multipliers = jnp.linalg.solve(safe, rhs[..., None])[..., 0, 0]
    flow1 = jnp.maximum(multipliers, 0.0)[:, :, None] * b1
    candidates = s0[:, None, :] - jnp.einsum("nij,nfj->nfi", Ce, flow1)
    yields = jnp.einsum("nfi,nji->nfj", candidates, A) - hard[:, None, :]
    active_yield = jnp.diagonal(yields[:, :, :9], axis1=1, axis2=2)
    lambda_tol = tol / jnp.maximum(jnp.max(jnp.abs(Ce), axis=(-2, -1)), 1e-300)
    feasible = (
        nonsingular
        & (multipliers >= -lambda_tol[:, None])
        & (jnp.max(yields, axis=-1) <= 4 * tol[:, None])
        & (jnp.abs(active_yield) <= 4 * tol[:, None])
    )
    yielding = jnp.max(jnp.einsum("nfi,ni->nf", A, s0) - hard, axis=-1) > tol
    has_single = jnp.any(feasible, axis=-1)
    first = jnp.argmax(feasible.astype(jnp.int32), axis=-1)
    batch = jnp.arange(n, dtype=jnp.int32)
    base_return = jnp.where((yielding & has_single)[:, None], candidates[batch, first], s0)
    base_flow = jnp.where((yielding & has_single)[:, None], flow1[batch, first], 0.0)
    base_faces = jnp.where(
        (yielding & has_single)[:, None], jax.nn.one_hot(first, 9, dtype=bool), False
    )
    base_good = (~yielding) | has_single
    fallback = yielding & ~has_single
    fallback_count = jnp.sum(fallback.astype(jnp.int32))

    def dense(_):
        return (*jax.vmap(_joint_return_full)(s0, A, B, Ce, hard, tol), fallback_count)

    def compact(_):
        ids = jnp.nonzero(fallback, size=_JOINT_RETURN_PACK, fill_value=0)[0]
        valid_slot = jnp.arange(_JOINT_RETURN_PACK, dtype=jnp.int32) < fallback_count
        out = jax.vmap(_joint_return_full)(s0[ids], A[ids], B[ids], Ce[ids], hard[ids], tol[ids])
        rr, ff, gg, fm = out
        # Drop padded writes; assign directly to preserve fallback arithmetic.
        write_ids = jnp.where(valid_slot, ids, n)
        ret = base_return.at[write_ids].set(rr, mode="drop")
        flow = base_flow.at[write_ids].set(ff, mode="drop")
        face = base_faces.at[write_ids].set(fm, mode="drop")
        good = base_good.at[write_ids].set(gg, mode="drop")
        return ret, flow, good, face, fallback_count

    return jax.lax.cond(
        fallback_count > _JOINT_RETURN_PACK,
        dense,
        lambda _: jax.lax.cond(
            fallback_count > 0,
            compact,
            lambda __: (base_return, base_flow, base_good, base_faces, fallback_count),
            None,
        ),
        None,
    )


def _joint_return_batch(s0, A, B, Ce, hard, tol):
    return _joint_return_batch_with_count(s0, A, B, Ce, hard, tol)[:4]


def _update(old, strain, J, dt, G, nu, phi, psi, cohesion, density, T, *, return_principal=False):
    n = strain.shape[0]

    def points(x):
        x = jnp.asarray(x, dtype=strain.dtype)
        return jnp.broadcast_to(x, (n,)) if x.ndim == 0 else x

    G, nu, phi, psi, cohesion, density, T = map(points, (G, nu, phi, psi, cohesion, density, T))
    lam = 2 * G * nu / (1 - 2 * nu)
    Ce = lam[:, None, None] * jnp.ones((n, 3, 3), dtype=strain.dtype) + 2 * G[
        :, None, None
    ] * jnp.eye(3, dtype=strain.dtype)
    trial = (
        old.stress
        + 2 * G[:, None, None] * strain
        + lam[:, None, None]
        * jnp.trace(strain, axis1=-2, axis2=-1)[:, None, None]
        * jnp.eye(3, dtype=strain.dtype)
    )
    values, vectors = jnp.linalg.eigh(trial)
    s0, axes = values[:, ::-1], vectors[:, :, ::-1]
    A_mc, B_mc = jax.vmap(_surface_matrices)(phi, psi)
    A_t = jnp.broadcast_to(jnp.eye(3, dtype=strain.dtype), (n, 3, 3))
    A, B = jnp.concatenate((A_mc, A_t), axis=1), jnp.concatenate((B_mc, A_t), axis=1)
    hard_mc = jnp.broadcast_to((2 * cohesion * jnp.cos(phi))[:, None], (n, 6))
    hard = jnp.concatenate((hard_mc, jnp.broadcast_to(T[:, None], (n, 3))), axis=1)
    tol = 1e-10 * jnp.maximum(jnp.max(jnp.abs(s0), axis=-1), jnp.max(jnp.abs(hard), axis=-1))
    returned, flow, good, return_faces = _joint_return_batch(s0, A, B, Ce, hard, tol)
    stress = jnp.einsum("pik,pk,pjk->pij", axes, returned, axes)
    dep = jnp.einsum("pik,pk,pjk->pij", axes, flow, axes)
    dep_eq = jnp.sqrt(jnp.maximum((2 / 3) * jnp.sum(dep * dep, axis=(-2, -1)), 0))
    pressure = -jnp.trace(stress, axis1=-2, axis2=-1) / 3
    updated = old._replace(
        stress=stress,
        plastic_tensor=old.plastic_tensor + dep,
        plastic_strain=old.plastic_strain + dep_eq,
        plastic_rate=dep_eq / dt,
        pressure=pressure,
        valid=old.valid & good,
        return_tolerance=tol,
        return_faces=return_faces,
    )
    result = finish_update(old, updated, strain, density / J)
    return (result, returned, flow, tol) if return_principal else result


def update_mat173_tensile_cutoff(old, strain, J, dt, material):
    """Joint six-plane MC + associated principal tension cap, peak strength."""
    p = material.base
    return _update(
        old,
        strain,
        J,
        dt,
        p.shear,
        p.poisson,
        p.friction,
        p.dilation,
        p.cohesion,
        p.density,
        material.tensile_strength,
    )


def update_mat173_softening_tensile_cutoff(old, strain, J, dt, material):
    """Joint return with default combined or optional MC-only shear softening.

    Combined: deviatoric norm of total MC+Rankine flow drives softening.
    Separated: sum of per-face MC equivalents drives softening, with a
    separate tensile diagnostic and explicit post-saturation ambiguity marker.
    """
    p = material.base
    phi, cohesion, dilation, _ = softening_properties(old.hardening, p)
    trial, returned, flow, tolerance = _update(
        old,
        strain,
        J,
        dt,
        p.shear,
        p.poisson,
        phi,
        dilation,
        cohesion,
        p.density,
        material.tensile_strength,
        return_principal=True,
    )
    dep = trial.plastic_tensor - old.plastic_tensor
    eye = jnp.eye(3, dtype=strain.dtype)
    dev = dep - jnp.trace(dep, axis1=-2, axis2=-1)[:, None, None] * eye / 3
    combined_increment = jnp.sqrt(jnp.maximum((2 / 3) * jnp.sum(dev * dev, axis=(-2, -1)), 0.0))
    residual = softening_fraction(old.hardening, p) >= 1
    G = jnp.broadcast_to(jnp.asarray(p.shear, dtype=strain.dtype), old.hardening.shape)
    separated = jnp.asarray(material.separate_shear_softening)

    def separate(_):
        return separated_principal_increments(
            returned, flow, dilation, cohesion, tolerance, G, residual
        )

    def combined(_):
        return (
            combined_increment,
            jnp.zeros_like(old.hardening),
            jnp.ones_like(old.valid),
            jnp.zeros_like(old.valid),
            jnp.zeros_like(old.hardening),
        )

    increment, tension, partition_good, ambiguous, _ = jax.lax.cond(
        separated, separate, combined, operand=None
    )
    kappa = old.hardening + increment
    fraction = softening_fraction(kappa, p)
    return trial._replace(
        hardening=kappa,
        damage=fraction,
        tensile_plastic_strain=old.tensile_plastic_strain + tension,
        partition_ambiguous_steps=old.partition_ambiguous_steps + ambiguous.astype(jnp.int32),
        postpeak=old.postpeak | (fraction > 0),
        valid=trial.valid & partition_good & jnp.all(fraction + 1e-14 >= old.damage),
    )
