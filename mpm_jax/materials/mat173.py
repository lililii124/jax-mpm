"""Small-strain Mohr-Coulomb material adapted from MAT173's documented rule.

Stress is tension-positive. The return map uses the six linear principal-stress
surfaces and nonassociated flow directions in the MAT173 source. Finite-strain
host rotations are handled only through the repository's corotated increment;
this is not a reproduction of LS-DYNA's complete MAT173 update.
"""

from itertools import combinations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .history import finish_update


class MAT173(NamedTuple):
    shear: float
    poisson: float
    friction: float
    cohesion: float
    dilation: float
    density: float
    novoid: int


def mat173(*, shear_modulus, poisson, friction_angle, cohesion, dilation_angle,
           density, novoid=1):
    """Construct the selected MAT173 property branch (SI units, angles radians)."""
    values = (shear_modulus, poisson, friction_angle, cohesion, dilation_angle, density)
    if not np.all(np.isfinite(values)):
        raise ValueError("MAT173 properties must be finite")
    if shear_modulus <= 0 or density <= 0 or not -1 < poisson < .5:
        raise ValueError("MAT173 requires G>0, density>0 and -1<nu<0.5")
    if not 0 <= dilation_angle <= friction_angle < np.pi / 2 or cohesion < 0:
        raise ValueError("MAT173 requires 0<=psi<=phi<pi/2 and cohesion>=0")
    if novoid != 1:
        raise ValueError("This adapter implements only the delivered NOVOID=1 branch")
    return MAT173(*map(float, values), int(novoid))


_SURFACE_PAIRS = tuple((i, j) for i in range(3) for j in range(3) if i != j)
_ACTIVE_SETS = tuple(c for n in (1, 2, 3) for c in combinations(range(6), n))


def _surface_matrices(phi, psi):
    A = jnp.zeros((6, 3), dtype=phi.dtype)
    B = jnp.zeros((6, 3), dtype=phi.dtype)
    for k, (i, j) in enumerate(_SURFACE_PAIRS):
        A = A.at[k, i].set(1 + jnp.sin(phi)).at[k, j].set(-(1 - jnp.sin(phi)))
        B = B.at[k, i].set(1 + jnp.sin(psi)).at[k, j].set(-(1 - jnp.sin(psi)))
    return A, B


def _active_return(s0, A, B, Ce, hard, tol):
    """Enumerate admissible one-, two- and three-surface MC returns."""
    dtype = s0.dtype
    rows_a = jnp.zeros((len(_ACTIVE_SETS), 3, 3), dtype=dtype)
    rows_b = jnp.zeros_like(rows_a)
    active = jnp.zeros((len(_ACTIVE_SETS), 3), dtype=bool)
    active_surface = jnp.zeros((len(_ACTIVE_SETS), 6), dtype=bool)
    for k, ids in enumerate(_ACTIVE_SETS):
        for r, idx in enumerate(ids):
            rows_a = rows_a.at[k, r].set(A[idx])
            rows_b = rows_b.at[k, r].set(B[idx])
            active = active.at[k, r].set(True)
            active_surface = active_surface.at[k, idx].set(True)
    m = jnp.einsum("kri,ij,ksj->krs", rows_a, Ce, rows_b)
    m = jnp.where(active[:, :, None] & active[:, None, :], m, 0)
    m = m + jnp.eye(3, dtype=dtype)[None] * (~active)[:, :, None]
    matrix_scale = jnp.max(jnp.abs(m), axis=(-2, -1))
    active_count = jnp.sum(active, axis=-1)
    determinant_scale = 1e-13 * jnp.maximum(matrix_scale, 1e-300) ** active_count
    solvable = jnp.abs(jnp.linalg.det(m)) > determinant_scale
    safe_m = jnp.where(solvable[:, None, None], m, jnp.eye(3, dtype=dtype)[None])
    rhs = jnp.einsum("kri,i->kr", rows_a, s0) - hard
    rhs = jnp.where(active, rhs, 0)
    lam = jnp.linalg.solve(safe_m, rhs[..., None])[..., 0]
    lambda_tolerance = tol / jnp.maximum(jnp.max(jnp.abs(Ce)), 1e-300)
    flow = jnp.einsum("kr,kri->ki", jnp.maximum(lam, 0), rows_b)
    candidates = s0[None, :] - jnp.einsum("ij,kj->ki", Ce, flow)
    yields = candidates @ A.T - hard
    admissible = (
        solvable
        & jnp.all(jnp.where(active, lam >= -lambda_tolerance, True), axis=-1)
        & (jnp.max(yields, axis=-1) <= 4 * tol)
        & (jnp.max(jnp.where(active_surface, jnp.abs(yields), 0), axis=-1) <= 4 * tol)
    )
    index = jnp.argmax(admissible.astype(jnp.int32))
    return candidates[index], flow[index], admissible.any(), jnp.sum(active[index])


def update_mat173(old, strain, J, dt, p):
    """Incremental co-rotated small-strain Mohr-Coulomb update, batched over points."""
    G = jnp.asarray(p.shear, dtype=strain.dtype)
    nu = jnp.asarray(p.poisson, dtype=strain.dtype)
    phi = jnp.asarray(p.friction, dtype=strain.dtype)
    psi = jnp.asarray(p.dilation, dtype=strain.dtype)
    cohesion = jnp.asarray(p.cohesion, dtype=strain.dtype)
    lame = 2 * G * nu / (1 - 2 * nu)
    bulk = lame + 2 * G / 3
    eye = jnp.eye(3, dtype=strain.dtype)
    Ce = lame * jnp.ones((3, 3), dtype=strain.dtype) + 2 * G * eye

    # The current history stress is already expressed in the old co-rotated chart.
    trial = old.stress + 2 * G * strain + lame * jnp.trace(strain, axis1=-2, axis2=-1)[:, None, None] * eye
    values, vectors = jnp.linalg.eigh(trial)
    s0 = values[:, ::-1]
    axes = vectors[:, :, ::-1]
    A, B = _surface_matrices(phi, psi)
    hard = 2 * cohesion * jnp.cos(phi)
    trial_yield = s0 @ A.T - hard
    peak = jnp.maximum(jnp.max(trial_yield, axis=-1), 0)
    scale = jnp.maximum(jnp.max(jnp.abs(s0), axis=-1), hard)
    tol = 1e-10 * jnp.maximum(scale, jnp.asarray(1e-300, dtype=strain.dtype))
    yielding = jnp.max(trial_yield, axis=-1) > tol

    def return_one(args):
        principal, local_tol = args
        return _active_return(principal, A, B, Ce, hard, local_tol)

    returned, flow, has_return, active_count = jax.vmap(return_one)((s0, tol))
    mean = jnp.mean(s0, axis=-1)
    cap = jnp.where(phi > 0, cohesion / jnp.tan(phi), jnp.inf)
    hydro = (phi > 0) & (mean > cap)
    hydro_stress = jnp.broadcast_to(mean[:, None], s0.shape)
    hydro_flow = (s0 - mean[:, None]) / (2 * G)
    principal = jnp.where(yielding[:, None], returned, s0)
    plastic_principal = jnp.where(yielding[:, None], flow, 0)
    principal = jnp.where(hydro[:, None], hydro_stress, principal)
    plastic_principal = jnp.where(hydro[:, None], hydro_flow, plastic_principal)

    stress = jnp.einsum("pik,pk, pjk->pij", axes, principal, axes)
    dep = jnp.einsum("pik,pk,pjk->pij", axes, plastic_principal, axes)
    dep_eq = jnp.sqrt(jnp.maximum((2 / 3) * jnp.sum(dep * dep, axis=(-2, -1)), 0))
    pressure = -jnp.trace(stress, axis1=-2, axis2=-1) / 3
    new = old._replace(
        stress=stress,
        plastic_tensor=old.plastic_tensor + dep,
        plastic_strain=old.plastic_strain + dep_eq,
        plastic_rate=dep_eq / dt,
        pressure=pressure,
        valid=old.valid & (~yielding | has_return | hydro),
    )
    return finish_update(old, new, strain, p.density / J)
