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
    """Construct the selected MAT173 branch, optionally with per-point properties.

    All property arrays must be one-dimensional and broadcast to one common
    point count. This represents explicit material zoning; it adds no softening
    or evolution law.
    """
    raw = tuple(np.asarray(value, dtype=np.float64) for value in
                (shear_modulus, poisson, friction_angle, cohesion, dilation_angle, density))
    if any(value.ndim > 1 for value in raw):
        raise ValueError("MAT173 properties must be scalars or one-dimensional point arrays")
    try:
        arrays = np.broadcast_arrays(*raw)
    except ValueError as error:
        raise ValueError("MAT173 point-property arrays must share a common length") from error
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("MAT173 properties must be finite")
    shear, nu, phi, cohesion_array, psi, density_array = arrays
    if np.any(shear <= 0) or np.any(density_array <= 0) or np.any((nu <= -1) | (nu >= .5)):
        raise ValueError("MAT173 requires G>0, density>0 and -1<nu<0.5")
    if (np.any(psi < 0) or np.any(psi > phi) or np.any(phi >= np.pi / 2)
            or np.any(cohesion_array < 0)):
        raise ValueError("MAT173 requires 0<=psi<=phi<pi/2 and cohesion>=0")
    if novoid != 1:
        raise ValueError("This adapter implements only the delivered NOVOID=1 branch")
    values = tuple(float(value) if value.ndim == 0 else value.copy() for value in arrays)
    return MAT173(*values, int(novoid))


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
    count = strain.shape[0]
    def points(value):
        value = jnp.asarray(value, dtype=strain.dtype)
        return jnp.broadcast_to(value, (count,)) if value.ndim == 0 else value
    G, nu, phi, psi, cohesion, density = map(points,
        (p.shear, p.poisson, p.friction, p.dilation, p.cohesion, p.density))
    lame = 2 * G * nu / (1 - 2 * nu)
    bulk = lame + 2 * G / 3
    eye = jnp.eye(3, dtype=strain.dtype)
    Ce = lame[:, None, None] * jnp.ones((count, 3, 3), dtype=strain.dtype) + 2 * G[:, None, None] * eye

    # The current history stress is already expressed in the old co-rotated chart.
    trial = (old.stress + 2 * G[:, None, None] * strain
             + lame[:, None, None] * jnp.trace(strain, axis1=-2, axis2=-1)[:, None, None] * eye)
    values, vectors = jnp.linalg.eigh(trial)
    s0 = values[:, ::-1]
    axes = vectors[:, :, ::-1]
    A, B = jax.vmap(_surface_matrices)(phi, psi)
    hard = 2 * cohesion * jnp.cos(phi)
    trial_yield = jnp.einsum("pi,pki->pk", s0, A) - hard[:, None]
    peak = jnp.maximum(jnp.max(trial_yield, axis=-1), 0)
    scale = jnp.maximum(jnp.max(jnp.abs(s0), axis=-1), hard)
    tol = 1e-10 * jnp.maximum(scale, jnp.asarray(1e-300, dtype=strain.dtype))
    yielding = jnp.max(trial_yield, axis=-1) > tol

    returned, flow, has_return, active_count = jax.vmap(_active_return)(s0, A, B, Ce, hard, tol)
    mean = jnp.mean(s0, axis=-1)
    cap = jnp.where(phi > 0, cohesion / jnp.tan(phi), jnp.inf)
    hydro = (phi > 0) & (mean > cap)
    hydro_stress = jnp.broadcast_to(mean[:, None], s0.shape)
    hydro_flow = (s0 - mean[:, None]) / (2 * G[:, None])
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
    return finish_update(old, new, strain, density / J)
