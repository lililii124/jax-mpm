"""Bonded matrix and locally oriented elastic yarns."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .elastic import _positive, neo_hookean, orthotropic


class WovenElastic(NamedTuple):
    matrix_mu: float
    matrix_lam: float
    yarn_stiffness: jax.Array
    orientation: jax.Array
    phase: jax.Array
    matrix_density: float
    yarn_density: float


def woven_elastic(
    *,
    matrix_young,
    matrix_poisson,
    matrix_density,
    yarn_young,
    yarn_poisson,
    yarn_shear,
    yarn_density,
    orientation,
    phase,
):
    """Phase zero is matrix; phase one is yarn. Q columns are reference axes."""
    _positive(matrix_density=matrix_density, yarn_density=yarn_density)
    matrix = neo_hookean(matrix_young, matrix_poisson)
    yarn = orthotropic(yarn_young, yarn_poisson, yarn_shear)
    q = np.asarray(orientation)
    labels = np.asarray(phase)
    if q.ndim != 3 or q.shape[1:] != (3, 3) or labels.shape != (len(q),):
        raise ValueError("orientation must be (N,3,3) and phase must be (N,)")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("phase must contain only matrix=0 or yarn=1")
    for start in range(0, len(q), 16384):
        block = q[start : start + 16384]
        if (
            not np.all(np.isfinite(block))
            or not np.allclose(
                np.swapaxes(block, -1, -2) @ block, np.eye(3), atol=1e-7, rtol=0
            )
            or not np.allclose(np.linalg.det(block), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError("each reference orientation must be a proper rotation")
    return WovenElastic(
        matrix.mu,
        matrix.lam,
        yarn.stiffness,
        jnp.asarray(q, dtype=yarn.stiffness.dtype),
        jnp.asarray(labels, dtype=jnp.uint8),
        matrix_density,
        yarn_density,
    )


def _yarn_strain(F, material):
    green = 0.5 * (jnp.swapaxes(F, -1, -2) @ F - jnp.eye(3, dtype=F.dtype))
    q = material.orientation
    local = jnp.swapaxes(q, -1, -2) @ green @ q
    return jnp.stack(
        (
            local[..., 0, 0],
            local[..., 1, 1],
            local[..., 2, 2],
            2 * local[..., 1, 2],
            2 * local[..., 0, 2],
            2 * local[..., 0, 1],
        ),
        axis=-1,
    )


def woven_energy(F, material):
    """Energy per reference volume; yarns use orthotropic Saint Venant–Kirchhoff."""
    strain = _yarn_strain(F, material)
    stress = strain @ material.yarn_stiffness.T
    yarn = 0.5 * jnp.sum(strain * stress, axis=-1)
    J = jnp.linalg.det(F)
    logJ = jnp.log(jnp.maximum(J, jnp.finfo(F.dtype).tiny))
    matrix = (
        0.5 * material.matrix_mu * (jnp.sum(F * F, axis=(-1, -2)) - 3)
        - material.matrix_mu * logJ
        + 0.5 * material.matrix_lam * logJ**2
    )
    return jnp.where(material.phase == 1, yarn, matrix)


def woven_kirchhoff_stress(F, material):
    """Objective Kirchhoff stress for small elastic strains and finite rotations."""
    s = _yarn_strain(F, material) @ material.yarn_stiffness.T
    local = jnp.stack(
        (
            s[..., 0],
            s[..., 5],
            s[..., 4],
            s[..., 5],
            s[..., 1],
            s[..., 3],
            s[..., 4],
            s[..., 3],
            s[..., 2],
        ),
        axis=-1,
    ).reshape(F.shape)
    q = material.orientation
    second_piola = q @ local @ jnp.swapaxes(q, -1, -2)
    yarn = F @ second_piola @ jnp.swapaxes(F, -1, -2)
    J = jnp.linalg.det(F)
    logJ = jnp.log(jnp.maximum(J, jnp.finfo(F.dtype).tiny))
    identity = jnp.eye(3, dtype=F.dtype)
    matrix = (
        material.matrix_mu * (F @ jnp.swapaxes(F, -1, -2) - identity)
        + material.matrix_lam * logJ[..., None, None] * identity
    )
    return jnp.where((material.phase == 1)[..., None, None], yarn, matrix)


def woven_wave_speed_bound(material):
    """Upper bound for the initial acoustic speeds, not nonlinear stability."""
    scale = np.diag((1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)))
    kelvin = scale @ np.asarray(material.yarn_stiffness) @ scale
    yarn = float(np.linalg.eigvalsh(kelvin).max()) / material.yarn_density
    bulk = material.matrix_lam + 2 * material.matrix_mu / 3
    matrix = max(3 * bulk, 2 * material.matrix_mu) / material.matrix_density
    return float(np.sqrt(max(matrix, yarn)))
