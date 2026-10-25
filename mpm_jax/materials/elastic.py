"""Elastic constitutive laws. Stresses are positive in tension."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Material(NamedTuple):
    mu: jax.Array | float
    lam: jax.Array | float
    bulk: jax.Array | float


class LinearMaterial(NamedTuple):
    stiffness: jax.Array
    orientation: jax.Array


class BlatzKo(NamedTuple):
    shear: float
    poisson: float


def blatz_ko(shear=1e6, poisson=0.463):
    """Compressible rubber branch used by MAT007 (reference nu=.463)."""
    _positive(shear=shear)
    if _concrete(poisson) and (not np.isfinite(poisson) or not 0 < poisson < 0.5):
        raise ValueError("Blatz-Ko requires 0 < poisson < .5")
    return BlatzKo(shear, poisson)


def _concrete(*values):
    return not any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves(values))


def _positive(**values):
    if _concrete(values):
        for name, value in values.items():
            if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
                raise ValueError(f"{name} must be finite and positive")


def neo_hookean(young=2e4, poisson=0.2):
    _positive(young=young)
    if _concrete(poisson) and (not np.isfinite(poisson) or not -1 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    return Material(
        young / (2 * (1 + poisson)),
        young * poisson / ((1 + poisson) * (1 - 2 * poisson)),
        young / (3 * (1 - 2 * poisson)),
    )


elastic = neo_hookean


def fluid(bulk=2e5):
    _positive(bulk=bulk)
    return Material(0.0, 0.0, bulk)


def orthotropic(young, poisson, shear, orientation=None):
    """E=(E1,E2,E3), nu=(nu12,nu13,nu23), G=(G12,G13,G23).

    nu_ij = -epsilon_j/epsilon_i under uniaxial stress in direction i.
    Columns of orientation are the material axes in global coordinates.
    Voigt order is xx, yy, zz, yz, xz, xy; strains use engineering shear.
    """
    E, nu, G = (jnp.asarray(v) for v in (young, poisson, shear))
    if any(v.shape != (3,) for v in (E, nu, G)):
        raise ValueError("young, poisson and shear must each contain three values")
    _positive(young=E, shear=G)
    S = jnp.diag(jnp.concatenate((1 / E, 1 / G[jnp.array([2, 1, 0])])))
    for i, j, k in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
        S = S.at[i, j].set(-nu[k] / E[i]).at[j, i].set(-nu[k] / E[i])
    Q = jnp.eye(3, dtype=S.dtype) if orientation is None else jnp.asarray(orientation)
    if Q.shape != (3, 3):
        raise ValueError("orientation must be a 3 by 3 rotation matrix")
    if _concrete(S, Q):
        if not np.all(np.isfinite(S)) or np.min(np.linalg.eigvalsh(S)) <= 0:
            raise ValueError("engineering constants must define a positive definite compliance")
        if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(Q), 1):
            raise ValueError("orientation must be a proper orthonormal rotation")
    return LinearMaterial(jnp.linalg.inv(S), Q)


def linear_elastic(young=2e4, poisson=0.2):
    p = neo_hookean(young, poisson)
    return orthotropic((young, young, young), (poisson, poisson, poisson), (p.mu,) * 3)


def transverse_isotropic(
    young_longitudinal, young_transverse, poisson_lt, poisson_tt, shear_lt, orientation=None
):
    """Transverse isotropy about local axis 3; nu_lt = -epsilon_t/epsilon_l."""
    El, Et = young_longitudinal, young_transverse
    return orthotropic(
        (Et, Et, El),
        (poisson_tt, poisson_lt * Et / El, poisson_lt * Et / El),
        (Et / (2 * (1 + poisson_tt)), shear_lt, shear_lt),
        orientation,
    )


def linear_stress(strain, material):
    """Cauchy stress from an infinitesimal strain tensor in global axes."""
    Q = material.orientation
    local = Q.T @ strain @ Q
    e = jnp.stack(
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
    s = e @ material.stiffness.T
    tensor = jnp.stack(
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
    ).reshape(strain.shape)
    return Q @ tensor @ Q.T


def kirchhoff_stress(F, material, kind):
    """tau = J sigma; 2D linear elasticity uses plane strain."""
    if kind == "woven-composite":
        from .woven import woven_kirchhoff_stress

        return woven_kirchhoff_stress(F, material)
    dim = F.shape[-1]
    J = jnp.linalg.det(F)
    eye = jnp.eye(dim, dtype=F.dtype)
    if kind == "orthotropic-svk":
        return stvenant_kirchhoff_stress(F, material)
    if kind == "fluid":
        return (material.bulk * J * (J - 1))[..., None, None] * eye
    if kind == "blatz-ko":
        B = F @ jnp.swapaxes(F, -1, -2)
        power = 1 - 1 / (1 - 2 * material.poisson)
        return material.shear * (
            B - jnp.maximum(J, jnp.finfo(F.dtype).tiny)[..., None, None] ** power * eye
        )
    if kind in ("linear", "orthotropic", "transverse-isotropic"):
        strain = 0.5 * (F + jnp.swapaxes(F, -1, -2)) - eye
        if dim == 2:
            strain = jnp.pad(strain, ((0, 0), (0, 1), (0, 1)))
        return J[..., None, None] * linear_stress(strain, material)[..., :dim, :dim]
    if kind not in ("elastic", "neo-hookean"):
        raise ValueError(f"{kind} requires a history-dependent constitutive update")
    B = F @ jnp.swapaxes(F, -1, -2)
    logJ = jnp.log(jnp.maximum(J, jnp.finfo(F.dtype).tiny))
    return material.mu * (B - eye) + material.lam * logJ[..., None, None] * eye


def stvenant_kirchhoff_energy(F, material):
    """Reference-volume energy for the3D orthotropic MAT002 solid branch."""
    if F.shape[-2:] != (3, 3):
        raise ValueError("St.Venant-Kirchhoff requires3D deformation gradients")
    strain = .5 * (jnp.swapaxes(F, -1, -2) @ F - jnp.eye(3, dtype=F.dtype))
    return .5 * jnp.sum(strain * linear_stress(strain, material), axis=(-2, -1))


def stvenant_kirchhoff_stress(F, material):
    """Kirchhoff stress F S F^T, S=C:E, E=(F^T F-I)/2."""
    if F.shape[-2:] != (3, 3):
        raise ValueError("St.Venant-Kirchhoff requires3D deformation gradients")
    strain = .5 * (jnp.swapaxes(F, -1, -2) @ F - jnp.eye(3, dtype=F.dtype))
    return F @ linear_stress(strain, material) @ jnp.swapaxes(F, -1, -2)
