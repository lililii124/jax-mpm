"""Crack-band-regularized peak-to-residual Mohr-Coulomb variant.

This is a new constitutive extension and is not part of the delivered MAT173
adapter. Softening is driven by accumulated equivalent *deviatoric* plastic
strain times a material-point characteristic length, so the transition is
parameterized by a plastic slip distance rather than a raw local strain.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .mat173 import MAT173, mat173, update_mat173


class MAT173Softening(NamedTuple):
    shear: object
    poisson: object
    friction_peak: object
    friction_residual: object
    cohesion_peak: object
    cohesion_residual: object
    dilation_peak: object
    dilation_residual: object
    density: object
    characteristic_length: object
    fracture_slip: object
    novoid: int


def mat173_softening(
    *,
    shear_modulus,
    poisson,
    friction_peak,
    friction_residual,
    cohesion_peak,
    cohesion_residual,
    dilation_peak,
    dilation_residual,
    density,
    characteristic_length,
    fracture_slip,
    novoid=1,
):
    """Create a linearly softening MC material with crack-band length scaling.

    Angles are radians; lengths and stresses use the caller's consistent units.
    The peak and residual surfaces both pass through the MAT173 adapter's
    parameter checks. ``fracture_slip`` is the plastic slip distance over
    which peak strength is reduced to residual strength.
    """
    peak = mat173(
        shear_modulus=shear_modulus,
        poisson=poisson,
        friction_angle=friction_peak,
        cohesion=cohesion_peak,
        dilation_angle=dilation_peak,
        density=density,
        novoid=novoid,
    )
    residual = mat173(
        shear_modulus=shear_modulus,
        poisson=poisson,
        friction_angle=friction_residual,
        cohesion=cohesion_residual,
        dilation_angle=dilation_residual,
        density=density,
        novoid=novoid,
    )
    raw = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            peak.shear,
            peak.poisson,
            peak.friction,
            residual.friction,
            peak.cohesion,
            residual.cohesion,
            peak.dilation,
            residual.dilation,
            peak.density,
            characteristic_length,
            fracture_slip,
        )
    )
    try:
        arrays = np.broadcast_arrays(*raw)
    except ValueError as error:
        raise ValueError("softening properties must share a common point count") from error
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("softening properties must be finite")
    (shear, nu, phi_p, phi_r, c_p, c_r, psi_p, psi_r, density_array, length, slip) = arrays
    if np.any(length <= 0) or np.any(slip <= 0):
        raise ValueError("characteristic length and fracture slip must be positive")
    if np.any(phi_r > phi_p) or np.any(c_r > c_p) or np.any(psi_r > psi_p):
        raise ValueError("residual friction, cohesion and dilation cannot exceed peak values")
    values = tuple(
        float(v) if v.ndim == 0 else v.copy()
        for v in (shear, nu, phi_p, phi_r, c_p, c_r, psi_p, psi_r, density_array, length, slip)
    )
    return MAT173Softening(*values, int(novoid))


def softening_fraction(kappa, material):
    """Return the bounded crack-band softening fraction at old history state."""
    return jnp.clip(kappa * material.characteristic_length / material.fracture_slip, 0.0, 1.0)


def softening_properties(kappa, material):
    """Return friction, cohesion and dilation at a stored plastic shear strain."""
    fraction = softening_fraction(kappa, material)

    def points(value):
        value = jnp.asarray(value, dtype=fraction.dtype)
        return jnp.broadcast_to(value, fraction.shape) if value.ndim == 0 else value

    phi = points(material.friction_peak) + fraction * (
        points(material.friction_residual) - points(material.friction_peak)
    )
    cohesion = points(material.cohesion_peak) + fraction * (
        points(material.cohesion_residual) - points(material.cohesion_peak)
    )
    dilation = points(material.dilation_peak) + fraction * (
        points(material.dilation_residual) - points(material.dilation_peak)
    )
    return phi, cohesion, dilation, fraction


def update_mat173_softening(old, strain, J, dt, p):
    """Explicitly evolve peak/residual properties around the existing MC return."""
    count = strain.shape[0]

    def points(value):
        value = jnp.asarray(value, dtype=strain.dtype)
        return jnp.broadcast_to(value, (count,)) if value.ndim == 0 else value

    phi, cohesion, dilation, _ = softening_properties(old.hardening, p)
    mc = MAT173(
        points(p.shear), points(p.poisson), phi, cohesion, dilation, points(p.density), p.novoid
    )
    trial = update_mat173(old, strain, J, dt, mc)
    plastic_increment = trial.plastic_tensor - old.plastic_tensor
    eye = jnp.eye(3, dtype=strain.dtype)
    deviator = (
        plastic_increment
        - jnp.trace(plastic_increment, axis1=-2, axis2=-1)[:, None, None] * eye / 3
    )
    delta_kappa = jnp.sqrt(jnp.maximum((2 / 3) * jnp.sum(deviator * deviator, axis=(-2, -1)), 0.0))
    kappa = old.hardening + delta_kappa
    fraction = softening_fraction(kappa, p)
    damage_monotone = jnp.all(fraction + 1e-14 >= old.damage)
    return trial._replace(
        hardening=kappa,
        damage=fraction,
        postpeak=old.postpeak | (fraction > 0),
        valid=trial.valid & damage_monotone,
    )
