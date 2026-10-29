"""Polar rotation and right Hencky increment for a full 3x3 plane-strain state."""

import jax.numpy as jnp


def corotated_increment_plane_strain(F, previous_F):
    a, b = previous_F[..., 0, 0], previous_F[..., 0, 1]
    c, d = previous_F[..., 1, 0], previous_F[..., 1, 1]
    det_previous = a*d-b*c
    inverse = jnp.stack((d, -b, -c, a), axis=-1).reshape(
        previous_F.shape[:-2] + (2, 2)
    ) / det_previous[..., None, None]
    delta = F[..., :2, :2] @ inverse
    a, b = delta[..., 0, 0], delta[..., 0, 1]
    c, d = delta[..., 1, 0], delta[..., 1, 1]
    determinant = a*d-b*c
    diagonal, off_diagonal = a+d, c-b
    norm = jnp.sqrt(diagonal**2 + off_diagonal**2)
    R = jnp.stack((diagonal, -off_diagonal, off_diagonal, diagonal), axis=-1)
    R = R.reshape(delta.shape) / norm[..., None, None]
    C00, C01, C11 = a*a+c*c, a*b+c*d, b*b+d*d
    mean = .5*(C00+C11)
    deviation = .5*(C00-C11)
    ratio_squared = (deviation**2+C01**2)/mean**2
    small = ratio_squared < 1e-8
    safe_ratio = jnp.sqrt(jnp.where(small, .25, ratio_squared))
    quotient = jnp.where(
        small, 1+ratio_squared/3+ratio_squared**2/5,
        jnp.arctanh(safe_ratio)/safe_ratio,
    )
    coefficient = .5*quotient/mean
    isotropic = .5*jnp.log(determinant)
    E = jnp.stack((
        isotropic+coefficient*deviation, coefficient*C01,
        coefficient*C01, isotropic-coefficient*deviation,
    ), axis=-1).reshape(delta.shape)
    strain = jnp.zeros_like(F).at[..., :2, :2].set(E)
    rotation = jnp.zeros_like(F).at[..., :2, :2].set(R).at[..., 2, 2].set(1)
    valid = (det_previous > 0) & (determinant > 0)
    valid = valid & jnp.all(jnp.isfinite(E), axis=(-2, -1))
    for tensor in (F, previous_F):
        valid = valid & (jnp.abs(tensor[..., 2, 2]-1) <= 1e-12)
        valid = valid & jnp.all(jnp.abs(tensor[..., 2, :2]) <= 1e-12, axis=-1)
        valid = valid & jnp.all(jnp.abs(tensor[..., :2, 2]) <= 1e-12, axis=-1)
    return strain, rotation, valid
