"""Material-gradient transport used by the unadmitted geometric DELTA candidate.

L is a physical spatial gradient, distinct from the APIC transfer coefficient C.
For end-kick particle motion x'=x+dt*v_star, A=I+dt*L_star. The new spatial
gradient is L_star A^-1, and F'=A F. Transfer C is a separate carry. Analytic
checks pass; actual impact accuracy and convergence remain unverified.
"""
from typing import NamedTuple
import jax.numpy as jnp
from .fragment_multifield_v6_mpm import cayley_kinematics


class GeometricResult(NamedTuple):
    F: object
    L: object
    rotation: object
    strain: object
    valid: object


def geometric_kinematics(F,L,increment_gradient,dt):
    eye=jnp.eye(3,dtype=F.dtype)
    star=L+increment_gradient
    A=eye+dt*star
    half=eye+.5*dt*star
    right_solve=lambda numerator,denominator:jnp.swapaxes(jnp.linalg.solve(
        jnp.swapaxes(denominator,-1,-2),jnp.swapaxes(numerator,-1,-2)),-1,-2)
    following=right_solve(star,A)
    midpoint=right_solve(star,half)
    _,rotation,strain=cayley_kinematics(F,midpoint,dt)
    new_F=A@F
    valid=(jnp.all(jnp.isfinite(new_F))&jnp.all(jnp.isfinite(following))
        &jnp.all(jnp.isfinite(rotation))&jnp.all(jnp.isfinite(strain))
        &jnp.isfinite(dt)&(dt>=0)
        &jnp.all(jnp.linalg.det(A)>0)&jnp.all(jnp.linalg.det(half)>0))
    return GeometricResult(new_F,following,rotation,strain,valid)
