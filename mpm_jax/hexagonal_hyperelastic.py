"""Assumed finite-strain sixfold hyperelastic solid, SI units."""
import jax
import jax.numpy as jnp


def energy_density(F, mu=1.e6, lam=2.333333333e6, kf=5.e6,
                   strain_scale=.03, angle=0.):
    directions = angle+jnp.arange(3)*2*jnp.pi/3
    a = jnp.stack((jnp.cos(directions), jnp.sin(directions), jnp.zeros(3)), axis=1)
    E = .5*(F.T@F-jnp.eye(3, dtype=F.dtype))
    e = jnp.einsum('ai,ij,aj->a', a, E, a)
    logJ = jnp.log(jnp.linalg.det(F))
    matrix = .5*mu*(jnp.sum(F*F)-3)-mu*logJ+.5*lam*logJ**2
    return matrix+kf/(4*strain_scale**2)*jnp.sum(jnp.maximum(e, 0.)**4)

first_piola = jax.grad(energy_density, argnums=0)


def stress(F, **parameters):
    P = first_piola(F, **parameters)
    return P@F.T/jnp.linalg.det(F)
