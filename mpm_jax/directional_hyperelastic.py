"""Assumed normalized directional hyperelastic families for plate comparisons."""
import jax
import jax.numpy as jnp

FAMILIES = {
    'hexagonal': ((0.,120.,240.), (1.,1.,1.)),
    'square': ((0.,90.), (1.5,1.5)),
    'rectangular': ((0.,90.), (2.25,.75)),
    'octagonal': ((0.,45.,90.,135.), (.75,.75,.75,.75)),
    'isotropic': ((), ()),
}


def energy_density(F, mu=1.e6, lam=2.333333333e6, kf=5.e6,
                   strain_scale=.03, angle=0., symmetry='hexagonal', exponent=4):
    """All reinforced families have sum(weights)=3; exponent shared across cases.

    Wdir=kf/(p es^(p-2))*sum(w_i max(a_i.E.a_i,0)^p).
    p=4 reproduces the original three-direction example exactly.
    The isotropic control is the unreinforced Neo-Hookean matrix.
    """
    if exponent not in (4,6):raise ValueError('Use common exponent 4 or 6')
    degrees, weights = FAMILIES[symmetry]
    logJ = jnp.log(jnp.linalg.det(F))
    matrix = .5*mu*(jnp.sum(F*F)-3)-mu*logJ+.5*lam*logJ**2
    if not degrees:return matrix
    theta=angle+jnp.deg2rad(jnp.asarray(degrees,dtype=F.dtype))
    a=jnp.stack((jnp.cos(theta),jnp.sin(theta),jnp.zeros(len(degrees))),axis=1)
    E=.5*(F.T@F-jnp.eye(3,dtype=F.dtype))
    e=jnp.einsum('ai,ij,aj->a',a,E,a)
    return matrix+kf/(exponent*strain_scale**(exponent-2))*jnp.sum(jnp.asarray(weights)*jnp.maximum(e,0.)**exponent)

first_piola=jax.grad(energy_density,argnums=0)


def stress(F, **parameters):
    return first_piola(F,**parameters)@F.T/jnp.linalg.det(F)
