"""Mass weighted finite section centroid constraints for reference grid dynamics.

Each row constrains only the material mass centroid's y displacement.  The
operator is assembled from the same reference quadratic MPM interpolation as
the TL force, so the transpose used for projection is its exact adjoint.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class GuideOperator(NamedTuple):
    nodes: jax.Array
    coefficients: jax.Array
    labels: jax.Array
    inverse_mass: jax.Array
    guide_count: int


def make_guide_operator(reference, volume, density, support, nodal_mass, guides):
    """Build sparse rows as particle-stencil entries; no locked section nodes."""
    volume=jnp.asarray(volume);guides=jnp.asarray(guides,jnp.int32)
    if guides.ndim!=2 or guides.shape[0]<1 or guides.shape[1]<1:
        raise ValueError('guides must be a nonempty (sections, particles-per-section) array')
    labels=jnp.repeat(jnp.arange(guides.shape[0],dtype=jnp.int32),guides.shape[1]*support.indices.shape[1])
    particle_ids=guides.reshape(-1)
    if bool(jnp.any((particle_ids<0)|(particle_ids>=len(reference)))):
        raise ValueError('guide particle index out of range')
    pmass=density*volume[particle_ids]
    band_mass=jnp.sum(pmass.reshape(guides.shape),axis=1)
    if bool(jnp.any(band_mass<=0)):
        raise ValueError('guide section has nonpositive material mass')
    coeff=(pmass[:,None]*support.weights[particle_ids]/jnp.repeat(band_mass,guides.shape[1])[:,None]).reshape(-1)
    nodes=support.indices[particle_ids].reshape(-1)
    return GuideOperator(nodes,coeff,labels,1/jnp.where(nodal_mass>0,nodal_mass,1.),guides.shape[0])


def _apply_transpose(operator, multipliers, node_count):
    return jnp.zeros((node_count,),multipliers.dtype).at[operator.nodes].add(
        operator.coefficients*multipliers[operator.labels],mode='drop')


def sample(operator, values):
    """Apply G to a nodal scalar field, preserving all finite guide rows."""
    values=jnp.asarray(values)
    return jnp.zeros((operator.guide_count,),values.dtype).at[operator.labels].add(
        operator.coefficients*values[operator.nodes],mode='drop')


def project(operator, values, target, *, tolerance=1e-11, iterations=96):
    """Mass metric projection of nodal y values onto Gv=target.

    Conjugate gradients solve (G M^-1 G^T) lambda = target-Gv.  The correction
    is M^-1 G^T lambda, which preserves x/z and leaves within-section modes free.
    """
    values=jnp.asarray(values);target=jnp.asarray(target,dtype=values.dtype)
    if values.ndim!=2 or values.shape[1]!=3 or target.shape!=(operator.guide_count,):
        raise ValueError('nodal values must be (nodes,3), with one target per guide')
    def G(y):return sample(operator,y)
    def gram(x):
        z=_apply_transpose(operator,x,len(values))*operator.inverse_mass
        return G(z)
    rhs=target-G(values[:,1]);x=jnp.zeros_like(rhs);r=rhs;p=r;rr=jnp.vdot(r,r)
    def body(_,carry):
        x,r,p,rr=carry;Ap=gram(p);den=jnp.vdot(p,Ap)
        alpha=jnp.where((rr>0)&(den>0),rr/jnp.where(den>0,den,1.),0.)
        xn=x+alpha*p;rn=r-alpha*Ap;rrn=jnp.vdot(rn,rn)
        beta=jnp.where(rr>0,rrn/jnp.where(rr>0,rr,1.),0.)
        return xn,rn,rn+beta*p,rrn
    x,r,_,_=jax.lax.fori_loop(0,iterations,body,(x,r,p,rr))
    correction=_apply_transpose(operator,x,len(values))*operator.inverse_mass
    projected=values.at[:,1].add(correction)
    residual=G(projected[:,1])-target
    return projected,x,residual
