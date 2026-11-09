"""Local elastic crack-tip diagnostics; not a substitute for an interaction integral."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TipFit(NamedTuple):
    mode_I: jax.Array
    mode_II: jax.Array
    T: jax.Array
    relative_residual: jax.Array
    condition: jax.Array
    sample_volume: jax.Array
    valid: jax.Array


def fit_tip_fields(position,stress,volume,tip,direction,radius):
    """Fit KI, KII and nonsingular T stress over 0.4R < r < R.

    Reports the fit residual: a cohesive zone or nearby boundary may invalidate
    the elastic asymptotic interpretation even with a well-conditioned fit.
    """
    direction=direction/jnp.maximum(jnp.linalg.norm(direction),1e-100)
    normal=jnp.array([-direction[1],direction[0]])
    basis=jnp.stack((direction,normal),axis=1)
    offset=position-tip
    local=offset@basis
    distance=jnp.linalg.norm(local,axis=-1)
    angle=jnp.arctan2(local[:,1],local[:,0])
    half=.5*angle
    c,s=jnp.cos(half),jnp.sin(half)
    c3,s3=jnp.cos(3*half),jnp.sin(3*half)
    root=jnp.sqrt(2.)
    mode_i=jnp.stack((c*(1-s*s3),c*(1+s*s3),root*c*s*c3),axis=-1)
    mode_ii=jnp.stack((-s*(2+c*c3),s*c*c3,root*c*(1-s*s3)),axis=-1)
    scale=jnp.sqrt(radius/jnp.maximum(distance,1e-100))
    columns=jnp.stack((scale[:,None]*mode_i,scale[:,None]*mode_ii,
                       jnp.broadcast_to(jnp.array([1.,0.,0.]),mode_i.shape)),axis=-1)
    local_stress=jnp.einsum('ai,pab,bj->pij',basis,stress,basis)
    values=jnp.stack((local_stress[:,0,0],local_stress[:,1,1],root*local_stress[:,0,1]),axis=-1)
    weights=volume*jnp.where((distance>.4*radius)&(distance<radius),(1-(distance/radius)**2)**2,0.)
    sampled=jnp.sum(weights)
    weights/=jnp.where(sampled>0,sampled,1.)
    matrix=jnp.einsum('p,pai,paj->ij',weights,columns,columns)
    rhs=jnp.einsum('p,pai,pa->i',weights,columns,values)
    eigenvalues=jnp.linalg.eigvalsh(matrix)
    condition=eigenvalues[-1]/jnp.maximum(eigenvalues[0],1e-100)
    valid=(sampled>0)&(eigenvalues[0]>1e-10)&(condition<1e8)
    coefficient=jnp.linalg.solve(jnp.where(valid,matrix,jnp.eye(3)),jnp.where(valid,rhs,0.))
    predicted=jnp.einsum('pai,i->pa',columns,coefficient)
    error=jnp.sum(weights[:,None]*(predicted-values)**2)
    norm=jnp.sum(weights[:,None]*values**2)
    residual=jnp.sqrt(error/jnp.maximum(norm,1e-100))
    factor=jnp.sqrt(2*jnp.pi*radius)
    return TipFit(coefficient[0]*factor,coefficient[1]*factor,coefficient[2],residual,condition,sampled,valid)


def mts_angle(mode_i,mode_ii):
    """Classic maximum tangential-stress kink angle for nonnegative mode I."""
    opening=jnp.maximum(mode_i,0.)
    denominator=opening+jnp.sqrt(opening*opening+8*mode_ii*mode_ii)
    return 2*jnp.arctan(-2*mode_ii/jnp.maximum(denominator,1e-100))
