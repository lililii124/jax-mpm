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
    gram_diagonal: jax.Array
    diagonal: bool
    duplicate_entries_merged: int
    max_guides_per_node: int


def make_guide_operator(reference, volume, density, support, nodal_mass, guides):
    """Build sparse rows as particle-stencil entries; no locked section nodes."""
    import numpy as np
    volume=jnp.asarray(volume);guides=jnp.asarray(guides,jnp.int32)
    if guides.ndim!=2 or guides.shape[0]<1 or guides.shape[1]<1:
        raise ValueError('guides must be a nonempty (sections, particles-per-section) array')
    guide_ids=np.asarray(guides,dtype=np.int32)
    particle_ids=guide_ids.reshape(-1)
    if np.any((particle_ids<0)|(particle_ids>=len(reference))):
        raise ValueError('guide particle index out of range')
    volume_np=np.asarray(volume);weights_np=np.asarray(support.weights);indices_np=np.asarray(support.indices)
    nodal_mass_np=np.asarray(nodal_mass);guide_count=guide_ids.shape[0]
    pmass=float(density)*volume_np[particle_ids]
    band_mass=pmass.reshape(guide_ids.shape).sum(axis=1)
    if np.any(band_mass<=0):
        raise ValueError('guide section has nonpositive material mass')
    coeff=(pmass[:,None]*weights_np[particle_ids]/np.repeat(band_mass,guide_ids.shape[1])[:,None]).reshape(-1)
    nodes=indices_np[particle_ids].reshape(-1)
    raw_labels=np.repeat(np.arange(guide_count,dtype=np.int32),guide_ids.shape[1]*indices_np.shape[1])
    if np.any((nodes<0)|(nodes>=len(nodal_mass_np))):raise ValueError('Guide stencil contains an invalid packed node')
    # Coalesce all repeated particle-stencil contributions to one coefficient
    # per (guide,node), which is required by the mass-metric Gram diagonal.
    nnode=len(nodal_mass_np);keys=raw_labels.astype(np.int64)*nnode+nodes
    unique,inverse=np.unique(keys,return_inverse=True);coeff=np.bincount(inverse,weights=coeff)
    labels=(unique//nnode).astype(np.int32);nodes=(unique%nnode).astype(np.int32)
    _,owners=np.unique(nodes,return_counts=True);max_owners=int(owners.max(initial=0))
    diagonal=max_owners<=1
    if np.any(nodal_mass_np[nodes]<=0):raise ValueError('Guide constraint contains a zero-mass node')
    inverse_mass=1/np.where(nodal_mass_np>0,nodal_mass_np,1.)
    gram_diagonal=np.bincount(labels,weights=coeff**2*inverse_mass[nodes],minlength=guide_count)
    if np.any(gram_diagonal<=0):raise ValueError('Guide Gram diagonal must be strictly positive')
    return GuideOperator(jnp.asarray(nodes),jnp.asarray(coeff),jnp.asarray(labels),jnp.asarray(inverse_mass),
        guide_count,jnp.asarray(gram_diagonal),diagonal,int(len(keys)-len(unique)),max_owners)


def _apply_transpose(operator, multipliers, node_count):
    return jnp.zeros((node_count,),multipliers.dtype).at[operator.nodes].add(
        operator.coefficients*multipliers[operator.labels],mode='drop')


def nodal_impulse(operator, multipliers, node_count):
    """Return the exact nodal impulse G^T lambda for a guide multiplier."""
    return _apply_transpose(operator,multipliers,node_count)


def sample(operator, values):
    """Apply G to a nodal scalar field, preserving all finite guide rows."""
    values=jnp.asarray(values)
    return jnp.zeros((operator.guide_count,),values.dtype).at[operator.labels].add(
        operator.coefficients*values[operator.nodes],mode='drop')


def project(operator, values, target, *, tolerance=1e-11, iterations=96):
    """Mass metric projection of nodal y values onto Gv=target.

    If coalesced guide supports are disjoint, the Gram is exactly diagonal and
    is solved by division.  Otherwise Jacobi-preconditioned CG uses the true
    residual for early stopping.  Both paths apply the exact G/G^T adjoint.
    """
    values=jnp.asarray(values);target=jnp.asarray(target,dtype=values.dtype)
    if values.ndim!=2 or values.shape[1]!=3 or target.shape!=(operator.guide_count,):
        raise ValueError('nodal values must be (nodes,3), with one target per guide')
    def G(y):return sample(operator,y)
    rhs=target-G(values[:,1])
    if operator.diagonal:
        x=rhs/operator.gram_diagonal
    else:
        def gram(q):
            z=_apply_transpose(operator,q,len(values))*operator.inverse_mass
            return G(z)
        x=jnp.zeros_like(rhs);r=rhs;z=r/operator.gram_diagonal;p=z;rz=jnp.vdot(r,z)
        rhs_norm=jnp.maximum(jnp.linalg.norm(rhs),1e-300)
        def condition(carry):k,_,r,_,_,_=carry;return (k<iterations)&(jnp.linalg.norm(r)>tolerance*rhs_norm)
        def body(carry):
            k,x,r,z,p,rz=carry;Ap=gram(p);den=jnp.vdot(p,Ap)
            alpha=jnp.where(den>0,rz/jnp.where(den>0,den,1.),0.)
            xn=x+alpha*p;rn=r-alpha*Ap;zn=rn/operator.gram_diagonal;rzn=jnp.vdot(rn,zn)
            beta=jnp.where(rz>0,rzn/jnp.where(rz>0,rz,1.),0.)
            return k+1,xn,rn,zn,zn+beta*p,rzn
        _,x,_,_,_,_=jax.lax.while_loop(condition,body,(jnp.asarray(0),x,r,z,p,rz))
    correction=_apply_transpose(operator,x,len(values))*operator.inverse_mass
    projected=values.at[:,1].add(correction)
    residual=G(projected[:,1])-target
    return projected,x,residual
