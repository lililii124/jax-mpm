"""Three-field frictionless nodal contact; no Hertz/adhesion approximation.

Mass-gradient normals follow contact.py. Dual projected Gauss-Seidel solves
all three unilateral constraints together, including shared-strand coupling.
Inputs use identical global lattice nodes, shape (3,n[,3]).
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

PAIRS=((0,1),(0,2),(1,2))

class ContactResult(NamedTuple):
    velocity: object
    pair_impulse: object
    fixture_impulse: object
    closing_residual: object
    valid: object
    sweeps: object


def project(mass,velocity,gradient,centres,clamped,*,tolerance=1e-8,max_sweeps=64):
    inv=jnp.where((mass>0)&~clamped,1/jnp.maximum(mass,1e-300),0.)
    normals=[]; active=[]
    for a,b in PAIRS:
        raw=gradient[b]/jnp.maximum(mass[b],1e-300)[:,None]-gradient[a]/jnp.maximum(mass[a],1e-300)[:,None]
        raw=jnp.where((jnp.linalg.norm(raw,axis=1)>1e-12)[:,None],raw,centres[b]-centres[a])
        norm=jnp.linalg.norm(raw,axis=1)
        normals.append(raw/jnp.maximum(norm,1e-300)[:,None])
        active.append((mass[a]>0)&(mass[b]>0)&(norm>1e-12))
    normals=jnp.stack(normals);active=jnp.stack(active)
    multipliers=jnp.zeros_like(active,dtype=mass.dtype)
    def residual(v,lam):
        values=[]
        for k,(a,b) in enumerate(PAIRS):
            c=jnp.sum((v[a]-v[b])*normals[k],axis=1)
            values.append(jnp.where(active[k],jnp.where(lam[k]>0,jnp.abs(c),jnp.maximum(c,0)),0))
        return jnp.max(jnp.stack(values))
    def body(carry):
        it,v,lam,_=carry
        for k,(a,b) in enumerate(PAIRS):
            c=jnp.sum((v[a]-v[b])*normals[k],axis=1)
            denom=inv[a]+inv[b]
            new=jnp.where(active[k]&(denom>0),jnp.maximum(lam[k]+c/jnp.maximum(denom,1e-300),0),0)
            impulse=(new-lam[k])[:,None]*normals[k]
            v=v.at[a].add(-inv[a,:,None]*impulse).at[b].add(inv[b,:,None]*impulse)
            lam=lam.at[k].set(new)
        return it+1,v,lam,residual(v,lam)
    it,v,lam,r=jax.lax.while_loop(lambda q:(q[0]<max_sweeps)&(q[3]>tolerance),body,(jnp.array(0),velocity,multipliers,residual(velocity,multipliers)))
    pair=lam[:,:,None]*normals
    # Pair impulses remain equal and opposite even at prescribed nodes.
    # The fixture cancels impulses on prescribed degrees of freedom.
    fixture=jnp.zeros_like(velocity)
    for k,(a,b) in enumerate(PAIRS):
        fixture=fixture.at[a].add(jnp.where(clamped[a,:,None],pair[k],0)).at[b].add(jnp.where(clamped[b,:,None],-pair[k],0))
    return ContactResult(v,pair,fixture,r,jnp.isfinite(r)&(r<=tolerance),it)


def mass_gradient(state,support,layout):
    """Negative spatial mass gradient, using the exact quadratic basis derivative."""
    from itertools import product
    f=-support.dpos[:,0,:]/layout.dx
    w=jnp.stack((.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2),axis=1)
    dw=jnp.stack((f-1.5,2*(1-f),f-.5),axis=1)/layout.dx
    offsets=jnp.asarray(list(product(range(3),repeat=3)))
    result=[]
    for axis in range(3):
        weight=jnp.ones_like(support.weights)
        for d in range(3):weight=weight*(dw if d==axis else w)[:,offsets[:,d],d]
        result.append(jnp.zeros(layout.node_count,dtype=state.x.dtype).at[support.indices].add(-state.mass[:,None]*weight,mode='drop'))
    return jnp.stack(result,axis=1)
