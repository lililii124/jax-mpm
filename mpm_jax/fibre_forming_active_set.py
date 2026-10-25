"""Same three-constraint mass-metric QP; bounded exact fallback after64 PGS.

Eight active sets are enumerated. Rank-deficient sets are rejected explicitly;
there is no pseudoinverse, mass cutoff, tolerance enlargement or changed target.
"""
from typing import NamedTuple
from itertools import product
import jax
import jax.numpy as jnp
from .fibre_forming_contact import project as project_pgs,PAIRS


class ContactResult(NamedTuple):
    velocity: object
    pair_impulse: object
    fixture_impulse: object
    closing_residual: object
    valid: object
    sweeps: object
    exact_nodes: object
    exact_successes: object
    capacity_valid: object
    rank_rejections: object


def solve_nodes(mass,velocity,normals,active,clamped,tolerance=1e-8):
    """Independent tiny QPs: (batch,body,xyz), normals(batch,pair,xyz)."""
    count=mass.shape[0]
    inverse=jnp.where((mass>0)&~clamped,1/jnp.maximum(mass,1e-300),0.)
    B=jnp.zeros((count,3,3,3),dtype=velocity.dtype)
    for k,(a,b) in enumerate(PAIRS):
        B=B.at[:,k,a,:].set(normals[:,k,:]).at[:,k,b,:].set(-normals[:,k,:])
    Q=jnp.einsum('nkar,na,nlar->nkl',B,inverse,B)
    c=jnp.einsum('nkar,nar->nk',B,velocity)
    scale=jnp.sqrt(jnp.maximum(jnp.diagonal(Q,axis1=1,axis2=2),0.))
    safe_scale=jnp.where(scale>0,scale,1.)
    H=Q/(safe_scale[:,:,None]*safe_scale[:,None,:]);rhs=c/safe_scale
    identity=jnp.eye(3,dtype=velocity.dtype)
    best_velocity=velocity;best_lambda=jnp.zeros_like(c);best_objective=jnp.full(count,jnp.inf,dtype=velocity.dtype)
    found=jnp.zeros(count,dtype=bool);rank_rejections=jnp.zeros(count,dtype=jnp.int32)
    best_residual=jnp.full(count,jnp.inf,dtype=velocity.dtype)
    for subset in product((False,True),repeat=3):
        mask=jnp.asarray(subset,dtype=bool);weight=mask.astype(velocity.dtype)
        matrix=H*weight[None,:,None]*weight[None,None,:]+identity*(1-weight)[None,:,None]
        eigen=jnp.linalg.eigvalsh(matrix)
        full_rank=eigen[:,0]>64*jnp.finfo(velocity.dtype).eps*jnp.maximum(eigen[:,-1],1.)
        allowed=jnp.all(~mask[None,:]|active,axis=1)
        rank_rejections=rank_rejections+(allowed&~full_rank).astype(jnp.int32)
        matrix=jnp.where(full_rank[:,None,None],matrix,identity)
        scaled=jnp.linalg.solve(matrix,(rhs*weight[None,:])[...,None])[...,0]
        lam=(scaled/safe_scale)*weight[None,:]
        trial=velocity-inverse[:,:,None]*jnp.einsum('nkar,nk->nar',B,lam)
        closing=jnp.einsum('nkar,nar->nk',B,trial)
        residual=jnp.max(jnp.where(active,jnp.where(lam>0,jnp.abs(closing),jnp.maximum(closing,0.)),0.),axis=1)
        feasible=allowed&full_rank&jnp.all(lam>=0,axis=1)&jnp.isfinite(residual)&(residual<=tolerance)
        objective=.5*jnp.sum(mass*jnp.sum((trial-velocity)**2,axis=2),axis=1)
        choose=feasible&(objective<best_objective)
        best_velocity=jnp.where(choose[:,None,None],trial,best_velocity)
        best_lambda=jnp.where(choose[:,None],lam,best_lambda)
        best_residual=jnp.where(choose,residual,best_residual)
        best_objective=jnp.where(choose,objective,best_objective);found=found|feasible
    return best_velocity,best_lambda,found,best_residual,rank_rejections


def project(mass,velocity,gradient,centres,clamped,*,tolerance=1e-8,max_sweeps=64,fallback_capacity=256):
    base=project_pgs(mass,velocity,gradient,centres,clamped,tolerance=tolerance,max_sweeps=max_sweeps)
    zero=jnp.asarray(0,dtype=jnp.int32)
    def unchanged(_):return ContactResult(*base,zero,zero,jnp.asarray(True),zero)
    def fallback(_):
        nodes=mass.shape[1];normals=[];active=[];residual=[]
        for k,(a,b) in enumerate(PAIRS):
            raw=gradient[b]/jnp.maximum(mass[b],1e-300)[:,None]-gradient[a]/jnp.maximum(mass[a],1e-300)[:,None]
            raw=jnp.where((jnp.linalg.norm(raw,axis=1)>1e-12)[:,None],raw,centres[b]-centres[a]);norm=jnp.linalg.norm(raw,axis=1)
            n=raw/jnp.maximum(norm,1e-300)[:,None];normals.append(n)
            present=(mass[a]>0)&(mass[b]>0)&(norm>1e-12);active.append(present)
            closing=jnp.sum((base.velocity[a]-base.velocity[b])*n,axis=1)
            lam=jnp.sum(base.pair_impulse[k]*n,axis=1)
            residual.append(jnp.where(present,jnp.where(lam>0,jnp.abs(closing),jnp.maximum(closing,0.)),0.))
        normals=jnp.stack(normals);active=jnp.stack(active);r=jnp.max(jnp.stack(residual),axis=0)
        bad=r>tolerance;number=jnp.sum(bad,dtype=jnp.int32)
        indices=jnp.nonzero(bad,size=fallback_capacity,fill_value=nodes)[0]
        real=indices<nodes;gather=jnp.where(real,indices,0);scatter=jnp.where(real,indices,nodes)
        packed_mass=mass[:,gather].T;packed_velocity=velocity[:,gather,:].transpose(1,0,2)
        packed_normals=normals[:,gather,:].transpose(1,0,2)
        v,lam,found,_,rank=solve_nodes(packed_mass,packed_velocity,packed_normals,active[:,gather].T,clamped[:,gather].T,tolerance)
        found=found&real
        old_v=base.velocity[:,gather,:].transpose(1,0,2)
        v=jnp.where(found[:,None,None],v,old_v)
        impulses=lam[:,:,None]*packed_normals
        old_impulse=base.pair_impulse[:,gather,:].transpose(1,0,2)
        impulses=jnp.where(found[:,None,None],impulses,old_impulse)
        final_velocity=base.velocity.at[:,scatter,:].set(v.transpose(1,0,2),mode='drop')
        pair=base.pair_impulse.at[:,scatter,:].set(impulses.transpose(1,0,2),mode='drop')
        fixture=jnp.zeros_like(velocity);errors=[]
        for k,(a,b) in enumerate(PAIRS):
            fixture=fixture.at[a].add(jnp.where(clamped[a,:,None],pair[k],0.)).at[b].add(jnp.where(clamped[b,:,None],-pair[k],0.))
            closing=jnp.sum((final_velocity[a]-final_velocity[b])*normals[k],axis=1)
            multiplier=jnp.sum(pair[k]*normals[k],axis=1)
            errors.append(jnp.where(active[k],jnp.where(multiplier>0,jnp.abs(closing),jnp.maximum(closing,0.)),0.))
        final_r=jnp.max(jnp.stack(errors));capacity_valid=number<=fallback_capacity
        valid=capacity_valid&jnp.all(~real|found)&jnp.isfinite(final_r)&(final_r<=tolerance)
        return ContactResult(final_velocity,pair,fixture,final_r,valid,base.sweeps,number,jnp.sum(found,dtype=jnp.int32),capacity_valid,jnp.sum(jnp.where(real,rank,0),dtype=jnp.int32))
    return jax.lax.cond(base.valid,unchanged,fallback,None)
