"""Solenoidal kinematic velocity extension for RT0 interface transport only.

Physical pressure velocity, its faces, momentum and energy are not updated.
The dimensionless free-face metric below is an extension choice, not a fluid
inertia or a physical pressure. Incompatible sealed regions are rejected.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from .resin_mac import divergence,transpose_divergence


class Extension(NamedTuple):
    velocity: object
    cells: object
    relative_residual: object
    maximum_divergence: object
    protected_change: object
    compatibility_residual: object
    valid: object


def extend(x,active,velocity,fixed,fluid,solid,layout,physical_divergence,*,tolerance=1e-8,max_iterations=400):
    shape=layout.shape;h=layout.dx;index=jnp.floor((x-jnp.asarray(layout.origin))/h).astype(jnp.int32);inside=jnp.all((index>=0)&(index<jnp.asarray(shape)),axis=1);linear=jnp.sum(index*jnp.asarray((shape[1]*shape[2],shape[2],1)),axis=1);linear=jnp.where(inside,linear,shape[0]*shape[1]*shape[2]);owned=jnp.zeros(shape[0]*shape[1]*shape[2],dtype=jnp.int32).at[linear].max(active.astype(jnp.int32),mode='drop').reshape(shape)>0
    halo=jax.lax.reduce_window(owned.astype(jnp.int32),0,jax.lax.max,(3,3,3),(1,1,1),'SAME')>0
    cells=(halo|fluid)&~solid;unknown=cells&~fluid;protected=[]
    for c in range(3):
        left=jnp.pad(fluid,tuple((1,0) if k==c else (0,0) for k in range(3)));right=jnp.pad(fluid,tuple((0,1) if k==c else (0,0) for k in range(3)));protected.append(fixed[c]|left|right)
    free=tuple((~f).astype(x.dtype) for f in protected)
    a,b,c=free;diagonal=(a[1:]+a[:-1]+b[:,1:]+b[:,:-1]+c[:,:,1:]+c[:,:,:-1])/h**2
    solve_mask=unknown&(diagonal>0);D=lambda u:divergence(u,h)/h**3;DT=lambda p:tuple(g/h**3 for g in transpose_divergence(p,h));original=D(velocity)
    compatibility=jnp.max(jnp.where(unknown&~solve_mask,jnp.abs(original),0.));compatibility_tolerance=jnp.maximum(10*physical_divergence,1e-10)
    rhs=jnp.where(solve_mask,original,0.);diag=jnp.where(solve_mask,diagonal,1.)
    def operator(p):
        grad=DT(jnp.where(solve_mask,p,0.));return jnp.where(solve_mask,D(tuple(f*g for f,g in zip(free,grad))),p)
    def solve(_):
        potential,_=cg(operator,rhs,x0=jnp.zeros_like(rhs),M=lambda z:z/diag,tol=tolerance,atol=0.,maxiter=max_iterations);return potential
    potential=jax.lax.cond(jnp.linalg.norm(rhs)>0,solve,lambda _:jnp.zeros_like(rhs),None);gradient=DT(potential);extended=tuple(u-f*g for u,f,g in zip(velocity,free,gradient));residual=operator(potential)-rhs;relative=jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(rhs),1e-300)
    maxdiv=jnp.max(jnp.where(cells,jnp.abs(D(extended)),0.));change=jnp.max(jnp.stack([jnp.max(jnp.where(p,jnp.abs(b-a),0.)) for a,b,p in zip(velocity,extended,protected)]));valid=jnp.all(~active|inside)&jnp.isfinite(relative)&(relative<=10*tolerance)&(compatibility<=compatibility_tolerance)&(change==0.)&jnp.all(jnp.stack([jnp.all(jnp.isfinite(u)) for u in extended]))
    return Extension(extended,cells,relative,maxdiv,change,compatibility,valid)
