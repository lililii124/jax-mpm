"""Midpoint bulk dynamics on moving material domains with an explicit energy ledger.

The change in the affine kinetic moment is recorded as a numerical geometry
term. It is not fracture dissipation or a proven conservative transport rule.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import grid_boundaries
from .cohesive_crack import midpoint_bulk
from .domain_transfer import batch_stencils,particle_to_grid,grid_to_fragment,velocity_gradient,internal_force


def kinetic_energy(state,config,moment=None):
    if moment is None:
        moment=.25*config.dx**2*jnp.eye(2)+state.F@state.reference_covariance@state.F.swapaxes(-1,-2)
    affine=jnp.trace(state.C@moment@state.C.swapaxes(-1,-2),axis1=-2,axis2=-1)
    return .5*jnp.sum(state.mass*(jnp.sum(state.v**2,axis=1)+affine))


def elastic_energy(state,material):
    logJ=jnp.log(jnp.linalg.det(state.F))
    density=.5*material.mu*(jnp.sum(state.F**2,axis=(1,2))-2)-material.mu*logJ+.5*material.lam*logJ**2
    return jnp.sum(state.reference_area*density)


class Work(NamedTuple):
    boundary: jax.Array
    gravity: jax.Array
    external: jax.Array
    transfer_loss: jax.Array
    boundary_projection_loss: jax.Array
    geometry_energy: jax.Array
    bulk_error: jax.Array
    ledger_error: jax.Array
    iterations: jax.Array
    residual: jax.Array


@partial(jax.jit,static_argnames=('config','n_fields','tile_radius','batch_size'))
def step(state,material,config,time,fields=None,force=None,dt=None,n_fields=1,tile_radius=1,batch_size=32):
    """One trial step; callers commit only if every returned state.valid is true.

    Boundary boxes act on each nodal field in physical coordinates. Optional
    nodal force is held fixed over the step. No interface/contact law is added
    implicitly between separate fields.
    """
    if config.dim!=2 or config.material not in ('elastic','neo-hookean') or n_fields<1:
        raise ValueError('Domain midpoint dynamics requires 2D neo-Hookean elasticity')
    dt=jnp.asarray(config.dt if dt is None else dt,dtype=state.x.dtype)
    fields=jnp.zeros(len(state.x),dtype=jnp.int32) if fields is None else jnp.asarray(fields)
    if fields.shape!=(len(state.x),) or not jnp.issubdtype(fields.dtype,jnp.integer):
        raise ValueError('Require one integer field index per fragment')
    size=(config.n_grid+1)**2;total=n_fields*size
    support=batch_stencils(state,config,tile_radius,batch_size)
    support=support._replace(indices=support.indices+fields[:,None]*size)
    mass,momentum=particle_to_grid(state,support,total)
    occupied=mass>0;denominator=jnp.where(occupied,mass,1.)[:,None]
    force=jnp.zeros_like(momentum) if force is None else jnp.asarray(force,dtype=momentum.dtype)
    if force.shape!=momentum.shape:
        raise ValueError('Nodal force must match the complete field grid')
    transferred=momentum/denominator

    def constrain(velocity,t):
        return jnp.concatenate([grid_boundaries(velocity[k*size:(k+1)*size],config,t,mass[k*size:(k+1)*size])
                                for k in range(n_fields)])

    initial=constrain(transferred,time)
    initial_impulse=mass[:,None]*(initial-transferred)

    def evaluate(velocity):
        middle=.5*(initial+velocity)
        vp,_=grid_to_fragment(middle,support)
        gradient=velocity_gradient(middle,support)
        F,tau,valid=midpoint_bulk(state.F,gradient,dt,material)
        bulk=internal_force(tau,state.reference_area,support,total)
        raw=jnp.where(occupied[:,None],initial+dt*((bulk+force)/denominator+jnp.asarray(config.gravity)),0.)
        bounded=2*constrain(.5*(initial+raw),time+.5*dt)-initial
        return bounded,(middle,vp,F,bulk,raw,valid)

    def iterate(carry):
        velocity,count,_=carry
        following,_=evaluate(velocity)
        residual=jnp.max(jnp.abs(following-velocity))/(1+jnp.max(jnp.abs(following)))
        return following,count+1,residual

    velocity,iterations,residual=jax.lax.while_loop(lambda c:(c[1]<32)&(c[2]>1e-9),iterate,
                                                   (initial,jnp.asarray(0),jnp.asarray(jnp.inf)))
    bounded,(middle,vp,F,bulk,raw,constitutive_valid)=evaluate(velocity)
    v,C=grid_to_fragment(velocity,support)
    following=state._replace(x=state.x+dt*vp,v=v,C=C,F=F)
    valid=jnp.all(state.valid)&jnp.all(support.valid)&constitutive_valid&(residual<=1e-8)&jnp.isfinite(dt)&(dt>0)
    valid &= jnp.all((fields>=0)&(fields<n_fields))&jnp.all(jnp.isfinite(force))
    valid &= jnp.all(jnp.where(occupied[:,None],True,force==0))
    for array in (following.x,v,C,F):
        valid &= jnp.all(jnp.isfinite(array))
    vertices=following.x[:,None,:]+jnp.einsum('pij,pkj->pki',F,following.reference.vertices-following.reference_centroid[:,None,:])
    active_vertex=following.active[:,None]&(jnp.arange(vertices.shape[1])[None,:]<following.reference.count[:,None])
    in_grid=jnp.all((vertices>=.5*config.dx)&(vertices<config.domain_size-.5*config.dx),axis=2)
    valid &= jnp.all(jnp.where(active_vertex,in_grid,True))
    following=following._replace(valid=following.valid&valid)
    boundary=jnp.sum(mass[:,None]*(bounded-raw)*middle)+jnp.sum(initial_impulse*initial)
    gravity=dt*jnp.sum(mass[:,None]*jnp.asarray(config.gravity)*middle)
    external=dt*jnp.sum(force*middle)
    K0=kinetic_energy(state,config);Kfrozen=kinetic_energy(following,config,support.moment)
    K1=kinetic_energy(following,config)
    transfer=K0-.5*jnp.sum(mass[:,None]*transferred**2)+.5*jnp.sum(mass[:,None]*velocity**2)-Kfrozen
    projection=.5*jnp.sum(mass[:,None]*(initial-transferred)**2)
    geometry=K1-Kfrozen
    change=elastic_energy(following,material)-elastic_energy(state,material)
    bulk_error=dt*jnp.sum(bulk*middle)+change
    ledger_error=K1-K0+change+transfer+projection-geometry-boundary-gravity-external
    return following,Work(boundary,gravity,external,transfer,projection,geometry,bulk_error,ledger_error,iterations,residual)
