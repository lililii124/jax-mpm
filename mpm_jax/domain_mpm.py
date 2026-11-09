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
from . import implicit_contact,normal_cohesive
from .boundaries import VelocityBox,CyclicVelocityBox


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


class ContactWork(NamedTuple):
    stored_change: jax.Array
    power_error: jax.Array
    iterations: jax.Array
    residual: jax.Array
    normal_resultant: jax.Array
    net_moment: jax.Array
    softening_number: jax.Array


@partial(jax.jit,static_argnames=('config','n_fields','tile_radius','batch_size','cohesive_law'))
def step(state,material,config,time,fields=None,force=None,dt=None,n_fields=1,tile_radius=1,batch_size=32,
         contact=None,contact_stiffness=None,cohesive_law=None):
    """One trial step; callers commit only if every returned state.valid is true.

    Boundary boxes act on each nodal field in physical coordinates. Optional
    nodal force is held fixed over the step. No interface/contact law is added
    implicitly between separate fields. Supplying contact adds a fixed-normal
    implicit interface and returns (state, interface, work, contact_work).
    """
    if config.dim!=2 or config.material not in ('elastic','neo-hookean') or n_fields<1:
        raise ValueError('Domain midpoint dynamics requires 2D neo-Hookean elasticity')
    if cohesive_law is not None:
        if contact is None or not isinstance(contact,normal_cohesive.Interface):
            raise ValueError('A normal-cohesive interface is required for cohesive evolution')
        contact_stiffness=cohesive_law.stiffness
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
    if contact is not None:
        if n_fields!=2 or config.boundary_cells or any(not isinstance(b,(VelocityBox,CyclicVelocityBox)) for b in config.boundaries):
            raise ValueError('Implicit contact requires two fields and prescribed velocity boxes only')
        if contact_stiffness is None:
            raise ValueError('An explicit contact stiffness is required')
        free=constrain(jnp.ones_like(initial),time+.5*dt)-constrain(jnp.zeros_like(initial),time+.5*dt)
        contact_trace=implicit_contact.trace(contact,mass,free,config)

    def evaluate(velocity):
        middle=.5*(initial+velocity)
        vp,_=grid_to_fragment(middle,support)
        gradient=velocity_gradient(middle,support)
        F,tau,valid=midpoint_bulk(state.F,gradient,dt,material)
        bulk=internal_force(tau,state.reference_area,support,total)
        raw=jnp.where(occupied[:,None],initial+dt*((bulk+force)/denominator+jnp.asarray(config.gravity)),0.)
        bounded=2*constrain(.5*(initial+raw),time+.5*dt)-initial
        if contact is not None:
            solved=implicit_contact.solve(contact,contact_trace,initial,bounded,contact_stiffness,dt,config.dx,cohesive_law)
            raw=raw+dt*solved.force/denominator
            bounded=solved.velocity
            valid &= solved.valid
            return bounded,(middle,vp,F,bulk,raw,valid,solved)
        return bounded,(middle,vp,F,bulk,raw,valid,None)

    def iterate(carry):
        velocity,count,_=carry
        following,_=evaluate(velocity)
        residual=jnp.max(jnp.abs(following-velocity))/(1+jnp.max(jnp.abs(following)))
        return following,count+1,residual

    velocity,iterations,residual=jax.lax.while_loop(lambda c:(c[1]<32)&(c[2]>1e-9),iterate,
                                                   (initial,jnp.asarray(0),jnp.asarray(jnp.inf)))
    bounded,(middle,vp,F,bulk,raw,constitutive_valid,solved)=evaluate(velocity)
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
    work=Work(boundary,gravity,external,transfer,projection,geometry,bulk_error,ledger_error,iterations,residual)
    if contact is None:
        return following,work
    new_contact=implicit_contact.advect(contact,contact_trace,middle,dt,transport_tangent=cohesive_law is not None)
    if cohesive_law is None:
        surface_change=implicit_contact.stored_energy(new_contact,contact_stiffness)-implicit_contact.stored_energy(contact,contact_stiffness)
    else:
        new_contact=normal_cohesive.update_history(new_contact)
        surface_change=sum(normal_cohesive.energy(new_contact,cohesive_law))-sum(normal_cohesive.energy(contact,cohesive_law))
    power_error=surface_change+dt*jnp.sum(solved.force*middle)
    work=work._replace(ledger_error=work.ledger_error+surface_change)
    contact_valid=constitutive_valid&jnp.isfinite(power_error)
    energy_scale=jnp.abs(surface_change)+jnp.abs(dt*jnp.sum(solved.force*middle))+K0+1e-20
    contact_valid &= jnp.abs(power_error)<=1e-8*energy_scale
    new_contact=new_contact._replace(valid=new_contact.valid&valid&contact_valid)
    following=following._replace(valid=following.valid&contact_valid)
    index=jnp.arange(size);nodes=config.dx*jnp.stack((index//(config.n_grid+1),index%(config.n_grid+1)),axis=1)
    f=solved.force.reshape(2,size,2)
    moment=jnp.sum(nodes[None,:,0]*f[:,:,1]-nodes[None,:,1]*f[:,:,0])
    contact_work=ContactWork(surface_change,power_error,solved.iterations,solved.residual,
                             jnp.sum(f[0],axis=0)@contact.normal,moment,solved.softening_number)
    return following,new_contact,work,contact_work
