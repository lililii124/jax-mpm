"""Fixed-normal frictionless contact with an implicit, work-conjugate trace.

This is a planar contact operator, not a cohesive fracture-growth law.
Quadratic traces are unfiltered; zero-mass support invalidates a step.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .core import stencil


class Interface(NamedTuple):
    plus: jax.Array
    minus: jax.Array
    area: jax.Array
    normal: jax.Array
    valid: jax.Array


def line_interface(start,end,normal,config,gap=0.):
    """Place three Gauss points per straight interval split at spline knots."""
    start,end,normal=map(lambda a:np.asarray(a,dtype=float),(start,end,normal))
    edge=end-start;length=np.linalg.norm(edge)
    if start.shape!=(2,) or end.shape!=(2,) or normal.shape!=(2,) or not np.isfinite(np.r_[start,end,normal,gap]).all():
        raise ValueError('Require finite planar endpoints, normal and gap')
    if length<=0 or np.linalg.norm(normal)<=0:
        raise ValueError('Interface length and normal must be nonzero')
    normal=normal/np.linalg.norm(normal)
    if abs(edge@normal)>1e-12*length:
        raise ValueError('The interface normal must be perpendicular to the segment')
    fractions=[0.,1.]
    knots=(np.arange(config.n_grid)+.5)*config.dx
    for shift in (-.5*gap*normal,.5*gap*normal):
        for axis in (0,1):
            if abs(edge[axis])>1e-15*length:
                values=(knots-start[axis]-shift[axis])/edge[axis]
                fractions.extend(values[(values>1e-12)&(values<1-1e-12)].tolist())
    fractions=np.unique(fractions)
    gauss,weights=np.polynomial.legendre.leggauss(3)
    middle=.5*(fractions[1:]+fractions[:-1]);half=.5*np.diff(fractions)
    t=(middle[:,None]+half[:,None]*gauss).ravel()
    area=(length*half[:,None]*weights).ravel()
    points=start+t[:,None]*edge
    return Interface(jnp.asarray(points+.5*gap*normal),jnp.asarray(points-.5*gap*normal),
                     jnp.asarray(area),jnp.asarray(normal),jnp.asarray(True))


def opening(interface):
    return (interface.plus-interface.minus)@interface.normal


def stored_energy(interface,stiffness):
    return .5*stiffness*jnp.sum(interface.area*jnp.minimum(opening(interface),0.)**2)


def discrete_pressure(old,new,stiffness):
    """Secant of the unilateral quadratic potential, with its new-gap tangent."""
    both=(old<=0)&(new<=0)
    entering=(old>=0)&(new<0)
    leaving=(old<0)&(new>0)
    denominator=jnp.where(entering|leaving,new-old,1.)
    enter=.5*stiffness*new**2/denominator
    leave=-.5*stiffness*old**2/denominator
    pressure=jnp.where(both,.5*stiffness*(old+new),jnp.where(entering,enter,jnp.where(leaving,leave,0.)))
    derivative=jnp.where(both,.5*stiffness,
                         jnp.where(entering,.5*stiffness*new*(new-2*old)/denominator**2,
                                   jnp.where(leaving,.5*stiffness*old**2/denominator**2,0.)))
    return pressure,derivative


class Trace(NamedTuple):
    plus_indices: jax.Array
    minus_indices: jax.Array
    plus_weights: jax.Array
    minus_weights: jax.Array
    nodes: jax.Array
    matrix: jax.Array
    inverse_mass: jax.Array
    compliance: jax.Array
    valid: jax.Array


def trace(interface,mass,free,config):
    """Condense to at most 18Q nodes, without dropping small masses or weights."""
    size=(config.n_grid+1)**2
    if mass.shape!=(2*size,) or free.shape!=(2*size,2):
        raise ValueError('Planar contact requires two complete grid fields')
    plus=stencil(interface.plus,config);minus=stencil(interface.minus,config)
    minus_indices=minus.indices+size
    indices=jnp.concatenate((plus.indices,minus_indices),axis=1)
    values=jnp.concatenate((plus.weights,-minus.weights),axis=1)
    nodes,inverse=jnp.unique(indices.ravel(),return_inverse=True,size=indices.size,fill_value=len(mass))
    scalar=jnp.zeros((len(interface.area),indices.size),dtype=mass.dtype)
    scalar=scalar.at[jnp.arange(len(interface.area))[:,None],inverse.reshape(indices.shape)].add(values)
    matrix=scalar[:,:,None]*interface.normal
    local_mass=jnp.pad(mass,(0,1))[nodes]
    local_free=jnp.pad(free,((0,1),(0,0)))[nodes]
    inv=jnp.where(local_mass[:,None]>0,local_free/jnp.where(local_mass[:,None]>0,local_mass[:,None],1.),0.)
    weighted=matrix*jnp.sqrt(inv)[None,:,:]
    compliance=jnp.einsum('qnd,rnd->qr',weighted,weighted)
    supported=jnp.all((values==0)|(mass[indices]>0))
    valid=interface.valid&plus.valid&minus.valid&supported&jnp.all(interface.area>0)
    valid &= jnp.isfinite(interface.area).all()&jnp.all((free==0)|(free==1))
    valid &= jnp.abs(jnp.sum(interface.normal**2)-1)<1e-12
    return Trace(plus.indices,minus_indices,plus.weights,minus.weights,nodes,matrix,inv,compliance,valid)


def face_velocities(velocity,support):
    plus=jnp.sum(support.plus_weights[:,:,None]*velocity[support.plus_indices],axis=1)
    minus=jnp.sum(support.minus_weights[:,:,None]*velocity[support.minus_indices],axis=1)
    return plus,minus


def advect(interface,support,velocity,dt,transport_tangent=False):
    """Move the normal traces at fixed common spatial tangential coordinates."""
    plus,minus=face_velocities(velocity,support)
    tangent=.5*(plus+minus)
    tangent=tangent-(tangent@interface.normal)[:,None]*interface.normal
    plus=(plus@interface.normal)[:,None]*interface.normal
    minus=(minus@interface.normal)[:,None]*interface.normal
    if transport_tangent:
        plus=plus+tangent;minus=minus+tangent
    return interface._replace(plus=interface.plus+dt*plus,minus=interface.minus+dt*minus)


def scatter(traction,support,area,shape):
    local=-jnp.einsum('qnd,q->nd',support.matrix,area*traction)
    return jnp.zeros(shape,dtype=traction.dtype).at[support.nodes].add(local,mode='drop')


class Solve(NamedTuple):
    velocity: jax.Array
    force: jax.Array
    gap: jax.Array
    iterations: jax.Array
    residual: jax.Array
    valid: jax.Array
    softening_number: jax.Array


def solve(interface,support,initial,free_velocity,stiffness,dt,dx,cohesive_law=None):
    """Solve the contact gap after eliminating unconstrained grid velocities."""
    old=opening(interface);plus,minus=face_velocities(.5*(initial+free_velocity),support)
    rhs=old+dt*((plus-minus)@interface.normal);alpha=.5*dt**2
    scale=dx+jnp.max(jnp.abs(old))+jnp.max(jnp.abs(rhs))

    def traction(gap):
        if cohesive_law is None:
            return discrete_pressure(old,gap,stiffness)
        from .normal_cohesive import discrete_traction
        return discrete_traction(old,gap,interface.maximum,interface.bonded,cohesive_law)

    def equation(gap):
        pressure,derivative=traction(gap)
        residual=gap-rhs+alpha*(support.compliance@(interface.area*pressure))
        tangent=jnp.eye(len(old))+alpha*support.compliance*(interface.area*derivative)[None,:]
        return residual,tangent

    def iterate(carry):
        gap,count,_=carry
        residual,tangent=equation(gap)
        change=jnp.linalg.solve(tangent,-residual)
        baseline=jnp.max(jnp.abs(residual))
        def trial(factor):
            value=gap+factor*change
            error=jnp.max(jnp.abs(equation(value)[0]))
            return value,error
        value,error=trial(1.)
        def backtrack(c):
            factor,_,_,attempt=c;factor=.5*factor
            value,error=trial(factor)
            return factor,value,error,attempt+1
        _,value,error,_=jax.lax.while_loop(lambda c:(c[2]>baseline)&(c[3]<16),backtrack,
                                          (jnp.asarray(1.),value,error,jnp.asarray(0)))
        return value,count+1,error/scale

    residual0=jnp.max(jnp.abs(equation(rhs)[0]))/scale
    gap,iterations,residual=jax.lax.while_loop(lambda c:(c[1]<32)&(c[2]>1e-12),iterate,
                                             (rhs,jnp.asarray(0),residual0))
    pressure,_=traction(gap)
    force=scatter(pressure,support,interface.area,initial.shape)
    local_force=jnp.pad(force,((0,1),(0,0)))[support.nodes]
    correction=dt*support.inverse_mass*local_force
    velocity=free_velocity.at[support.nodes].add(correction,mode='drop')
    valid=support.valid&jnp.isfinite(stiffness)&(stiffness>0)&jnp.isfinite(dt)&(dt>0)&(residual<=1e-10)
    valid &= jnp.all(jnp.isfinite(velocity))&jnp.all(jnp.isfinite(force))
    softening_number=jnp.asarray(0.,dtype=old.dtype)
    if cohesive_law is not None:
        root_area=jnp.sqrt(interface.area)
        weighted=root_area[:,None]*support.compliance*root_area[None,:]
        spectral_bound=jnp.max(jnp.sum(jnp.abs(weighted),axis=1))
        negative_slope=.5*cohesive_law.strength/(cohesive_law.final_opening-cohesive_law.peak_opening)
        softening_number=alpha*negative_slope*spectral_bound
        # A sufficient bound for a positive effective inertial tangent. Do not
        # accept an arbitrary Newton root after softening makes it indefinite.
        valid &= softening_number<.95
        valid &= jnp.all(jnp.isfinite(interface.maximum))&jnp.all(interface.maximum>=0)
        valid &= jnp.all((~interface.bonded)|(interface.maximum>=jnp.maximum(old,0.)))
    return Solve(velocity,force,gap,iterations,residual,valid,softening_number)
