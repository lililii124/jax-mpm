"""Reference-domain fragments with conservative affine velocity splitting."""

from typing import Any,NamedTuple

import jax
import jax.numpy as jnp

from .cut_geometry import Polygon,bulk_quadrature,split


class Moments(NamedTuple):
    area: jax.Array
    centroid: jax.Array
    covariance: jax.Array
    valid: jax.Array


def moments(domain):
    points,weights=bulk_quadrature(domain)
    area=weights.sum();denominator=jnp.where(area>0,area,1.)
    centroid=jnp.sum(weights[:,None]*points,axis=0)/denominator
    relative=points-centroid
    covariance=jnp.einsum('q,qi,qj->ij',weights,relative,relative)/denominator
    valid=domain.valid&(area>=0)&jnp.isfinite(area)&jnp.all(jnp.isfinite(centroid))&jnp.all(jnp.isfinite(covariance))
    return Moments(area,centroid,covariance,valid)


class Fragment(NamedTuple):
    reference: Polygon
    reference_area: jax.Array
    reference_centroid: jax.Array
    reference_covariance: jax.Array
    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    mass: jax.Array
    history: Any
    valid: jax.Array

    @property
    def active(self):
        return (self.mass>0)&(self.reference_area>0)


def fragment(reference,density,x=None,v=None,F=None,C=None,history=None):
    """Initialize one constant-density material domain; history is intensive."""
    geometry=moments(reference);dtype=reference.vertices.dtype
    x=geometry.centroid if x is None else jnp.asarray(x,dtype=dtype)
    v=jnp.zeros(2,dtype=dtype) if v is None else jnp.asarray(v,dtype=dtype)
    F=jnp.eye(2,dtype=dtype) if F is None else jnp.asarray(F,dtype=dtype)
    C=jnp.zeros((2,2),dtype=dtype) if C is None else jnp.asarray(C,dtype=dtype)
    if x.shape!=(2,) or v.shape!=(2,) or F.shape!=(2,2) or C.shape!=(2,2):
        raise ValueError('Fragments require planar positions, velocities and 2x2 tensors')
    density=jnp.asarray(density,dtype=dtype)
    valid=geometry.valid&(geometry.area>0)&jnp.isfinite(density)&(density>0)&(jnp.linalg.det(F)>0)
    valid &= jnp.all(jnp.isfinite(x))&jnp.all(jnp.isfinite(v))&jnp.all(jnp.isfinite(F))&jnp.all(jnp.isfinite(C))
    return Fragment(reference,geometry.area,geometry.centroid,geometry.covariance,x,v,F,C,
                    density*geometry.area,history,valid)


def current_domain(state):
    points=state.x+(state.reference.vertices-state.reference_centroid)@state.F.T
    return Polygon(points,state.reference.count,state.valid)


def kinetic_moment(state,dx):
    """Moment of the volume-averaged quadratic grid basis, not just area inertia."""
    return .25*dx**2*jnp.eye(2,dtype=state.F.dtype)+state.F@state.reference_covariance@state.F.T


class Invariants(NamedTuple):
    mass: jax.Array
    linear_momentum: jax.Array
    angular_momentum: jax.Array
    kinetic_energy: jax.Array
    reference_area: jax.Array


def invariants(state,dx):
    moment=kinetic_moment(state,dx)
    cross=moment@state.C.T
    angular=state.mass*(state.x[0]*state.v[1]-state.x[1]*state.v[0]+cross[0,1]-cross[1,0])
    kinetic=.5*state.mass*(state.v@state.v+jnp.trace(state.C@moment@state.C.T))
    return Invariants(state.mass,state.mass*state.v,angular,kinetic,state.reference_area)


def split_fragment(state,normal,offset):
    """Return two independent fragments without deleting small positive pieces.

    The cut is specified in current coordinates. Constitutive F and intensive
    history are inherited. This is an instantaneous conservative split, not a
    crack-growth criterion or a time update for the evolving kinetic moment.
    """
    normal=jnp.asarray(normal,dtype=state.x.dtype)
    reference_normal=state.F.T@normal
    reference_offset=offset-normal@state.x+reference_normal@state.reference_centroid
    plus,minus=split(state.reference,reference_normal,reference_offset)
    domains=jax.tree.map(lambda a,b:jnp.stack((a,b)),plus,minus)

    def child(domain):
        geometry=moments(domain);active=geometry.area>0
        center=jnp.where(active,geometry.centroid,state.reference_centroid)
        shift=state.F@(center-state.reference_centroid)
        fraction=geometry.area/jnp.where(state.reference_area>0,state.reference_area,1.)
        return Fragment(domain,geometry.area,center,geometry.covariance,state.x+shift,
                        state.v+state.C@shift,state.F,state.C,state.mass*fraction,state.history,
                        state.valid&geometry.valid)
    return jax.vmap(child)(domains)
