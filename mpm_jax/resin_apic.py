"""Quadratic staggered APIC transfers for the independent viscous fill stage."""
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple
import jax.numpy as jnp


@dataclass(frozen=True)
class MACLayout:
    shape: tuple
    origin: tuple
    dx: float


class Support(NamedTuple):
    indices: object
    weights: object
    dpos: object
    valid: object


def component_shape(layout,component):
    return tuple(n+(d==component) for d,n in enumerate(layout.shape))


def support(x,active,layout,component):
    shape=component_shape(layout,component)
    offset=jnp.asarray(tuple(0. if d==component else .5 for d in range(3)),dtype=x.dtype)
    q=(x-jnp.asarray(layout.origin,dtype=x.dtype))/layout.dx-offset
    base=jnp.floor(q-.5).astype(jnp.int32);fraction=q-base
    axis=jnp.stack((.5*(1.5-fraction)**2,.75-(fraction-1)**2,.5*(fraction-.5)**2),axis=1)
    offsets=jnp.asarray(list(product(range(3),repeat=3)),dtype=jnp.int32)
    nodes=base[:,None,:]+offsets
    inside=jnp.all((nodes>=0)&(nodes<jnp.asarray(shape)),axis=2)
    strides=jnp.asarray((shape[1]*shape[2],shape[2],1))
    indices=jnp.sum(nodes*strides,axis=2);size=shape[0]*shape[1]*shape[2]
    indices=jnp.where(inside,indices,size)
    weights=jnp.prod(axis[:,offsets,jnp.arange(3)],axis=2)
    dpos=(offsets-fraction[:,None,:])*layout.dx
    valid=jnp.all(inside|~active[:,None])&jnp.all(jnp.isfinite(x))
    return Support(indices,weights,dpos,valid)


def particle_to_faces(x,v,C,mass,active,layout):
    masses=[];momenta=[];supports=[]
    effective=jnp.where(active,mass,0.)
    for component in range(3):
        stencil=support(x,active,layout,component);shape=component_shape(layout,component);size=shape[0]*shape[1]*shape[2]
        local=v[:,component,None]+jnp.sum(C[:,component,None,:]*stencil.dpos,axis=2)
        m=jnp.zeros(size,dtype=x.dtype).at[stencil.indices].add(effective[:,None]*stencil.weights,mode='drop')
        p=jnp.zeros(size,dtype=x.dtype).at[stencil.indices].add(effective[:,None]*stencil.weights*local,mode='drop')
        masses.append(m.reshape(shape));momenta.append(p.reshape(shape));supports.append(stencil)
    return tuple(masses),tuple(momenta),tuple(supports)


def faces_to_particles(velocity,supports,layout):
    v=[];C=[]
    for component in range(3):
        stencil=supports[component]
        gathered=jnp.take(velocity[component].ravel(),stencil.indices,mode='fill',fill_value=0.)
        v.append(jnp.sum(stencil.weights*gathered,axis=1))
        C.append((4/layout.dx**2)*jnp.sum((stencil.weights*gathered)[:,:,None]*stencil.dpos,axis=1))
    return jnp.stack(v,axis=1),jnp.stack(C,axis=1)


def cell_mass(x,mass,active,layout):
    stencil=support(x,active,layout,3);shape=layout.shape;size=shape[0]*shape[1]*shape[2]
    value=jnp.zeros(size,dtype=x.dtype).at[stencil.indices].add(jnp.where(active,mass,0.)[:,None]*stencil.weights,mode='drop')
    return value.reshape(shape),stencil.valid


def kinetic_particle(mass,v,C,active,dx):
    return .5*jnp.sum(jnp.where(active,mass,0.)*(jnp.sum(v*v,axis=1)+dx*dx/4*jnp.sum(C*C,axis=(1,2))))


def kinetic_grid(mass,velocity):
    return .5*sum(jnp.sum(m*v*v) for m,v in zip(mass,velocity))
