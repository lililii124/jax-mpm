"""Bounded exact linear-flight events at fixed voxel faces for resin particles.

Zero restitution normal impulse, unchanged tangential velocity and APIC affine
velocity. This enforces the existing impermeable voxel boundary on particle
advection; zero face velocity alone is insufficient for quadratic interpolation.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class Collision(NamedTuple):
    x: object
    v: object
    impulse: object
    torque_impulse: object
    kinetic_loss: object
    events: object
    valid: object


def advect(x,v,mass,active,solid,layout,dt):
    origin=jnp.asarray(layout.origin,dtype=x.dtype);shape=jnp.asarray(layout.shape)
    cells=jnp.floor((x-origin)/layout.dx).astype(jnp.int32)
    def occupied(index):
        inside=jnp.all((index>=0)&(index<shape),axis=1);safe=jnp.clip(index,0,shape-1)
        return solid[safe[:,0],safe[:,1],safe[:,2]],inside
    initial_solid,inside=occupied(cells)
    valid=jnp.all(~active|(inside&~initial_solid))&jnp.all(~active|(jnp.linalg.norm(v,axis=1)*dt/layout.dx<=.25))
    remaining=jnp.where(active,dt,0.);impulse=jnp.zeros_like(x);torque=jnp.zeros_like(x);loss=jnp.zeros(len(x),dtype=x.dtype);events=jnp.zeros(len(x),dtype=jnp.int32)
    def body(_,carry):
        position,velocity,index,left,I,T,L,count,ok=carry
        direction=jnp.sign(velocity).astype(jnp.int32)
        face=origin+(index+(direction>0).astype(jnp.int32))*layout.dx
        times=jnp.where(velocity!=0,(face-position)/jnp.where(velocity!=0,velocity,1.),jnp.inf)
        axis=jnp.argmin(times,axis=1);tau=jnp.min(times,axis=1)
        event=(left>0)&(tau<=left);advance=jnp.where(event,jnp.maximum(tau,0.),left)
        position=position+advance[:,None]*velocity;left=left-advance
        selector=jax.nn.one_hot(axis,3,dtype=x.dtype);speed=jnp.sum(velocity*selector,axis=1);sign=jnp.sign(speed).astype(jnp.int32)
        next_index=index+selector.astype(jnp.int32)*sign[:,None]
        wall,inside=occupied(next_index);hit=event&wall
        ok=ok&jnp.all(~event|inside)
        delta=jnp.where(hit[:,None],-speed[:,None]*selector,0.)
        event_impulse=mass[:,None]*delta
        T=T+jnp.cross(position,event_impulse);I=I+event_impulse;L=L+jnp.where(hit,.5*mass*speed**2,0.);count=count+hit.astype(jnp.int32)
        velocity=velocity+delta
        # One floating-point neighbour selects the correct side of the exact
        # event face. No finite safety gap or arbitrary position clipping.
        boundary=jnp.sum(face*selector,axis=1)
        toward=boundary+jnp.where(hit,-1.,1.)*sign*layout.dx
        boundary=jnp.nextafter(boundary,toward)
        position=jnp.where(event[:,None]&selector.astype(bool),boundary[:,None],position)
        index=jnp.where((event&~wall)[:,None],next_index,index)
        return position,velocity,index,left,I,T,L,count,ok
    position,velocity,index,left,I,T,L,count,valid=jax.lax.fori_loop(0,4,body,(x,v,cells,remaining,impulse,torque,loss,events,valid))
    endsolid,endinside=occupied(jnp.floor((position-origin)/layout.dx).astype(jnp.int32))
    valid=valid&jnp.all(~active|(endinside&~endsolid))&jnp.all(left<=dt*1e-12)&jnp.all(jnp.isfinite(position))
    return Collision(position,velocity,jnp.sum(I,axis=0),jnp.sum(T,axis=0),jnp.sum(L),jnp.sum(count),valid)
