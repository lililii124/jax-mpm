"""Actual material-collar rotation through an affine grid velocity field."""
import jax.numpy as jnp
from .fibre_mpm import grid_coordinates


def angle(time,duration=.020,ramp=.001):
    # Trapezoidal angular speed; exactly one turn at duration.
    t=jnp.clip(time,0,duration)
    primitive=jnp.where(t<ramp,t*t/(2*ramp),jnp.where(t<=duration-ramp,t-ramp/2,duration-ramp-(duration-t)**2/(2*ramp)))
    return 2*jnp.pi*primitive/(duration-ramp)


def prescribe(velocity,mass,state,support,layout,masks,dt,duration):
    coordinates,_=grid_coordinates(layout,jnp.arange(layout.node_count))
    selected_all=jnp.zeros(layout.node_count,dtype=bool)
    valid=jnp.asarray(True)
    out=velocity
    for side,mask in enumerate(masks):
        selected=jnp.zeros(layout.node_count,dtype=jnp.int32).at[support.indices].max(jnp.broadcast_to(mask[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
        selected=selected&(mass>0)
        valid=valid&~jnp.any(selected&selected_all)
        delta=angle(state.time+dt,duration)-angle(state.time,duration) if side else jnp.asarray(0.)
        c,s=jnp.cos(delta),jnp.sin(delta)
        Q=jnp.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        target=(coordinates@Q.T-coordinates)/dt
        # 0.5 percent tensile extension of the 21 mm free span, smooth ramp.
        def extension(t):
            z=jnp.clip(t/.002,0,1)
            return .000105*z*z*(3-2*z)
        speed=(extension(state.time+dt)-extension(state.time))/dt
        target=target.at[:,0].add((1 if side else -1)*speed/2)
        out=jnp.where(selected[:,None],target,out)
        selected_all=selected_all|selected
    return out,selected_all,valid
