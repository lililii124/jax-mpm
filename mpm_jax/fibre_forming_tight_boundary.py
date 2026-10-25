"""Independent symmetric axial-feed boundary for actual dry tight forming."""
import jax.numpy as jnp
from .fibre_mpm import grid_coordinates
from .fibre_forming_boundary import angle


ORBIT=.001174124516259575
FREE_LENGTH=.021
RAMP=.005


def extension(time,duration):
    theta=angle(time,duration,RAMP)
    return jnp.sqrt(FREE_LENGTH**2-(theta*ORBIT)**2)-FREE_LENGTH


def prescribe(velocity,mass,state,support,layout,masks,dt,duration):
    coordinates,_=grid_coordinates(layout,jnp.arange(layout.node_count))
    selected_all=jnp.zeros(layout.node_count,dtype=bool);valid=jnp.asarray(True);out=velocity
    for side,mask in enumerate(masks):
        selected=jnp.zeros(layout.node_count,dtype=jnp.int32).at[support.indices].max(jnp.broadcast_to(mask[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
        selected=selected&(mass>0);valid=valid&~jnp.any(selected&selected_all)
        delta=angle(state.time+dt,duration,RAMP)-angle(state.time,duration,RAMP) if side else jnp.asarray(0.)
        c,s=jnp.cos(delta),jnp.sin(delta);Q=jnp.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
        target=(coordinates@Q.T-coordinates)/dt
        speed=(extension(state.time+dt,duration)-extension(state.time,duration))/dt
        target=target.at[:,0].add((1 if side else -1)*speed/2)
        out=jnp.where(selected[:,None],target,out);selected_all=selected_all|selected
    return out,selected_all,valid
