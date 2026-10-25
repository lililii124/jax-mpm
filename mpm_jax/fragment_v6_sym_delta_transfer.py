"""Unadmitted candidate; zero-force kinetic energy need not be nonincreasing.

The signed G2P correction is numerical bookkeeping, never thermodynamic heat.
Symmetric APIC and skew DELTA components are orthogonal for D=dx^2 I/4.
"""
from typing import NamedTuple
import jax.numpy as jnp
from .fragment_multifield_v6_mpm import transfer_to_particles


class TransferDiagnostics(NamedTuple):
    linear_defect_by_field: object
    angular_defect_by_field: object
    symmetric_affine_change_from_delta_J: object
    orthogonal_identity_residual_J: object
    correction_identity_residual_J: object


def transfer_sym_delta(state,old_velocity,final_velocity,support,grid):
    sym=lambda a:.5*(a+jnp.swapaxes(a,-1,-2))
    _,aC,_=transfer_to_particles(final_velocity,support,grid)
    dv,dC,_=transfer_to_particles(final_velocity-old_velocity,support,grid)
    v,delta_C=state.v+dv,state.C+dC
    C=sym(aC)+.5*(delta_C-jnp.swapaxes(delta_C,-1,-2))
    kinetic=lambda vv,cc:.5*jnp.sum(vv*vv,axis=-1)+grid.layout.dx**2/8*jnp.sum(cc*cc,axis=(-2,-1))
    gather=lambda a:jnp.take(a,support.indices,axis=0,mode='fill',fill_value=0.)
    change=gather(final_velocity-old_velocity)
    final=gather(final_velocity)
    grid_specific=.5*jnp.sum(support.weights*jnp.sum(final*final,axis=-1),axis=-1)
    correction=grid_specific-kinetic(v,C)
    symmetric_change=grid.layout.dx**2/8*jnp.sum(state.mass[:,None,None]*(sym(aC)**2-sym(delta_C)**2))
    orthogonal=jnp.sum(state.mass*(kinetic(v,C)-kinetic(v,delta_C)))-symmetric_change
    correction_identity=jnp.sum(state.mass*(correction-(grid_specific-kinetic(v,delta_C))))+symmetric_change
    spin=lambda cc:jnp.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    before=state.mass[:,None]*(jnp.cross(state.x,state.v)+grid.layout.dx**2/4*spin(state.C))
    after=state.mass[:,None]*(jnp.cross(state.x,v)+grid.layout.dx**2/4*spin(C))
    wm=state.mass[:,None]*support.weights
    torque=jnp.sum(wm[:,:,None]*jnp.cross(state.x[:,None,:]+support.dpos,change),axis=1)
    impulse=jnp.sum(wm[:,:,None]*change,axis=1)
    field_sum=lambda a:jnp.zeros((grid.layout.fibre_count,)+a.shape[1:],dtype=a.dtype).at[state.body_id].add(a)
    linear=field_sum(state.mass[:,None]*(v-state.v)-impulse)
    angular=field_sum(after-before-torque)
    return v,C,correction,TransferDiagnostics(linear,angular,symmetric_change,orthogonal,correction_identity)
