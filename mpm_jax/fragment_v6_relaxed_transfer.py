"""Conservative complete-map blend with a frozen cell-crossing time."""
import math
from typing import NamedTuple
import jax.numpy as jnp
from .fragment_multifield_v6_mpm import transfer_to_particles


class TransferDiagnostics(NamedTuple):
    alpha: object
    blending_variance_J: object
    convex_energy_identity_residual_J: object
    angular_defect_by_field: object
    angular_scale_by_field: object


def transfer_relaxed(state,old_velocity,final_velocity,support,grid,*,dt,relaxation_time):
    """One scalar mixes BOTH v and C; no independently tuned coefficients.

    alpha=exp(-dt/tau0), tau0 is the independently frozen original cell-crossing
    time stable_dt/CFL. Signed correction is K_grid(final)-K_particle(blend).
    It belongs to the numerical reservoir; never to thermodynamic heat.
    """
    if not math.isfinite(relaxation_time) or relaxation_time<=0: raise ValueError('Frozen positive cell-crossing time required')
    av,aC,_=transfer_to_particles(final_velocity,support,grid)
    dv,dC,_=transfer_to_particles(final_velocity-old_velocity,support,grid)
    fv,fC=state.v+dv,state.C+dC
    alpha=jnp.exp(-dt/relaxation_time)
    v=(1-alpha)*av+alpha*fv
    C=(1-alpha)*aC+alpha*fC
    kinetic=lambda vv,cc:.5*jnp.sum(vv*vv,axis=-1)+grid.layout.dx**2/8*jnp.sum(cc*cc,axis=(-2,-1))
    gather=lambda a:jnp.take(a,support.indices,axis=0,mode='fill',fill_value=0.)
    old,final=gather(old_velocity),gather(final_velocity)
    grid_specific=.5*jnp.sum(support.weights*jnp.sum(final*final,axis=-1),axis=-1)
    correction=grid_specific-kinetic(v,C)
    variance=alpha*(1-alpha)*jnp.sum(state.mass*kinetic(fv-av,fC-aC))
    convex=jnp.sum(state.mass*(kinetic(v,C)-(1-alpha)*kinetic(av,aC)-alpha*kinetic(fv,fC)))+variance
    spin=lambda cc:jnp.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    before=state.mass[:,None]*(jnp.cross(state.x,state.v)+grid.layout.dx**2/4*spin(state.C))
    after=state.mass[:,None]*(jnp.cross(state.x,v)+grid.layout.dx**2/4*spin(C))
    torque=jnp.sum((state.mass[:,None]*support.weights)[:,:,None]*jnp.cross(state.x[:,None,:]+support.dpos,final-old),axis=1)
    field_sum=lambda a:jnp.zeros((grid.layout.fibre_count,)+a.shape[1:],dtype=a.dtype).at[state.body_id].add(a)
    defect=field_sum(after-before-torque)
    scale=field_sum(jnp.linalg.norm(before,axis=-1)+jnp.linalg.norm(after,axis=-1)+jnp.linalg.norm(torque,axis=-1))
    return v,C,correction,TransferDiagnostics(alpha,variance,convex,defect,scale)
