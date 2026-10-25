"""Reference-grid MPM with lumped nodal inertia and velocity-Verlet dynamics."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import State,stencil
from .materials import INCREMENTAL_MODELS,initial_history,kirchhoff_stress,update_history


class ReferenceGrid(NamedTuple):
    positions:jax.Array
    indices:jax.Array
    weights:jax.Array
    gradients:jax.Array
    volume:jax.Array
    particle_mass:jax.Array
    nodal_mass:jax.Array
    nodes:jax.Array
    valid:jax.Array


class NodalBoundary(NamedTuple):
    fixed:jax.Array
    displacement:jax.Array
    velocity:jax.Array
    acceleration:jax.Array


class TLState(NamedTuple):
    displacement:jax.Array
    velocity:jax.Array
    force:jax.Array
    F:jax.Array
    history:object
    time:jax.Array
    valid:jax.Array
    impulse:jax.Array
    angular_impulse:jax.Array
    body_work:jax.Array
    boundary_work:jax.Array


def reference_grid(positions,volume,density,config):
    """The grid bounds refer to material coordinates, not the moving body."""
    positions=jnp.asarray(positions);volume=jnp.broadcast_to(jnp.asarray(volume,dtype=positions.dtype),positions.shape[:1])
    if config.dim!=3:raise ValueError('Reference-grid dynamics currently require3D')
    if positions.ndim!=2 or positions.shape[1]!=config.dim:raise ValueError('Reference coordinates do not match dimension')
    if config.boundaries or config.boundary_cells:raise ValueError('Pass material-node displacement/velocity/acceleration through NodalBoundary')
    support=stencil(positions,config);mass=density*volume;size=(config.n_grid+1)**config.dim
    nodal_mass=jnp.zeros(size,dtype=positions.dtype).at[support.indices].add(mass[:,None]*support.weights)
    node=jnp.arange(size);strides=(config.n_grid+1)**jnp.arange(config.dim-1,-1,-1)
    nodes=(((node[:,None]//strides)% (config.n_grid+1))*config.dx).astype(positions.dtype)
    affine=jnp.einsum('pki,pkj->pij',nodes[support.indices],support.gradients)
    tolerance=jnp.maximum(1e-12,64*jnp.finfo(positions.dtype).eps)
    valid=(support.valid&jnp.all(mass>0)&jnp.all(jnp.isfinite(positions))
        &(jnp.max(abs(support.weights.sum(1)-1))<tolerance)
        &(jnp.max(abs(support.gradients.sum(1)))*config.dx<tolerance)
        &(jnp.max(abs(affine-jnp.eye(config.dim)))<tolerance)
        &(abs(nodal_mass.sum()-mass.sum())<tolerance*mass.sum()))
    return ReferenceGrid(positions,support.indices,support.weights,support.gradients,volume,mass,nodal_mass,nodes,valid)


def deformation(displacement,reference):
    return jnp.eye(displacement.shape[1],dtype=displacement.dtype)+jnp.einsum('pki,pkj->pij',displacement[reference.indices],reference.gradients)


def sample_nodal(values,reference):
    return jnp.sum(reference.weights[:,:,None]*values[reference.indices],axis=1)


def forces(F,history,reference,material,config,body_acceleration):
    if config.material in INCREMENTAL_MODELS:
        tau=jnp.linalg.det(F)[:,None,None]*history.stress
    else:tau=kirchhoff_stress(F,material,config.material)
    cofactor=jnp.stack((jnp.cross(F[:,:,1],F[:,:,2]),jnp.cross(F[:,:,2],F[:,:,0]),jnp.cross(F[:,:,0],F[:,:,1])),axis=-1)
    J=jnp.sum(F[:,:,0]*cofactor[:,:,0],axis=-1)
    P=(tau@cofactor)/J[:,None,None]
    internal=-reference.volume[:,None,None]*jnp.einsum('pij,pkj->pki',P,reference.gradients)
    body=reference.particle_mass[:,None]*body_acceleration
    load=internal+reference.weights[:,:,None]*body[:,None,:]
    return jnp.zeros_like(reference.nodes).at[reference.indices].add(load)


def initial_state(reference,material,config,body_acceleration,history=None):
    displacement=jnp.zeros_like(reference.nodes);F=deformation(displacement,reference)
    if config.material in INCREMENTAL_MODELS and history is None:
        rho=float(reference.particle_mass.sum()/reference.volume.sum())
        history=initial_history(len(reference.positions),reference.positions.dtype,material,rho)
    force=forces(F,history,reference,material,config,body_acceleration)
    return TLState(displacement,jnp.zeros_like(displacement),force,F,history,jnp.asarray(0.),
        reference.valid&jnp.all(jnp.isfinite(force)),jnp.zeros(config.dim),jnp.zeros(3),jnp.asarray(0.),jnp.asarray(0.))


def sample_particles(state,reference):
    """Sample for output; inertia and kinetic energy remain nodal quantities."""
    x=reference.positions+sample_nodal(state.displacement,reference)
    v=sample_nodal(state.velocity,reference)
    return State(x,v,state.F,jnp.zeros_like(state.F),reference.particle_mass,reference.volume,state.history,state.time,state.valid)


def momentum(state,reference):
    linear=jnp.sum(reference.nodal_mass[:,None]*state.velocity,axis=0)
    angular=jnp.sum(jnp.cross(reference.nodes+state.displacement,reference.nodal_mass[:,None]*state.velocity),axis=0)
    return linear,angular


def step(state,reference,material,config,body_start,body_end,boundary_start=None,boundary_end=None):
    """Advance one step; optional boundary data prescribe material-node DOFs."""
    if config.dim!=3:raise ValueError('The nodal angular-momentum ledger currently requires3D')
    if config.failure_mode!='retain':raise ValueError('Reference-grid fracture/ballistic remapping requires a separate model')
    dt=config.dt;M=reference.nodal_mass[:,None];inverse=1/jnp.where(M>0,M,1.)
    a0=state.force*inverse
    if boundary_start is not None:a0=jnp.where(boundary_start.fixed,boundary_start.acceleration,a0)
    half=state.velocity+.5*dt*a0;u=state.displacement+dt*half
    if boundary_end is not None:
        u=jnp.where(boundary_end.fixed,boundary_end.displacement,u)
        half=jnp.where(boundary_end.fixed,(u-state.displacement)/dt,half)
    F=deformation(u,reference);history=state.history
    if config.material in INCREMENTAL_MODELS:history=update_history(history,F,state.F,dt,material,config.material)
    force=forces(F,history,reference,material,config,body_end);a1=force*inverse
    if boundary_end is not None:a1=jnp.where(boundary_end.fixed,boundary_end.acceleration,a1)
    velocity=half+.5*dt*a1
    if boundary_end is not None:velocity=jnp.where(boundary_end.fixed,boundary_end.velocity,velocity)
    mask=jnp.zeros_like(u,dtype=bool)
    if boundary_start is not None:mask=mask|boundary_start.fixed
    if boundary_end is not None:mask=mask|boundary_end.fixed
    J0=jnp.where(mask,M*(half-state.velocity)-.5*dt*state.force,0.)
    J1=jnp.where(mask,M*(velocity-half)-.5*dt*force,0.)
    R0=jnp.zeros_like(u) if boundary_start is None else jnp.where(boundary_start.fixed,M*a0-state.force,0.)
    R1=jnp.zeros_like(u) if boundary_end is None else jnp.where(boundary_end.fixed,M*a1-force,0.)
    x0=reference.positions+sample_nodal(state.displacement,reference)
    x1=reference.positions+sample_nodal(u,reference)
    f0=reference.particle_mass[:,None]*body_start;f1=reference.particle_mass[:,None]*body_end
    impulse=.5*dt*jnp.sum(f0+f1,axis=0)+jnp.sum(J0+J1,axis=0)
    torque=.5*dt*jnp.sum(jnp.cross(x0,f0)+jnp.cross(x1,f1),axis=0)
    torque+=jnp.sum(jnp.cross(reference.nodes+state.displacement,J0)+jnp.cross(reference.nodes+u,J1),axis=0)
    body_work=jnp.sum(.5*(f0+f1)*(x1-x0));boundary_work=jnp.sum(.5*(R0+R1)*(u-state.displacement))
    valid=(state.valid&jnp.all(jnp.linalg.det(F)>0)&jnp.all(jnp.isfinite(F))
        &jnp.all(jnp.isfinite(u))&jnp.all(jnp.isfinite(velocity))&jnp.all(jnp.isfinite(force)))
    if config.material in INCREMENTAL_MODELS:valid=valid&jnp.all(history.valid)
    following=TLState(u,velocity,force,F,history,state.time+dt,valid,state.impulse+impulse,state.angular_impulse+torque,
        state.body_work+body_work,state.boundary_work+boundary_work)
    P,L=momentum(following,reference)
    row=dict(time_s=following.time,valid=valid,min_J=jnp.min(jnp.linalg.det(F)),
        kinetic_J=.5*jnp.sum(M*velocity**2),momentum_error=jnp.linalg.norm(P-following.impulse),
        angular_momentum_error=jnp.linalg.norm(L-following.angular_impulse),body_work_J=body_work,boundary_work_J=boundary_work)
    return following,row
