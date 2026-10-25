"""Unique ownership packing of every shared contact node, without mass cutoff."""
from typing import NamedTuple
import jax.numpy as jnp
from .fibre_forming_active_set import project as project_full


class ContactResult(NamedTuple):
    velocity: object
    pair_impulse: object
    fixture_impulse: object
    closing_residual: object
    valid: object
    sweeps: object
    exact_nodes: object
    exact_successes: object
    capacity_valid: object
    rank_rejections: object
    shared_nodes: object
    shared_capacity_valid: object


def project(mass,velocity,gradient,centres,clamped,*,tolerance=1e-8,max_sweeps=64,fallback_capacity=256,shared_capacity=16384):
    nodes=mass.shape[1]
    shared=jnp.sum((mass>0).astype(jnp.int32),axis=0)>=2
    number=jnp.sum(shared,dtype=jnp.int32)
    indices=jnp.nonzero(shared,size=shared_capacity,fill_value=nodes)[0]
    real=indices<nodes;gather=jnp.where(real,indices,0);scatter=jnp.where(real,indices,nodes)
    packed_mass=jnp.where(real[None,:],mass[:,gather],0.)
    packed_velocity=jnp.where(real[None,:,None],velocity[:,gather,:],0.)
    packed_gradient=jnp.where(real[None,:,None],gradient[:,gather,:],0.)
    packed_clamped=jnp.where(real[None,:],clamped[:,gather],False)
    answer=project_full(packed_mass,packed_velocity,packed_gradient,centres,packed_clamped,tolerance=tolerance,max_sweeps=max_sweeps,fallback_capacity=fallback_capacity)
    v=velocity.at[:,scatter,:].set(answer.velocity,mode='drop')
    pair=jnp.zeros_like(velocity).at[:,scatter,:].set(answer.pair_impulse,mode='drop')
    fixture=jnp.zeros_like(velocity).at[:,scatter,:].set(answer.fixture_impulse,mode='drop')
    capacity=number<=shared_capacity
    return ContactResult(v,pair,fixture,answer.closing_residual,answer.valid&capacity,answer.sweeps,answer.exact_nodes,answer.exact_successes,answer.capacity_valid,answer.rank_rejections,number,capacity)
