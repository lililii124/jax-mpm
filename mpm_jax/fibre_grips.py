"""Sparse reference-endcap selection, equivalent to sequential disjoint clamps."""
import numpy as np


def sparse_reference_grips(particle_masks,velocities):
    import jax.numpy as jnp
    from .fibre_mpm import BoundaryResult
    masks=tuple(np.asarray(mask,dtype=bool) for mask in particle_masks);velocities=tuple(velocities)
    if not masks or len(masks)!=len(velocities) or any(mask.shape!=masks[0].shape or mask.ndim!=1 for mask in masks):raise ValueError('Matching reference masks and velocities required')
    selected=[np.flatnonzero(mask) for mask in masks];particle_ids=jnp.asarray(np.concatenate(selected),dtype=jnp.int32)
    labels=jnp.asarray(np.concatenate([np.full(len(ids)*27,i,np.int32) for i,ids in enumerate(selected)]));count=len(masks)
    def boundary(velocity,mass,state,support,layout,time):
        if masks[0].shape!=state.mass.shape:raise ValueError('Reference mask size mismatch')
        nodes=support.indices[particle_ids].reshape(-1)
        low=jnp.full(layout.node_count,count,jnp.int32).at[nodes].min(labels,mode='drop')
        high=jnp.full(layout.node_count,-1,jnp.int32).at[nodes].max(labels,mode='drop')
        active=(high>=0)&(mass>0);valid=~jnp.any(active&(low!=high))
        targets=jnp.stack([jnp.asarray(fn(time),dtype=velocity.dtype) for fn in velocities])
        if targets.shape!=(count,3):raise ValueError('Guide velocities must have three components')
        group=jnp.maximum(high,0);corrected=jnp.where(active[:,None],targets[group],velocity)
        delta=mass[:,None]*(corrected-velocity);node_work=.5*mass*jnp.sum((corrected-velocity)*(corrected+velocity),axis=1)
        occurrence=jnp.arange(len(nodes),dtype=jnp.int32)
        first=jnp.full(layout.node_count,len(nodes),jnp.int32).at[nodes].min(occurrence,mode='drop')
        unique=occurrence==jnp.take(first,nodes,mode='fill',fill_value=len(nodes))
        chosen=unique&jnp.take(active,nodes,mode='fill',fill_value=False)
        owner=jnp.maximum(jnp.take(high,nodes,mode='fill',fill_value=0),0)
        impulses=jnp.zeros((count,3),velocity.dtype).at[owner].add(jnp.where(chosen[:,None],jnp.take(delta,nodes,axis=0,mode='fill',fill_value=0.),0.))
        work=jnp.zeros(count,velocity.dtype).at[owner].add(jnp.where(chosen,jnp.take(node_work,nodes,mode='fill',fill_value=0.),0.))
        return BoundaryResult(corrected,impulses,work,valid)
    boundary.group_count=count
    return boundary
