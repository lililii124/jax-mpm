"""Mass-metric ideal rolling-bearing normal constraints on finite sections.

Only each guide band's mass-weighted centroid y velocity is constrained. The
constraint is G2P-adjoint, leaves x/z and section deformation free, and supplies
an explicit normal reaction and kinetic projection work. Disjoint guide support
and no overlap with end grips are required; no sequential approximate QP.
"""
import numpy as np


def normal_guides(end_masks,end_velocities,guides,guide_velocities,release_times,*,mass,wave_speed):
    import jax.numpy as jnp
    from .fibre_grips import sparse_reference_grips
    from .fibre_mpm import BoundaryResult
    ends=sparse_reference_grips(end_masks,end_velocities);count=len(guides)
    if count!=len(guide_velocities) or count!=len(release_times):raise ValueError('Complete finite guide definitions required')
    guide_array=jnp.asarray(guides,jnp.int32)
    if guide_array.ndim!=2 or guide_array.shape[0]!=count:raise ValueError('Equal finite section bands required')
    ids=guide_array.reshape(-1);particle_labels=jnp.repeat(jnp.arange(count,dtype=jnp.int32),guide_array.shape[1]);labels=jnp.repeat(particle_labels,27)
    band_mass=jnp.sum(jnp.asarray(mass)[guide_array],axis=1);release=jnp.asarray(release_times)
    end_ids=jnp.asarray(np.concatenate([np.flatnonzero(m) for m in end_masks]),jnp.int32);Mc=jnp.sum(mass)*wave_speed
    def targets(time):
        end=jnp.stack([jnp.asarray(fn(time)) for fn in end_velocities]);normal=jnp.stack([jnp.asarray(fn(time)) for fn in guide_velocities]);guide=jnp.zeros((count,3),end.dtype).at[:,1].set(normal)
        return jnp.concatenate((end,guide))
    def boundary(velocity,node_mass,state,support,layout,time):
        end=ends(velocity,node_mass,state,support,layout,time);v=end.velocity;engaged=time<release
        nodes=support.indices[ids].reshape(-1);weights=(state.mass[ids,None]*support.weights[ids]/band_mass[particle_labels,None]*engaged[particle_labels,None]).reshape(-1)
        coefficient=jnp.zeros(layout.node_count,v.dtype).at[nodes].add(weights,mode='drop')
        low=jnp.full(layout.node_count,count,jnp.int32).at[nodes].min(labels,mode='drop');high=jnp.full(layout.node_count,-1,jnp.int32).at[nodes].max(labels,mode='drop')
        tagged=(coefficient>0)&(node_mass>0);end_nodes=support.indices[end_ids].reshape(-1);end_active=jnp.zeros(layout.node_count,bool).at[end_nodes].set(True,mode='drop')
        occurrence=jnp.arange(len(nodes),dtype=jnp.int32);first=jnp.full(layout.node_count,len(nodes),jnp.int32).at[nodes].min(occurrence,mode='drop');unique=occurrence==jnp.take(first,nodes,mode='fill',fill_value=len(nodes));live=unique&jnp.take(tagged,nodes,mode='fill',fill_value=False)
        a=jnp.take(coefficient,nodes,mode='fill',fill_value=0.);m=jnp.take(node_mass,nodes,mode='fill',fill_value=1.);vy=jnp.take(v[:,1],nodes,mode='fill',fill_value=0.)
        q=jnp.zeros(count,v.dtype).at[labels].add(jnp.where(live,a*vy,0.));D=jnp.zeros(count,v.dtype).at[labels].add(jnp.where(live,a*a/jnp.where(m>0,m,1.),0.));partition=jnp.zeros(count,v.dtype).at[labels].add(jnp.where(live,a,0.))
        target=targets(time)[ends.group_count:,1];impulse=jnp.where(engaged,(target-q)/jnp.where(D>0,D,1.),0.);owner=jnp.maximum(high,0)
        delta=jnp.where(tagged,coefficient/jnp.where(node_mass>0,node_mass,1.)*impulse[owner],0.);corrected=v.at[:,1].add(delta)
        q_after=jnp.zeros(count,v.dtype).at[labels].add(jnp.where(live,a*jnp.take(corrected[:,1],nodes,mode='fill',fill_value=0.),0.))
        guide_impulse=jnp.zeros((count,3),v.dtype).at[:,1].set(impulse);guide_work=impulse*q+.5*impulse**2*D
        actual=jnp.sum(node_mass[:,None]*(corrected-velocity),axis=0);group_impulse=jnp.concatenate((end.group_impulses,guide_impulse));group_work=jnp.concatenate((end.group_work,guide_work))
        valid=end.valid&~jnp.any(tagged&(low!=high))&~jnp.any(tagged&end_active)&jnp.all(jnp.where(engaged,D>0,True))&jnp.all(jnp.where(engaged,abs(partition-1.)<=1e-12,True))&jnp.all(jnp.where(engaged,abs(q_after-target)<=1e-10*wave_speed,True))&(jnp.linalg.norm(actual-group_impulse.sum(0))<=1e-10*Mc)
        return BoundaryResult(corrected,group_impulse,group_work,valid)
    boundary.group_count=ends.group_count+count;boundary.targets=targets;boundary.guide_count=count
    return boundary
