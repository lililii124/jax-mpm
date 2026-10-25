"""Physical normal-bearing command and released-geometry observations."""
import numpy as np
from .triaxial_weaving_geometry import fixture_position


def make_boundary(geometry,plan,config):
    import jax.numpy as jnp
    from mpm_jax.fibre_normal_guides import normal_guides
    meta=geometry['metadata'];times=jnp.asarray(meta['fixture_times']);positions=jnp.asarray(meta['fixture_positions']);guides=meta['placement_guides']
    def end_speed(i):return lambda t:(fixture_position(t,times,positions,jnp)[i]-fixture_position(t-config.dt,times,positions,jnp)[i])/config.dt
    def guide_speed(g):
        fid=g['fibre_id'];release=g['release_time_s']
        return lambda t:(fixture_position(jnp.minimum(t,release),times,positions,jnp)[2*fid,1]-fixture_position(jnp.minimum(t-config.dt,release),times,positions,jnp)[2*fid,1])/config.dt
    return normal_guides(geometry['masks'],tuple(end_speed(i) for i in range(len(geometry['masks']))),[g['particle_ids'] for g in guides],tuple(guide_speed(g) for g in guides),[g['release_time_s'] for g in guides],mass=geometry['mass'],wave_speed=plan['elastic_wave_speed_m_s'])


def observe_guides(state,geometry):
    meta=geometry['metadata'];time=float(state.time);x=np.asarray(state.x);times=meta['fixture_times'];positions=np.asarray(meta['fixture_positions']);target=fixture_position(time,times,positions);centres=[];active=[];errors=[];drifts=[]
    for g in meta['placement_guides']:
        p=np.asarray(g['particle_ids']);centre=np.average(x[p],axis=0,weights=np.asarray(state.mass)[p]);reference=np.asarray(g['reference_centroid_m']);engaged=time<g['release_time_s'];expected=reference[1]+target[2*g['fibre_id'],1]-positions[0,2*g['fibre_id'],1]
        centres.append(centre);active.append(engaged)
        if engaged:errors.append(abs(centre[1]-expected));drifts.append(np.linalg.norm((centre-reference)[[0,2]]))
    return dict(time_s=time,active_guides=int(sum(active)),maximum_active_centroid_y_error_m=max(errors,default=0.),maximum_active_XZ_drift_m=max(drifts,default=0.),centres=np.asarray(centres),active=np.asarray(active),release_state='First families released' if time>=min(g['release_time_s'] for g in meta['placement_guides']) else 'Placement support active',scope='Actual mass-weighted material-band centroids; no coordinate reset or finished-shape interpolation')
