"""Two compatible real feed batches for a finite dry three-direction station.

This is a fresh process version. Only end fixtures are commanded; body points
must evolve with MPM. Same-family cords feed together because their required
crossing-order vectors against installed families are identical.
"""
import numpy as np
from .triaxial_weaving_geometry import make_process


def make_batch_process(*,maximum_machine_speed=3.,**kwargs):
    if maximum_machine_speed<=0:raise ValueError('Positive machine speed required')
    geometry=make_process(maximum_feed_speed=maximum_machine_speed,**kwargs)
    meta=geometry['metadata'];fibres=meta['fibres'];origin=np.asarray(meta['origin_m'])
    caps=np.asarray(meta['fixture_positions'][0]);current=caps.copy()
    times=[0.];knots=[current.copy()];stages=[];proof=[]
    installed=[f['id'] for f in fibres if f['family_id']==0]
    closed=np.zeros(len(fibres),int)
    def append(name,minimum,**extra):
        travel=float(np.linalg.norm(current-knots[-1],axis=1).max())
        duration=max(minimum,np.pi*travel/(2*maximum_machine_speed))
        stages.append(dict(name=name,start=times[-1],end=times[-1]+duration,
            maximum_fixture_travel_m=travel,peak_fixture_speed_m_s=np.pi*travel/(2*duration),**extra))
        times.append(times[-1]+duration);knots.append(current.copy())
    for family in (1,2):
        batch=[f['id'] for f in fibres if f['family_id']==family]
        vectors=[tuple(next(sign for _,sign,other in fibres[old]['crossings'] if other==incoming) for old in installed) for incoming in batch]
        if len(set(vectors))!=1:raise ValueError('Batch has conflicting installed-yarn shed requirements')
        proof.append(dict(incoming=batch,installed=installed.copy(),common_shed_sign=list(vectors[0])))
        for old,sign in zip(installed,vectors[0]):current[2*old:2*old+2,1]=origin[1]+sign*2*meta['diameter_m']
        for incoming in batch:current[2*incoming:2*incoming+2,1]=origin[1]
        append('open_and_align_family_'+str(family),.0004)
        for incoming in batch:
            f=fibres[incoming];current[2*incoming:2*incoming+2,[0,2]]+=f['feed_distance_m']*np.asarray(f['direction'])
        append('parallel_feed_family_'+str(family),meta['minimum_feed_time_s'],fibre_ids=batch)
        for old in installed:closed[old]+=len(batch)
        for incoming in batch:closed[incoming]=len(installed)
        installed+=batch
        for old in installed:
            f=fibres[old];direction=np.array([f['direction'][0],0.,f['direction'][1]])
            current[2*old:2*old+2]=caps[2*old:2*old+2]+f['feed_distance_m']*direction
            fraction=min(1.,closed[old]/len(f['crossings']))
            if f['family_id']==0:
                current[2*old]+=fraction*f['feed_reserve_m']/2*direction
                current[2*old+1]-=fraction*f['feed_reserve_m']/2*direction
            else:current[2*old]+=fraction*f['feed_reserve_m']*direction
            current[2*old:2*old+2,1]=origin[1]
        append('close_and_supply_reserve_family_'+str(family),.0004)
        append('hold_after_family_'+str(family),.00015)
    meta.update(case='finite-triaxial-weaving-process-v3-parallel-family-feed',
        process_version='v3 fresh process; two compatible same-family batches; all machine motion rate limited',
        fixture_times=times,fixture_positions=np.asarray(knots),stages=stages,duration_s=times[-1],
        batch_compatibility=proof,maximum_machine_speed_m_s=maximum_machine_speed,
        qualification='Source-only geometry; actual dynamic contact/order and time-step checks pending')
    # The original v2 full swept bounds do not describe this new simultaneous
    # motion. Execution must rebuild from actual state and this version's sweep.
    geometry.pop('swept_reference_bounds',None)
    return geometry
