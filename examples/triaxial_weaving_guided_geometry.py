"""Independent v4: finite normal rolling guides for placement, then release.

Guides hold sampled cross-section centroids in y, leaving x/z and all other
3D deformation free. Split guide halves retract outside the future fabric
before feeding. Particle coordinates are never assigned machine trajectories.
"""
import numpy as np
from .triaxial_weaving_batch_plan import plan_batch
from .triaxial_weaving_process import json_value


def plan_guided(per_family=8,nominal_cfl=.08):
    if nominal_cfl!=.08:raise ValueError('Original fine .08 only; no time-step scan')
    geometry,plan=plan_batch(per_family=per_family,nominal_cfl=nominal_cfl)
    meta=geometry['metadata'];old_times=meta['fixture_times'];old_positions=np.asarray(meta['fixture_positions']);old_stages=meta['stages'];times=[0.];positions=[old_positions[0].copy()];stages=[];release={}
    for index,old in enumerate(old_stages):
        row=dict(old);row.update(start=times[-1],end=times[-1]+old['end']-old['start']);stages.append(row);times.append(row['end']);positions.append(old_positions[index+1].copy())
        if old['name'].startswith('open_and_align_family_'):
            family=int(old['name'].split('_')[-1]);release[family]=times[-1]
            stages.append(dict(name='release_and_retract_placement_guides_family_'+str(family),start=times[-1],end=times[-1]+.001,maximum_fixture_travel_m=0.,peak_fixture_speed_m_s=0.,split_time_s=.00015,normal_retract_time_s=.0008,split_stroke_m=.000225,normal_stow_stroke_m=.0015,maximum_carrier_speed_m_s=float(max(np.pi*.000225/(2*.00015),np.pi*.0015/(2*.0008)))))
            times.append(stages[-1]['end']);positions.append(positions[-1].copy())
    guides=[];reports=[];origin=np.asarray(meta['origin_m'])[[0,2]];cross=meta['cross_section_particles'];clearance=.0006
    for fibre in meta['fibres']:
        first,last=fibre['particle_slice'];centres=geometry['x'][first:last].reshape(fibre['longitudinal_sections'],cross,3).mean(1);distance=np.full(len(centres),np.inf)
        for other in meta['fibres']:
            if other['family_id']==fibre['family_id']:continue
            direction=np.asarray(other['direction']);normal=np.array([-direction[1],direction[0]]);base=np.asarray(other['base'])+origin
            distance=np.minimum(distance,abs((centres[:,[0,2]]-base)@normal))
        eligible=distance>=clearance;eligible[:8]=False;eligible[-8:]=False;indices=np.flatnonzero(eligible);groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1);selected=[]
        for group in groups:
            if not len(group):continue
            pieces=max(1,int(np.ceil(len(group)*.00006/.0015)))
            selected.extend(int(part[len(part)//2]) for part in np.array_split(group,pieces) if len(part))
        if not selected:raise ValueError('No noninterfering finite guide stations')
        axis=np.asarray(fibre['direction']);coordinate=centres[:,[0,2]]@axis;anchors=np.r_[coordinate[1],coordinate[selected],coordinate[-2]];gap=float(np.diff(anchors).max())
        if gap>.004:raise ValueError('Unsupported guide span exceeds predeclared4mm placement design bound')
        for section in selected:
            ids=list(range(first+section*cross,first+(section+1)*cross));guides.append(dict(id=len(guides),fibre_id=fibre['id'],section=section,particle_ids=ids,reference_centroid_m=centres[section].tolist(),normal=[0.,1.,0.],release_time_s=release[1 if fibre['family_id'] in (0,1) else 2],minimum_projected_clearance_m=float(distance[section]),ring_radius_m=fibre['radius_m'],ring_wall_thickness_m=.05*meta['diameter_m'],lateral_split_direction=[float(fibre['normal'][0]),0.,float(fibre['normal'][1])]))
        reports.append(dict(fibre_id=fibre['id'],stations=len(selected),maximum_unsupported_span_m=gap))
    meta.update(case='finite-triaxial-weaving-v4-normal-placement-guides',process_version='Independent v4 supported placement, split-guide release/stow, real subsequent dry feeding',fixture_times=times,fixture_positions=np.asarray(positions),stages=stages,duration_s=times[-1],placement_guides=guides,guide_station_reports=reports,guide_projected_clearance_m=clearance,guide_maximum_XZ_drift_m=.00005,guide_idealization='Frictionless floating rolling-bearing cross-section-centroid y constraints; x/z and cross-sectional deformation free, reaction applied by mass-metric grid projection. Normal latch opens at release; split halves retract0.225mm in0.15ms then stow1.5mm normally in0.8ms before feed. Not a resolved solid-machine contact model.',initial_geometry_unchanged_from_v3=True)
    plan.update(case=meta['case'],duration_s=times[-1],protocol=json_value(stages),guide_policy=json_value(dict(count=len(guides),releases=release,stations=reports,projected_clearance_m=clearance,normal_only=True,reengage_woven_fibres=False)),first_release_time_s=release[1],first_release_checked_time_s=release[1]+.001)
    plan['full_steps']=int(np.ceil(plan['duration_s']/plan['dt']));plan['placement_prefix_steps']=int(np.ceil(plan['first_release_checked_time_s']/plan['dt']))
    return geometry,plan
