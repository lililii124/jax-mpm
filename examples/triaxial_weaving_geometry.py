"""Unwoven finite straight cords and physical end-fixture commands.

The periodic centreline is used ONLY to choose crossing order and feed reserve.
No particle is assigned a finished crimp or interpolated toward a woven shape.
"""
import numpy as np
from .triaxial_fibre_geometry import _frame, _disk_sampling, centreline, _segment_distance


def family_frame(family):
    direction,normal=_frame(family)
    return (-direction,-normal) if family==2 else (direction,normal)


def fixture_position(time, times, positions, xp=np):
    """Smooth interpolation of MACHINE END FIXTURES, never particle positions."""
    times=xp.asarray(times);positions=xp.asarray(positions)
    index=xp.clip(xp.searchsorted(times,time,side='right')-1,0,len(times)-2)
    phase=xp.clip((time-times[index])/(times[index+1]-times[index]),0.,1.)
    blend=.5*(1-xp.cos(xp.pi*phase))
    return positions[index]*(1-blend)+positions[index+1]*blend


def make_process(*,per_family=2,pitch=.0027,diameter=.000225,spacing=.00006,
                 density=1500.,shed_diameters=2.,feed_time=.0018,
                 opening_time=.0004,closing_time=.0004,hold_time=.00015):
    if not isinstance(per_family,int) or not 1<=per_family<=12:raise ValueError('Finite station supports1..12 cords per family')
    if min(pitch,diameter,spacing,density,feed_time,opening_time,closing_time,hold_time)<=0:
        raise ValueError('Positive geometry/material/timing required')
    if diameter/pitch>.09:raise ValueError('Pitch must leave a slender bending span')
    radius=diameter/2;span=4*pitch;origin=np.array([.025,.004,.025])
    indices=list(range(-(per_family//2),-(per_family//2)+per_family))
    fibres=[]
    for family in range(3):
        d,n=family_frame(family)
        for index in indices:
            fibres.append(dict(id=len(fibres),family_id=family,periodic_line_index=index,
                direction=d,normal=n,base=(index+.5)*pitch*n,crossings=[]))
    crossing=[]
    for left in fibres:
        for right in fibres[left['id']+1:]:
            if left['family_id']==right['family_id']:continue
            t,s=np.linalg.solve(np.column_stack([left['direction'],-right['direction']]),right['base']-left['base'])
            heights=[]
            for fibre,parameter in [(left,t),(right,s)]:
                f=fibre['family_id'];i=fibre['periodic_line_index']
                if f==2:i,parameter=-i-1,-parameter
                heights.append(float(centreline(f,i,parameter,pitch=pitch,crimp=radius)[0][1]))
            sign=int(np.sign(heights[0]-heights[1]))
            if sign==0:raise ValueError('Ambiguous design crossing order')
            point=left['base']+t*left['direction']
            item=dict(fibre_ids=[left['id'],right['id']],parameters=[float(t),float(s)],
                plane_xz=(point+origin[[0,2]]).tolist(),expected_left_above_right=sign)
            crossing.append(item)
            left['crossings'].append((float(t),sign,right['id']))
            right['crossings'].append((float(s),-sign,left['id']))
    span=max(span,2*(max(abs(t) for f in fibres for t,_,_ in f['crossings'])+pitch))
    radii,angles,rings=_disk_sampling(radius,spacing);cross_count=len(radii)
    positions=[];volumes=[];ids=[];families=[];masks=[];candidates=[]
    particle_offset=0
    for fibre in fibres:
        ordered=sorted(fibre['crossings']);lo=min(p[0] for p in ordered);hi=max(p[0] for p in ordered)
        final_end=(-span/2,span/2) if fibre['family_id']==0 else (hi+pitch-span,hi+pitch)
        vertices=[(final_end[0],0.),*[(t,radius*sign) for t,sign,_ in ordered],(final_end[1],0.)]
        lengths=np.linalg.norm(np.diff(np.asarray(vertices),axis=0),axis=1)
        reserve=1.05*(float(lengths.sum())-span)+.0002*span
        if reserve<=0 or reserve>.05*span:raise ValueError('Invalid finite feed reserve')
        if fibre['family_id']==0:
            start,end=-span/2-reserve/2,span/2+reserve/2;park=0.;travel=0.
        else:
            end=lo-pitch;start=end-span-reserve;travel=hi+pitch-end
            rank=indices.index(fibre['periodic_line_index'])
            park=(1 if fibre['family_id']==1 else -1)*(3+2*rank)*diameter
        count=int(np.ceil((end-start)/spacing));parameters=start+(np.arange(count)+.5)*(end-start)/count
        d,n=fibre['direction'],fibre['normal'];path=fibre['base'][None]+parameters[:,None]*d
        centre=np.column_stack([path[:,0],np.full(count,park),path[:,1]])+origin
        lateral=np.array([n[0],0.,n[1]]);vertical=np.array([0.,1.,0.])
        offsets=radii[:,None]*(np.cos(angles)[:,None]*vertical+np.sin(angles)[:,None]*lateral)
        x=(centre[:,None,:]+offsets).reshape(-1,3);number=len(x)
        volume=np.full(number,np.pi*radius**2*(end-start)/number)
        positions.append(x);volumes.append(volume);ids.append(np.full(number,fibre['id'],np.int32));families.append(np.full(number,fibre['family_id'],np.int16))
        fibre.update(radius_m=radius,particle_count=number,longitudinal_sections=count,
            particle_slice=[particle_offset,particle_offset+number],parameter_range_m=[start,end],
            reference_arclength_m=end-start,feed_distance_m=travel,feed_reserve_m=reserve,
            initial_centreline_endpoints_m=np.stack([centre[0],centre[-1]]).tolist())
        particle_offset+=number
    x=np.concatenate(positions);volume=np.concatenate(volumes);fibre_id=np.concatenate(ids);family_id=np.concatenate(families)
    caps=[]
    for fibre in fibres:
        start,end=fibre['particle_slice'];cap=3*cross_count
        for first,last in [(start,start+cap),(end-cap,end)]:
            mask=np.zeros(len(x),bool);mask[first:last]=True;masks.append(mask);caps.append(x[mask].mean(0))
    caps=np.asarray(caps);current=caps.copy();times=[0.];knots=[current.copy()];stages=[];installed=[f['id'] for f in fibres if f['family_id']==0];closed=np.zeros(len(fibres),int)
    def append(label,duration):
        stages.append(dict(name=label,start=times[-1],end=times[-1]+duration))
        times.append(times[-1]+duration);knots.append(current.copy())
    def set_slack(fibre,frac):
        i=fibre['id'];d=np.array([fibre['direction'][0],0.,fibre['direction'][1]])
        current[2*i:2*i+2]=caps[2*i:2*i+2]+fibre['feed_distance_m']*d
        if fibre['family_id']==0:
            current[2*i]+=frac*fibre['feed_reserve_m']/2*d
            current[2*i+1]-=frac*fibre['feed_reserve_m']/2*d
        else:current[2*i]+=frac*fibre['feed_reserve_m']*d
    order=[next(f['id'] for f in fibres if f['family_id']==family and f['periodic_line_index']==index) for index in indices for family in (1,2)]
    for incoming in order:
        for i in installed:
            if fibres[i]['family_id']==fibres[incoming]['family_id']:continue
            sign=next(sign for _,sign,other in fibres[i]['crossings'] if other==incoming)
            current[2*i:2*i+2,1]=origin[1]+sign*shed_diameters*diameter
        current[2*incoming:2*incoming+2,1]=origin[1]
        append(f'open_and_align_yarn_{incoming}',opening_time)
        d=fibres[incoming]['direction'];current[2*incoming:2*incoming+2,[0,2]]+=fibres[incoming]['feed_distance_m']*d
        append(f'feed_yarn_{incoming}',feed_time)
        for i in installed:
            if fibres[i]['family_id']!=fibres[incoming]['family_id']:closed[i]+=1
        closed[incoming]=len([i for i in installed if fibres[i]['family_id']!=fibres[incoming]['family_id']])
        installed.append(incoming)
        for i in installed:
            set_slack(fibres[i],min(1.,closed[i]/len(fibres[i]['crossings'])))
            current[2*i:2*i+2,1]=origin[1]
        append(f'close_and_supply_crimp_reserve_{incoming}',closing_time)
        append(f'hold_after_yarn_{incoming}',hold_time)
    for left in fibres:
        for right in fibres[left['id']+1:]:
            a=np.asarray(left['initial_centreline_endpoints_m']);b=np.asarray(right['initial_centreline_endpoints_m'])
            distance,(_,wa),(_,wb)=_segment_distance(a,b,closest=True)
            if distance<=diameter:raise ValueError('Initial unwoven cords overlap')
            if left['family_id']!=right['family_id']:
                candidates.append(dict(fibre_ids=[left['id'],right['id']],reference_centreline_points_m=[(a[0]+wa*(a[1]-a[0])).tolist(),(b[0]+wb*(b[1]-b[0])).tolist()]))
    positions_knots=np.asarray(knots);swept=[]
    for fibre in fibres:
        i=fibre['id'];sample=np.concatenate([x[fibre_id==i],positions_knots[:,2*i:2*i+2].reshape(-1,3)])
        margin=np.array([.2*pitch,2*diameter,.2*pitch])+radius
        swept.append(np.column_stack([sample.min(0)-margin,sample.max(0)+margin]))
    metadata=dict(case='finite-triaxial-weaving-process-v1',matrix_present=False,
        nominal_family_angles_deg=[0.,60.,-60.],weave_plane='XZ',thickness_axis='Y',
        fibres=fibres,cross_section_particles=cross_count,cross_section_ring_counts=rings,
        diameter_m=diameter,pitch_m=pitch,origin_m=origin,particle_count=len(x),
        reference_volume_m3=float(volume.sum()),density_kg_m3=density,
        initial_geometry='separate straight cords; no finished weave or initial crimp',
        topology_pattern='periodic centreline used only for discrete over/under commands and feed reserve',
        fixture_times=times,fixture_positions=positions_knots,stages=stages,crossings=crossing,
        duration_s=times[-1],contact_seed_count=len(candidates),maximum_fixture_displacement_m=float(np.linalg.norm(positions_knots-caps[None],axis=-1).max()))
    return dict(x=x,volume=volume,mass=density*volume,fibre_id=fibre_id,family_id=family_id,
        masks=tuple(masks),contact_candidates=candidates,metadata=metadata,
        swept_reference_bounds=np.asarray(swept))
