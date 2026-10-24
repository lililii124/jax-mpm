"""Three independent circular helical fibres under axial end-collar tension.

This is a stress-free manufactured helical reference, not a simulated twisting
process. Normal Hertz contact is local and frictionless; extended ribbon contact
is rejected rather than converted into an unjustified single point force.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from .single_fibre_tension import digest,descriptor,save_json,save_npz,motion,runtime_record

ROOT=Path(__file__).resolve().parents[1]
SOURCE_FILES=('examples/three_strand_tension.py','examples/single_fibre_tension.py',
 'mpm_jax/rope_hertz_contact.py','mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py',
 'mpm_jax/fibre_coupling.py','mpm_jax/fibre_contact.py','mpm_jax/fibre_patch_contact.py',
 'mpm_jax/contact_patches.py','mpm_jax/fibre_jkr_regions.py','mpm_jax/fibre_region_identity.py')


def source_record():
    files={n:digest(ROOT/n) for n in SOURCE_FILES}
    return {'files':files,'fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def centreline(t,fibre,radius,pitch):
    t=np.asarray(t);angle=2*np.pi*t/pitch+2*np.pi*fibre/3
    return np.column_stack((t,radius*np.cos(angle),radius*np.sin(angle)))


def make_geometry(*,length=.024,diameter=.002,pitch=.008,compactness=.985,spacing=.0001,density=1000.,grip=.0015):
    from scipy.optimize import minimize_scalar
    from .triaxial_fibre_geometry import _segment_distance
    if not np.isfinite((length,diameter,pitch,compactness,spacing,density,grip)).all() or min(length,diameter,pitch,spacing,density,grip)<=0 or not 0<compactness<=1 or 2*grip>=length:
        raise ValueError('Positive finite dimensions/density;0<compactness<=1 and disjoint end collars required')
    r=diameter/2;k=2*np.pi/pitch;clearance=diameter*(1/compactness-1)+.01*spacing
    if pitch<=3*(diameter+clearance):raise ValueError('Pitch must exceed3*(diameter+clearance) for conservative no-overlap helical packing')
    def minimum_distance(orbit):
        # Global infinite-helix pair minimum in the nearest period; farther
        # axial offsets add at least a full pitch of separation.
        def f(d):return d*d+2*orbit*orbit*(1-np.cos(k*d+2*np.pi/3))
        grid=np.linspace(-pitch,pitch,257);values=f(grid);candidates=[float(values[0]),float(values[-1])]
        for i in np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1:
            v=minimize_scalar(f,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-15});candidates.append(float(v.fun))
        return math.sqrt(min(candidates))
    lo,hi=r,diameter*2
    while minimum_distance(hi)<diameter+clearance:hi*=1.5
    for _ in range(55):
        mid=(lo+hi)/2
        if minimum_distance(mid)<diameter+clearance:lo=mid
        else:hi=mid
    orbit=hi*(1+1e-5);metric=math.sqrt(1+(orbit*k)**2);curvature=orbit*k*k/(1+(orbit*k)**2)
    if curvature*r>=.75:raise ValueError('Tube curvature exceeds admitted nonfolding local geometry range')
    count=math.ceil(length*metric/spacing);t=(np.arange(count)+.5)*length/count;ds=metric*length/count
    samples=round(diameter/spacing)
    if samples<16 or samples%2 or not np.isclose(samples*spacing,diameter):raise ValueError('Diameter requires an even integer >=16 particle samples')
    values=(np.arange(samples)+.5)*spacing-r;u,v=np.meshgrid(values,values,indexing='ij');cross=np.column_stack((u.ravel(),v.ravel()));cross=cross[np.sum(cross**2,axis=1)<=r*r]
    xall=[];vall=[];ids=[];sections=[];axial=[];fibres=[];tangents=[]
    for fibre in range(3):
        angle=k*t+2*np.pi*fibre/3;centre=centreline(t,fibre,orbit,pitch)
        tangent=np.column_stack((np.ones(count),-orbit*k*np.sin(angle),orbit*k*np.cos(angle)))/metric
        normal=np.column_stack((np.zeros(count),-np.cos(angle),-np.sin(angle)));binormal=np.cross(tangent,normal)
        xyz=centre[:,None,:]+cross[None,:,0,None]*normal[:,None,:]+cross[None,:,1,None]*binormal[:,None,:]
        volume=np.broadcast_to(ds*spacing**2*(1-curvature*cross[:,0]),(count,len(cross))).copy()
        n=count*len(cross);xall.append(xyz.reshape(-1,3));vall.append(volume.ravel());ids.append(np.full(n,fibre,dtype=np.int32));sections.append(np.repeat(np.arange(count,dtype=np.int32),len(cross)));axial.append(np.repeat(t,len(cross)));tangents.append(np.repeat(tangent,len(cross),axis=0))
        fibres.append({'id':fibre,'family_id':fibre,'longitudinal_sections':count,'particle_count':n,'particle_slice':[fibre*n,(fibre+1)*n],'radius_m':r,'reference_tube_bounds_m':np.column_stack((xyz.min((0,1))-.002,xyz.max((0,1))+.002)).tolist()})
    x=np.concatenate(xall);volume=np.concatenate(vall);ids=np.concatenate(ids);axial=np.concatenate(axial)
    # Certified continuous-tube clearance from all finite polyline segment pairs,
    # with the exact circular-helix sagitta bounding the curve/polyline error.
    proof_t=np.linspace(0,length,513);proof_curves=[centreline(proof_t,i,orbit,pitch) for i in range(3)]
    sagitta=orbit*(1-math.cos(k*(proof_t[1]-proof_t[0])/2));proof=[];seeds=[]
    for a,b in ((0,1),(0,2),(1,2)):
        distance=_segment_distance(proof_curves[a],proof_curves[b]);lower=distance-2*sagitta
        if lower<=diameter:raise ValueError('Continuous circular tubes overlap or clearance proof is inconclusive')
        proof.append({'fibre_ids':[a,b],'polyline_min_distance_m':distance,'continuous_min_distance_lower_bound_m':lower,'surface_clearance_lower_bound_m':lower-diameter})
        seeds.append({'fibre_ids':[a,b],'reference_centreline_points_m':[centreline(np.array([length/2]),i,orbit,pitch)[0].tolist() for i in (a,b)]})
    masks=tuple((ids==f)&(axial<grip if side==0 else axial>length-grip) for f in range(3) for side in (0,1))
    meta={'length_m':length,'diameter_m':diameter,'radius_m':r,'pitch_m':pitch,'compactness':compactness,'orbit_radius_m':orbit,'turns':length/pitch,'helix_angle_deg':math.degrees(math.atan(orbit*k)),
          'spacing_m':spacing,'grip_length_m':grip,'free_span_m':length-2*grip,'section_arc_spacing_m':ds,'sections_per_strand':count,'cross_section_particles':len(cross),'fibres':fibres,'clearance_proof':proof,
          'tube_jacobian_minimum':1-curvature*r,'sampled_minimum_area_m2':len(cross)*spacing**2,'sampled_end_area_m2':len(cross)*spacing**2,'nominal_area_m2':math.pi*r*r,
          'sampling':'True normal-plane circular Cartesian sections, constant arc spacing; reference volume ds*du*dv*(1-kappa*u), exact positive circular-helix tube Jacobian. Contact centres use equal-area section averages, physical inertia uses actual quadrature masses.',
          'reference_state':'Stress-free helical reference F=I; no manufactured twisting residual stress; six end collars retain twist and prescribe only axial translation.'}
    return {'x':x,'volume':volume,'mass':density*volume,'fibre_id':ids,'family_id':ids.copy(),'masks':masks,'axial_parameter':axial,'section':np.concatenate(sections),'tangent':np.concatenate(tangents),'metadata':meta,'contact_candidates':seeds}


def plan_case(*,length=.024,diameter=.002,pitch=.008,compactness=.985,dx=.0002,spacing=.0001,model='hertz',frames=60):
    if model not in ('hertz','jkr') or type(frames) is not int or frames<2:raise ValueError('Hertz/JKR and >=2 output intervals required')
    if diameter/dx<10 or not np.isclose(dx/spacing,2):raise ValueError('At least10 cells across strand diameter and two particles/cell required')
    material={'young':2e6,'poisson':.25,'density':1000.,'tensile_strength':1e5,'fracture_energy':1.,'characteristic_length':dx}
    if dx>=2*material['young']*material['fracture_energy']/material['tensile_strength']**2:raise ValueError('Crack-band h bound violated')
    geometry=make_geometry(length=length,diameter=diameter,pitch=pitch,compactness=compactness,spacing=spacing)
    if len(geometry['x'])>450000:raise ValueError('Formal particle upper bound450000 exceeded')
    meta=geometry['metadata'];gauge=meta['free_span_m'];c=math.sqrt(2e6*1.2/1000);load_time=20*gauge/c;duration=load_time*1.25;steps=math.ceil(duration/(.15*dx/c));dt=duration/steps
    bounds=np.asarray([f['reference_tube_bounds_m'] for f in meta['fibres']]);shape=np.ceil(bounds[:,:,1]/dx).astype(int)-np.floor(bounds[:,:,0]/dx).astype(int)+13;nodes=int(np.prod(shape,axis=1).sum())
    segments=meta['sections_per_strand']-1;coverage_bound=3*segments*segments
    if nodes>1800000 or coverage_bound>600000:raise ValueError('Grid/contact coverage resource upper bound exceeded')
    plan={'scope':'formal-three-turn' if length>=3*pitch else 'geometry-resource-precheck','series':'three-strand-tension-v1','case':'three-strand-tension','model':model,'duration_s':duration,'frames':frames,'saved_states':frames+1,'steps':steps,'dt':dt,'frame_steps':np.rint(np.linspace(0,steps,frames+1)).astype(int).tolist(),'particles':len(geometry['x']),'geometry':meta,
          'grid':{'dx_m':dx,'padding_cells':6,'bounds_m':bounds.tolist(),'nodes':nodes,'max_nodes':1800000},'material':material,
          'material_assumptions':'Three independent identical material parameter instances/history fields; illustrative uncalibrated E2MPa/ft0.1MPa/Gf1Jm^-2 elastic-brittle baseline; h=dx, no particle erosion.',
          'softening':{'onset_strain':.05,'onset_displacement_m':dx*.05,'failure_displacement_m':2e-5,'failure_strain':2e-5/dx,'maximum_h_m':.0004},
          'protocol':{'load_time_s':load_time,'stroke_m':.12*gauge,'hold_time_s':duration-load_time,'boundary':'Each strand has independent tagged left/right reference collars. Six collars move +/-half stroke in x and are fixed in yz; no twist release or artificial strand bonding.'},
          'contact':{'model':model,'work_of_adhesion_J_per_m2':0. if model=='hertz' else .02,'coverage_skin_m':max(.003,3*.12*gauge),'capacity':4096,'max_patches':32,'max_iterations':256,'tracking_gap_m':1e-5,'maximum_coverage_pairs':600000,
                     'scope':'Frictionless connected local circular-equivalent Hertz, or persistent JKR with full region checkpoints. Extended line/ribbon contact (>max(2 section cells,4 local contact radii)) rejected. No friction, exact elliptical pressure or rope strength calibration claimed.'},
          'wave_speed_m_per_s':c,'loading_wave_transits':20.,'source':source_record(),'geometry_scale':1.,'production_eligible':False,
          'resource':{'uss_limit_bytes':6*1024**3,'raw_trajectory_bytes':(frames+1)*len(geometry['x'])*45,'coverage_pair_upper_bound':coverage_bound,'particle_state_subtotal_bytes':len(geometry['x'])*400,'stencil_subtotal_bytes':len(geometry['x'])*27*(4+8+24),'initial_stage_frame':2,'scope':'Analytic storage subtotals, not peak prediction. Must admit actual f2 timing/USS/contact coverage before full continuation.'},
          'render_plan':{'renderer':'ParaView','field':'family_id','colors_rgb':[[.045,.36,.67],[.94,.36,.08],[.32,.66,.48]],'frames':frames+1,'geometry_scale':1,'axes':False,'annotations':False,'resolution':1200},
          'risks':['Extended helical line contact may invalidate the local Hertz/JKR approximation and stop the solve; a future distributed surface/line law may be required.','Fixed reference circular radius ignores finite radial contraction; low/moderate cross-sectional strain only.','Stress-free helix does not include fabrication torsion/residual stress or frictional strand locking.','Failed sections in active contact are rejected; complete rupture may require a future failed-fragment surface contact model.','Within-strand self-contact is unsupported; the initial pitch/curvature bounds exclude it, but a severely buckled state requires rejection or a future self-contact model.','No complete physical performance claim until full audited61 frames and visual review.']}
    return geometry,plan


def fracture_review(geometry,x,failed):
    reports=[]
    for fibre in range(3):
        selected=geometry['fibre_id']==fibre;section=geometry['section'][selected];alive=np.bincount(section,weights=(~failed[selected]),minlength=geometry['metadata']['sections_per_strand']);cuts=np.flatnonzero(alive==0)
        reports.append({'strand':fibre,'fully_failed_sections':cuts.tolist(),'intact_axial_section_path':not bool(len(cuts)),'failed_particles':int(failed[selected].sum())})
    return {'strands':reports,'scope':'Per-strand complete-section material failure/path diagnostic, not a claim about frictional bundle connectivity or all3D MPM support paths.'}


def run_case(geometry,plan,output,*,resume=False,stop_after_frame=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import (FibreConfig,FibreState,build_layout,make_state,reference_grips,step_with_diagnostics,validate_state,stored_energy)
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle,brittle_energy
    from mpm_jax.fibre_coupling import build_contact_data,FibreCouplingHistory
    from mpm_jax.fibre_contact import fibre_contact_parameters,FibreContactHistory
    from mpm_jax.contact_patches import patch_neighbours
    from mpm_jax.fibre_patch_contact import initial_patch_history
    from mpm_jax.rope_hertz_contact import evaluate_rope_contact
    contact_plan=plan['contact']
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'checkpoint.npz').exists() and not resume:raise ValueError('Existing state requires explicit resume')
    runtime=runtime_record();signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    material=isotropic_brittle(**plan['material']);grid=plan['grid'];layout=build_layout(geometry['x'],geometry['fibre_id'],dx=grid['dx_m'],padding_cells=6,reference_bounds=grid['bounds_m'],max_nodes=grid['max_nodes'])
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout);config=FibreConfig(plan['dt'])
    def speed(t):return (motion(t,plan['protocol'],jnp)-motion(t-config.dt,plan['protocol'],jnp))/config.dt
    velocities=tuple((lambda t,side=side:jnp.array([(-1 if side==0 else 1)*speed(t)/2,0.,0.])) for fibre in range(3) for side in (0,1))
    boundary=reference_grips(geometry['masks'],velocities)
    data=build_contact_data(geometry,search_window=8,coverage_skin=contact_plan['coverage_skin_m'],max_coverage_pairs=contact_plan['maximum_coverage_pairs'])
    neighbours=patch_neighbours(np.asarray(data.coverage_pairs),np.asarray(data.segment_fibre))
    law=fibre_contact_parameters(effective_modulus=material.young/(2*(1-material.poisson**2)),effective_radius=plan['geometry']['radius_m'],work_of_adhesion=contact_plan['work_of_adhesion_J_per_m2'],model=plan['model'])
    if plan['model']=='jkr':
        from mpm_jax.fibre_jkr_regions import initial_jkr_region_history
        contact=initial_jkr_region_history(data,max_patches=contact_plan['max_patches'])
    else:contact=initial_patch_history(data,max_patches=contact_plan['max_patches'])
    @jax.jit
    def evaluate(s,h):return evaluate_rope_contact(s,h,data,law,neighbours,capacity=contact_plan['capacity'],max_patches=contact_plan['max_patches'],max_iterations=contact_plan['max_iterations'],model=plan['model'],tracking_gap=contact_plan['tracking_gap_m'])
    n=plan['particles'];frames=plan['saved_states'];shapes={'x':(frames,n,3),**{k:(frames,n) for k in ('damage','failed','speed','temperature')}}
    types={'x':np.float64,'damage':np.float64,'failed':np.uint8,'speed':np.float64,'temperature':np.float32}
    rows=[];records=[];times=[];start=0;solver_seconds=0.;progress=(jnp.array(0,dtype=jnp.int64),jnp.zeros((6,3)),jnp.zeros(6),jnp.zeros(3),jnp.zeros((3,3)),jnp.zeros(3),jnp.asarray(True),contact)
    static={'phase':np.ones(n,dtype=np.int8),'yarn_id':geometry['fibre_id'],'family_id':geometry['family_id'],'grip_id':sum((i+1)*mask.astype(np.int8) for i,mask in enumerate(geometry['masks']))}
    arrays={}
    try:
        if resume:
            status=json.loads((folder/'status.json').read_text())
            if status['phase']!='checkpoint-ready' or status['signature']!=signature:raise ValueError('Only a same-source accepted complete-frame prefix may resume')
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
                if str(cp['signature'])!=signature or json.loads(str(cp['runtime']))!=runtime:raise ValueError('Runtime/source/physical request mismatch')
                values={k:jnp.asarray(cp['state_'+k]) for k in state._fields if k!='history'}
                history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in state.history._fields));state=FibreState(**values,history=history)
                law_history=FibreContactHistory(jnp.asarray(cp['contact_law_active']),jnp.asarray(cp['contact_law_valid']))
                contact=contact._replace(law=law_history,**{k:jnp.asarray(cp['contact_history_'+k]) for k in contact._fields if k!='law'})
                progress=tuple(jnp.asarray(cp['progress_'+str(i)]) for i in range(7))+(contact,);start=int(cp['frame']);solver_seconds=float(cp['solver_seconds'])
                tables=json.loads(str(cp['tables']));rows,records,times=tables['rows'],tables['records'],tables['times']
            if start!=status['frame'] or int(progress[0])!=plan['frame_steps'][start] or len(rows)!=start+1 or len(times)!=start+1 or len(records)!=start+1:raise ValueError('Checkpoint/frame tables disagree')
            validate_state(state,layout)
            for k in shapes:
                a=np.load(folder/f'trajectory_{k}.npy',mmap_mode='r+');arrays[k]=a
                if a.shape!=shapes[k] or a.dtype!=np.dtype(types[k]):raise ValueError('Raw trajectory shape changed')
                for i in range(start+1):
                    if descriptor(a[i])!=records[i][k]:raise ValueError('Raw saved prefix changed')
            if not np.array_equal(arrays['x'][start],np.asarray(state.x)) or not np.array_equal(arrays['damage'][start],np.asarray(state.history.damage)):raise ValueError('Checkpoint/raw endpoint mismatch')
            if not np.array_equal(np.asarray(state.mass),geometry['mass']) or not np.array_equal(np.asarray(state.volume),geometry['volume']):raise ValueError('Reference mass/volume changed')
        else:
            if any(folder.iterdir()):raise ValueError('Fresh output must be empty')
            arrays={k:np.lib.format.open_memmap(folder/f'trajectory_{k}.npy',mode='w+',dtype=types[k],shape=shapes[k]) for k in shapes}
            save_json(folder/'plan.json',plan);save_npz(folder/'particle_fields.npz',**static);save_json(folder/'contact-geometry.json',{'coverage_pairs':descriptor(np.asarray(data.coverage_pairs)),'neighbours':descriptor(neighbours),'reference_centres':descriptor(np.asarray(data.coverage_reference)),'actual_coverage_pairs':len(data.coverage_pairs)})
        end=plan['frames'] if stop_after_frame is None else stop_after_frame
        if type(end) is not int or not start<end<=plan['frames']:raise ValueError('Endpoint must advance whole saved frames within fixed full plan')
        @jax.jit
        def advance_block(s,p,count):
            def body(_,carry):
                old,total=carry
                def attempt(_):
                    applied,attachment,metrics,extent,unsupported=evaluate(old,total[7])
                    new,diag=step_with_diagnostics(old,material,layout,config,applied,boundary=boundary)
                    new=new._replace(valid=new.valid&metrics.valid)
                    # Stored elastic energy at fixed F before/after damage isolates
                    # the irreversible constitutive loss from transfer/integration.
                    undamaged=new.history._replace(damage=jnp.zeros_like(new.history.damage),failed=jnp.zeros_like(new.history.failed))
                    loss_particle=new.volume*brittle_energy(new.F,undamaged,material)*(new.history.damage-old.history.damage)
                    loss=jnp.zeros(3).at[new.fibre_id].add(loss_particle)
                    fixture_speeds=jnp.stack(tuple(v(new.time) for v in velocities))
                    contact_impulse=config.dt*jnp.zeros((3,3)).at[old.fibre_id].add(applied)
                    contact_work=jnp.zeros(3).at[old.fibre_id].add(jnp.sum(applied*(new.x-old.x),axis=1))
                    updated=(total[0]+1,total[1]+diag.group_impulses,total[2]+jnp.sum(diag.group_impulses*fixture_speeds,axis=1),total[3]+loss,total[4]+contact_impulse,total[5]+contact_work,total[6]&new.valid,attachment)
                    return new,updated
                return jax.lax.cond(old.valid,attempt,lambda _:carry,None)
            return jax.lax.fori_loop(0,count,body,(s,p))
        def record(frame,previous_impulse,previous_step):
            x,v,mass,F,C=(np.asarray(getattr(state,k)) for k in ('x','v','mass','F','C'))
            D=np.asarray(state.history.damage);failed=np.asarray(state.history.failed);t=float(state.time);steps=int(progress[0])
            values={'x':x,'damage':D,'failed':failed.astype(np.uint8),'speed':np.linalg.norm(v,axis=1),'temperature':np.full(n,293.15,dtype=np.float32)}
            for k,a in arrays.items():a[frame]=values[k];a.flush()
            records.append({k:descriptor(a[frame]) for k,a in arrays.items()});times.append(t)
            interval=(steps-previous_step)*config.dt;force=(np.asarray(progress[1])-previous_impulse)/interval if interval else np.zeros((6,3))
            U=float(stored_energy(state,material));K=.5*float(np.sum(mass*np.sum(v*v,axis=1)))+grid['dx_m']**2/8*float(np.sum(mass*np.sum(C*C,axis=(1,2))))
            momentum=np.sum(mass[:,None]*v,axis=0);impulse=np.sum(np.asarray(progress[1]),axis=0)
            stroke=float(motion(t,plan['protocol']));targets=tuple(np.array([(-1 if side==0 else 1)*stroke/2,0.,0.]) for fibre in range(3) for side in (0,1))
            grip_error=max(float(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-target,axis=1))) for mask,target in zip(geometry['masks'],targets))
            row={'time':t,'accepted_steps':steps,'grip_displacement_m':stroke,'left_reaction_N':float(force[::2,0].sum()),'right_reaction_N':float(force[1::2,0].sum()),'tensile_load_N':float((force[1::2,0].sum()-force[::2,0].sum())/2),
                 'mass_kg':float(mass.sum()),'mass_relative_drift':float((mass.sum()-geometry['mass'].sum())/geometry['mass'].sum()),'min_J':float(np.linalg.det(F).min()),'kinetic_energy_J':K,'elastic_energy_J':U,
                 'fixture_work_J':float(np.sum(np.asarray(progress[2]))),'damage_dissipation_J':float(np.sum(np.asarray(progress[3]))),'energy_ledger_residual_J':float(np.sum(np.asarray(progress[2])))+float(np.sum(np.asarray(progress[5])))-K-U-float(np.sum(np.asarray(progress[3]))),
                 'momentum_residual_Ns':float(np.max(np.abs(momentum-impulse))),'grip_position_error_m':grip_error,'max_damage':float(D.max()),'failed_particles':int(failed.sum())}
            contact_force,_,metrics,extent,unsupported=evaluate(state,progress[7])
            if not bool(metrics.valid):raise FloatingPointError('Saved-state contact invalid, extended line contact or failed-section contact unsupported')
            row.update(contact_work_J=float(np.sum(np.asarray(progress[5]))),contact_force_balance_N=float(np.linalg.norm(np.asarray(metrics.force_residual))),contact_torque_balance_Nm=float(np.linalg.norm(np.asarray(metrics.torque_residual))),minimum_contact_gap_m=float(metrics.minimum_coverage_gap),active_contact_regions=int(metrics.patch_count),maximum_connected_contact_extent_m=float(extent),contact_valid=bool(metrics.valid))
            energy_density=np.asarray(brittle_energy(state.F,state.history,material))
            for fibre in range(3):
                mask=geometry['fibre_id']==fibre;grip_impulse=np.asarray(progress[1])[2*fibre:2*fibre+2].sum(0);contact_impulse=np.asarray(progress[4])[fibre];p=np.sum(mass[mask,None]*v[mask],axis=0)
                row.update({f'strand{fibre}_fixture_work_J':float(np.asarray(progress[2])[2*fibre:2*fibre+2].sum()),f'strand{fibre}_elastic_energy_J':float(np.sum(geometry['volume'][mask]*energy_density[mask])),f'strand{fibre}_kinetic_energy_J':.5*float(np.sum(mass[mask]*np.sum(v[mask]**2,axis=1)))+grid['dx_m']**2/8*float(np.sum(mass[mask]*np.sum(C[mask]**2,axis=(1,2)))),f'strand{fibre}_tensile_load_N':float((force[2*fibre+1,0]-force[2*fibre,0])/2),f'strand{fibre}_mass_kg':float(mass[mask].sum()),f'strand{fibre}_max_damage':float(D[mask].max()),f'strand{fibre}_failed_particles':int(failed[mask].sum()),f'strand{fibre}_damage_dissipation_J':float(progress[3][fibre]),f'strand{fibre}_contact_work_J':float(progress[5][fibre]),f'strand{fibre}_momentum_residual_Ns':float(np.linalg.norm(p-grip_impulse-contact_impulse))})
            rows.append(row)
            with (folder/'observables.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            # Existing VTK/render tooling expects diagnostics and a trajectory index.
            with (folder/'diagnostics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            save_npz(folder/'trajectory.npz',time=np.asarray(times),body=np.zeros(n,dtype=np.int32))
            leaves={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
            final={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};final.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()});save_npz(folder/'final_state_0.npz',**final)
            leaves.update({'progress_'+str(i):np.asarray(v) for i,v in enumerate(progress[:7])});leaves.update({'contact_history_'+k:np.asarray(v) for k,v in progress[7]._asdict().items() if k!='law'});leaves.update(contact_law_active=np.asarray(progress[7].law.active),contact_law_valid=np.asarray(progress[7].law.valid));leaves.update(frame=frame,signature=np.array(signature),runtime=np.array(json.dumps(runtime,sort_keys=True)),solver_seconds=solver_seconds,tables=np.array(json.dumps({'rows':rows,'records':records,'times':times})))
            save_npz(folder/'checkpoint.npz',**leaves)
        if not resume:record(0,np.zeros((6,3)),0)
        for frame in range(start+1,end+1):
            oldimp=np.asarray(progress[1]).copy();oldstep=int(progress[0]);target=plan['frame_steps'][frame]
            while int(progress[0])<target:
                started=time.monotonic();state,progress=advance_block(state,progress,jnp.asarray(min(16,target-int(progress[0])),dtype=jnp.int32));state.x.block_until_ready();solver_seconds+=time.monotonic()-started
                if not bool(state.valid):raise FloatingPointError('Invalid constitutive/grid state; no resumable checkpoint for failed step')
            validate_state(state,layout);record(frame,oldimp,oldstep);print(json.dumps({'frame':frame,'steps':int(progress[0]),'solver_seconds':solver_seconds}),flush=True)
        if source_record()!=plan['source']:raise ValueError('Source changed while computing')
        complete=end==plan['frames'];review=fracture_review(geometry,np.asarray(state.x),np.asarray(state.history.failed))
        scale=max(max(abs(r['fixture_work_J'])+abs(r['contact_work_J']) for r in rows),1e-20)
        checks={'full_plan_completed':complete,'formal_three_complete_turns':plan['scope']=='formal-three-turn','mass_conserved':max(abs(r['mass_relative_drift']) for r in rows)<1e-12,'positive_orientation':min(r['min_J'] for r in rows)>0,
                'three_independent_material_fields':layout.fibre_count==3,'per_strand_momentum':max(r[f'strand{i}_momentum_residual_Ns'] for r in rows for i in range(3))<1e-10,
                'formal_six_grips':max(r['grip_position_error_m'] for r in rows)<1e-9,'contact_valid':all(r['contact_valid'] for r in rows),'contact_action_reaction':max(r['contact_force_balance_N'] for r in rows)<1e-10,
                'energy_ledger_within15percent_work':max(abs(r['energy_ledger_residual_J']) for r in rows)<=.15*scale}
        summary={'case':'three-strand-tension','completed':complete,'segment_complete':not complete,'phase':'completed' if complete else 'checkpoint-ready','particles':[n],'steps':int(progress[0]),'target_steps':plan['steps'],'simulation_time':float(state.time),'solver_seconds':solver_seconds,'dtype':'float64','runtime':runtime,'definition':plan,'source':plan['source'],'materials':[dict(plan['material']) for _ in range(3)],
                 'configs':[{'dim':3,'dt':config.dt,'domain_size':plan['geometry']['length_m'],'n_grid':round(plan['geometry']['length_m']/grid['dx_m']),'failure_mode':'retain','grid_type':'three-independent-packed-fibre-fields'}],
                 'geometry_scale':1,'particle_spacing_m':plan['geometry']['spacing_m'],'particle_fields':{'file':'particle_fields.npz','fields':{k:descriptor(v) for k,v in static.items()}},
                 'fracture_review':review,'checks':checks,'numerical_loading_accepted':all(checks.values()),'all_three_strands_fractured':all(not s['intact_axial_section_path'] for s in review['strands']),'production_eligible':False,'public_visual_review_passed':False,
                 'energy_scope':'Fixture impulse times actual finite-step grip speed plus measured contact force work, minus stored+kinetic+constitutive damage loss. Curvature-dependent contact potential does not complete a conservative total energy.15percent dynamic residual is an explicit gate.',
                 'peak_tensile_load_N':max(r['tensile_load_N'] for r in rows),'initial':rows[0],'final':rows[-1]}
        save_json(folder/'summary.json',summary);save_json(folder/'status.json',{'phase':summary['phase'],'frame':end,'signature':signature,'steps':int(progress[0])});return summary
    except BaseException:
        save_json(folder/'status.json',{'phase':'failed','frame':len(rows)-1,'signature':signature,'never_resumable':True})
        raise
    finally:
        for a in arrays.values():a.flush();a._mmap.close()



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/three-strand-tension'));p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-frame',type=int);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--length',type=float,default=.024);p.add_argument('--diameter',type=float,default=.002);p.add_argument('--pitch',type=float,default=.008);p.add_argument('--compactness',type=float,default=.985);p.add_argument('--dx',type=float,default=.0002);p.add_argument('--spacing',type=float,default=.0001);p.add_argument('--model',choices=['hertz','jkr'],default='hertz')
    a=p.parse_args();geometry,plan=plan_case(length=a.length,diameter=a.diameter,pitch=a.pitch,compactness=a.compactness,dx=a.dx,spacing=a.spacing,model=a.model)
    if a.dry_run:print(json.dumps(plan,indent=2))
    else:run_case(geometry,plan,a.output,resume=a.resume,stop_after_frame=a.stop_after_frame)


if __name__=='__main__':main()
