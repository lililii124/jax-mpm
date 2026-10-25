"""Common compliant-fixture press/hold/normal-pull-off fibre comparison.

Four physical requests share geometry/material/actuator path/fixture/camera.
The robust Hertz/JKR pair is screened away from snap. Separate MD/SC comparison
remains disabled until their applicability and BEM mesh convergence qualify.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from .fibre_pull_off import make_geometry
from .single_fibre_tension import digest,descriptor,save_json,save_npz,runtime_record

ROOT=Path(__file__).resolve().parents[1]
MODELS=('hertz','jkr','md','self-consistent')
SOURCE_FILES=('examples/fibre_adhesion_contrast.py','examples/fibre_pull_off.py','examples/single_fibre_tension.py',
 'tools/scan_fibre_adhesion_contrast.py','tools/scan_fibre_pair_fixture.py','mpm_jax/fibre_mpm.py','mpm_jax/fibre_coupling.py','mpm_jax/fibre_contact.py','mpm_jax/fibre_contrast_coupling.py','mpm_jax/local_contact_models.py','mpm_jax/maugis_dugdale.py','mpm_jax/cohesive_bem.py','mpm_jax/materials/isotropic_brittle.py')


def source_record():
    files={n:digest(ROOT/n) for n in SOURCE_FILES}
    return {'files':files,'fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def motion(t,protocol,xp=np):
    result=xp.asarray(0.)
    for left,right,a,b in zip(protocol['times_s'][:-1],protocol['times_s'][1:],protocol['actuator_y_m'][:-1],protocol['actuator_y_m'][1:]):
        q=xp.clip((t-left)/(right-left),0.,1.)
        result=result+.5*(b-a)*(1-xp.cos(xp.pi*q))
    return result


def plan_case(model='jkr',*,young=4e5,work=.08,tabor=5.,fixture_stiffness=None,fixture_damping=.0015,hold_actuator=.00045,rings=256,duration=.1,frames=60):
    if model not in MODELS or min(young,work,tabor,duration)<=0 or fixture_damping<0:raise ValueError('Positive material, adhesion, Tabor and duration; nonnegative fixture damping required')
    if tabor<5:raise ValueError('Four-model quantitative comparison requires JKR asymptotic-screen Tabor>=5; small-Tabor differences are not admissible JKR evidence')
    R=.001;length=.032;grip=.003;dx=.0002;spacing=.0001;nu=.25;rho=1000.;gap=.00001
    geometry=make_geometry(spacing=spacing,length=length,radius=R,initial_gap=gap,density=rho,grip_length=grip)
    reduced=young/(2*(1-nu**2));interaction=(R*work**2/(reduced**2*tabor**3))**(1/3)
    a0=(9*np.pi*work*R*R/(2*reduced))**(1/3)
    if a0/R>.18:raise ValueError('Local circular contact radius is too large relative to actual fibre radius')
    gauge=length-2*grip;beam_stiffness=192*young*(np.pi*R**4/4)/(2*gauge**3)
    stiffness=1/(1/.5-1/beam_stiffness) if fixture_stiffness is None else fixture_stiffness
    if stiffness<=0:raise ValueError('Positive physical fixture spring required')
    effective=1/(1/beam_stiffness+1/stiffness)
    c=math.sqrt(young*(1-nu)/((1+nu)*(1-2*nu)*rho));steps=math.ceil(duration/(.15*dx/c));dt=duration/steps
    material={'young':young,'poisson':nu,'density':rho,'tensile_strength':3e5,'fracture_energy':100.,'characteristic_length':dx}
    bounds=np.asarray([[[-length/2,length/2],[-.004,.006],[-.002,.002]],[[-.002,.002],[-.003,.008],[-length/2,length/2]]]);shape=np.ceil(bounds[:,:,1]/dx).astype(int)-np.floor(bounds[:,:,0]/dx).astype(int)+13
    common={'geometry':geometry['metadata'],'material':material,'fixture':{'spring_stiffness_N_per_m':stiffness,'parallel_damper_Ns_per_m':fixture_damping,'beam_relative_stiffness_estimate_N_per_m':beam_stiffness,'series_stiffness_estimate_N_per_m':effective,
            'boundary':'Lower fibre both end collars fixed xyz. Upper fibre end collars have fixed xz and one common free rigid y translation; spring/damper links that y DOF to the prescribed actuator. No upper y displacement is prescribed directly.'},
            'protocol':{'times_s':(duration*np.asarray([0,.15,.25,.55,.85,1.])).tolist(),'actuator_y_m':[0.,-.00015,-.00015,hold_actuator,hold_actuator,.0012],'scope':'Identical press,hold,slow pull to common near-threshold plateau,hold,final separation. Hold level is a scan-based prototype and requires shared-path MPM pilot confirmation; never shift individual model loading.'},
            'contact_parameters':{'effective_modulus_Pa':reduced,'reference_effective_radius_m':R,'work_of_adhesion_J_per_m2':work,'interaction_range_m':interaction,'Tabor_mu':tabor,'outer_radius_m':.0003,'rings':rings,'local_a0_over_R':a0/R,'max_sweeps':8192,'equilibrium_tolerance':1e-8},
            'camera':{'focus_bounds_m':[-.003,.003,-.0018,.0042,-.003,.003],'field':'family_id','geometry_scale':1,'axes':False,'annotations':False,'resolution':1200,'colors':'Same fixed lower/upper fibre family palette for all4 models'},'duration_s':duration,'saved_states':frames+1,'steps':steps,'dt':dt}
    from tools.scan_fibre_pair_fixture import state,jkr_branch
    held=state('jkr',young,work,effective,gap+hold_actuator,True)
    critical=jkr_branch(young,work,effective)[1]
    pair_screen={'actual_centre_separation_difference_m':abs(gap+hold_actuator+held['indentation_m']),'linear_beam_bending_m':abs(held['force_N'])/beam_stiffness,'hold_contact_force_N':held['force_N'],'hold_contact_radius_m':held['radius_m'],'stable_hold_margin_m':critical-gap-hold_actuator,'scope':'Analytic stable-contact plus linear beam/common spring screen; actual MPM deformation/dynamics remain unverified.'}
    pair_pass=model in ('hertz','jkr') and held['active'] and pair_screen['actual_centre_separation_difference_m']>.0003 and critical>1.3*(gap+hold_actuator) and held['radius_m']/R<.2
    return geometry,{'series':'fibre-adhesion-contrast-v1','model':model,'case':'fibre-adhesion-contrast','physical_common':common,'common_request_sha256':hashlib.sha256(json.dumps(common,sort_keys=True).encode()).hexdigest(),
      'material':material,'geometry':geometry['metadata'],'fixture':common['fixture'],'protocol':common['protocol'],'contact':common['contact_parameters'],'camera':common['camera'],
      'duration_s':duration,'frames':frames,'saved_states':frames+1,'steps':steps,'dt':dt,'frame_steps':np.rint(np.linspace(0,steps,frames+1)).astype(int).tolist(),'particles':len(geometry['x']),
      'grid':{'dx_m':dx,'padding_cells':6,'bounds_m':bounds.tolist(),'max_nodes':1500000},'source':source_record(),
      'comparison_group':'hertz-jkr-adsorption-baseline-pair' if model in ('hertz','jkr') else 'unqualified-four-model-research','pair_reduced_screen':pair_screen,'group_scan_qualified':pair_pass,'qualification_blocker':'Tabor5 local scan converges with8192 sweeps at strict1e-8 tolerance, but SC128/256/512-ring threshold drifts48/111um exceed the2.58um JKR/MD distinction. A shared compliant fixture can expose a physical time window only after convergence/pilot validation; no artificial contrast or altered per-model conditions allowed.',
      'production_eligible':False,'public_visual_review_passed':False,'resource':{'uss_limit_bytes':6*1024**3,'raw_trajectory_bytes':(frames+1)*len(geometry['x'])*45,'first_prefix_frame':2},
      'scope':'Hertz versus JKR adsorption/nonadsorption pair first; MD/SC only when individually applicable and mesh-qualified. Illustrative same-geometry soft-fixture dynamic normal pull-off, not axial interfacial shear pullout. No fabrication pretension or calibrated strength. Material damage is measured separately from contact detachment.'}


def run_case(geometry,plan,output,*,resume=False,stop_after_frame=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import (FibreConfig,FibreState,BoundaryResult,build_layout,make_state,step_with_diagnostics,validate_state,stored_energy)
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle,brittle_energy
    from mpm_jax.fibre_coupling import build_contact_data,initial_contact_history,FibreCouplingHistory
    from mpm_jax.fibre_contrast_coupling import evaluate_contact
    from mpm_jax.local_contact_models import local_contact_model
    from mpm_jax.fibre_contact import FibreContactHistory
    from examples.triaxial_failure import restore_contact_history,restore_contact_model
    if not plan['group_scan_qualified'] and (stop_after_frame is None or stop_after_frame>2):raise ValueError(plan['qualification_blocker'])
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'checkpoint.npz').exists() and not resume:raise ValueError('Existing state requires explicit resume')
    runtime=runtime_record();signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    material=isotropic_brittle(**plan['material']);grid=plan['grid'];layout=build_layout(geometry['x'],geometry['fibre_id'],dx=grid['dx_m'],padding_cells=6,reference_bounds=grid['bounds_m'],max_nodes=grid['max_nodes'])
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout);config=FibreConfig(plan['dt'])
    local=plan['contact'];capture=local['interaction_range_m'] if plan['model'] in ('md','self-consistent') else 0.
    data=build_contact_data(geometry,search_window=15,coverage_skin=.006,proximity_margin=capture)
    law=local_contact_model(plan['model'],effective_modulus=local['effective_modulus_Pa'],effective_radius=local['reference_effective_radius_m'],work_of_adhesion=local['work_of_adhesion_J_per_m2'],interaction_range=local['interaction_range_m'],outer_radius=local['outer_radius_m'],rings=local['rings'])
    contact=initial_contact_history(data,law)
    @jax.jit
    def evaluate(current,history):return evaluate_contact(current,history,data,law)
    lower,upper=(jnp.asarray(m) for m in geometry['grip_masks']);upper_mass=jnp.sum(state.mass*upper)
    def collar_boundary(velocity,mass,current,support,layout,instant):
        assigned=jnp.zeros(layout.node_count,dtype=bool);impulses=[];work=[];valid=jnp.asarray(True)
        for i,mask in enumerate((lower,upper)):
            selected=jnp.zeros(layout.node_count,dtype=jnp.int32).at[support.indices].max(jnp.broadcast_to(mask[:,None],support.weights.shape).astype(jnp.int32),mode='drop')>0
            selected=selected&(mass>0);valid=valid&~jnp.any(selected&assigned)
            free_y=jnp.sum(jnp.where(selected,mass*velocity[:,1],0))/jnp.maximum(jnp.sum(jnp.where(selected,mass,0)),1e-30)
            target=jnp.zeros(3).at[1].set(jnp.where(i==1,free_y,0))
            corrected=jnp.where(selected[:,None],target,velocity);impulses.append(jnp.sum(mass[:,None]*(corrected-velocity),axis=0));work.append(.5*jnp.sum(mass*jnp.sum((corrected-velocity)*(corrected+velocity),axis=1)));velocity=corrected;assigned|=selected
        return BoundaryResult(velocity,jnp.stack(impulses),jnp.stack(work),valid)
    def actuator(current):
        y=jnp.sum(current.mass*upper*(current.x[:,1]-jnp.asarray(geometry['x'][:,1])))/upper_mass
        velocity=jnp.sum(current.mass*upper*current.v[:,1])/upper_mass
        target=motion(current.time,plan['protocol'],jnp);next_target=motion(current.time+config.dt,plan['protocol'],jnp);target_speed=(next_target-target)/config.dt
        extension=target-y;relative=target_speed-velocity
        force=plan['fixture']['spring_stiffness_N_per_m']*extension+plan['fixture']['parallel_damper_Ns_per_m']*relative
        particle_force=jnp.zeros_like(current.x).at[:,1].set(current.mass*upper/upper_mass*force)
        return particle_force,force,target_speed,relative,extension
    n=plan['particles'];frames=plan['saved_states'];shapes={'x':(frames,n,3),**{k:(frames,n) for k in ('damage','failed','speed','temperature')}}
    types={'x':np.float64,'damage':np.float64,'failed':np.uint8,'speed':np.float64,'temperature':np.float32}
    rows=[];records=[];times=[];start=0;solver_seconds=0.;progress=(jnp.array(0,dtype=jnp.int64),jnp.zeros((2,3)),jnp.zeros(2),jnp.array(0.),jnp.zeros((2,3)),jnp.zeros(2),jnp.array(0.),jnp.array(0.),contact)
    static={'phase':np.ones(n,dtype=np.int8),'yarn_id':geometry['fibre_id'],'family_id':geometry['family_id'],'grip_id':geometry['grip_masks'][0].astype(np.int8)+2*geometry['grip_masks'][1].astype(np.int8)}
    arrays={}
    try:
        if resume:
            status=json.loads((folder/'status.json').read_text())
            if status['phase']!='checkpoint-ready' or status['signature']!=signature:raise ValueError('Only a same-source accepted complete-frame prefix may resume')
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
                if str(cp['signature'])!=signature or json.loads(str(cp['runtime']))!=runtime:raise ValueError('Runtime/source/physical request mismatch')
                values={k:jnp.asarray(cp['state_'+k]) for k in state._fields if k!='history'}
                history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in state.history._fields));state=FibreState(**values,history=history)
                law=restore_contact_model(law,cp)
                contact=FibreCouplingHistory(restore_contact_history(contact.law,cp,jnp,parameters=law),jnp.asarray(cp['contact_radius']),jnp.asarray(cp['contact_segments']),jnp.asarray(cp['contact_valid']))
                progress=tuple(jnp.asarray(cp['progress_'+str(i)]) for i in range(8))+(contact,);start=int(cp['frame']);solver_seconds=float(cp['solver_seconds'])
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
            save_json(folder/'plan.json',plan);save_npz(folder/'particle_fields.npz',**static)
        end=plan['frames'] if stop_after_frame is None else stop_after_frame
        if type(end) is not int or not start<end<=plan['frames']:raise ValueError('Endpoint must advance whole saved frames within fixed full plan')
        @jax.jit
        def advance_block(s,p,count):
            def body(_,carry):
                old,total=carry
                def attempt(_):
                    contact_force,attachment,metrics=evaluate(old,total[8])
                    fixture_force,force_scalar,target_speed,relative,extension=actuator(old)
                    new,diag=step_with_diagnostics(old,material,layout,config,contact_force+fixture_force,boundary=collar_boundary)
                    new=new._replace(valid=new.valid&metrics.valid)
                    # Stored elastic energy at fixed F before/after damage isolates
                    # the irreversible constitutive loss from transfer/integration.
                    undamaged=new.history._replace(damage=jnp.zeros_like(new.history.damage),failed=jnp.zeros_like(new.history.failed))
                    loss=jnp.sum(new.volume*brittle_energy(new.F,undamaged,material)*(new.history.damage-old.history.damage))
                    contact_impulse=config.dt*jnp.zeros((2,3)).at[old.fibre_id].add(contact_force)
                    contact_work=jnp.zeros(2).at[old.fibre_id].add(jnp.sum(contact_force*(new.x-old.x),axis=1))
                    fixture_body_work=jnp.sum(fixture_force*(new.x-old.x))
                    updated=(total[0]+1,total[1]+diag.group_impulses,total[2]+contact_work,total[3]+loss,total[4]+contact_impulse,total[5]+jnp.array([0.,force_scalar*config.dt]),total[6]+force_scalar*target_speed*config.dt,total[7]+plan['fixture']['parallel_damper_Ns_per_m']*relative**2*config.dt,attachment)
                    return new,updated
                return jax.lax.cond(old.valid,attempt,lambda _:carry,None)
            return jax.lax.fori_loop(0,count,body,(s,p))
        # Private prototype instrumentation: separate compilation from actual
        # integration throughput before sizing a bounded physical hold pilot.
        compile_started=time.monotonic()
        advance_executable=advance_block.lower(state,progress,jnp.asarray(16,dtype=jnp.int32)).compile()
        compilation_seconds=time.monotonic()-compile_started
        save_json(folder/f'compilation-profile-from-frame-{start}.json',
                  {'seconds':compilation_seconds,'physical_steps_executed':0,'source':plan['source']})
        def record(frame,previous_impulse,previous_step):
            x,v,mass,F,C=(np.asarray(getattr(state,k)) for k in ('x','v','mass','F','C'))
            D=np.asarray(state.history.damage);failed=np.asarray(state.history.failed);t=float(state.time);steps=int(progress[0])
            values={'x':x,'damage':D,'failed':failed.astype(np.uint8),'speed':np.linalg.norm(v,axis=1),'temperature':np.full(n,293.15,dtype=np.float32)}
            for k,a in arrays.items():a[frame]=values[k];a.flush()
            records.append({k:descriptor(a[frame]) for k,a in arrays.items()});times.append(t)
            interval=(steps-previous_step)*config.dt;force=(np.asarray(progress[1])-previous_impulse)/interval if interval else np.zeros((2,3))
            U=float(stored_energy(state,material));K=.5*float(np.sum(mass*np.sum(v*v,axis=1)))+grid['dx_m']**2/8*float(np.sum(mass*np.sum(C*C,axis=(1,2))))
            momentum=np.sum(mass[:,None]*v,axis=0);impulse=np.sum(np.asarray(progress[1]),axis=0)
            stroke=float(motion(t,plan['protocol']));mean_y=float(np.sum(mass*np.asarray(upper)*(x[:,1]-geometry['x'][:,1]))/float(upper_mass))
            grip_error=max(float(np.max(np.linalg.norm((x-geometry['x'])[np.asarray(lower)],axis=1))),float(np.max(np.linalg.norm((x-geometry['x'])[np.asarray(upper)][:,[0,2]],axis=1))))
            contact_force,_,metrics=evaluate(state,progress[8]);_,fixture_force,_,_,extension=actuator(state)
            spring=.5*plan['fixture']['spring_stiffness_N_per_m']*float(extension)**2
            row={'time':t,'accepted_steps':steps,'actuator_displacement_y_m':stroke,'actual_upper_collar_displacement_y_m':mean_y,'fixture_force_N':float(fixture_force),'contact_force_N':float(metrics.normal_force[0]),'contact_radius_m':float(metrics.contact_radius[0]),'contact_gap_m':float(metrics.gap[0]),'contact_active':bool(metrics.active[0]),'contact_valid':bool(metrics.valid),
                 'mass_kg':float(mass.sum()),'mass_relative_drift':float((mass.sum()-geometry['mass'].sum())/geometry['mass'].sum()),'min_J':float(np.linalg.det(F).min()),'kinetic_energy_J':K,'elastic_energy_J':U,'spring_energy_J':spring,'actuator_work_J':float(progress[6]),'damper_dissipation_J':float(progress[7]),'damage_dissipation_J':float(progress[3]),'actual_contact_work_J':float(np.sum(np.asarray(progress[2]))),
                 'energy_ledger_residual_J':float(progress[6])+float(np.sum(np.asarray(progress[2])))-K-U-spring-float(progress[7])-float(progress[3]),'momentum_residual_Ns':float(np.max(np.abs(momentum-np.asarray(state.boundary_impulse)-np.asarray(state.external_impulse)))),'grip_position_error_m':grip_error,'max_damage':float(D.max()),'failed_particles':int(failed.sum())}
            for body,label in ((0,'lower'),(1,'upper')):
                mask=geometry['fibre_id']==body;centre=mask&(np.abs(geometry['x'][:,0 if body==0 else 2])<=plan['geometry']['particle_spacing_m'])
                row[label+'_centre_displacement_y_m']=float(np.mean((x-geometry['x'])[centre,1]));row[label+'_momentum_residual_Ns']=float(np.linalg.norm(np.sum(mass[mask,None]*v[mask],axis=0)-np.asarray(progress[1])[body]-np.asarray(progress[4])[body]-np.array([0.,float(progress[5][body]),0.])))
            if not row['contact_valid']:raise FloatingPointError('Local contact failed before saved-frame admission')
            rows.append(row)
            with (folder/'observables.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            # Existing VTK/render tooling expects diagnostics and a trajectory index.
            with (folder/'diagnostics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            save_npz(folder/'trajectory.npz',time=np.asarray(times),body=np.zeros(n,dtype=np.int32))
            leaves={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
            final={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};final.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()});save_npz(folder/'final_state_0.npz',**final)
            leaves.update({'progress_'+str(i):np.asarray(v) for i,v in enumerate(progress[:8])});leaves.update(contact_radius=np.asarray(progress[8].radius),contact_segments=np.asarray(progress[8].segments),contact_valid=np.asarray(progress[8].valid));leaves.update({'contact_local_'+k:np.asarray(v) for k,v in progress[8].law._asdict().items() if k not in ('active','valid')});leaves.update(contact_active=np.asarray(progress[8].law.active),contact_law_valid=np.asarray(progress[8].law.valid));
            if hasattr(law,'base'):leaves.update({'contact_bem_'+k:np.asarray(v) for k,v in law.base._asdict().items()})
            leaves.update(frame=frame,signature=np.array(signature),runtime=np.array(json.dumps(runtime,sort_keys=True)),solver_seconds=solver_seconds,tables=np.array(json.dumps({'rows':rows,'records':records,'times':times})))
            save_npz(folder/'checkpoint.npz',**leaves)
        if not resume:record(0,np.zeros((2,3)),0)
        for frame in range(start+1,end+1):
            oldimp=np.asarray(progress[1]).copy();oldstep=int(progress[0]);target=plan['frame_steps'][frame]
            while int(progress[0])<target:
                block_start=int(progress[0]);started=time.monotonic();state,progress=advance_executable(state,progress,jnp.asarray(min(16,target-int(progress[0])),dtype=jnp.int32));state.x.block_until_ready();block_seconds=time.monotonic()-started;solver_seconds+=block_seconds
                with (folder/'integration-block-profile.jsonl').open('a') as profile:
                    profile.write(json.dumps({'start_step':block_start,'end_step':int(progress[0]),'seconds':block_seconds})+'\n')
                if not bool(state.valid):raise FloatingPointError('Invalid constitutive/grid state; no resumable checkpoint for failed step')
            validate_state(state,layout);record(frame,oldimp,oldstep);print(json.dumps({'frame':frame,'steps':int(progress[0]),'solver_seconds':solver_seconds}),flush=True)
        if source_record()!=plan['source']:raise ValueError('Source changed while computing')
        complete=end==plan['frames'];active=[i for i,r in enumerate(rows) if r['contact_active']];detach=next((r['time'] for i,r in enumerate(rows) if i>0 and rows[i-1]['contact_active'] and not r['contact_active']),None)
        scale=max(max(abs(r['actuator_work_J'])+abs(r['actual_contact_work_J']) for r in rows),1e-20)
        checks={'full_plan_completed':complete,'common_scan_qualified':plan['group_scan_qualified'],'contact_valid':all(r['contact_valid'] for r in rows),'mass_conserved':max(abs(r['mass_relative_drift']) for r in rows)<1e-12,'formal_lower_and_upper_xz_constraints':max(r['grip_position_error_m'] for r in rows)<1e-9,'per_body_momentum':max(max(r['lower_momentum_residual_Ns'],r['upper_momentum_residual_Ns']) for r in rows)<1e-10,'undamaged_material':not np.asarray(state.history.failed).any() and not np.asarray(state.history.damage).any(),'energy_ledger_within15percent_work':max(abs(r['energy_ledger_residual_J']) for r in rows)<.15*scale}
        summary={'case':'fibre-adhesion-contrast/'+plan['model'],'completed':complete,'segment_complete':not complete,'phase':'completed' if complete else 'checkpoint-ready','particles':[n],'steps':int(progress[0]),'target_steps':plan['steps'],'simulation_time':float(state.time),'solver_seconds':solver_seconds,'runtime':runtime,'definition':plan,'source':plan['source'],'common_request_sha256':plan['common_request_sha256'],'materials':[plan['material']],
                 'configs':[{'dim':3,'dt':config.dt,'domain_size':plan['geometry']['length_m'],'n_grid':round(plan['geometry']['length_m']/grid['dx_m']),'failure_mode':'retain','grid_type':'independent-packed-rectangular-fibre-fields'}],'geometry_scale':1,'particle_spacing_m':plan['geometry']['particle_spacing_m'],'particle_fields':{'file':'particle_fields.npz','fields':{k:descriptor(v) for k,v in static.items()}},
                 'checks':checks,'numerical_comparison_qualified':all(checks.values()),'production_eligible':False,'public_visual_review_passed':False,'first_saved_detachment_time_s':detach,'minimum_saved_contact_force_N':min(r['contact_force_N'] for r in rows),'maximum_actual_relative_centre_bending_m':max(abs(r['upper_centre_displacement_y_m']-r['actual_upper_collar_displacement_y_m']-r['lower_centre_displacement_y_m']) for r in rows),'initial':rows[0],'final':rows[-1],
                 'scope':'Resource-prefix prototype only until strict local SC mesh/domain convergence and common MPM path qualification. Saved detachment time is frame-sampled, not a substep event time; old complete results remain authoritative separate experiments.'}
        save_json(folder/'summary.json',summary);save_json(folder/'status.json',{'phase':summary['phase'],'frame':end,'signature':signature,'steps':int(progress[0])});return summary
    except BaseException:
        save_json(folder/'status.json',{'phase':'failed','frame':len(rows)-1,'signature':signature,'never_resumable':True})
        raise
    finally:
        for a in arrays.values():a.flush();a._mmap.close()



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,default='jkr');p.add_argument('--output',type=Path,default=Path('outputs/fibre-adhesion-contrast'));p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-frame',type=int);p.add_argument('--dry-run',action='store_true');p.add_argument('--young',type=float,default=4e5);p.add_argument('--work',type=float,default=.08);p.add_argument('--tabor',type=float,default=5.);p.add_argument('--fixture-stiffness',type=float);p.add_argument('--fixture-damping',type=float,default=.0015);p.add_argument('--hold-actuator',type=float,default=.00045);p.add_argument('--rings',type=int,default=256)
    a=p.parse_args();geometry,plan=plan_case(a.model,young=a.young,work=a.work,tabor=a.tabor,fixture_stiffness=a.fixture_stiffness,fixture_damping=a.fixture_damping,hold_actuator=a.hold_actuator,rings=a.rings)
    if a.dry_run:print(json.dumps(plan,indent=2))
    else:run_case(geometry,plan,a.output/a.model,resume=a.resume,stop_after_frame=a.stop_after_frame)


if __name__=='__main__':main()
