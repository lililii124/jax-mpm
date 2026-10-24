"""Three-dimensional circular single-fibre tension with audited crack-band failure.

No contact, particle erosion or imposed crack. A declared smooth central waist
selects localization. Parameters are an illustrative material-family benchmark.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ('examples/single_fibre_tension.py','mpm_jax/fibre_mpm.py',
                'mpm_jax/materials/isotropic_brittle.py')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def source_record():
    files={name:digest(ROOT/name) for name in SOURCE_FILES}
    return {'files':files,'fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_npz(path,**data):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)


def descriptor(a):
    a=np.ascontiguousarray(a)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}


def plan_case(dx=.0002,spacing=.0001,frames=60):
    if not (np.isfinite(dx) and np.isfinite(spacing) and dx>0 and spacing>0) or type(frames) is not int or frames<2:
        raise ValueError('Positive finite grid/sampling and at least two output intervals required')
    length,radius,grip=.024,.001,.002
    if not np.isclose(dx/spacing,2) or not np.isclose(2*radius/spacing,round(2*radius/spacing)):
        raise ValueError('Use two particles per cell and an integer diameter sampling')
    if 2*radius/dx<10:raise ValueError('At least ten grid cells across the diameter are required')
    axial=(np.arange(round(length/spacing))+.5)*spacing-length/2
    radial=(np.arange(round(2*radius/spacing))+.5)*spacing-radius
    xx,yy,zz=np.meshgrid(axial,radial,radial,indexing='ij')
    local_radius=radius*(1-.15*np.exp(-(xx/.0008)**4))
    occupied=yy*yy+zz*zz<=local_radius**2
    lattice=np.argwhere(occupied).astype(np.int32)
    x=np.column_stack((xx[occupied],yy[occupied],zz[occupied]))
    volume=np.full(len(x),spacing**3);ids=np.zeros(len(x),dtype=np.int32)
    masks=(x[:,0]<-length/2+grip,x[:,0]>length/2-grip)
    material={'young':2e6,'poisson':.25,'density':1000.,'tensile_strength':1e5,
              'fracture_energy':1.,'characteristic_length':dx}
    c=math.sqrt(material['young']*(1-material['poisson'])/((1+material['poisson'])*(1-2*material['poisson']))/material['density'])
    duration=.012;load_time=.009;stroke=.0032
    steps=math.ceil(duration/(.15*dx/c));dt=duration/steps
    delta0=dx*material['tensile_strength']/material['young'];deltaf=2*material['fracture_energy']/material['tensile_strength']
    if deltaf<=delta0:raise ValueError('Invalid crack-band softening length')
    bounds=np.array([[[-length/2-stroke/2,length/2+stroke/2],[-.002,.002],[-.002,.002]]])
    gridshape=np.ceil(bounds[:,:,1]/dx).astype(int)-np.floor(bounds[:,:,0]/dx).astype(int)+13
    shape=occupied.shape;counts=np.bincount(lattice[:,0],minlength=len(axial));area=counts*spacing**2
    plan={'series':'single-fibre-tension-v1','case':'single-fibre-tension','duration_s':duration,
          'frames':frames,'saved_states':frames+1,'steps':steps,'dt':dt,
          'frame_steps':np.rint(np.linspace(0,steps,frames+1)).astype(int).tolist(),
          'particles':len(x),'geometry':{'length_m':length,'radius_m':radius,'diameter_m':2*radius,
          'grip_length_m':grip,'free_span_m':length-2*grip,'waist_radius_reduction':.15,
          'waist_half_width_m':.0008,'waist_definition':'r(x)=R*(1-0.15*exp(-(x/0.8mm)^4)); circular cross sections',
          'nominal_area_m2':math.pi*radius**2,'sampled_minimum_area_m2':float(area.min()),
          'sampled_end_area_m2':float(area.max()),'lattice_shape':list(shape),'spacing_m':spacing,
          'diameter_grid_cells':2*radius/dx,'diameter_particle_samples':2*radius/spacing},
          'grid':{'dx_m':dx,'padding_cells':6,'bounds_m':bounds.tolist(),'nodes':int(np.prod(gridshape)), 'max_nodes':500000},
          'material':material,'material_assumptions':'Illustrative uncalibrated isotropic elastic-brittle baseline; E,nu,rho share crossed-fibre material family, but ft=0.1MPa and Gf=1J/m2 deliberately differ. Not a measured single-fibre strength or a calibration of old contact results.',
          'softening':{'onset_strain':material['tensile_strength']/material['young'],
          'onset_displacement_m':delta0,'failure_displacement_m':deltaf,'failure_strain':deltaf/dx,
          'maximum_h_m':2*material['young']*material['fracture_energy']/material['tensile_strength']**2,
          'scope':'Existing maximum tensile Biot-strain history, irreversible isotropic damage and linear equivalent-displacement softening, h=grid dx. Gf refers to monotonic uniaxial reference-area envelope; distributed dynamic dissipation is audited separately.'},
          'protocol':{'load_time_s':load_time,'stroke_m':stroke,'hold_time_s':duration-load_time,
          'boundary':'Both 2mm reference end collars rigid in xyz; symmetric +/-x half stroke; smooth half-cosine ramp then hold. Exact finite-step prescribed displacement increments/dt.'},
          'wave_speed_m_per_s':c,'loading_wave_transits':load_time*c/(length-2*grip),
          'source':source_record(),'geometry_scale':1.,'production_eligible':False,
          'resource':{'uss_limit_bytes':6*1024**3,'raw_trajectory_bytes':(frames+1)*len(x)*(24+8+8+4+1),
          'note':'USS enforced by external one-second process-tree monitor. Grid/array counts are storage, not a proven peak-memory estimate.'},
          'render_plan':{'renderer':'ParaView','field':'damage','frames':frames+1,'geometry_scale':1,'axes':False,'annotations':False,'resolution':1200,'camera':'Shared full true-scale rod; optional independent central detail without geometry exaggeration'},
          'publication_requirements':['completed full physical plan','all saved states/hash inventory valid','full sampled cross-section failed','no intact reference lattice path between grips','positive measured axial opening','force envelope falls below10percent peak','mass/momentum/energy ledger acceptance','actual ParaView61 frames independently reviewed']}
    geometry={'x':x,'volume':volume,'mass':material['density']*volume,'fibre_id':ids,'masks':masks,'lattice':lattice}
    return geometry,plan


def motion(t,protocol,xp=np):
    phase=xp.clip(t/protocol['load_time_s'],0.,1.)
    return .5*protocol['stroke_m']*(1-xp.cos(xp.pi*phase))


def fracture_review(geometry,x,failed):
    from scipy.ndimage import binary_propagation
    lattice=geometry['lattice'];shape=tuple(lattice.max(0)+1)
    intact=np.zeros(shape,dtype=bool);intact[tuple(lattice[~failed].T)]=True
    seed=np.zeros(shape,dtype=bool);seed[tuple(lattice[geometry['masks'][0]&~failed].T)]=True
    connected=binary_propagation(seed,mask=intact)
    right=lattice[geometry['masks'][1]&~failed]
    spans=bool(connected[tuple(right.T)].any()) if len(right) else False
    total=np.bincount(lattice[:,0],minlength=shape[0]);alive=np.bincount(lattice[~failed,0],minlength=shape[0])
    cuts=np.flatnonzero((total>0)&(alive==0));opening=0.
    for cut in cuts:
        left=(lattice[:,0]<cut)&~failed;rightmask=(lattice[:,0]>cut)&~failed
        if left.any() and rightmask.any():opening=max(opening,float(x[rightmask,0].min()-x[left,0].max()))
    return {'intact_reference_path_between_grips':spans,'fully_failed_axial_sections':cuts.tolist(),
            'measured_intact_fragment_gap_m':opening,
            'scope':'Six-neighbour intact reference-lattice connectivity plus stronger entire sampled cross-section failure; no erased particles, imposed crack surface or rescaled displacement. Gap is between intact fragments and may contain stress-free failed particles.'}


def runtime_record():
    return {'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy')}}


def run_case(geometry,plan,output,*,resume=False,stop_after_frame=None):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import (FibreConfig,FibreState,build_layout,make_state,reference_grips,step_with_diagnostics,validate_state,stored_energy)
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle,brittle_energy
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'checkpoint.npz').exists() and not resume:raise ValueError('Existing state requires explicit resume')
    runtime=runtime_record();signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    material=isotropic_brittle(**plan['material']);grid=plan['grid'];layout=build_layout(geometry['x'],geometry['fibre_id'],dx=grid['dx_m'],padding_cells=6,reference_bounds=grid['bounds_m'],max_nodes=grid['max_nodes'])
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout);config=FibreConfig(plan['dt'])
    def speed(t):return (motion(t,plan['protocol'],jnp)-motion(t-config.dt,plan['protocol'],jnp))/config.dt
    boundary=reference_grips(geometry['masks'],(lambda t:jnp.array([-speed(t)/2,0.,0.]),lambda t:jnp.array([speed(t)/2,0.,0.])))
    n=plan['particles'];frames=plan['saved_states'];shapes={'x':(frames,n,3),**{k:(frames,n) for k in ('damage','failed','speed','temperature')}}
    types={'x':np.float64,'damage':np.float64,'failed':np.uint8,'speed':np.float64,'temperature':np.float32}
    rows=[];records=[];times=[];start=0;solver_seconds=0.;progress=(jnp.array(0,dtype=jnp.int64),jnp.zeros((2,3)),jnp.zeros(2),jnp.array(0.))
    static={'phase':np.ones(n,dtype=np.int8),'yarn_id':np.zeros(n,dtype=np.int32),'family_id':np.zeros(n,dtype=np.int32),'grip_id':geometry['masks'][0].astype(np.int8)+2*geometry['masks'][1].astype(np.int8)}
    arrays={}
    try:
        if resume:
            status=json.loads((folder/'status.json').read_text())
            if status['phase']!='checkpoint-ready' or status['signature']!=signature:raise ValueError('Only a same-source accepted complete-frame prefix may resume')
            with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
                if str(cp['signature'])!=signature or json.loads(str(cp['runtime']))!=runtime:raise ValueError('Runtime/source/physical request mismatch')
                values={k:jnp.asarray(cp['state_'+k]) for k in state._fields if k!='history'}
                history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in state.history._fields));state=FibreState(**values,history=history)
                progress=tuple(jnp.asarray(cp['progress_'+str(i)]) for i in range(4));start=int(cp['frame']);solver_seconds=float(cp['solver_seconds'])
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
                    new,diag=step_with_diagnostics(old,material,layout,config,boundary=boundary)
                    # Stored elastic energy at fixed F before/after damage isolates
                    # the irreversible constitutive loss from transfer/integration.
                    undamaged=new.history._replace(damage=jnp.zeros_like(new.history.damage),failed=jnp.zeros_like(new.history.failed))
                    loss=jnp.sum(new.volume*brittle_energy(new.F,undamaged,material)*(new.history.damage-old.history.damage))
                    velocities=jnp.stack((jnp.array([-speed(new.time)/2,0.,0.]),jnp.array([speed(new.time)/2,0.,0.])))
                    updated=(total[0]+1,total[1]+diag.group_impulses,total[2]+jnp.sum(diag.group_impulses*velocities,axis=1),total[3]+loss)
                    return new,updated
                return jax.lax.cond(old.valid,attempt,lambda _:carry,None)
            return jax.lax.fori_loop(0,count,body,(s,p))
        def record(frame,previous_impulse,previous_step):
            x,v,mass,F,C=(np.asarray(getattr(state,k)) for k in ('x','v','mass','F','C'))
            D=np.asarray(state.history.damage);failed=np.asarray(state.history.failed);t=float(state.time);steps=int(progress[0])
            values={'x':x,'damage':D,'failed':failed.astype(np.uint8),'speed':np.linalg.norm(v,axis=1),'temperature':np.full(n,293.15,dtype=np.float32)}
            for k,a in arrays.items():a[frame]=values[k];a.flush()
            records.append({k:descriptor(a[frame]) for k,a in arrays.items()});times.append(t)
            interval=(steps-previous_step)*config.dt;force=(np.asarray(progress[1])-previous_impulse)/interval if interval else np.zeros((2,3))
            U=float(stored_energy(state,material));K=.5*float(np.sum(mass*np.sum(v*v,axis=1)))+grid['dx_m']**2/8*float(np.sum(mass*np.sum(C*C,axis=(1,2))))
            momentum=np.sum(mass[:,None]*v,axis=0);impulse=np.sum(np.asarray(progress[1]),axis=0)
            stroke=float(motion(t,plan['protocol']));targets=(np.array([-stroke/2,0.,0.]),np.array([stroke/2,0.,0.]))
            grip_error=max(float(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-target,axis=1))) for mask,target in zip(geometry['masks'],targets))
            row={'time':t,'accepted_steps':steps,'grip_displacement_m':stroke,'left_reaction_N':float(force[0,0]),'right_reaction_N':float(force[1,0]),'tensile_load_N':float((force[1,0]-force[0,0])/2),
                 'mass_kg':float(mass.sum()),'mass_relative_drift':float((mass.sum()-geometry['mass'].sum())/geometry['mass'].sum()),'min_J':float(np.linalg.det(F).min()),'kinetic_energy_J':K,'elastic_energy_J':U,
                 'fixture_work_J':float(np.sum(np.asarray(progress[2]))),'damage_dissipation_J':float(progress[3]),'energy_ledger_residual_J':float(np.sum(np.asarray(progress[2])))-K-U-float(progress[3]),
                 'momentum_residual_Ns':float(np.max(np.abs(momentum-impulse))),'grip_position_error_m':grip_error,'max_damage':float(D.max()),'failed_particles':int(failed.sum())}
            rows.append(row)
            with (folder/'observables.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            # Existing VTK/render tooling expects diagnostics and a trajectory index.
            with (folder/'diagnostics.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            save_npz(folder/'trajectory.npz',time=np.asarray(times),body=np.zeros(n,dtype=np.int32))
            leaves={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
            final={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};final.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()});save_npz(folder/'final_state_0.npz',**final)
            leaves.update({'progress_'+str(i):np.asarray(v) for i,v in enumerate(progress)});leaves.update(frame=frame,signature=np.array(signature),runtime=np.array(json.dumps(runtime,sort_keys=True)),solver_seconds=solver_seconds,tables=np.array(json.dumps({'rows':rows,'records':records,'times':times})))
            save_npz(folder/'checkpoint.npz',**leaves)
        if not resume:record(0,np.zeros((2,3)),0)
        for frame in range(start+1,end+1):
            oldimp=np.asarray(progress[1]).copy();oldstep=int(progress[0]);target=plan['frame_steps'][frame]
            while int(progress[0])<target:
                started=time.monotonic();state,progress=advance_block(state,progress,jnp.asarray(min(16,target-int(progress[0])),dtype=jnp.int32));state.x.block_until_ready();solver_seconds+=time.monotonic()-started
                if not bool(state.valid):raise FloatingPointError('Invalid constitutive/grid state; no resumable checkpoint for failed step')
            validate_state(state,layout);record(frame,oldimp,oldstep);print(json.dumps({'frame':frame,'steps':int(progress[0]),'solver_seconds':solver_seconds}),flush=True)
        if source_record()!=plan['source']:raise ValueError('Source changed while computing')
        complete=end==plan['frames'];review=fracture_review(geometry,np.asarray(state.x),np.asarray(state.history.failed))
        peak=max(r['tensile_load_N'] for r in rows);tail=max(abs(r['tensile_load_N']) for r in rows[-5:]);ledger_scale=max(max(abs(r['fixture_work_J']) for r in rows),1e-20)
        checks={'full_plan_completed':complete,'mass_conserved':max(abs(r['mass_relative_drift']) for r in rows)<1e-12,'positive_orientation':min(r['min_J'] for r in rows)>0,
                'momentum_closure':max(r['momentum_residual_Ns'] for r in rows)<1e-10,'formal_grips':max(r['grip_position_error_m'] for r in rows)<1e-9,
                'nonnegative_damage_dissipation':all(r['damage_dissipation_J']>=0 for r in rows),'energy_ledger_within15percent_input_work':max(abs(r['energy_ledger_residual_J']) for r in rows)<=.15*ledger_scale,
                'cross_section_fully_failed':bool(review['fully_failed_axial_sections']),'axial_path_broken':not review['intact_reference_path_between_grips'],
                'positive_fragment_gap':review['measured_intact_fragment_gap_m']>3*grid['dx_m'],'load_drop90percent':peak>0 and tail<.1*peak}
        summary={'case':'single-fibre-tension','completed':complete,'segment_complete':not complete,'phase':'completed' if complete else 'checkpoint-ready','particles':[n],'steps':int(progress[0]),'target_steps':plan['steps'],'simulation_time':float(state.time),'solver_seconds':solver_seconds,'dtype':'float64','runtime':runtime,'definition':plan,'source':plan['source'],'materials':[plan['material']],
                 'configs':[{'dim':3,'dt':config.dt,'domain_size':plan['geometry']['length_m'],'n_grid':round(plan['geometry']['length_m']/grid['dx_m']),'failure_mode':'retain','grid_type':'independent-packed-rectangular-fibre-fields'}],
                 'geometry_scale':1,'particle_spacing_m':plan['geometry']['spacing_m'],'particle_fields':{'file':'particle_fields.npz','fields':{k:descriptor(v) for k,v in static.items()}},
                 'fracture_review':review,'checks':checks,'numerical_fracture_accepted':all(checks.values()),'production_eligible':False,'public_visual_review_passed':False,
                 'energy_scope':'Fixture impulse times actual prescribed finite-step velocity; constitutive damage loss at fixed updated F; residual includes APIC transfer/time integration. No global energy conservation asserted;15percent input-work gate is an explicit acceptance threshold.',
                 'assumed_single_band_fracture_work_J':plan['material']['fracture_energy']*plan['geometry']['sampled_minimum_area_m2'],'actual_damage_loss_over_assumed_single_band_work':float(progress[3])/(plan['material']['fracture_energy']*plan['geometry']['sampled_minimum_area_m2']),'opening_acceptance_m':3*grid['dx_m'],'peak_tensile_load_N':peak,'tail_maximum_abs_load_N':tail,'initial':rows[0],'final':rows[-1]}
        save_json(folder/'summary.json',summary);save_json(folder/'status.json',{'phase':summary['phase'],'frame':end,'signature':signature,'steps':int(progress[0])});return summary
    except BaseException:
        save_json(folder/'status.json',{'phase':'failed','frame':len(rows)-1,'signature':signature,'never_resumable':True})
        raise
    finally:
        for a in arrays.values():a.flush();a._mmap.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/single-fibre-tension'));p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-frame',type=int);p.add_argument('--dry-run',action='store_true');a=p.parse_args();geometry,plan=plan_case()
    if a.dry_run:print(json.dumps(plan,indent=2))
    else:run_case(geometry,plan,a.output,resume=a.resume,stop_after_frame=a.stop_after_frame)


if __name__=='__main__':main()
