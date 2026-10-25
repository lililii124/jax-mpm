"""First-error-stopped real dynamic window from the authenticated 125 ns seed.

Two constrained kicks surround ballistic drift and the verified midpoint
material return. Contact force and stress use the SAME kick midpoint velocity.
Only independently derived zero-time losses and viscosity enter the old
noncalorimetric ledger; no balance remainder is fed into any state variable.
"""
import argparse,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from tools.metal_affine_resource import require_commit_headroom,commit_headroom_bytes
require_commit_headroom()
import numpy as np
from tools.metal_affine_dynamic_numpy import DynamicFrame,kinetic,whiten,unpack,synchronize,moments,thermal,cfl_limit,stress_force


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--epoch',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dt',type=float,required=True);p.add_argument('--steps',type=int,default=1)
    a=p.parse_args()
    if not (0<a.dt<=2e-12 and 1<=a.steps<=4 and a.dt*a.steps<=8e-12):raise ValueError('Extreme short-window bounds required')
    if a.output.exists():raise FileExistsError('Preserve prior raw data')
    a.output.mkdir(parents=True)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(a.state)!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Authenticated seed required')
    admission=json.loads(a.admission.read_text());params=admission['material'];tol=float(admission['energy_roundoff_J'])
    if digest(a.epoch)!=admission['input_sha256']['result/dt1/epoch-initial.npz']:raise ValueError('Authenticated boundary reference required')
    s=dict(np.load(a.state,allow_pickle=False));epoch=dict(np.load(a.epoch,allow_pickle=False))
    if any(not np.array_equal(s[k],epoch[k]) for k in s if k in epoch):raise ValueError('Reference and state differ')
    fixed=(epoch['physical_body']==1)&((abs(epoch['reference'][:,0])>=.009)|(abs(epoch['reference'][:,2])>=.009))
    np.savez_compressed(a.output/'000-original.npz',**s)
    seed={k:v.copy() for k,v in s.items()}
    for name,value in epoch.items():
        if name not in s:s[name]=value.copy()
    s['C_seed']=s['C'].copy();s['reference_root']=np.linalg.inv(s['F'])*.00025/2
    s['D0']=s['reference_root']@s['reference_root'].swapaxes(-1,-2);s['W']=s['C']@s['F']
    y=whiten(s);synchronize(s,y)
    base_numeric=float(seed['numerical_reservoir']);base_boundary=float(seed['boundary_reservoir'])
    base_boundary_loss=float(seed['boundary_projection_loss']);base_impulse=seed['boundary_impulse'].copy()
    ledger=dict(grid_projection=0.,boundary_projection=0.,contact_projection=0.,viscosity=0.,
        boundary_impulse=np.zeros(3),boundary_torque=np.zeros(3),contact_loss_by_body=np.zeros(2),contact_coupling_by_body=np.zeros(2))
    report=dict(schema='material-affine-real-dynamic125-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initial_sha256=digest(a.state),reference_sha256=digest(a.epoch),dt_s=a.dt,steps_requested=a.steps,
        fixed_particles=int(fixed.sum()),material_parameters=params,energy_roundoff_J=tol,
        initialization='W=Cseed Fseed; E0=Fseed^-1 h/2; D0=E0 E0^T; all original histories preserved before instantaneous projection/contact',
        accepted_steps=0,status='running',stages=[],steps=[],production_admitted=False,public_media_allowed=False)
    stage=0;frame=None
    def progress(label):
        print(json.dumps(dict(progress=label,accepted_steps=report['accepted_steps'],commit_headroom_bytes=commit_headroom_bytes())),flush=True)
    def serial(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value))
    def write_report():
        report['ledger']=ledger;report['last_time_s']=float(s['time'])
        (a.output/'result.json').write_text(json.dumps(report,indent=2,default=serial)+'\n')
    def gate(condition,message):
        if not bool(condition):raise ValueError(message)
    def update_ledger_fields():
        # Every increment below is computed from an explicit operator or
        # viscosity power. Neither the total energy error nor material defect
        # is used to update either reservoir, heat, cold work or fracture.
        s['numerical_reservoir']=np.asarray(base_numeric+ledger['grid_projection']+ledger['contact_projection']+ledger['viscosity'])
        s['boundary_projection_loss']=np.asarray(base_boundary_loss+ledger['boundary_projection'])
        s['boundary_reservoir']=np.asarray(base_boundary+ledger['boundary_projection'])
        s['boundary_impulse']=base_impulse+ledger['boundary_impulse']
        s['contact_projection_loss']=seed['contact_projection_loss']+ledger['contact_loss_by_body']
        s['contact_coupling_work']=seed['contact_coupling_work']+ledger['contact_coupling_by_body']
    def energy():return kinetic(y)+float(np.sum(s['mass']*s['internal_energy']))+float(s['numerical_reservoir'])+float(s['boundary_reservoir'])
    initial_energy=energy();initial_momentum,initial_angular=moments(s,y)
    def save(label,extra=None):
        nonlocal stage
        synchronize(s,y);update_ledger_fields()
        data=dict(s);data.update(candidate_white=y,candidate_fixed_particles=fixed,candidate_accepted_steps=np.asarray(report['accepted_steps']))
        if extra:data.update(extra)
        np.savez_compressed(a.output/f'{stage:03}-{label}.npz',**data);stage+=1
        progress(label);write_report()
    def boundary_reaction(delta):
        dp,dl=moments(s,delta);ledger['boundary_impulse']+=dp;ledger['boundary_torque']+=dl
    def project_geometry(label):
        nonlocal y
        before=y.copy();y,row,reaction=frame.project(y)
        ledger['grid_projection']+=row['grid_projection_loss_J'];ledger['boundary_projection']+=row['boundary_projection_loss_J'];boundary_reaction(reaction)
        gate(abs(row['grid_projection_identity_J'])<=tol,'Grid projection energy identity failed')
        gate(abs(row['boundary_projection_identity_J'])<=tol,'Fixed-node projection energy identity failed')
        gate(np.all(unpack(s,y)[0][fixed]==0) and np.all(unpack(s,y)[1][fixed]==0),'Original fixed-particle velocity/gradient not exactly zero')
        report['stages'].append(dict(stage=label,geometry=frame.report,projection=row))
        save(label,dict(candidate_white_before=before,candidate_boundary_reaction=reaction,
            candidate_contact_pairs=frame.contact.pairs,candidate_contact_normals=frame.contact.normal,
            candidate_eliminated_nodes=frame.allowed.clamped_nodes,candidate_node_coordinates=frame.contact.node_coordinates,
            candidate_node_body=frame.contact.node_body,candidate_node_mass=frame.contact.grid_mass))
    def instantaneous_contact(label):
        nonlocal y
        before=y.copy();y,row,reaction,physical=frame.restricted.apply(y)
        delta=y-before;loss=float(.5*np.sum(delta*delta));ledger['contact_projection']+=loss;boundary_reaction(reaction)
        body=s['body_id'];perparticle=.5*np.sum(delta.reshape(len(body),4,3)**2,axis=(1,2))
        oldK=.5*np.sum(before.reshape(len(body),4,3)**2,axis=(1,2));newK=.5*np.sum(y.reshape(len(body),4,3)**2,axis=(1,2))
        for b in (0,1):
            part=float(np.sum(perparticle[body==b]));ledger['contact_loss_by_body'][b]+=part
            ledger['contact_coupling_by_body'][b]+=-float(np.sum((newK-oldK)[body==b]))-part
        gate(row['valid'],'Original contact tolerance failed')
        gate(row['original_sample_closing_m_s']<=.005 and row['restricted_sample_difference_m_s']<=.005,'Original full contact sampler tolerance failed')
        gate(abs(kinetic(y)-kinetic(before)+loss)<=tol,'Instantaneous contact power identity failed')
        gate(abs(row['boundary_virtual_work_J'])<=tol,'Contact boundary reaction has nonzero allowed work')
        report['stages'].append(dict(stage=label,contact=row,derived_projection_loss_J=loss))
        save(label,dict(candidate_white_before=before,candidate_contact_physical_impulse=physical,candidate_boundary_reaction=reaction))
    def coupled_kick(label,halfdt):
        nonlocal y
        synchronize(s,y);before=y.copy();raw,tau,viscous,q=stress_force(s,y,params)
        force,reaction_force,force_rows=frame.force(raw)
        temporary=y+halfdt*force;y,row,reaction_contact,physical_contact=frame.restricted.apply(temporary)
        reaction=halfdt*reaction_force+reaction_contact;boundary_reaction(reaction)
        midpoint=.5*(before+y);_,Wmid=unpack(s,midpoint);Lmid=Wmid@np.linalg.inv(s['F'])
        work=float(halfdt*np.sum(s['volume'][:,None,None]*tau*Lmid))
        viscosity=float(halfdt*np.sum(s['volume'][:,None,None]*viscous*Lmid));ledger['viscosity']+=viscosity
        contact_work=float(np.sum(midpoint*physical_contact));boundary_work=float(np.sum(midpoint*reaction))
        identity=kinetic(y)-kinetic(before)+work-contact_work-boundary_work
        gate(row['valid'],'Original contact tolerance failed after force')
        gate(row['original_sample_closing_m_s']<=.005 and row['restricted_sample_difference_m_s']<=.005,'Original full contact sampler tolerance failed after force')
        gate(abs(identity)<=tol,'Coupled physical kick work failed')
        gate(abs(boundary_work)<=tol,'Fixed-node reaction has nonzero admissible work')
        gate(contact_work<=tol,'Contact force integration creates excessive positive work')
        gate(viscosity>=-tol,'Original nonnegative viscosity work failed')
        data=dict(stage=label,stress_work_J=work,viscosity_work_J=viscosity,contact_force_work_J=contact_work,
            boundary_force_work_J=boundary_work,kick_work_identity_J=identity,contact=row,force_projection=force_rows,
            projection_variance_diagnostic_J=float(.5*np.sum((y-temporary)**2)))
        report['stages'].append(data)
        save(label,dict(candidate_white_before=before,candidate_raw_force=raw,candidate_projected_force=force,
            candidate_stress=tau,candidate_viscous_stress=viscous,candidate_viscosity=q,
            candidate_contact_physical_impulse=physical_contact,candidate_boundary_reaction=reaction))
        validate_state()
        return data
    def validate_state():
        synchronize(s,y);J,T,entropy=thermal(s,params)
        gate(all(np.all(np.isfinite(v)) for v in s.values()),'Nonfinite candidate state')
        gate(np.array_equal(s['mass'],seed['mass']) and np.array_equal(s['volume'],seed['volume']),'Mass/reference volume changed')
        gate(np.all(s['mass']>0) and np.allclose(s['mass'],params['density']*s['volume'],rtol=32*np.finfo(float).eps,atol=0),'Original reference mass-density gate failed')
        gate(np.all((J>=params['minimum_J'])&(J<=params['maximum_J'])) and np.all(T>0),'Original EOS material domain failed')
        gate(np.all((s['damage']>=seed['damage'])&(s['damage']<=1)) and np.all(s['plastic_strain']>=seed['plastic_strain']),'Original irreversible material history failed')
        gate(np.all(s['cold_work']>=0) and np.all(s['fracture_energy']>=0),'Original cold/fracture energy positivity failed')
        scale=max(1.,float(np.max(abs(s['tau']))))
        gate(np.max(abs(s['tau']-s['tau'].swapaxes(-1,-2)))<=1e-12*scale and np.max(abs(np.trace(s['tau'],axis1=-2,axis2=-1)))<=1e-12*scale,'Original symmetric deviatoric stress gate failed')
        gate(np.all(s['v'][fixed]==0) and np.all(s['W'][fixed]==0),'Original fixed-particle condition failed')
        gate(a.dt<=cfl_limit(s,params)*(1+1e-12),'Original/physical CFL gate failed')
        return dict(min_J=float(J.min()),max_J=float(J.max()),max_temperature=float(T.max()),minimum_cfl_limit_s=cfl_limit(s,params),max_damage=float(s['damage'].max()))
    try:
        save('mapped-initial')
        frame=DynamicFrame(s,fixed,progress);project_geometry('initial-space-boundary');instantaneous_contact('initial-impact-contact')
        gate(abs(energy()-initial_energy)<=tol,'Initial projection/contact energy ledger failed')
        for step in range(a.steps):
            require_commit_headroom();start_energy=energy();start_time=float(s['time']);start_state_J=np.linalg.det(s['F']);start_material=float(np.sum(s['mass']*s['internal_energy']))
            validate_state();kick0=coupled_kick(f'step-{step}-half-kick',a.dt/2)
            synchronize(s,y);s['candidate_requested_dt_s']=np.asarray(a.dt);s['candidate_previous_entropy']=thermal(s,params)[2]
            request=a.output/f'step-{step}-material-request.npz';np.savez_compressed(request,**s)
            drift_x=s['x'].copy();drift_F=s['F'].copy();drift_v=s['v'].copy();drift_W=s['W'].copy()
            frame.release();frame=None;gc.collect();progress('material-worker-start')
            response=a.output/f'step-{step}-material-response.npz'
            command=[sys.executable,'-m','tools.metal_affine_material_step_worker','--request',str(request.resolve()),'--admission',str(a.admission.resolve()),'--output',str(response.resolve())]
            completed=subprocess.run(command,capture_output=True,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
            (a.output/f'step-{step}-material-worker.log').write_text(completed.stdout+completed.stderr)
            gate(completed.returncode==0,'Material worker rejected stage; raw request/response/log preserved')
            s=dict(np.load(response,allow_pickle=False));y=whiten(s)
            gate(np.max(abs(s['F']-drift_F-a.dt*drift_W))<=64*np.finfo(float).eps*max(float(np.max(abs(drift_F))),1.),'Physical Fdot=W drift failed')
            gate(np.max(abs(s['x']-drift_x-a.dt*drift_v))<=64*np.finfo(float).eps*max(float(np.max(abs(drift_x))),1e-30),'Physical particle drift failed')
            gate(np.array_equal(s['W'],drift_W),'Force-free material-stage W changed')
            gate(bool(s['candidate_material_valid']),'Original material return gate failed')
            material_work=float(np.sum(s['volume']*s['candidate_material_work_density']))
            material_change=float(np.sum(s['mass']*s['internal_energy']))-start_material
            save(f'step-{step}-drift-material')
            frame=DynamicFrame(s,fixed,progress);project_geometry(f'step-{step}-new-space-boundary');instantaneous_contact(f'step-{step}-geometric-contact')
            kick1=coupled_kick(f'step-{step}-final-half-kick',a.dt/2)
            diagnostics=validate_state();end_energy=energy();residual=end_energy-start_energy;global_residual=end_energy-initial_energy
            momentum,angular=moments(s,y);momentum_residual=momentum-initial_momentum-ledger['boundary_impulse']
            angular_residual=angular-initial_angular-ledger['boundary_torque']
            momentum_scale=float(np.sum(s['mass']*(np.linalg.norm(seed['v'],axis=1)+np.linalg.norm(s['v'],axis=1)))+np.linalg.norm(ledger['boundary_impulse']))
            row=dict(step=step+1,start_time_s=start_time,time_s=float(s['time']),energy_residual_J=residual,global_energy_residual_J=global_residual,
                material_work_J=material_work,material_internal_change_J=material_change,material_work_remainder_J=material_work-material_change,
                force_time_work_remainder_J=kick0['stress_work_J']+kick1['stress_work_J']-kick0['viscosity_work_J']-kick1['viscosity_work_J']-material_work,
                contact_force_time_work_J=kick0['contact_force_work_J']+kick1['contact_force_work_J'],
                momentum_residual=momentum_residual.tolist(),angular_residual=angular_residual.tolist(),diagnostics=diagnostics)
            report['steps'].append(row);write_report()
            gate(abs(residual)<=tol and abs(global_residual)<=tol,'Original full energy accounting tolerance failed')
            gate(np.linalg.norm(momentum_residual)<=1e-10*max(momentum_scale,1e-30),'Original momentum gate failed')
            angular_scale=float(np.sum(s['mass']*(np.linalg.norm(s['x'],axis=1)*(np.linalg.norm(s['v'],axis=1)+np.linalg.norm(seed['v'],axis=1))))+np.linalg.norm(ledger['boundary_torque']))
            gate(np.linalg.norm(angular_residual)<=64*np.finfo(float).eps*max(angular_scale,1e-30),'Angular momentum with boundary torque failed')
            report['accepted_steps']+=1;save(f'step-{step}-accepted')
            if step+1<a.steps:frame.release();frame=None;frame=DynamicFrame(s,fixed,progress)
        report['status']='bounded-window-passed'
    except Exception as error:
        report['status']='rejected';report['first_error']=dict(type=type(error).__name__,message=str(error),stage=stage)
        np.savez_compressed(a.output/'first-rejected-state.npz',**s,candidate_white=y)
    finally:
        if frame is not None:frame.release()
        write_report();print(json.dumps(dict(status=report['status'],accepted_steps=report['accepted_steps'],first_error=report.get('first_error'))),flush=True)
    if report['status']!='bounded-window-passed':raise SystemExit(2)


if __name__=='__main__':main()
