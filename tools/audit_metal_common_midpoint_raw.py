"""Read-only physical-field/work audit; no step, return solve or factorization."""
import argparse,hashlib,itertools,json,math,os,sys,tarfile
from pathlib import Path
from types import SimpleNamespace
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.sparse import coo_matrix


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def audit(task,expected,oldseed,baseline,output):
    contract=json.loads((task/'contract.json').read_text());report=json.loads((task/'result/result.json').read_text())
    assert report['status']=='bounded-window-passed' and report['dt_s']==2e-12
    for name,pin in contract['files'].items():assert sha(task/name)==pin
    with tarfile.open(task/'source.tar') as t:
        for item in t.getmembers():
            if item.isfile():
                b=t.extractfile(item).read();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                assert blob==expected[item.name],item.name
                assert hashlib.sha256(b).hexdigest()==sha(task/'source'/item.name)
    source_count=len(expected)
    sys.path.insert(0,str(task/'source'))
    from tools.metal_frozen_transfer import support
    from tools.metal_affine_contact_numpy import CompatibleContact
    p=json.loads((task/'inputs/admission.json').read_text())['material'];tol=report['energy_roundoff_J'];dt=report['dt_s'];h=.00025
    load=lambda name:dict(np.load(task/'result'/name,allow_pickle=False))
    seed=dict(np.load(task/'inputs/post-contact.npz',allow_pickle=False));old=dict(np.load(oldseed,allow_pickle=False))
    for key in ('x','v','F','W','tau','plastic_strain','cold_work','fracture_energy','damage','internal_energy','mass','volume','reference_root','D0'):
        assert np.array_equal(seed[key],old[key]),key
    m=seed['mass'];n=len(m);root=seed['reference_root'];iroot=np.linalg.inv(root);fixed=(seed['physical_body']==1)&((abs(seed['reference'][:,0])>=.009)|(abs(seed['reference'][:,2])>=.009))
    assert np.array_equal(fixed,seed['candidate_fixed_particles']) and fixed.sum()==4864
    def white(s):return np.concatenate((s['v'][:,:,None],s['W']@root),axis=2).transpose(0,2,1).reshape(-1,3)*np.repeat(np.sqrt(m),4)[:,None]
    def vw(y):
        a=(y/np.repeat(np.sqrt(m),4)[:,None]).reshape(n,4,3).transpose(0,2,1);return a[:,:,0],a[:,:,1:]@iroot
    def K(y):return .5*float(np.sum(y*y))
    def moment(s,y):
        v,W=vw(y);B=W@s['D0']@s['F'].swapaxes(-1,-2);spin=np.stack((B[:,2,1]-B[:,1,2],B[:,0,2]-B[:,2,0],B[:,1,0]-B[:,0,1]),-1)
        summ=lambda a:np.array([math.fsum(a[:,j]) for j in range(3)])
        return summ(m[:,None]*v),summ(m[:,None]*(np.cross(s['x'],v)+spin))
    def therm(s):
        J=np.linalg.det(s['F']);elastic=np.sum(s['tau']**2,axis=(1,2))/(4*p['shear_modulus']*p['density']*np.maximum(1-s['damage'],1e-12))
        vol=np.where(J>1,1-s['damage'],1)*p['bulk_modulus']*(1-J)**2/(2*p['density'])
        heat=s['internal_energy']-s['cold_work']-s['fracture_energy']-elastic-vol
        return J,p['reference_temperature']+heat/p['heat_capacity'],heat,elastic,vol
    def fields(s):
        assert all(np.isfinite(v).all() for v in s.values())
        for key in ('mass','volume','body_id','physical_body','reference','reference_root','D0'):assert np.array_equal(s[key],seed[key]),key
        J,T,_,_,_=therm(s);assert ((J>=p['minimum_J'])&(J<=p['maximum_J'])).all() and (T>0).all()
        assert (s['damage']>=seed['damage']).all() and (s['damage']<=1).all() and (s['plastic_strain']>=seed['plastic_strain']).all()
        assert (s['cold_work']>=0).all() and (s['fracture_energy']>=0).all()
        scale=max(1.,float(abs(s['tau']).max()));assert abs(s['tau']-s['tau'].swapaxes(-1,-2)).max()<=1e-12*scale and abs(np.trace(s['tau'],axis1=-2,axis2=-1)).max()<=1e-12*scale
        assert np.all(s['v'][fixed]==0) and np.all(s['W'][fixed]==0)
        assert np.array_equal(s['x'][fixed],seed['x'][fixed]) and np.array_equal(s['F'][fixed],seed['F'][fixed])
    def sampler(s):
        w,r,ids,wm,gm=support(s['x'],s['body_id'],m,h);active=gm>0;lookup=np.full(len(gm),-1,int);lookup[active]=np.arange(active.sum());ids=lookup[ids];nodes=int(active.sum())
        rows=np.broadcast_to(np.arange(n)[:,None,None]*4+np.arange(4)[None,None,:],(n,27,4));cols=np.broadcast_to(ids[:,:,None],rows.shape)
        grad=4/h**2*np.einsum('pki,pij->pkj',r,s['F']@root);values=np.sqrt(m)[:,None,None]*w[:,:,None]*np.concatenate((np.ones((*w.shape,1)),grad),axis=2);keep=cols>=0
        Q=coo_matrix((values[keep],(rows[keep],cols[keep])),shape=(4*n,nodes)).tocsr()
        coords=np.zeros((nodes,3));body=np.zeros(nodes,int);good=ids>=0;coords[ids[good]]=(s['x'][:,None]+r)[good];body[ids[good]]=np.broadcast_to(s['body_id'][:,None],ids.shape)[good]
        phi=[]
        for b in np.unique(body):
            mask=body==b;c=coords[mask].mean(0);phi.append(mask.astype(float));phi.extend(mask*(coords[:,j]-c[j])/h for j in range(3))
        T=np.linalg.qr(Q@np.array(phi).T,mode='reduced')[0]
        op=SimpleNamespace(x=s['x'],mass=m,body=s['body_id'],dx=h,F=s['F'],reference_root=root,n=n,Q0=Q,T=T)
        contact=CompatibleContact(op);clamp=np.unique(ids[fixed]);clamp=clamp[clamp>=0]
        assert len(contact.pairs)==114 and not np.isin(contact.pairs,clamp).any()
        return contact,dict(nodes=nodes,clamped_nodes=len(clamp),pairs=len(contact.pairs))
    before=load('coupled-before-remap.npz');final=load('accepted.npz');fields(seed);fields(before);fields(final)
    s0,cov0=sampler(seed);P=np.zeros(3);L=np.zeros(3);channels=dict(grid=0.,boundary=0.,contact=0.,viscosity=0.,dynamic_contact=0.)
    def reaction(s,r):
        nonlocal P,L
        dp,dl=moment(s,r);P+=dp;L+=dl
    def projection(name,s):
        z=load(name);free=z['white']-z['reaction'];g=.5*float(np.sum((free-z['white_before'])**2));b=.5*float(np.sum(z['reaction']**2))
        assert abs(K(z['white'])-K(z['white_before'])+g+b)<=tol
        channels['grid']+=g;channels['boundary']+=b;reaction(s,z['reaction']);return z['white']
    def contact(name,s,op,mu):
        z=load(name);assert np.max(abs(z['white']-z['white_before']-z['physical']-z['reaction']))<1e-12
        assert np.max(abs(z['physical']+op.adjoint(mu)))<1e-12
        closing=op.sample(z['white']);assert np.all(mu>=0) and np.max(np.where(mu>0,abs(closing),np.maximum(closing,0)),initial=0)<=.005
        loss=.5*float(np.sum((z['white']-z['white_before'])**2));assert abs(K(z['white'])-K(z['white_before'])+loss)<=tol
        channels['contact']+=loss;reaction(s,z['reaction']);return z['white']
    yseed=white(seed);yp0=projection('initial-projection.npz',seed);req0=load('iteration-0-request.npz')
    assert np.max(abs(load('initial-projection.npz')['white_before']-yseed))<1e-12
    y0=contact('initial-contact.npz',seed,s0,req0['candidate_contact_multiplier'])
    heat_error=cold_error=fracture_error=0.;last=None
    for i in range(len(report['iterations'])):
        req=load(f'iteration-{i}-request.npz');res=load(f'iteration-{i}-response.npz');f=load(f'iteration-{i}-force.npz');fields(req);fields(res)
        assert bool(res['candidate_material_valid']) and np.max(abs(white(req)-f['midpoint_white']))<1e-12
        assert np.array_equal(req['F'],seed['F']) and np.array_equal(req['internal_energy'],seed['internal_energy'])
        assert np.max(abs(res['F']-seed['F']-dt*req['W']))<64*np.finfo(float).eps and np.max(abs(res['x']-seed['x']-dt*req['v']))<1e-17
        assert np.array_equal(req['W'],res['W']) and np.array_equal(req['v'],res['v'])
        _,_,H0,E0,V0=therm(req);J1,_,H1,E1,V1=therm(res)
        heat_error=max(heat_error,float(np.sum(m*abs(H1-H0-res['candidate_plastic_heat']))))
        cold_error=max(cold_error,float(np.sum(m*abs(res['cold_work']-req['cold_work']-res['candidate_plastic_heat']*(1-p['heat_fraction'])/p['heat_fraction']))))
        oldD=req['damage'];tau_before=res['tau']*((1-oldD)/np.maximum(1-res['damage'],1e-12))[:,None,None]
        E_before=np.sum(tau_before**2,axis=(1,2))/(4*p['shear_modulus']*p['density']*np.maximum(1-oldD,1e-12))
        V_before=np.where(J1>1,1-oldD,1)*p['bulk_modulus']*(1-J1)**2/(2*p['density'])
        fracture_error=max(fracture_error,float(np.sum(m*abs(res['fracture_energy']-req['fracture_energy']-(E_before-E1)-(V_before-V1)))))
        Fm=.5*(req['F']+res['F']);inv=np.linalg.inv(Fm);_,Wm=vw(f['midpoint_white']);Lm=Wm@inv;Jm=np.linalg.det(Fm)
        speed=h*np.maximum(-np.trace(Lm,axis1=-2,axis2=-1),0);q=p['density']/Jm*(p['viscosity_linear']*np.sqrt(p['bulk_modulus']*Jm**2/p['density'])*speed+p['viscosity_quadratic']*speed**2)
        visc=-Jm[:,None,None]*q[:,None,None]*inv.swapaxes(-1,-2);assert np.max(abs(visc-f['viscosity_piola']))<=1e-12*max(1.,abs(visc).max())
        piola=res['candidate_work_piola']+visc;cov=-seed['volume'][:,None,None]/np.sqrt(m)[:,None,None]*piola@iroot.swapaxes(-1,-2)
        raw=np.concatenate((np.zeros((n,3,1)),cov),axis=2).transpose(0,2,1).reshape(-1,3);assert np.max(abs(raw-f['raw_force']))<=1e-12*max(1.,abs(raw).max())
        assert np.max(abs(f['projected_force']-raw-f['reaction_force']))<=1e-12*max(1.,abs(raw).max())
        assert np.max(abs(f['proposed_white']-y0-dt*f['projected_force']-f['physical_contact']-f['reaction_contact']))<1e-12
        assert np.max(abs(f['physical_contact']+s0.adjoint(f['multiplier'])))<1e-12
        closing=s0.sample(f['proposed_white']);assert np.all(f['multiplier']>=0) and np.max(np.where(f['multiplier']>0,abs(closing),np.maximum(closing,0)),initial=0)<=.005
        last=(req,res,f,Fm,visc,Wm)
    assert heat_error<1e-12 and cold_error<1e-12 and fracture_error<1e-12
    req,res,f,Fm,visc,Wm=last;y1=f['proposed_white'];ym=.5*(y0+y1);vm,Wa=vw(ym)
    x_error=float(np.max(abs(before['x']-seed['x']-dt*vm)));F_error=float(np.max(abs(before['F']-seed['F']-dt*Wa)))
    assert x_error<=64*np.finfo(float).eps*max(abs(seed['x']).max(),1e-30) and F_error<=64*np.finfo(float).eps*max(abs(seed['F']).max(),1.)
    assert np.max(abs(white(before)-y1))<1e-12 and np.max(abs(before['candidate_white']-y1))<1e-12
    mid={**seed,'x':.5*(seed['x']+before['x']),'F':Fm};r=dt*f['reaction_force']+f['reaction_contact'];reaction(mid,r)
    work=float(np.sum(seed['volume']*res['candidate_material_work_density']));change=float(np.sum(m*(res['internal_energy']-seed['internal_energy'])))
    viscosity=dt*float(np.sum(seed['volume'][:,None,None]*visc*Wm));cw=float(np.sum(ym*f['physical_contact']));bw=float(np.sum(ym*r))
    defect=K(y1)-K(y0)+work+viscosity-cw-bw;assert abs(defect)<=tol and abs(work-change)<=tol and cw<=tol and abs(bw)<=tol and viscosity>=-tol
    contact_torque=moment(mid,f['physical_contact'])[1];channels['viscosity']=viscosity;channels['dynamic_contact']=-cw
    del s0
    s1,cov1=sampler(before);yp1=projection('end-projection.npz',before);yfinal=contact('end-contact.npz',before,s1,final['candidate_contact_multiplier'])
    assert np.max(abs(white(final)-yfinal))<1e-12 and np.array_equal(final['F'],before['F']) and np.array_equal(final['x'],before['x'])
    for name,value in channels.items():assert abs(value-report['ledger'][name])<1e-14
    p0,l0=moment(seed,yseed);p1,l1=moment(final,yfinal);dp=p1-p0-P;dl=l1-l0-L
    pscale=float(np.sum(m*(np.linalg.norm(req0['v'],axis=1)+np.linalg.norm(final['v'],axis=1))))+np.linalg.norm(P)
    lscale=float(np.sum(m*np.linalg.norm(final['x'],axis=1)*(np.linalg.norm(req0['v'],axis=1)+np.linalg.norm(final['v'],axis=1))))+np.linalg.norm(L)
    assert np.linalg.norm(dp)<=1e-10*max(pscale,1e-30) and np.linalg.norm(dl)<=64*np.finfo(float).eps*max(lscale,1e-30)
    energy=K(yfinal)-K(yseed)+float(np.sum(m*(final['internal_energy']-seed['internal_energy'])))+sum(channels.values());assert abs(energy)<=tol
    assert abs(float(final['numerical_reservoir']-seed['numerical_reservoir'])-channels['grid']-channels['contact']-channels['viscosity']-channels['dynamic_contact'])<1e-14
    assert abs(float(final['boundary_reservoir']-seed['boundary_reservoir'])-channels['boundary'])<1e-14
    expectedC=final['W']@final['D0']@final['F'].swapaxes(-1,-2)*(4/h**2)
    assert np.max(abs(final['C']-expectedC))<=1e-12*max(1.,abs(expectedC).max()) and np.array_equal(final['C'],final['C_transfer'])
    assert np.max(abs(final['L_physical']-final['W']@np.linalg.inv(final['F'])))<=1e-12*max(1.,abs(final['L_physical']).max())
    J,T,_,_,_=therm(final);cs=np.sqrt(p['bulk_modulus']*J*J/p['density']);wave=np.sqrt(cs*cs+4*p['shear_modulus']*np.maximum(J,1)/(3*p['density']))
    affine=1.5*np.sqrt(3)*h*np.linalg.norm(final['C'],axis=(1,2));limits=[]
    for gradient in (final['C'],final['L_physical']):
        compression=np.maximum(-np.trace(gradient,axis1=-2,axis2=-1),0)
        signal=wave+np.linalg.norm(final['v'],axis=1)+affine+p['viscosity_linear']*cs+2*p['viscosity_quadratic']*h*compression
        limits.append(.1*h/float(signal.max()))
    assert dt<=min(limits)
    oldreport=json.loads((baseline/'first2ps-result.json').read_text())
    with np.load(baseline/'first2ps-accepted.npz',allow_pickle=False) as previous:
        old_state_errors={k:float(np.max(abs(final[k]-previous[k]))) for k in ('x','v','W','F','tau','plastic_strain','damage','internal_energy')}
    result=dict(passed=True,scope='One actual common-time-level2ps step only; no dynamics rerun and no long-window admission.',source_files_verified=source_count,input_matches_original_postcontact_state=True,
        material_evaluations_verified=len(report['iterations']),fixed_particles=int(fixed.sum()),geometry_coverage=[cov0,cov1],
        x_midpoint_error_m=x_error,F_midpoint_error=F_error,plastic_only_heat_L1_error_J=heat_error,cold_work_L1_error_J=cold_error,fracture_release_L1_error_J=fracture_error,
        coupled_defect_J=defect,material_work_identity_J=work-change,contact_work_J=cw,contact_torque=contact_torque.tolist(),boundary_work_J=bw,
        independently_derived_channels_J=channels,energy_residual_J=energy,momentum_residual=dp.tolist(),angular_residual=dl.tolist(),angular_tolerance=64*np.finfo(float).eps*max(lscale,1e-30),
        old_step_force_time_remainder_J=oldreport['steps'][0]['force_time_work_remainder_J'],old_step_global_energy_residual_J=oldreport['steps'][0]['global_energy_residual_J'],
        original_CFL_limit_s=min(limits),min_J=float(J.min()),max_temperature=float(T.max()),max_damage=float(final['damage'].max()),old_final_state_differences=old_state_errors,
        global_energy_improvement_factor=abs(oldreport['steps'][0]['global_energy_residual_J'])/max(abs(energy),1e-30),numerical_heat_added_J=0.,physical_steps_recomputed=0,production_admitted=False)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('task','expected','oldseed','baseline','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();audit(a.task,json.loads(a.expected.read_text()),a.oldseed,a.baseline,a.output)
