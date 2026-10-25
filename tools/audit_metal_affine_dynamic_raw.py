"""Independent NumPy review of complete real dynamic stage checkpoints."""
import argparse,hashlib,itertools,json,math,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True)
    p.add_argument('--capsule',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior audit')
    manifest=json.loads((a.case/'whole-raw.json').read_text())
    if manifest['source_capsule_manifest_sha256']!=sha(a.capsule/'manifest.json'):raise ValueError('Different frozen capsule')
    for name,pin in manifest['files'].items():
        path=a.case/name
        if path.stat().st_size!=pin['bytes'] or sha(path)!=pin['sha256']:raise ValueError(name)
    folder=a.case/'result';r=json.loads((folder/'result.json').read_text());resource=json.loads((a.case/'resource.json').read_text())
    if r['status']!='bounded-window-passed' or resource['status']!='passed':raise ValueError('No complete accepted window')
    load=lambda p:dict(np.load(p,allow_pickle=False));seed=load(a.capsule/'inputs/initial.npz');epoch=load(a.capsule/'inputs/epoch-initial.npz')
    mapped=load(folder/'000-mapped-initial.npz');m=seed['mass'];n=len(m);params=r['material_parameters'];tol=r['energy_roundoff_J'];dt=r['dt_s'];h=.00025
    fixed=(epoch['physical_body']==1)&((abs(epoch['reference'][:,0])>=.009)|(abs(epoch['reference'][:,2])>=.009))
    def thermal(s):
        J=np.linalg.det(s['F']);el=np.sum(s['tau']**2,axis=(-2,-1))/(4*params['shear_modulus']*params['density']*np.maximum(1-s['damage'],1e-12))
        vol=np.where(J>1,1-s['damage'],1)*params['bulk_modulus']*(1-J)**2/(2*params['density'])
        heat=s['internal_energy']-s['cold_work']-s['fracture_energy']-el-vol
        return J,params['reference_temperature']+heat/params['heat_capacity'],heat
    def vw(s,y):
        v=(y/np.repeat(np.sqrt(m),4)[:,None]).reshape(n,4,3).transpose(0,2,1)
        return v[:,:,0],v[:,:,1:]@np.linalg.inv(s['reference_root'])
    def moment(s,y):
        v,W=vw(s,y);B=W@s['D0']@s['F'].swapaxes(-1,-2)
        spin=np.stack((B[:,2,1]-B[:,1,2],B[:,0,2]-B[:,2,0],B[:,1,0]-B[:,0,1]),-1)
        accurate=lambda x:np.array([math.fsum(x[:,j]) for j in range(3)])
        return accurate(m[:,None]*v),accurate(m[:,None]*(np.cross(s['x'],v)+spin))
    def kinetic(s):return float(.5*np.sum(m[:,None]*s['v']**2)+.5*np.sum(m[:,None,None]*(s['W']@s['reference_root'])**2))
    def energy(s):return kinetic(s)+float(np.sum(m*s['internal_energy']))+float(s['numerical_reservoir'])+float(s['boundary_reservoir'])
    channels=dict(grid_projection=0.,boundary_projection=0.,contact_projection=0.,viscosity=0.)
    boundaryP=np.zeros(3);boundaryL=np.zeros(3);coverage=[];max_power=max_contact=max_kkt=max_boundary=0.;checkpoints=0
    for path in sorted(folder.glob('[0-9][0-9][0-9]-*.npz')):
        if path.name=='000-original.npz':continue
        s=load(path);checkpoints+=1
        assert all(np.all(np.isfinite(v)) for v in s.values())
        assert np.array_equal(s['mass'],m) and np.array_equal(s['volume'],seed['volume'])
        assert np.array_equal(s['reference_root'],mapped['reference_root']) and np.array_equal(s['D0'],mapped['D0'])
        J,T,_=thermal(s);assert np.all((J>=params['minimum_J'])&(J<=params['maximum_J'])) and np.all(T>0)
        assert np.all((s['damage']>=seed['damage'])&(s['damage']<=1)) and np.all(s['plastic_strain']>=seed['plastic_strain'])
        stress_scale=max(1.,float(np.max(abs(s['tau']))))
        assert np.max(abs(s['tau']-s['tau'].swapaxes(-1,-2)))<=1e-12*stress_scale
        assert np.max(abs(np.trace(s['tau'],axis1=-2,axis2=-1)))<=1e-12*stress_scale
        assert np.all(s['cold_work']>=0) and np.all(s['fracture_energy']>=0)
        if 'mapped' not in path.name:
            assert np.all(s['v'][fixed]==0) and np.all(s['W'][fixed]==0)
            assert np.array_equal(s['x'][fixed],seed['x'][fixed]) and np.array_equal(s['F'][fixed],seed['F'][fixed])
        if 'candidate_boundary_reaction' in s:
            dp,dl=moment(s,s['candidate_boundary_reaction']);boundaryP+=dp;boundaryL+=dl
        if 'space-boundary' in path.name:
            after=s['candidate_white'];before=s['candidate_white_before'];free=after-s['candidate_boundary_reaction']
            channels['grid_projection']+=float(.5*np.sum((free-before)**2));channels['boundary_projection']+=float(.5*np.sum((after-free)**2))
            offset=np.array(list(itertools.product(range(3),repeat=3)));base=np.floor(s['x']/h-.5).astype(int);node=base[:,None,:]+offset;f=s['x']/h-base
            w1=np.stack((.5*(1.5-f)**2,.75-(f-1)**2,.5*(f-.5)**2),axis=1)
            w=np.prod(np.stack([w1[:,offset[:,c],c] for c in range(3)],-1),-1)
            keys=np.concatenate((np.broadcast_to(s['body_id'][:,None,None],(*node.shape[:2],1)),node),-1)
            unique,inv=np.unique(keys.reshape(-1,4),axis=0,return_inverse=True);inv=inv.reshape(n,27)
            massgrid=np.bincount(inv.ravel(),weights=(m[:,None]*w).ravel());active=massgrid>0;lookup=np.full(len(unique),-1,int);lookup[active]=np.arange(active.sum())
            assert np.array_equal(unique[active,1:],s['candidate_node_coordinates']) and np.array_equal(unique[active,0],s['candidate_node_body'])
            clamp=np.unique(lookup[inv][fixed]);clamp=clamp[clamp>=0];assert np.array_equal(clamp,s['candidate_eliminated_nodes'])
            table={tuple(unique[i,1:]):lookup[i] for i in np.flatnonzero(active&(unique[:,0]==1))}
            pairs=np.array([(lookup[i],table[tuple(unique[i,1:])]) for i in np.flatnonzero(active&(unique[:,0]==0)) if tuple(unique[i,1:]) in table])
            assert np.array_equal(pairs,s['candidate_contact_pairs'])
            coverage.append(dict(stage=path.name,nodes=int(active.sum()),clamped_nodes=len(clamp),pairs=len(pairs)))
        if 'impact-contact' in path.name or 'geometric-contact' in path.name:channels['contact_projection']+=float(.5*np.sum((s['candidate_white']-s['candidate_white_before'])**2))
        if 'candidate_contact_final_full_sample' in s:
            mu=s['candidate_contact_multiplier'];g=s['candidate_contact_final_full_sample'];assert np.all(mu>=0)
            closure=float(np.max(g,initial=0));kkt=max(closure,float(np.max(abs(g[mu>0]),initial=0)))
            assert kkt<=.005 and float(s['maximum_contact_closing_speed'])>=closure
            assert float(s['maximum_contact_complementarity_residual'])+1e-14>=kkt
            max_contact=max(max_contact,closure);max_kkt=max(max_kkt,kkt)
        if 'half-kick' in path.name:
            before=s['candidate_white_before'];after=s['candidate_white'];mid=.5*(before+after);_,W=vw(s,mid);L=W@np.linalg.inv(s['F'])
            work=float(dt/2*np.sum(s['volume'][:,None,None]*s['candidate_stress']*L));visc=float(dt/2*np.sum(s['volume'][:,None,None]*s['candidate_viscous_stress']*L))
            cp=float(np.sum(mid*s['candidate_contact_physical_impulse']));bp=float(np.sum(mid*s['candidate_boundary_reaction']))
            residual=float(.5*np.sum(after**2)-.5*np.sum(before**2)+work-cp-bp)
            assert abs(residual)<=tol and abs(bp)<=tol and cp<=tol and visc>=-tol
            channels['viscosity']+=visc;max_power=max(max_power,abs(residual));max_boundary=max(max_boundary,abs(bp))
    heat_error=cold_error=F_error=x_error=0.
    for step in range(r['accepted_steps']):
        before=load(folder/f'step-{step}-material-request.npz');after=load(folder/f'step-{step}-material-response.npz')
        _,_,h0=thermal(before);_,_,h1=thermal(after)
        heat_error=max(heat_error,float(np.sum(m*abs(h1-h0-after['candidate_plastic_heat']))))
        cold_error=max(cold_error,float(np.sum(m*abs(after['cold_work']-before['cold_work']-after['candidate_plastic_heat']*(1-params['heat_fraction'])/params['heat_fraction']))))
        F_error=max(F_error,float(np.max(abs(after['F']-before['F']-dt*before['W']))));x_error=max(x_error,float(np.max(abs(after['x']-before['x']-dt*before['v']))))
        assert np.array_equal(after['W'],before['W']) and bool(after['candidate_material_valid'])
        assert np.all(after['plastic_strain']>=before['plastic_strain']) and np.all(after['damage']>=before['damage'])
    assert heat_error<1e-12 and cold_error<1e-12 and F_error<64*np.finfo(float).eps and x_error<1e-17
    for name,value in channels.items():assert abs(value-r['ledger'][name])<1e-14
    assert np.max(abs(boundaryP-r['ledger']['boundary_impulse']))<1e-20 and np.max(abs(boundaryL-r['ledger']['boundary_torque']))<1e-23
    final=load(sorted(folder.glob('*-accepted.npz'))[-1]);global_error=energy(final)-energy(mapped)
    p0,l0=moment(mapped,mapped['candidate_white']);p1,l1=moment(final,final['candidate_white'])
    momentum_residual=p1-p0-boundaryP;angular_residual=l1-l0-boundaryL
    momentum_scale=float(np.sum(m*(np.linalg.norm(seed['v'],axis=1)+np.linalg.norm(final['v'],axis=1))))+np.linalg.norm(boundaryP)
    angular_scale=float(np.sum(m*np.linalg.norm(final['x'],axis=1)*(np.linalg.norm(seed['v'],axis=1)+np.linalg.norm(final['v'],axis=1))))+np.linalg.norm(boundaryL)
    assert np.linalg.norm(momentum_residual)<=1e-10*max(momentum_scale,1e-30)
    assert np.linalg.norm(angular_residual)<=64*np.finfo(float).eps*max(angular_scale,1e-30)
    assert abs(global_error)<=tol
    assert abs(float(final['numerical_reservoir']-seed['numerical_reservoir'])-channels['grid_projection']-channels['contact_projection']-channels['viscosity'])<1e-14
    assert abs(float(final['boundary_reservoir']-seed['boundary_reservoir'])-channels['boundary_projection'])<1e-14
    result=dict(passed=True,dt_s=dt,steps=r['accepted_steps'],time_ns=float(final['time'])*1e9,verified_output_files=len(manifest['files']),verified_stage_checkpoints=checkpoints,
        coverage=coverage,fixed_particles=int(fixed.sum()),fixed_positions_F_preserved=True,fixed_v_W_exact_zero=True,
        maximum_contact_closing_m_s=max_contact,maximum_complementarity_m_s=max_kkt,maximum_joint_power_identity_J=max_power,maximum_boundary_virtual_work_J=max_boundary,
        plastic_heat_weighted_L1_error_J=heat_error,cold_work_weighted_L1_error_J=cold_error,F_drift_residual=F_error,x_drift_residual_m=x_error,
        independently_derived_channels_J=channels,global_energy_residual_J=global_error,global_momentum_residual=momentum_residual.tolist(),global_angular_residual=angular_residual.tolist(),
        new_replays_required=False,numerical_heat_added_J=0.,production_admitted=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
