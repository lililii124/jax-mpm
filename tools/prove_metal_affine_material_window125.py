"""Original-material prescribed-path window, not an accepted impact advance.

After the separately audited instantaneous contact, prescribe ballistic affine
motion for 0.04 ns. This isolates material/time work from force/contact errors.
The actual original return map and every material parameter are unchanged.
"""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
from tools.metal_affine_resource import require_commit_headroom
require_commit_headroom()
import numpy as np
from functools import partial
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as j
from mpm_jax.materials.linear_metal import linear_metal,deviatoric_return
from mpm_jax.fragment_multifield_v6_mpm import elastic_energy,damage_volume_energy,ImpactConfig
from mpm_jax.fragment_v6_midpoint_strength import midpoint_return


@partial(jax.jit,static_argnames=('midpoint_strength',))
def update(F,tau,ep,cold,fracture,damage,internal,W,dt,material,midpoint_strength=False):
    J=j.linalg.det(F);newF=F+dt*W;newJ=j.linalg.det(newF)
    midpoint=.5*(F+newF);L=W@j.linalg.inv(midpoint)
    skew=.5*(L-j.swapaxes(L,-1,-2));eye=j.eye(3)
    rotation=j.linalg.solve(eye-.5*dt*skew,eye+.5*dt*skew)
    strain=.5*dt*(L+j.swapaxes(L,-1,-2))
    heat=internal-cold-fracture-elastic_energy(tau,material,damage)-damage_volume_energy(J,damage,material)
    temperature=material.reference_temperature+heat/material.heat_capacity
    factor=1-damage
    return_function=midpoint_return if midpoint_strength else deviatoric_return
    returned=return_function(tau/j.maximum(factor,1e-12)[:,None,None],ep,strain,rotation,
        j.where(factor>0,newJ,1.),dt,temperature,material)
    nexttau=returned.kirchhoff_stress*factor[:,None,None]
    nextep=j.where(factor>0,returned.plastic_strain,ep)
    nextcold=cold+factor*returned.specific_cold_work
    plasticheat=factor*returned.specific_plastic_heat
    config=ImpactConfig()
    nextdamage=j.maximum(damage,j.clip((nextep-config.damage_onset_strain)/(config.damage_failure_strain-config.damage_onset_strain),0.,1.))
    before=elastic_energy(nexttau,material,damage)
    nexttau=nexttau*((1-nextdamage)/j.maximum(1-damage,1e-12))[:,None,None]
    elastic_release=j.maximum(before-elastic_energy(nexttau,material,nextdamage),0.)
    volume_release=j.maximum(damage_volume_energy(newJ,damage,material)-damage_volume_energy(newJ,nextdamage,material),0.)
    release=elastic_release+volume_release
    nextfracture=fracture+release
    nextinternal=damage_volume_energy(newJ,nextdamage,material)+heat+plasticheat+nextcold+nextfracture+elastic_energy(nexttau,material,nextdamage)
    p0=material.bulk_modulus*(1-J);p0=j.where(p0<0,(1-damage)*p0,p0)
    p1=material.bulk_modulus*(1-newJ);p1=j.where(p1<0,(1-nextdamage)*p1,p1)
    total0=tau-J[:,None,None]*p0[:,None,None]*eye
    total1=nexttau-newJ[:,None,None]*p1[:,None,None]*eye
    work_density=dt*j.sum(.5*(total0+total1)*L,axis=(-2,-1))
    volume_work=-.5*dt*(J*p0+newJ*p1)*j.trace(L,axis1=-2,axis2=-1)
    if midpoint_strength:
        work_density=volume_work+dt*j.sum(factor[:,None,None]*returned.midpoint_stress*L,axis=(-2,-1))
    volume_increment=damage_volume_energy(newJ,nextdamage,material)-damage_volume_energy(J,damage,material)+volume_release
    deviatoric_increment=elastic_energy(nexttau,material,nextdamage)-elastic_energy(tau,material,damage)+elastic_release+plasticheat+(nextcold-cold)
    BE=factor*(returned.specific_return_dissipation-returned.specific_plastic_work)
    valid=j.all(returned.valid|(factor==0))&j.all(j.isfinite(nextinternal))&j.all(nextdamage>=damage)&j.all(nextep>=ep)
    return (newF,nexttau,nextep,nextcold,nextfracture,nextdamage,nextinternal),work_density,plasticheat,BE,valid,(volume_work,volume_increment,deviatoric_increment)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--contact',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--midpoint-strength',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve raw evidence')
    a.output.mkdir(parents=True)
    sha=hashlib.sha256(a.state.read_bytes()).hexdigest()
    if sha!='3abe31b2fde3f8d93ec63b138e0d868011aef53ca18e5f3162c0b145b0dc2817':raise ValueError('Original state required')
    s=dict(np.load(a.state,allow_pickle=False));m=s['mass'];F0=s['F'];h=.00025
    white=np.load(a.contact,allow_pickle=False)['weighted_contact']
    values=(white/np.repeat(np.sqrt(m),4)[:,None]).reshape(len(m),4,3).transpose(0,2,1)
    velocity=values[:,:,0];W=values[:,:,1:]@F0*(2/h)
    admission=json.loads(a.admission.read_text());material=linear_metal(**admission['material'])
    report=dict(kind='prescribed kinematics; no force solve or accepted impact state',duration_s=4e-11,
        strength_integrator='candidate midpoint flow, original endpoint yield' if a.midpoint_strength else 'original backward Euler return',
        material_parameters=admission['material'],initial_state_sha256=sha,contact_npz_sha256=hashlib.sha256(a.contact.read_bytes()).hexdigest(),
        initialized_state_mapping='E0=F0^-1 h/2; D0=E0 E0^T; W0=C0 F0 before projection/contact; F0 and all material histories unchanged by both instantaneous operations',
        numerical_heat_added_J=0.,production_admitted=False,levels=[])
    for steps in (2,4,8):
        dt=report['duration_s']/steps
        current=tuple(j.asarray(s[k]) for k in ('F','tau','plastic_strain','cold_work','fracture_energy','damage','internal_energy'))
        work=heat=BE=bulkwork=bulkchange=devchange=0.;rows=[]
        for step in range(steps):
            following,density,plastic,be,valid,parts=update(*current,j.asarray(W),j.asarray(dt),material,midpoint_strength=a.midpoint_strength)
            following=tuple(np.asarray(x) for x in following)
            if not bool(valid):raise ValueError(f'Original material gate failed level {steps} step {step}')
            work+=float(np.sum(s['volume']*np.asarray(density)));heat+=float(np.sum(m*np.asarray(plastic)));BE+=float(np.sum(m*np.asarray(be)))
            bulkwork+=float(np.sum(s['volume']*np.asarray(parts[0])))
            bulkchange+=float(np.sum(m*np.asarray(parts[1])))
            devchange+=float(np.sum(m*np.asarray(parts[2])))
            change=float(np.sum(m*(following[-1]-s['internal_energy'])))
            rows.append(dict(step=step+1,time_s=(step+1)*dt,work_J=work,internal_change_J=change,remainder_J=work-change,plastic_heat_J=heat,backward_euler_return_dissipation_J=BE,
                volume_work_J=bulkwork,volume_internal_increment_J=bulkchange,volume_remainder_J=bulkwork-bulkchange,
                deviatoric_work_J=work-bulkwork,deviatoric_internal_increment_J=devchange,deviatoric_remainder_J=work-bulkwork-devchange))
            raw=dict(zip(('F','tau','plastic_strain','cold_work','fracture_energy','damage','internal_energy'),following))
            raw.update(stress_work_density_J_m3=np.asarray(density),plastic_heat_J_kg=np.asarray(plastic),
                return_extra_dissipation_J_kg=np.asarray(be),volume_work_density_J_m3=np.asarray(parts[0]),
                volume_internal_increment_J_kg=np.asarray(parts[1]),deviatoric_internal_increment_J_kg=np.asarray(parts[2]),
                valid=np.asarray(valid),dt_s=np.asarray(dt),time_s=np.asarray((step+1)*dt))
            np.savez_compressed(a.output/f'prescribed-{steps}-step-{step+1}.npz',**raw)
            current=following
        final=dict(s)
        for k,x in zip(('F','tau','plastic_strain','cold_work','fracture_energy','damage','internal_energy'),current):final[k]=x
        root=np.linalg.inv(F0)*h/2;D0=root@root.swapaxes(-1,-2)
        transfer=W@D0@current[0].swapaxes(-1,-2)*(4/h**2)
        final.update(x=s['x']+report['duration_s']*velocity,v=velocity,W=W,reference_root=root,D0=D0,
            C_seed=s['C'],C=transfer,C_transfer=transfer,
            L_physical=W@np.linalg.inv(current[0]),time=s['time']+report['duration_s'],
            prescribed_path=np.asarray(True),accepted_impact_state=np.asarray(False))
        np.savez_compressed(a.output/f'prescribed-{steps}-final.npz',**final)
        report['levels'].append(dict(steps=steps,dt_s=dt,trace=rows,maximum_damage=float(np.max(current[5])),minimum_J=float(np.min(np.linalg.det(current[0])))))
    report['source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        (Path(__file__),Path('mpm_jax/materials/shock_metal.py'),Path('mpm_jax/materials/linear_metal.py'),Path('mpm_jax/fragment_multifield_v6_mpm.py'),Path('mpm_jax/fragment_v6_midpoint_strength.py'))}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
