"""Run one forced3D manufactured cycle through the production MLS-MPM step."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config,State,step
from tools.check_elastic_force_refinement import MODELS,GRIDS,DENSITY,CHUNK,field,parameters,reference


PERIOD=.04
DIRECTION=np.array([1.,.6,-.4])
LIMITS=dict(finest_position_velocity_F_relative=.05,momentum_relative=1e-10,
            angular_momentum_relative=1e-10,forcing_reference_relative=5e-10)


def reference_data(X,name):
    material,C,Q,amplitude=parameters(name)
    parts={key:[] for key in ('u0','g','Ha','trH','aHa','c','linear','quadratic')}
    minimum=1.;forcing_error=0.
    for start in range(0,len(X),CHUNK):
        points=X[start:start+CHUNK];x,F,dF=field(points,amplitude)
        g=F[:,0,:]-np.eye(3)[0];H=dF[:,0,:,:];Ha=H@DIRECTION
        values=dict(u0=x-points,g=g,Ha=Ha,trH=np.trace(H,axis1=1,axis2=2),aHa=Ha@DIRECTION,c=g@DIRECTION)
        first=reference(points,name,C,Q,amplitude)[4]
        if name=='blatz-ko':
            values['linear']=np.zeros_like(points);values['quadratic']=np.zeros_like(points)
            power=1-1/(1-2*.463);J=1+values['c']
            closed=8e3/DENSITY*(DIRECTION*values['trH'][:,None]+(1-power)*(J**(power-1))[:,None]*Ha
                +(power-1)*(J**(power-2)*values['aHa'])[:,None]*g)
        else:
            second=reference(points,name,C,Q,2*amplitude)[4]
            values['quadratic']=.5*(second-2*first);values['linear']=2*first-.5*second
            q=.37;closed=q*values['linear']+q*q*values['quadratic']
            first=reference(points,name,C,Q,q*amplitude)[4]
        forcing_error=max(forcing_error,float(np.linalg.norm(closed-first)/max(np.linalg.norm(first),1e-30)))
        for key in parts:parts[key].append(values[key])
        minimum=min(minimum,float(np.linalg.det(field(points,2*amplitude)[1]).min()))
    data={key:jnp.asarray(np.concatenate(value)) for key,value in parts.items()}
    return material,data,dict(amplitude_m=amplitude,minimum_exact_J=minimum,independent_forcing_relative_error=forcing_error)


def body_acceleration(t,data,name):
    omega=2*jnp.pi/PERIOD;q=1-jnp.cos(omega*t)
    if name=='blatz-ko':
        power=1-1/(1-2*.463);J=1+q*data['c']
        internal=8e3/DENSITY*(q*data['trH'][:,None]*jnp.asarray(DIRECTION)
            +(1-power)*q*(J**(power-1))[:,None]*data['Ha']
            +(power-1)*q*q*(J**(power-2)*data['aHa'])[:,None]*data['g'])
    else:internal=q*data['linear']+q*q*data['quadratic']
    return omega*omega*jnp.cos(omega*t)*data['u0']-internal


def momenta(s,dx):
    linear=jnp.sum(s.mass[:,None]*s.v,axis=0)
    curl=jnp.stack([s.C[:,2,1]-s.C[:,1,2],s.C[:,0,2]-s.C[:,2,0],s.C[:,1,0]-s.C[:,0,1]],axis=1)
    angular=jnp.sum(jnp.cross(s.x,s.mass[:,None]*s.v)+s.mass[:,None]*(dx*dx/4)*curl,axis=0)
    return linear,angular


@partial(jax.jit,static_argnames=('config','count'))
def advance(s,impulse,angular_impulse,data,material,*,config,count):
    def body(carry,_):
        before,I,L=carry
        force=before.mass[:,None]*body_acceleration(before.time,data,config.material)
        I=I+config.dt*jnp.sum(force,axis=0)
        L=L+config.dt*jnp.sum(jnp.cross(before.x,force),axis=0)
        following=step(before,material,config,force)
        P,A=momenta(following,config.dx)
        row=dict(time_s=following.time,valid=following.valid,min_J=jnp.min(jnp.linalg.det(following.F)),
                 momentum_error=jnp.linalg.norm(P-I),angular_momentum_error=jnp.linalg.norm(A-L))
        return (following,I,L),row
    return jax.lax.scan(body,(s,impulse,angular_impulse),None,length=count)


@jax.jit
def errors(s,X,data):
    omega=2*jnp.pi/PERIOD;q=1-jnp.cos(omega*s.time)
    exact_x=X+q*data['u0'];exact_v=omega*jnp.sin(omega*s.time)*data['u0']
    exact_F=jnp.eye(3)+q*jnp.asarray(DIRECTION)[None,:,None]*data['g'][:,None,:]
    amplitude=jnp.maximum(jnp.linalg.norm(data['u0']),1e-30)
    deformation=jnp.maximum(jnp.linalg.norm(jnp.asarray(DIRECTION)[None,:,None]*data['g'][:,None,:]),1e-30)
    return dict(position=jnp.linalg.norm(s.x-exact_x)/(2*amplitude),
                velocity=jnp.linalg.norm(s.v-exact_v)/(omega*amplitude),
                F=jnp.linalg.norm(s.F-exact_F)/(2*deformation))


def save_json(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)+'\n';temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def save_checkpoint(path,s,I,L,current):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,step=current,impulse=np.asarray(I),angular_impulse=np.asarray(L),
            **{key:np.asarray(value) for key,value in s._asdict().items() if key!='history'})
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--grid',type=int,choices=GRIDS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after-steps',type=int);p.add_argument('--resume-from',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    n=3*a.grid//2;spacing=1/(2*a.grid);axes=.125+(np.arange(n)+.5)*spacing
    X=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
    material,data,reference_review=reference_data(X,a.model)
    if reference_review['independent_forcing_relative_error']>LIMITS['forcing_reference_relative']:raise ValueError('Independent forcing check failed')
    planned=5*a.grid;dt=PERIOD/planned
    config=Config(n_grid=a.grid,dim=3,dt=dt,gravity=(0,0,0),boundary_cells=0,material=a.model)
    count=len(X);mass=jnp.full(count,DENSITY*spacing**3);volume=mass/DENSITY
    state=State(jnp.asarray(X),jnp.zeros((count,3)),jnp.broadcast_to(jnp.eye(3),(count,3,3)),jnp.zeros((count,3,3)),mass,volume,None,jnp.array(0.),jnp.array(True))
    I=jnp.zeros(3);L=jnp.zeros(3);current=0;history=[]
    source=Path(__file__).resolve().parents[1]
    contract=dict(model=a.model,grid=a.grid,particles=count,planned_steps=planned,dt_s=dt,period_s=PERIOD,
        density_kg_m3=DENSITY,reference_review=reference_review,tolerances=LIMITS,
        source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
            ['tools/check_elastic_dynamic_mms.py','tools/check_elastic_force_refinement.py','mpm_jax/core.py','mpm_jax/materials/elastic.py']},
        boundary='Traction-free outer material faces; exact displacement/stress zero in exterior layer',
        forcing='Analytical reference body acceleration evaluated at fixed material coordinates; external force=m*b(X,t)',
        scope='Production3D explicit MLS-MPM manufactured dynamics, full prescribed cycle; no contact/fracture or general structural certification')
    if a.resume_from:
        if json.loads((a.resume_from/'contract.json').read_text())!=contract:raise ValueError('Resume contract mismatch')
        with np.load(a.resume_from/'accepted.npz') as cp:
            state=State(**{key:jnp.asarray(cp[key]) for key in State._fields if key!='history'},history=None)
            I=jnp.asarray(cp['impulse']);L=jnp.asarray(cp['angular_impulse']);current=int(cp['step'])
        history=json.loads((a.resume_from/'review.json').read_text())['history']
        if not state.valid or len(history)==0 or history[-1]['step']!=current:raise ValueError('Parent state is incomplete')
    save_json(a.output/'contract.json',contract)
    target=min(planned,a.stop_after_steps or planned)
    if not current<target<=planned:raise ValueError('Require an advancing bounded window')
    scale=float(mass.sum())*(2*np.pi/PERIOD)*float(jnp.max(jnp.linalg.norm(data['u0'],axis=1)))
    while current<target:
        number=min(20,target-current)
        (state,I,L),rows=advance(state,I,L,data,material,config=config,count=number)
        jax.block_until_ready(state);current+=number
        raw=jax.tree.map(np.asarray,rows);error={key:float(value) for key,value in errors(state,jnp.asarray(X),data).items()}
        checks=dict(valid=bool(raw['valid'].all()),positive_J=bool(raw['min_J'].min()>0),finite=all(np.isfinite(v).all() for v in jax.tree.leaves(state)),
            linear_momentum=float(raw['momentum_error'].max())/scale<=LIMITS['momentum_relative'],
            angular_momentum=float(raw['angular_momentum_error'].max())/scale<=LIMITS['angular_momentum_relative'])
        row=dict(step=current,time_s=float(state.time),elapsed_s=time.monotonic()-begin,errors=error,checks=checks,
            min_J=float(raw['min_J'].min()),momentum_error_relative=float(raw['momentum_error'].max())/scale,
            angular_momentum_error_relative=float(raw['angular_momentum_error'].max())/scale)
        history.append(row)
        maximum={key:max(item['errors'][key] for item in history) for key in error}
        report=dict(contract=contract,steps=current,full_cycle_complete=current==planned,history=history,maximum_errors=maximum,
            physical_checks_passed=all(all(item['checks'].values()) for item in history),
            finest_accuracy_target_met=all(value<=LIMITS['finest_position_velocity_F_relative'] for value in maximum.values()),
            exact_solution_peak_normalization=True,source_bound=True)
        save_checkpoint(a.output/'candidate.npz',state,I,L,current);save_json(a.output/'review.json',report)
        print(json.dumps(row),flush=True)
        if not all(checks.values()):raise ValueError('Original production-state or balance check failed')
        save_checkpoint(a.output/'accepted.npz',state,I,L,current)


if __name__=='__main__':main()
