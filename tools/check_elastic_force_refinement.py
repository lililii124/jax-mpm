"""Compare the production 3D MLS force transfer with an independent smooth field."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config, State, particle_to_grid, stencil, grid_to_particle
from mpm_jax.materials.elastic import orthotropic, transverse_isotropic, blatz_ko, kirchhoff_stress


MODELS = ('orthotropic', 'transverse-isotropic', 'blatz-ko')
GRIDS = (32, 48, 64)
DENSITY = 1000.
CHUNK = 32768
TOLERANCES = dict(finest_acceleration_L2=.05, stress_relative=5e-12,
                  analytic_divergence_FD_relative=1e-6, mass_relative=1e-12,
                  force_and_torque_relative=1e-10)


def field(X, amplitude):
    q=(X-.25)*2*np.pi;inside=(X>.25)&(X<.75)
    s,c=np.sin(q),np.cos(q);k=2*np.pi
    b=np.where(inside,s**4,0.)
    db=np.where(inside,4*k*s**3*c,0.)
    ddb=np.where(inside,4*k*k*(3*s*s*c*c-s**4),0.)
    f=np.prod(b,axis=-1);g=np.empty_like(X);H=np.empty(X.shape[:-1]+(3,3))
    for i in range(3):
        other=[j for j in range(3) if j!=i]
        g[...,i]=db[...,i]*b[...,other[0]]*b[...,other[1]]
        H[...,i,i]=ddb[...,i]*b[...,other[0]]*b[...,other[1]]
        for j in other:
            left=3-i-j
            H[...,i,j]=db[...,i]*db[...,j]*b[...,left]
    direction=np.array([1.,.6,-.4])
    x=X+amplitude*f[...,None]*direction
    F=np.eye(3)+amplitude*direction[:,None]*g[...,None,:]
    dF=amplitude*direction[:,None,None]*H[...,None,:,:]
    return x,F,dF


def parameters(name):
    angle=np.pi/6;c,s=np.cos(angle),np.sin(angle)
    if name=='blatz-ko':return blatz_ko(8e3),None,None,.002
    if name=='orthotropic':
        E=np.array([8e4,2e4,3e4]);nu=np.array([.2,.15,.25]);G=np.array([1e4,1.5e4,8e3])
        Q=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        material=orthotropic(E,nu,G,Q)
    else:
        El,Et,nult,nutt,Glt=8e4,2e4,.25,.3,1.2e4
        E=np.array([Et,Et,El]);nu=np.array([nutt,nult*Et/El,nult*Et/El]);G=np.array([Et/(2*(1+nutt)),Glt,Glt])
        Q=np.array([[c,0,s],[0,1.,0],[-s,0,c]])
        material=transverse_isotropic(El,Et,nult,nutt,Glt,Q)
    compliance=np.diag(np.r_[1/E,1/G[[2,1,0]]])
    for i,j,k in [(0,1,0),(0,2,1),(1,2,2)]:compliance[i,j]=compliance[j,i]=-nu[k]/E[i]
    return material,np.linalg.inv(compliance),Q,2e-5


def linear_response(strain, stiffness, Q):
    e=Q.T@strain@Q
    voigt=np.stack([e[...,0,0],e[...,1,1],e[...,2,2],2*e[...,1,2],2*e[...,0,2],2*e[...,0,1]],axis=-1)
    v=voigt@stiffness.T;stress=np.empty_like(strain)
    for i,j,k in [(0,0,0),(1,1,1),(2,2,2),(1,2,3),(0,2,4),(0,1,5)]:stress[...,i,j]=stress[...,j,i]=v[...,k]
    return Q@stress@Q.T


def reference(X, name, stiffness, Q, amplitude):
    x,F,dF=field(X,amplitude);J=np.linalg.det(F);A=np.linalg.inv(F);AT=A.swapaxes(-1,-2)
    if name=='blatz-ko':
        mu=8e3;power=1-1/(1-2*.463);jq=J**power
        P=mu*(F-jq[...,None,None]*AT)
        tau=mu*(F@F.swapaxes(-1,-2)-jq[...,None,None]*np.eye(3))
    else:
        sigma=linear_response(.5*(F+F.swapaxes(-1,-2))-np.eye(3),stiffness,Q)
        tau=J[...,None,None]*sigma;P=tau@AT
    divergence=np.zeros_like(X)
    for axis in range(3):
        d=dF[...,axis];tr=np.einsum('...ij,...ji->...',A,d)
        dAT=-AT@d.swapaxes(-1,-2)@AT
        if name=='blatz-ko':derivative=mu*(d-power*(jq*tr)[...,None,None]*AT-jq[...,None,None]*dAT)
        else:
            ds=linear_response(.5*(d+d.swapaxes(-1,-2)),stiffness,Q)
            derivative=(J*tr)[...,None,None]*sigma@AT+J[...,None,None]*(ds@AT+sigma@dAT)
        divergence+=derivative[..., :,axis]
    return x,F,tau,P,divergence/DENSITY


@partial(jax.jit,static_argnames=('config',))
def transfer(x,F,volume,material,*,config):
    count=len(x);mass=jnp.full(count,DENSITY*volume)
    state=State(x,jnp.zeros_like(x),F,jnp.zeros_like(F),mass,jnp.full(count,volume),None,jnp.array(0.),jnp.array(True))
    support=stencil(x,config)
    gm,gp=particle_to_grid(state,material,config,support)
    return gm,gp/config.dt,kirchhoff_stress(F,material,config.material),support.valid


@partial(jax.jit,static_argnames=('config',))
def sample(x,acceleration,*,config):
    return grid_to_particle(acceleration,stencil(x,config),config)[0]


def check_case(name,grid):
    begin=time.monotonic();material,C,Q,amplitude=parameters(name)
    spacing=1/(2*grid);n=3*grid//2
    axes=.125+(np.arange(n)+.5)*spacing
    X=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
    volume=spacing**3;config=Config(n_grid=grid,dim=3,dt=1e-6,gravity=(0,0,0),boundary_cells=0,material=name)
    size=(grid+1)**3;gm=jnp.zeros(size);gf=jnp.zeros((size,3));error2=reference2=0.;stress_error=stress_scale=0.;minJ=np.inf
    chunks=[];support_valid=True
    for start in range(0,len(X),CHUNK):
        reference_x=X[start:start+CHUNK]
        x,F,tau,P,exact=reference(reference_x,name,C,Q,amplitude)
        mass,force,actual_tau,valid=transfer(jnp.asarray(x),jnp.asarray(F),volume,material,config=config)
        gm+=mass;gf+=force;actual_tau=np.asarray(actual_tau)
        stress_error=max(stress_error,float(np.max(abs(actual_tau-tau))));stress_scale=max(stress_scale,float(np.max(abs(tau))))
        minJ=min(minJ,float(np.linalg.det(F).min()));support_valid &= bool(valid)
        chunks.append((start,len(x)))
    ga=gf/jnp.where(gm>0,gm,1.)[:,None]
    for start,count in chunks:
        x,F,tau,P,exact=reference(X[start:start+count],name,C,Q,amplitude)
        actual=np.asarray(sample(jnp.asarray(x),ga,config=config))
        error2+=float(np.sum((actual-exact)**2));reference2+=float(np.sum(exact**2))
    mass_total=float(gm.sum());force=np.asarray(gf);peak=float(np.max(np.linalg.norm(np.asarray(ga),axis=1)))
    node=np.arange(size);xyz=np.stack([node//(grid+1)**2,(node//(grid+1))%(grid+1),node%(grid+1)],axis=1)/grid
    force_scale=max(mass_total*peak,1e-30)
    linear=float(np.linalg.norm(force.sum(0)))/force_scale
    angular=float(np.linalg.norm(np.cross(xyz,force).sum(0)))/force_scale
    mass_error=abs(mass_total-DENSITY*.75**3)/(DENSITY*.75**3)
    # A separate centred finite difference checks the analytical Piola divergence.
    probes=np.random.default_rng(193).uniform(.31,.69,(24,3));eps=1e-6;fd=np.zeros_like(probes)
    for axis in range(3):
        shift=np.eye(3)[axis]*eps
        plus=reference(probes+shift,name,C,Q,amplitude)[3];minus=reference(probes-shift,name,C,Q,amplitude)[3]
        fd+=(plus-minus)[...,axis]/(2*eps*DENSITY)
    analytic=reference(probes,name,C,Q,amplitude)[4]
    fd_error=float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))
    result=dict(model=name,grid=grid,particles=len(X),spacing_m=spacing,amplitude_m=amplitude,minimum_J=minJ,
        acceleration_relative_L2=float(np.sqrt(error2/reference2)),stress_relative_max=stress_error/max(stress_scale,1e-30),
        independent_divergence_FD_relative=fd_error,mass_relative_error=mass_error,net_force_relative=linear,net_torque_relative=angular,
        full_support=support_valid,elapsed_s=time.monotonic()-begin)
    result['checks']=dict(finite=all(np.isfinite(v) for v in result.values() if isinstance(v,(int,float))),positive_J=minJ>0,
        full_support=support_valid,stress=result['stress_relative_max']<=TOLERANCES['stress_relative'],
        analytic_reference=fd_error<=TOLERANCES['analytic_divergence_FD_relative'],mass=mass_error<=TOLERANCES['mass_relative'],
        linear_momentum=linear<=TOLERANCES['force_and_torque_relative'],angular_momentum=angular<=TOLERANCES['force_and_torque_relative'])
    jax.clear_caches()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));p.add_argument('--grids',nargs='+',type=int,default=list(GRIDS));args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['tools/check_elastic_force_refinement.py','mpm_jax/core.py','mpm_jax/materials/elastic.py']}
    report=dict(scope='Frozen3D production MLS force operator refinement; not a dynamic time-integration or nonlinear structural convergence certificate',
        reference_domain_m=[[.125,.875]]*3,displacement_support_m=[[.25,.75]]*3,
        boundary='Displacement and stress vanish in an exterior layer; traction-free outer material faces; no imposed nodal reactions',
        field='u=A sin(pi*(X-.25)/.5)^4 sin(pi*(Y-.25)/.5)^4 sin(pi*(Z-.25)/.5)^4 (1,.6,-.4), zero outside support',
        reference='Independent NumPy compliance, Piola stress and analytical material divergence; centred finite-difference cross-check',
        tolerances=TOLERANCES,source_sha256=sources,cases=[],refinement={})
    def save():(args.output/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    for name in args.models:
        rows=[]
        for grid in args.grids:
            if grid not in GRIDS:raise ValueError('Use the declared32/48/64 grids')
            row=check_case(name,grid);rows.append(row);report['cases'].append(row);save();print(json.dumps(row),flush=True)
            if not all(row['checks'].values()):raise ValueError('Source/reference/operator invariant failed')
        errors=[v['acceleration_relative_L2'] for v in rows]
        check=dict(decreasing=all(a>b for a,b in zip(errors,errors[1:])),finest_within5percent=errors[-1]<=TOLERANCES['finest_acceleration_L2'])
        report['refinement'][name]=dict(errors=errors,checks=check)
        save()
    report['passed']=all(all(v['checks'].values()) for v in report['cases']) and all(all(v['checks'].values()) for v in report['refinement'].values())
    save()
    if not report['passed']:raise ValueError('Declared force refinement target not met')


if __name__=='__main__':main()
