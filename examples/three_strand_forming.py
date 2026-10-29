"""Actual one-turn end-driven three-fibre formation, initially straight.

Independent material grids, shared-node frictionless unilateral normal contact.
Illustrative elastic parameters, no adhesive or Hertz contact claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

SOURCE=('examples/three_strand_forming.py','mpm_jax/fibre_forming_contact.py','mpm_jax/fibre_forming_boundary.py','mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py')


def geometry(spacing=.0001,length=.024,radius=.001,orbit=.0014):
    values=(np.arange(round(2*radius/spacing))+.5)*spacing-radius
    u,v=np.meshgrid(values,values,indexing='ij')
    cross=np.column_stack((u.ravel(),v.ravel()))
    cross=cross[np.sum(cross**2,axis=1)<radius**2]
    sections=round(length/spacing)
    t=(np.arange(sections)+.5)*length/sections
    blocks=[]
    for f in range(3):
        phase=2*np.pi*f/3
        xyz=np.empty((sections,len(cross),3))
        xyz[:,:,0]=t[:,None]
        xyz[:,:,1:]=cross[None,:,:]+orbit*np.array([np.cos(phase),np.sin(phase)])
        blocks.append(xyz.reshape(-1,3))
    x=np.concatenate(blocks);ids=np.repeat(np.arange(3,dtype=np.int32),len(blocks[0]))
    section=np.tile(np.repeat(np.arange(sections),len(cross)),3)
    volume=np.full(len(x),spacing**2*length/sections)
    masks=(x[:,0]<.0015,x[:,0]>length-.0015)
    return dict(x=x,ids=ids,volume=volume,masks=masks,section=section,sections=sections)


def plan():
    g=geometry();dx=.0002;young=50000.;density=1000.;c=math.sqrt(1.2*young/density)
    duration=.020;steps=math.ceil(duration/(.15*dx/c))
    root=Path(__file__).resolve().parents[1]
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE}
    return g,dict(case='three-strand-forming',particles=len(g['x']),dx=dx,dt=duration/steps,steps=steps,duration=duration,frames=60,
        geometry=dict(length_m=.024,radius_m=.001,initial_orbit_m=.0014,spacing_m=.0001,initial_state='three parallel straight stress-free fibres; F=I'),
        material=dict(young=young,poisson=.25,density=density,tensile_strength=1e5,fracture_energy=100.,characteristic_length=dx),
        protocol=dict(turns=1,rotation_ramp_s=.001,axial_extension_m=.000105,extension_ramp_s=.002,collar_length_m=.0015),
        contact=dict(model='three-field mass-gradient nodal unilateral normal projection',friction=0.,adhesion=0.,tolerance_m_per_s=1e-8,max_sweeps=64,spatial_accuracy='grid-scale, h=0.2mm; no resolved Hertz pressure claim'),
        bounds=[[-.001,.025],[-.0035,.0035],[-.0035,.0035]],source=hashes,
        production_eligible=False,
        resource=dict(state_bytes_estimate=len(g['x'])*400,stencil_bytes_estimate=len(g['x'])*27*36,uss_limit_bytes=5*1024**3),
        colors_rgb=[[.045,.36,.67],[.94,.36,.08],[.32,.66,.48]])


def run(output,stop_steps=64,resume=False):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import (FibreConfig,FibreState,build_layout,make_state,stencil,particle_to_grid,grid_to_particle,stored_energy)
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle,update_brittle,BrittleHistory
    from mpm_jax.fibre_forming_contact import mass_gradient,project
    from mpm_jax.fibre_forming_boundary import prescribe,angle
    g,p=plan();folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    if any(folder.iterdir()) and not resume:raise ValueError('Fresh output must be empty')
    (folder/'plan.json').write_text(json.dumps(p,indent=2))
    material=isotropic_brittle(**p['material']);config=FibreConfig(p['dt'])
    bounds=np.broadcast_to(np.asarray(p['bounds']),(3,3,2)).copy()
    layout=build_layout(g['x'],g['ids'],dx=p['dx'],reference_bounds=bounds,padding_cells=3,max_nodes=1600000)
    if not np.all(np.asarray(layout.shape)==np.asarray(layout.shape)[0]) or not np.all(np.asarray(layout.lower)==np.asarray(layout.lower)[0]):raise ValueError('Contact requires identical global lattice maps')
    nodes=layout.node_count//3
    state=make_state(g['x'],g['volume'],g['ids'],material,layout)
    masks=tuple(jnp.asarray(m) for m in g['masks'])
    rows=[];start=0;totals=np.zeros(5)
    if resume:
        with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
            if str(cp['signature'])!=signature:raise ValueError('Frozen source/plan changed')
            leaves={k:jnp.asarray(cp['state_'+k]) for k in state._fields if k!='history'}
            leaves['history']=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in state.history._fields))
            state=FibreState(**leaves);start=int(cp['step']);totals=cp['totals'].copy()
        rows=json.loads((folder/'observables.json').read_text())
    if not start<stop_steps<=p['steps']:raise ValueError('Stop must advance within one-turn plan')
    @jax.jit
    def step(s):
        support=stencil(s.x,s.fibre_id,layout)
        mass,momentum=particle_to_grid(s,layout,config,support)
        free=momentum/jnp.where(mass>0,mass,1)[:,None]
        prescribed,clamped,bcvalid=prescribe(free,mass,s,support,layout,masks,config.dt,p['duration'])
        grad=mass_gradient(s,support,layout).reshape(3,nodes,3)
        centres=jnp.stack([jnp.sum(jnp.where((s.fibre_id==i)[:,None],s.mass[:,None]*s.x,0),axis=0)/jnp.sum(jnp.where(s.fibre_id==i,s.mass,0)) for i in range(3)])
        contact=project(mass.reshape(3,nodes),prescribed.reshape(3,nodes,3),grad,centres,clamped.reshape(3,nodes))
        velocity=contact.velocity.reshape(-1,3)
        initial_fixture=mass[:,None]*(prescribed-free)
        fixture=initial_fixture+contact.fixture_impulse.reshape(-1,3)
        impulse=jnp.sum(fixture,axis=0)
        # Prescribed support work uses imposed velocity; the initial kinetic
        # projection loss is separately logged, not hidden as contact energy.
        work=jnp.sum(fixture*prescribed)
        projection_loss=.5*jnp.sum(mass*jnp.sum((prescribed-free)**2,axis=1))
        grid_delta=.5*jnp.sum(mass*jnp.sum(velocity**2-free**2,axis=1))
        contact_loss=work-projection_loss-grid_delta
        v,C=grid_to_particle(velocity,support,layout)
        F=(jnp.eye(3)+config.dt*C)@s.F;x=s.x+config.dt*v
        history,tau=update_brittle(s.history,F,material)
        valid=s.valid&support.valid&stencil(x,s.fibre_id,layout).valid&bcvalid&contact.valid&jnp.all(history.valid)&jnp.all(jnp.isfinite(x))&jnp.all(jnp.isfinite(F))&jnp.all(jnp.linalg.det(F)>0)&jnp.all(history.damage==0)
        new=FibreState(x,v,F,C,s.mass,s.volume,s.fibre_id,history,tau,s.time+config.dt,valid,s.boundary_impulse+impulse,s.external_impulse,s.boundary_work+work)
        metrics=jnp.array([work,projection_loss,contact_loss,contact.closing_residual,contact.sweeps])
        return new,metrics
    def save(k):
        x=np.asarray(state.x);F=np.asarray(state.F);v=np.asarray(state.v);C=np.asarray(state.C);mass=np.asarray(state.mass)
        momentum=np.sum(mass[:,None]*v,axis=0)
        expected=g['x'].copy();theta=float(angle(state.time,p['duration']));c,s=np.cos(theta),np.sin(theta)
        Q=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        z=np.clip(float(state.time)/.002,0,1);ext=.000105*z*z*(3-2*z)
        errors=[]
        for side,mask in enumerate(g['masks']):
            target=g['x'][mask]@(Q.T if side else np.eye(3));target[:,0]+=(1 if side else -1)*ext/2
            errors.append(float(np.max(np.linalg.norm(x[mask]-target,axis=1))))
        centres=np.stack([x[g['ids']==i].reshape(g['sections'],-1,3).mean(axis=1) for i in range(3)])
        gaps=[np.linalg.norm(centres[a]-centres[b],axis=1).min()-.002 for a,b in ((0,1),(0,2),(1,2))]
        row=dict(step=k,time_s=float(state.time),right_angle_rad=theta,mass_kg=float(mass.sum()),mass_relative_error=float((mass.sum()-g['volume'].sum()*1000)/(g['volume'].sum()*1000)),momentum_residual_Ns=float(np.linalg.norm(momentum-np.asarray(state.boundary_impulse))),min_J=float(np.linalg.det(F).min()),max_damage=float(np.asarray(state.history.damage).max()),collar_error_m=max(errors),section_centre_gap_proxy_m=min(gaps),gap_scope='same-reference-section circular centre proxy only; not certified surface gap',kinetic_J=float(.5*np.sum(mass*np.sum(v*v,axis=1))+p['dx']**2/8*np.sum(mass*np.sum(C*C,axis=(1,2)))),elastic_J=float(stored_energy(state,material)),fixture_work_J=float(state.boundary_work),fixture_projection_loss_J=float(totals[1]),contact_projection_loss_J=float(totals[2]),max_closing_residual_m_per_s=float(totals[3]),max_contact_sweeps=int(totals[4]),valid=bool(state.valid))
        rows.append(row)
        np.savez_compressed(folder/f'frame-{k:06d}.npz',x=x,fibre_id=g['ids'],section=g['section'])
        leaves={'state_'+key:np.asarray(value) for key,value in state._asdict().items() if key!='history'}
        leaves.update({'history_'+key:np.asarray(value) for key,value in state.history._asdict().items()})
        np.savez_compressed(folder/'checkpoint.tmp.npz',**leaves,step=k,signature=signature,totals=totals)
        (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
        (folder/'observables.json').write_text(json.dumps(rows,indent=2))
        print(json.dumps(row),flush=True)
    if not resume:save(0)
    started=time.monotonic()
    save_steps=set(np.rint(np.linspace(0,p['steps'],61)).astype(int))|{stop_steps}
    for k in range(start+1,stop_steps+1):
        state,metrics=step(state);metrics=np.asarray(metrics)
        totals[:3]+=metrics[:3];totals[3:]=np.maximum(totals[3:],metrics[3:])
        if not bool(state.valid):raise FloatingPointError(f'Invalid forming state at step {k}; last checkpoint preserved')
        if k in save_steps:save(k)
    receipt=dict(accepted_interval_steps=[start,stop_steps],completed_turn=stop_steps==p['steps'],solver_seconds=time.monotonic()-started,grid_nodes=layout.node_count,latest=rows[-1],source=p['source'],production_eligible=False)
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2));return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs/three-strand-forming'));parser.add_argument('--stop-steps',type=int,default=64);parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true');a=parser.parse_args()
    if a.dry_run:print(json.dumps(plan()[1],indent=2))
    else:run(a.output,a.stop_steps,a.resume)

if __name__=='__main__':main()
