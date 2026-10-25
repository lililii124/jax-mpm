"""Bounded original-fibre trajectory with a per-step numerical energy split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from examples.single_fibre_tension import plan_case,motion,save_json,save_npz
from mpm_jax.fibre_mpm import (FibreConfig,FibreState,build_layout,make_state,reference_grips,
    stencil,particle_to_grid,step_with_diagnostics,stored_energy,validate_state)
from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle,brittle_energy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-from',type=Path)
    a=p.parse_args()
    if not 100<=a.steps<=19600 or a.steps%100:p.error('Use100-step blocks within the original physical plan')
    a.output.mkdir(exist_ok=False,parents=True)
    geometry,plan=plan_case();material=isotropic_brittle(**plan['material'])
    layout=build_layout(geometry['x'],geometry['fibre_id'],dx=plan['grid']['dx_m'],
        padding_cells=6,reference_bounds=plan['grid']['bounds_m'],max_nodes=500000)
    config=FibreConfig(plan['dt']);state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout)
    def speed(t):return (motion(t,plan['protocol'],jnp)-motion(t-config.dt,plan['protocol'],jnp))/config.dt
    boundary=reference_grips(geometry['masks'],(
        lambda t:jnp.array([-speed(t)/2,0.,0.]),lambda t:jnp.array([speed(t)/2,0.,0.])))
    def kinetic(s):return .5*jnp.sum(s.mass*jnp.sum(s.v*s.v,axis=1))+layout.dx**2/8*jnp.sum(s.mass[:,None,None]*s.C*s.C)
    def observed(s):
        support=stencil(s.x,s.fibre_id,layout)
        mass,momentum=particle_to_grid(s,layout,config,support)
        _,transport=particle_to_grid(s._replace(tau=jnp.zeros_like(s.tau)),layout,config,support)
        inverse=1/jnp.where(mass>0,mass,1)
        v0=transport*inverse[:,None];trial=momentum*inverse[:,None]
        bc=boundary(trial,mass,s,support,layout,s.time+config.dt)
        following,diagnostic=step_with_diagnostics(s,material,layout,config,boundary=boundary)
        grid_ke=lambda v:.5*jnp.sum(mass[:,None]*v*v)
        K0,K1=kinetic(s),kinetic(following)
        U0,U1=stored_energy(s,material),stored_energy(following,material)
        Ufixed=jnp.sum(s.volume*brittle_energy(following.F,s.history,material))
        targets=jnp.array([[-1.,0.,0.],[1.,0.,0.]])*speed(following.time)/2
        work=jnp.sum(diagnostic.group_impulses*targets)
        projection=.5*jnp.sum(mass[:,None]*(bc.velocity-trial)**2)
        p2g=K0-grid_ke(v0);g2p=grid_ke(bc.velocity)-K1
        splitting=-(grid_ke(trial)-grid_ke(v0)+Ufixed-U0)
        damage=Ufixed-U1;residual=work-(K1-K0)-(U1-U0)-damage
        parts=dict(fixture_work_J=work,p2g_loss_J=p2g,g2p_loss_J=g2p,
            boundary_projection_J=projection,force_elastic_defect_J=splitting,
            damage_loss_J=damage,residual_J=residual,
            split_identity_error_J=residual-p2g-g2p-projection-splitting,
            boundary_work_identity_error_J=work-grid_ke(bc.velocity)+grid_ke(trial)-projection,
            kinetic_J=K1,elastic_J=U1,min_J=jnp.min(jnp.linalg.det(following.F)),
            max_damage=jnp.max(following.history.damage),valid=following.valid)
        return following,parts
    chunks=[];start_step=0;resume_proof=None
    if a.resume_from is not None:
        parent=a.resume_from
        old_contract=json.loads((parent/'contract.json').read_text())
        if old_contract['plan']!=plan:raise ValueError('Resume physical plan or core source differs')
        with np.load(parent/'checkpoint.npz',allow_pickle=False) as cp:
            start_step=int(cp['step'])
            history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields))
            state=FibreState(**{k:jnp.asarray(cp[k]) for k in FibreState._fields if k!='history'},history=history)
        validate_state(state,layout)
        if abs(float(state.time)-start_step*config.dt)>1e-14:raise ValueError('Resume step/clock differs')
        if not np.array_equal(state.mass,geometry['mass']) or not np.array_equal(state.volume,geometry['volume']):raise ValueError('Resume reference mass/volume differs')
        with np.load(parent/'step_records.npz',allow_pickle=False) as old:
            previous={k:old[k] for k in old.files}
        if not all(len(v)==start_step and np.isfinite(v).all() for v in previous.values()) or not previous['valid'].all():raise ValueError('Incomplete or invalid previous history')
        chunks.append(previous)
        resume_proof=dict(parent_step=start_step,parent_checkpoint_sha256=hashlib.sha256((parent/'checkpoint.npz').read_bytes()).hexdigest(),
            parent_contract_sha256=hashlib.sha256((parent/'contract.json').read_bytes()).hexdigest(),
            core_material_geometry_loading_sources_identical=True,prior_steps_replayed=0,
            bridge='Observer runner adds restore only; original numerical step and physical plan unchanged')
    if not start_step<a.steps<=min(plan['steps'],start_step+2000):p.error('Advance at most2000 steps within original plan')
    compiled=jax.jit(lambda s:jax.lax.scan(lambda carry,_:observed(carry),s,None,length=100)).lower(state).compile()
    # One independent direct-step control establishes that telemetry preserves carry.
    direct,_=step_with_diagnostics(state,material,layout,config,boundary=boundary) if start_step==0 else (state,None)
    instrumented,_=jax.jit(observed)(state) if start_step==0 else (state,None)
    velocity_scale=plan['wave_speed_m_per_s'];mass_scale=float(geometry['mass'].sum())
    scales=dict(x=plan['geometry']['length_m'],v=velocity_scale,F=1.,C=velocity_scale/layout.dx,
        mass=float(geometry['mass'].max()),volume=float(geometry['volume'].max()),fibre_id=1.,
        tau=material.young,time=config.dt,valid=1.,boundary_impulse=mass_scale*velocity_scale,
        external_impulse=mass_scale*velocity_scale,boundary_work=material.young*float(geometry['volume'].sum()))
    def fields(s):
        values={k:v for k,v in s._asdict().items() if k!='history'}
        values.update({'history_'+k:v for k,v in s.history._asdict().items()})
        return values
    parity={} if start_step==0 else json.loads((a.resume_from/'first-step-control.json').read_text())
    for key,x in fields(direct).items():
        if start_step:break
        x,y=np.asarray(x),np.asarray(fields(instrumented)[key])
        exact=bool(np.array_equal(x,y));difference=float(np.max(np.abs(x.astype(float)-y.astype(float)),initial=0))
        scaled=difference/scales.get(key,1.)
        parity[key]=dict(bitwise_equal=exact,max_absolute_difference=difference,engineering_scaled_difference=scaled,
            passed=exact if x.dtype.kind in 'biu' else scaled<=1e-12)
    save_json(a.output/'first-step-control.json',parity)
    if not all(value['passed'] for value in parity.values()):raise ValueError('Telemetry changed original first-step carry beyond engineering-scaled1e-12')
    save_json(a.output/'contract.json',dict(plan=plan,steps=a.steps,source_sha256={
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        (*plan['source']['files'],'tools/observe_single_fibre_energy.py')},
        first_step_control=parity,resume=resume_proof,scope='Numerical error diagnosis only; no relaxation of original15% acceptance'))
    start=time.monotonic()
    for end in range(start_step+100,a.steps+1,100):
        state,rows=compiled(state);jax.block_until_ready(state)
        host=jax.tree.map(np.asarray,rows)
        if not host['valid'].all() or any(not np.isfinite(v).all() for v in host.values()):raise ValueError('Original state invalid or nonfinite telemetry')
        chunks.append(host);validate_state(state,layout)
        raw={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'}
        raw.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
        save_npz(a.output/'checkpoint.npz',step=end,**raw)
        values={key:np.concatenate([c[key] for c in chunks]) for key in host}
        save_npz(a.output/'step_records.npz',**values)
        additive=('fixture_work_J','p2g_loss_J','g2p_loss_J','boundary_projection_J','force_elastic_defect_J','damage_loss_J','residual_J')
        totals={key:float(values[key].sum()) for key in additive}
        stretches=np.linalg.svd(np.asarray(state.F),compute_uv=False);strain=stretches-1
        mu=material.young/(2*(1+material.poisson));lam=material.young*material.poisson/((1+material.poisson)*(1-2*material.poisson))
        independent_U=float(np.sum(np.asarray(state.volume)*(1-np.asarray(state.history.damage))*(.5*lam*strain.sum(1)**2+mu*np.sum(strain*strain,axis=1))))
        scale=max(abs(totals['fixture_work_J']),float(host['kinetic_J'][-1]+host['elastic_J'][-1]),1e-30)
        identity=float(np.max(abs(values['split_identity_error_J'])))
        if identity>1e-10*scale or abs(independent_U-float(host['elastic_J'][-1]))>1e-9*scale:raise ValueError('Independent energy or telescoping identity failed')
        review=dict(step=end,time_s=float(state.time),elapsed_s=time.monotonic()-start,totals_J=totals,
            independent_numpy_SVD_elastic_J=independent_U,kinetic_J=float(host['kinetic_J'][-1]),
            max_identity_error_J=identity,min_J=float(values['min_J'].min()),max_damage=float(values['max_damage'].max()),
            residual_over_fixture_work=totals['residual_J']/max(abs(totals['fixture_work_J']),1e-30),
            scope='Actual original-plan prefix, not full fracture admission; no numerical loss assigned to physical dissipation')
        save_json(a.output/'review.json',review)
        print(json.dumps({k:review[k] for k in ('step','time_s','elapsed_s','max_damage','residual_over_fixture_work')}),flush=True)


if __name__=='__main__':main()
