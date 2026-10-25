"""Compare reference-grid MPM with the existing3D manufactured solution."""
import argparse,hashlib,json,time
from functools import partial
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from mpm_jax.core import Config
from mpm_jax.materials.elastic import linear_stress
from mpm_jax.total_lagrangian import ReferenceGrid,TLState,reference_grid,initial_state,step,sample_particles
from tools.check_elastic_dynamic_mms import reference_data,body_acceleration,errors,PERIOD,LIMITS,save_json
from tools.check_elastic_force_refinement import MODELS,GRIDS,DENSITY


@partial(jax.jit,static_argnames=('config','count'))
def advance(state,reference,data,material,*,config,count):
    def body(s,_):
        return step(s,reference,material,config,body_acceleration(s.time,data,config.material),body_acceleration(s.time+config.dt,data,config.material))
    return jax.lax.scan(body,state,None,length=count)


def stored_energy(F,volume,material,kind):
    if kind=='blatz-ko':
        power=1-1/(1-2*material.poisson)
        W=.5*material.shear*(jnp.sum(F*F,axis=(-2,-1))-3)-material.shear/power*(jnp.linalg.det(F)**power-1)
    else:
        strain=.5*(F+jnp.swapaxes(F,-1,-2))-jnp.eye(3)
        W=.5*jnp.sum(strain*linear_stress(strain,material),axis=(-2,-1))
    return float(jnp.sum(volume*W))


def save_checkpoint(path,state,current):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,step=current,**{k:np.asarray(v) for k,v in state._asdict().items() if k!='history'})
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--grid',type=int,choices=GRIDS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--time-refinement',type=int,choices=[1,2],default=1);p.add_argument('--stop-after-steps',type=int);p.add_argument('--resume-from',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    n=3*a.grid//2;spacing=1/(2*a.grid);axes=.125+(np.arange(n)+.5)*spacing
    X=np.stack(np.meshgrid(axes,axes,axes,indexing='ij'),axis=-1).reshape(-1,3)
    material,data,reference_review=reference_data(X,a.model)
    if reference_review['independent_forcing_relative_error']>LIMITS['forcing_reference_relative']:raise ValueError('Independent forcing reference failed')
    planned=5*a.grid*a.time_refinement;dt=PERIOD/planned
    cfg=Config(n_grid=a.grid,dim=3,dt=dt,gravity=(0,0,0),boundary_cells=0,material=a.model)
    ref=reference_grid(X,jnp.full(len(X),spacing**3),DENSITY,cfg)
    if not bool(ref.valid):raise ValueError('Reference basis/quadrature failed')
    state=initial_state(ref,material,cfg,body_acceleration(0.,data,a.model));current=0;history=[]
    root=Path(__file__).resolve().parents[1]
    contract=dict(model=a.model,grid=a.grid,particles=len(X),planned_steps=planned,dt_s=dt,period_s=PERIOD,time_refinement=a.time_refinement,
        reference_review=reference_review,tolerances=LIMITS,backend='Total-Lagrangian quadratic MPM; reference quadrature, lumped nodal inertia, velocity-Verlet',
        mass_metric='Nodal lumped kinetic energy/angular momentum, not particle APIC affine inertia',
        boundary='Traction-free outer material faces, compact exact deformation; material-coordinate reference grid',
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
         ['tools/check_elastic_tl_mms.py','tools/check_elastic_dynamic_mms.py','tools/check_elastic_force_refinement.py','mpm_jax/total_lagrangian.py','mpm_jax/core.py','mpm_jax/materials/elastic.py']},
        scope='New spatial semidiscretization of the same continuum forcing/material data; not an APIC trajectory-equivalence claim')
    if a.resume_from:
        if json.loads((a.resume_from/'contract.json').read_text())!=contract:raise ValueError('Resume contract mismatch')
        with np.load(a.resume_from/'accepted.npz') as cp:
            state=TLState(**{key:jnp.asarray(cp[key]) for key in TLState._fields if key!='history'},history=None);current=int(cp['step'])
        history=json.loads((a.resume_from/'review.json').read_text())['history']
        if not bool(state.valid) or not history or history[-1]['step']!=current:raise ValueError('Invalid parent state')
    save_json(a.output/'contract.json',contract)
    target=min(planned,a.stop_after_steps or planned)
    if not current<target<=planned:raise ValueError('Require an advancing target')
    scale=float(ref.particle_mass.sum())*(2*np.pi/PERIOD)*float(jnp.max(jnp.linalg.norm(data['u0'],axis=1)))
    while current<target:
        count=min(20,target-current);state,rows=advance(state,ref,data,material,config=cfg,count=count)
        jax.block_until_ready(state);current+=count;raw=jax.tree.map(np.asarray,rows)
        error={k:float(v) for k,v in errors(sample_particles(state,ref),jnp.asarray(X),data).items()}
        checks=dict(valid=bool(raw['valid'].all()),positive_J=bool(raw['min_J'].min()>0),finite=all(np.isfinite(v).all() for v in jax.tree.leaves(state)),
            linear_momentum=float(raw['momentum_error'].max())/scale<=LIMITS['momentum_relative'],
            angular_momentum=float(raw['angular_momentum_error'].max())/scale<=LIMITS['angular_momentum_relative'])
        kinetic=float(raw['kinetic_J'][-1]);U=stored_energy(state.F,ref.volume,material,a.model)
        work=float(state.body_work+state.boundary_work)
        row=dict(step=current,time_s=float(state.time),elapsed_s=time.monotonic()-begin,errors=error,checks=checks,
            min_J=float(raw['min_J'].min()),momentum_error_relative=float(raw['momentum_error'].max())/scale,
            angular_momentum_error_relative=float(raw['angular_momentum_error'].max())/scale,
            kinetic_J=kinetic,elastic_J=U,external_work_J=work,energy_remainder_J=work-kinetic-U)
        history.append(row);maximum={k:max(item['errors'][k] for item in history) for k in error}
        peak_work=max(max(abs(item['external_work_J']) for item in history),1e-30)
        report=dict(contract=contract,steps=current,full_cycle_complete=current==planned,history=history,maximum_errors=maximum,
            physical_checks_passed=all(all(item['checks'].values()) for item in history),
            finest_accuracy_target_met=all(v<=LIMITS['finest_position_velocity_F_relative'] for v in maximum.values()),
            maximum_energy_remainder_over_peak_work=max(abs(item['energy_remainder_J']) for item in history)/peak_work,
            energy_scope='Exact finite-strain energy branch only for Blatz-Ko; infinitesimal strain-energy diagnostic for the two linear Cauchy laws')
        save_checkpoint(a.output/'candidate.npz',state,current);save_json(a.output/'review.json',report);print(json.dumps(row),flush=True)
        if not all(checks.values()):raise ValueError('Physical/balance check failed; rejected candidate retained')
        save_checkpoint(a.output/'accepted.npz',state,current)


if __name__=='__main__':main()
