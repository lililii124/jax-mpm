"""Total-Lagrangian MPM tensile failure of the reference circular fibre."""
import argparse,hashlib,json,sys,time
from functools import partial
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from .single_fibre_tension import plan_case,save_json,save_npz,motion,fracture_review
from mpm_jax.fibre_mpm import build_layout
from mpm_jax.fibre_tlmpm import make_solver,TLState
from mpm_jax.materials.isotropic_brittle import isotropic_brittle,BrittleHistory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stop-after-steps',type=int,default=2000);p.add_argument('--resume-from',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    geometry,plan=plan_case();material=isotropic_brittle(**plan['material'])
    layout=build_layout(geometry['x'],geometry['fibre_id'],dx=plan['grid']['dx_m'],padding_cells=6,
        reference_bounds=plan['grid']['bounds_m'],max_nodes=500000)
    state,step,particles,energy,spatial=make_solver(geometry['x'],geometry['volume'],layout,material,geometry['masks'],
        plan['dt'],plan['protocol'],duration=plan['duration_s'])
    spatial={k:float(v) for k,v in spatial.items()};planned=int(spatial['planned_steps']);dt=spatial['dt']
    target=min(a.stop_after_steps,planned)
    if target<=0:p.error('A positive advancing target is required')
    if (spatial['partition_error']>1e-12 or spatial['gradient_constant_error']>1e-9 or spatial['gradient_affine_error']>1e-12
        or abs(spatial['nodal_mass']-spatial['reference_mass'])>1e-12*spatial['reference_mass']
        or spatial['acoustic_64_iteration_relative_change']>.02):raise ValueError('Reference basis or acoustic power iteration not qualified')
    root=Path(__file__).resolve().parents[1]
    source={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('mpm_jax/fibre_tlmpm.py','examples/single_fibre_tlmpm.py','mpm_jax/materials/isotropic_brittle.py','examples/single_fibre_tension.py','mpm_jax/fibre_mpm.py')}
    contract=dict(plan=plan,source_sha256=source,spatial=spatial,
        method='Total-Lagrangian quadratic B-spline MPM; fixed reference quadrature, lumped nodal inertia, velocity-Verlet',
        energy_metric='Lumped nodal kinetic energy plus reference material energy; particle APIC affine energy is not used',
        scope='New numerical trajectory; original material, waist, physical displacement/time law; not equivalent to old updated-grid APIC trajectory')
    chunks=[];start_step=0
    if a.resume_from:
        old=json.loads((a.resume_from/'contract.json').read_text())
        if old!=contract:raise ValueError('Resume source/material/geometry/time grid mismatch')
        with np.load(a.resume_from/'accepted.npz',allow_pickle=False) as cp:
            start_step=int(cp['step']);history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields))
            state=TLState(**{k:jnp.asarray(cp[k]) for k in TLState._fields if k!='history'},history=history)
        with np.load(a.resume_from/'step_records.npz',allow_pickle=False) as old:
            previous={k:old[k] for k in old.files}
        if not all(len(v)==start_step and np.isfinite(v).all() for v in previous.values()) or not previous['valid'].all():raise ValueError('Incomplete or invalid parent records')
        if abs(float(state.time)-start_step*dt)>1e-14 or not state.valid:raise ValueError('Parent clock/validity mismatch')
        chunks.append(previous)
        save_json(a.output/'parent.json',dict(checkpoint_sha256=hashlib.sha256((a.resume_from/'accepted.npz').read_bytes()).hexdigest(),start_step=start_step,prior_steps_replayed=0))
    if not start_step<target<=min(planned,start_step+2000):p.error('Advance at most2000 steps per bounded window')
    save_json(a.output/'contract.json',contract)
    if start_step==0:
        x0,v0=map(np.asarray,particles(state))
        save_npz(a.output/'frame-000000.npz',time=0.,x=x0,v=v0,damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed))
        save_npz(a.output/'reference.npz',x=geometry['x'],volume=geometry['volume'],mass=geometry['mass'],
            lattice=geometry['lattice'],left_grip=geometry['masks'][0],right_grip=geometry['masks'][1])
    @partial(jax.jit,static_argnames=('count',))
    def batch(state,*,count):return jax.lax.scan(lambda s,_:step(s),state,None,length=count)
    # lower.compile does not execute a warm physical step.
    compiled=batch.lower(state,count=min(100,target-start_step)).compile()
    initial_K,initial_U=energy(state);begin=time.monotonic()
    current=start_step
    while current<target:
        count=min(100,target-current)
        if count==100 or count==min(100,target-start_step):state,rows=compiled(state)
        else:state,rows=batch(state,count=count)
        jax.block_until_ready(state);current+=count;host=jax.tree.map(np.asarray,rows);chunks.append(host)
        values={key:np.concatenate([chunk[key] for chunk in chunks]) for key in host}
        raw={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'}
        raw.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
        save_npz(a.output/'candidate.npz',step=current,**raw);save_npz(a.output/'step_records.npz',**values)
        x,v=map(np.asarray,particles(state));stroke=float(motion(float(state.time),plan['protocol']))
        grip=max(float(np.max(np.linalg.norm(x[mask]-geometry['x'][mask]-np.array([sign*stroke/2,0.,0.]),axis=1))) for mask,sign in zip(geometry['masks'],(-1,1)))
        K,U=map(float,energy(state));work=float(state.work);scale=max(float(np.max(abs(values['fixture_work_total_J']))),1e-30)
        residual=float(np.max(abs(values['residual_J'])))/scale
        checks=dict(valid=bool(values['valid'].all()),finite=all(np.isfinite(v).all() for v in jax.tree.leaves(state)),
            grips=grip<1e-9,momentum=float(values['momentum_error'].max())<1e-10,
            angular_momentum=float(values['angular_momentum_error'].max())<1e-10,
            energy_within_original15percent=residual<=.15,positive_J=float(values['min_J'].min())>0,
            nonnegative_damage=float(state.damage_work)>=0)
        review=dict(step=current,time_s=float(state.time),target_steps=planned,dt_s=dt,elapsed_s=time.monotonic()-begin,
            initial_window_K_J=float(initial_K),initial_window_U_J=float(initial_U),kinetic_J=K,elastic_J=U,work_J=work,
            damage_work_J=float(state.damage_work),maximum_energy_residual_over_peak_work=residual,
            final_energy_residual_over_peak_work=float(host['residual_J'][-1])/scale,max_damage=float(values['max_damage'].max()),
            min_J=float(values['min_J'].min()),grip_error_m=grip,checks=checks,full_plan_completed=current==planned,
            peak_tensile_load_N=float(values['tensile_load_N'].max()),
            fracture=fracture_review(geometry,x,np.asarray(state.history.failed)),
            scope='Actual total-Lagrangian MPM trajectory; no space/time convergence certificate')
        save_json(a.output/'review.json',review);print(json.dumps({k:review[k] for k in ('step','time_s','elapsed_s','max_damage','maximum_energy_residual_over_peak_work','checks')}),flush=True)
        if not all(checks.values()):raise ValueError('Original physical admission failed; raw candidate retained')
        save_npz(a.output/'accepted.npz',step=current,**raw)
        if current%400==0 or current==target:save_npz(a.output/f'frame-{current:06d}.npz',time=float(state.time),x=x,v=v,damage=np.asarray(state.history.damage),failed=np.asarray(state.history.failed))


if __name__=='__main__':main()
