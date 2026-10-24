"""Exploratory fixed-epoch fragment fields, independent of observation cadence."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from examples.metal_fragment_multifield_compact import reference_edges,split_fields


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--parent-sha',required=True);p.add_argument('--parent-recovery-receipt-sha');p.add_argument('--resolution',choices=('formal',),default='formal')
 p.add_argument('--epoch-ns',type=float,choices=(125.,62.5),default=125.);p.add_argument('--iterations',type=int,choices=(256,512,1024,2048),default=1024);p.add_argument('--contact-tolerance',type=float,choices=(.005,.0005),default=.005)
 p.add_argument('--end-ns',type=float,default=2000.);p.add_argument('--frames',type=int,default=16);args=p.parse_args()
 import jax;jax.config.update('jax_enable_x64',True)
 import jax.numpy as jnp
 import jaxlib
 from examples.metal_sphere_impact_fracture import plan_case,save_json,save_npz
 from mpm_jax import fragment_multifield_v6_mpm as solver
 from mpm_jax.materials.linear_metal import linear_metal
 from tools.audit_metal_fragments import components
 end_tick=round(args.end_ns*1000);epoch_tick=round(args.epoch_ns*1000)
 if not 0<end_tick<=2_000_000 or args.frames!=16:raise ValueError('Require positive horizon<=2us and2..60 observation intervals')
 dx=.00025;width=.02
 geometry,plan=plan_case(sphere_diameter=.002,plate_size=(width,width),plate_thickness=.0005,gap=3*dx,clamp_width=.001,dx=dx,spacing=(dx,dx/2,dx),frames=args.frames,end_time=2e-6)
 if args.output.exists():raise FileExistsError('Fresh protocol only; old proof never resumed')
 args.output.mkdir(parents=True)
 save_json(args.output/'runtime.json',dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64)))
 parent=args.parent.resolve();parent_checkpoint=parent/'result/checkpoint.npz'
 if hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()!=args.parent_sha:raise ValueError('Exact complete parent checkpoint required')
 monitor=json.loads((parent/'monitor.json').read_text())
 if monitor['source_changed'] or (parent/'result/failure.json').exists():raise ValueError('Changed source or physical failed candidate cannot resume')
 if monitor['returncode']!=0 or monitor['reason'] is not None:
  pause=parent/'run-receipt.json'
  if not args.parent_recovery_receipt_sha or hashlib.sha256(pause.read_bytes()).hexdigest()!=args.parent_recovery_receipt_sha:raise ValueError('Independently admitted resource-pause receipt required')
  recovery=json.loads(pause.read_text())
  if recovery.get('resumable_complete_checkpoint') is not True or recovery['checkpoint']['sha256']!=args.parent_sha or recovery.get('physical_gate_failed') is not False:raise ValueError('Only a validated complete resource-paused checkpoint can resume')
 parent_source=json.loads((parent/'source.json').read_text())
 for name,expected in parent_source.items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Parent source changed:'+name)
 runtime=json.loads((args.output/'runtime.json').read_text())
 if runtime!=json.loads((parent/'result/runtime.json').read_text()):raise ValueError('Exact parent runtime required')
 old_assumptions=json.loads((parent/'result/assumptions.json').read_text())
 if old_assumptions['topology_epoch_ns']!=125. or old_assumptions['contact_iterations']!=1024 or old_assumptions['contact_tolerance_m_s']!=.005 or args.epoch_ns!=125. or args.iterations!=1024 or args.contact_tolerance!=.005:raise ValueError('Unchanged numerical protocol required')
 with np.load(parent_checkpoint,allow_pickle=False) as cp:
  expected=set(solver.ImpactState._fields)|{'physical_body','reference','edges','alive'}
  if set(cp.files)!=expected or not all(np.isfinite(cp[k]).all() for k in cp.files):raise ValueError('Parent carry field set/finite check failed')
  state=solver.ImpactState(*(jnp.asarray(cp[name]) for name in solver.ImpactState._fields));physical=cp['physical_body'];reference=cp['reference'];edges=cp['edges'];alive=cp['alive']
 if not np.array_equal(reference,geometry['x']) or not np.array_equal(physical,geometry['body_id']) or not np.array_equal(np.asarray(state.volume),geometry['volume']):raise ValueError('Reference geometry/mass protocol changed')
 start_tick=round(float(state.time)*1e12)
 if not bool(state.valid) or start_tick%125000 or not 500000<=start_tick<end_tick<=2000000:raise ValueError('Parent must be a later accepted complete epoch')
 if float(state.maximum_contact_closing_speed)>.005 or float(state.maximum_contact_complementarity_residual)>.005:raise ValueError('Parent contact gate failed')
 import shutil
 shutil.copy2(parent_checkpoint,args.output/'parent-checkpoint.npz');shutil.copy2(parent_checkpoint,args.output/'checkpoint.npz')
 save_json(args.output/'lineage.json',dict(parent=str(parent),parent_checkpoint_sha256=args.parent_sha,parent_source_sha256=hashlib.sha256((parent/'source.json').read_bytes()).hexdigest(),start_tick_ps=start_tick,stop_tick_ps=end_tick,physics_unchanged=True,old_failed_state_restored=False))
 material=linear_metal(**plan['material']);config=solver.ImpactConfig(contact_iterations=args.iterations,contact_tolerance=args.contact_tolerance)
 output_ticks=list(range(start_tick+125000,end_tick+1,125000))
 epoch_ticks=list(range(start_tick+epoch_tick,end_tick+1,epoch_tick));epoch_ticks=sorted(set(epoch_ticks+[end_tick]))
 targets=sorted(set(output_ticks+epoch_ticks));grid=None;rows=[];begun=time.monotonic();current_tick=start_tick
 save_json(args.output/'assumptions.json',dict(geometry=plan,topology_epoch_ns=args.epoch_ns,observation_times_s=[t*1e-12 for t in output_ticks],contact_iterations=args.iterations,dual_cache='Previous-step normal dual multiplier, dt-scaled; reprojected using current mass/normal; reset at every topology epoch',contact_solver='Mass-normalized accelerated projected dual gradient with restart and actual occupied-degree metric; fixed-DOF inverse mass0; full opposite impulses/external reactions separate',contact_tolerance_m_s=args.contact_tolerance,colored_slot_limit=2000000,damage_onset=.10,damage_failure=.30,bond_break_damage=.95,bond_opening_ratio=1.10,max_fields=8192,node_limit=2000000,pair_limit=2000000,virtual_int32_map_limit=32000000,intact_J_domain=[.4,2.],failed_tensile_J_domain='finiteJ>=1, zero tensile traction; no intact upper bound',heat='Only surviving-fraction plastic work. Fracture release stored separately.',boundary='Original plate edge reference particles fixed xyz; no backing',scope='Exploratory damage hypotheses; no fracture-energy calibration, mesh objectivity, phase or IJIE validation. Numerical reservoir is not physical heat.'))
 def checkpoint():
  save_npz(args.output/'checkpoint.npz',**{k:np.asarray(v) for k,v in state._asdict().items()},physical_body=physical,reference=reference,edges=edges,alive=alive)
 def record(tick):
  solver.validate_state(state,material,grid)
  if float(state.maximum_contact_closing_speed)>config.contact_tolerance or float(state.maximum_contact_complementarity_residual)>config.contact_tolerance:raise RuntimeError('Contact threshold not attained')
  audit=components(np.asarray(state.x),np.asarray(state.mass),np.asarray(state.v),physical,separation_radius=1.05*dx)
  radial=[dict(radius_factor=factor,**components(np.asarray(state.x),np.asarray(state.mass),np.asarray(state.v),physical,separation_radius=factor*dx)) for factor in (1.05,1.5,2.)]
  save_json(args.output/f'spatial-{tick//125000:03}.json',dict(time_s=float(state.time),three_radii=radial))
  ledger=np.asarray(solver.energy_ledger(state,grid));thermo=solver.thermodynamics(state,material)
  mass=np.asarray(state.mass);velocity=np.asarray(state.v)
  row=dict(frame=tick//125000,time_s=float(state.time),fields=int(len(bodies)),nodes=grid.layout.node_count,contact_candidate_pairs=len(grid.pairs),broken_bonds=int((~alive).sum()),spatial_groups_1_05dx=len(audit['components']),mass_kg=float(mass.sum()),momentum_residual_Ns=float(np.linalg.norm(np.sum(mass[:,None]*(velocity-geometry['v']),axis=0)-np.asarray(state.boundary_impulse))),physical_energy_J=float(ledger[:3].sum()),boundary_reservoir_J=float(ledger[3]),numerical_reservoir_J=float(ledger[4]),accounted_energy_J=float(ledger.sum()),fracture_energy_J=float(np.sum(mass*np.asarray(state.fracture_energy))),maximum_temperature_K=float(thermo.temperature.max()),minimum_J=float(np.linalg.det(np.asarray(state.F)).min()),maximum_J=float(np.linalg.det(np.asarray(state.F)).max()),maximum_contact_closing_history_m_s=float(state.maximum_contact_closing_speed),maximum_contact_complementarity_history_m_s=float(state.maximum_contact_complementarity_residual),maximum_contact_iterations_used=float(state.maximum_contact_iterations),boundary_impulse_Ns=np.asarray(state.boundary_impulse).tolist(),boundary_projection_loss_J=float(state.boundary_projection_loss),boundary_coupling_work_J=float(state.boundary_coupling_work),elapsed_s=time.monotonic()-begun)
  radial_reference=np.maximum(np.abs(reference[:,0]),np.abs(reference[:,2]));bins=[]
  for low,high in zip((0.,.002,.004,.006,.008),(.002,.004,.006,.008,.0101)):
   select=(physical==1)&(radial_reference>=low)&(radial_reference<high);bins.append(dict(lower_m=low,upper_m=high,kinetic_energy_J=float(.5*np.sum(mass[select]*np.sum(velocity[select]**2,axis=1))),signed_y_momentum_Ns=float(np.sum(mass[select]*velocity[select,1]))))
  row['plate_reference_annuli']=bins
  rows.append(row);save_json(args.output/'metrics.json',rows)
  save_npz(args.output/f'frame-{row["frame"]:03}.npz',**{k:np.asarray(v) for k,v in state._asdict().items()},physical_body=physical)
  save_json(args.output/f'groups-{row["frame"]:03}.json',audit);checkpoint();print(json.dumps(row),flush=True)
 for target_tick in targets:
  if grid is None:
   x=reference if state is None else np.asarray(state.x);damage=np.zeros(len(x)) if state is None else np.asarray(state.damage)
   labels,alive,bodies=split_fields(x,reference,edges,alive,damage,physical,max_fields=8192)
   epoch_end=min(current_tick+epoch_tick,end_tick);duration=(epoch_end-current_tick)*1e-12
   velocity=geometry['v'] if state is None else np.asarray(state.v);predicted=x+duration*velocity
   lower=np.full((len(bodies),3),np.inf);upper=np.full_like(lower,-np.inf)
   np.minimum.at(lower,labels,np.minimum(x,predicted));np.maximum.at(upper,labels,np.maximum(x,predicted))
   bounds=np.stack((lower-2*dx,upper+2*dx),axis=-1)
   grid=solver.build_grid(x,labels,dx=dx,padding_cells=3,reference_bounds=bounds,max_nodes=2_000_000,field_physical_body=bodies,contact_sweeps=(x,predicted))
   fixed=(physical==1)&((np.abs(reference[:,0])>=width/2-.001)|(np.abs(reference[:,2])>=width/2-.001))
   support=solver.stencil(jnp.asarray(x),jnp.asarray(labels),grid.layout);mask=np.zeros((grid.layout.node_count,3),bool)
   nodes=np.asarray(support.indices)[fixed].ravel();nodes=nodes[(nodes>=0)&(nodes<grid.layout.node_count)];mask[nodes]=True
   grid=replace(grid,clamped=jnp.asarray(mask))
   if state is None:state=solver.make_state(x,geometry['volume'],labels,material,grid,v=geometry['v'])
   else:state=state._replace(body_id=jnp.asarray(labels),contact_lambda=jnp.zeros(len(grid.pairs),dtype=state.x.dtype),contact_cache_dt=jnp.asarray(0.))
  if target_tick>current_tick:
   candidate,steps=solver.advance_to(state,material,grid,config,jnp.asarray(target_tick*1e-12),max_steps=20000)
   candidate.x.block_until_ready()
   if not bool(candidate.valid):
    save_npz(args.output/'failed_state.npz',**{k:np.asarray(v) for k,v in candidate._asdict().items()})
    save_json(args.output/'failure.json',dict(last_accepted_time_s=float(state.time),candidate_time_s=float(candidate.time),target_time_s=target_tick*1e-12,scope='Original failed candidate retained; no rollback as advancement'))
    raise RuntimeError('Step/support/contact/material gate rejected; failed raw retained')
   state=candidate;current_tick=target_tick
  if target_tick in output_ticks:record(target_tick)
  else:checkpoint()
  if target_tick in epoch_ticks and target_tick<end_tick:grid=None


if __name__=='__main__':main()
