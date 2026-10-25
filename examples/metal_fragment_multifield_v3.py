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
 p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',choices=('medium','10k','25k'),default='medium')
 p.add_argument('--epoch-ns',type=float,choices=(125.,62.5),default=125.);p.add_argument('--iterations',type=int,choices=(8,16,32),default=8)
 p.add_argument('--end-ns',type=float,default=2000.);p.add_argument('--frames',type=int,default=8);args=p.parse_args()
 import jax;jax.config.update('jax_enable_x64',True)
 import jax.numpy as jnp
 import jaxlib
 from examples.metal_sphere_impact_fracture import plan_case,save_json,save_npz
 from mpm_jax import fragment_multifield_v3_mpm as solver
 from mpm_jax.materials.linear_metal import linear_metal
 from tools.audit_metal_fragments import components
 end_tick=round(args.end_ns*1000);epoch_tick=round(args.epoch_ns*1000)
 if not 0<end_tick<=2_000_000 or not 2<=args.frames<=60:raise ValueError('Require positive horizon<=2us and2..60 observation intervals')
 dx=dict(medium=.00025,**{'10k':1/6000.,'25k':.000125})[args.resolution];width=.0065 if args.resolution=='25k' else .006
 geometry,plan=plan_case(sphere_diameter=.002,plate_size=(width,width),plate_thickness=.0005,gap=3*dx,clamp_width=.001,dx=dx,spacing=(dx,dx/2,dx),frames=args.frames,end_time=end_tick*1e-12)
 if args.output.exists():raise FileExistsError('Fresh protocol only; old proof never resumed')
 args.output.mkdir(parents=True)
 save_json(args.output/'runtime.json',dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64)))
 physical=geometry['body_id'];reference=geometry['x'];edges=reference_edges(reference,physical,1.05*dx);alive=np.ones(len(edges),bool)
 material=linear_metal(**plan['material']);config=solver.ImpactConfig(contact_iterations=args.iterations)
 output_ticks=[round(i*end_tick/args.frames) for i in range(args.frames+1)]
 epoch_ticks=list(range(0,end_tick+1,epoch_tick));epoch_ticks=sorted(set(epoch_ticks+[end_tick]))
 targets=sorted(set(output_ticks+epoch_ticks));state=None;grid=None;rows=[];begun=time.monotonic();current_tick=0
 save_json(args.output/'assumptions.json',dict(geometry=plan,topology_epoch_ns=args.epoch_ns,observation_times_s=[t*1e-12 for t in output_ticks],contact_iterations=args.iterations,relaxation='Inverse maximum actual occupied-node contact degree; independent of candidate padding',damage_onset=.10,damage_failure=.30,bond_break_damage=.95,bond_opening_ratio=1.10,max_fields=8192,node_limit=2000000,pair_limit=2000000,virtual_int32_map_limit=32000000,intact_J_domain=[.4,2.],failed_tensile_J_domain='finiteJ>=1, zero tensile traction; no intact upper bound',heat='Only surviving-fraction plastic work. Fracture release stored separately.',boundary='Original plate edge reference particles fixed xyz; no backing',scope='Exploratory damage hypotheses; no fracture-energy calibration, mesh objectivity, phase or IJIE validation. Numerical reservoir is not physical heat.'))
 def checkpoint():
  save_npz(args.output/'checkpoint.npz',**{k:np.asarray(v) for k,v in state._asdict().items()},physical_body=physical,reference=reference,edges=edges,alive=alive)
 def record(tick):
  solver.validate_state(state,material,grid)
  audit=components(np.asarray(state.x),np.asarray(state.mass),np.asarray(state.v),physical,separation_radius=1.05*dx)
  ledger=np.asarray(solver.energy_ledger(state,grid));thermo=solver.thermodynamics(state,material)
  mass=np.asarray(state.mass);velocity=np.asarray(state.v)
  row=dict(frame=output_ticks.index(tick),time_s=float(state.time),fields=int(len(bodies)),nodes=grid.layout.node_count,contact_candidate_pairs=len(grid.pairs),broken_bonds=int((~alive).sum()),spatial_groups_1_05dx=len(audit['components']),mass_kg=float(mass.sum()),momentum_residual_Ns=float(np.linalg.norm(np.sum(mass[:,None]*(velocity-geometry['v']),axis=0)-np.asarray(state.boundary_impulse))),physical_energy_J=float(ledger[:3].sum()),boundary_reservoir_J=float(ledger[3]),numerical_reservoir_J=float(ledger[4]),accounted_energy_J=float(ledger.sum()),fracture_energy_J=float(np.sum(mass*np.asarray(state.fracture_energy))),maximum_temperature_K=float(thermo.temperature.max()),minimum_J=float(np.linalg.det(np.asarray(state.F)).min()),maximum_J=float(np.linalg.det(np.asarray(state.F)).max()),maximum_contact_closing_history_m_s=float(state.maximum_contact_closing_speed),elapsed_s=time.monotonic()-begun)
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
   else:state=state._replace(body_id=jnp.asarray(labels))
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
