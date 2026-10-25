"""Analytic constrained-contact impulses plus an integrated fixed-grid energy check."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 import jax;jax.config.update('jax_enable_x64',True)
 import jax.numpy as j
 from mpm_jax import fragment_multifield_v6_mpm as m
 from mpm_jax.materials.linear_metal import linear_metal
 if args.output.exists():raise FileExistsError('Fresh proof destination only')
 args.output.mkdir(parents=True)
 cases=[('free_equalization',[2.,5.],[[3.,0,0],[-1.,0,0]],[[False]*3,[False]*3],[[1.,0,0]],[[1/7,0,0],[1/7,0,0]]),('fixed_wall',[2.,5.],[[3.,0,0],[0.,0,0]],[[False]*3,[True]*3],[[1.,0,0]],[[0.,0,0],[0.,0,0]]),('partial_fixed',[2.,5.],[[2.,0,0],[0.,0,0]],[[False]*3,[True,False,False]],[[1.,1.,0]],[[7/6,-5/6,0],[0,1/3,0]]),('three_field_fixed',[1.,1.,1.],[[3.,0,0],[1.,0,0],[0.,0,0]],[[False]*3,[False]*3,[True]*3],[[1.,0,0]]*3,[[0.,0,0]]*3)]
 rows=[]
 for name,masses,velocities,masks,normals,expected in cases:
  mass=j.asarray(masses);v=j.asarray(velocities);mask=j.asarray(masks);pairs=j.array([[0,1]]) if len(masses)==2 else j.array([[0,1],[0,2],[1,2]])
  colors,active=m.color_contact_pairs(np.asarray(pairs),np.zeros(len(masses),dtype=np.int32))
  result,loss,internal,valid,closure,residual,sweeps,dual=m.paired_contact(v,mass,j.asarray(normals),pairs,mask,256,1e-10,j.asarray(colors),j.asarray(active))
  result.block_until_ready();actual=mass[:,None]*(result-v);reaction=actual-internal
  error=float(j.max(j.abs(result-j.asarray(expected))));momentum=float(j.linalg.norm(actual.sum(0)-reaction.sum(0)));internal_sum=float(j.linalg.norm(internal.sum(0)))
  before=float(.5*j.sum(mass*j.sum(v*v,axis=1)));after=float(.5*j.sum(mass*j.sum(result*result,axis=1)))
  assert bool(valid) and error<1e-8 and momentum<1e-10 and internal_sum<1e-10 and after<=before+1e-10
  assert float(j.max(j.abs(j.where(mask,0.,reaction))))<1e-10
  warm=m.paired_contact(v,mass,j.asarray(normals),pairs,mask,256,1e-10,j.asarray(colors),j.asarray(active),j.full(len(pairs),10.))
  warm[0].block_until_ready();warm_error=float(j.max(j.abs(warm[0]-j.asarray(expected))))
  assert bool(warm[3]) and warm_error<1e-8
  rows.append(dict(case=name,warm_start_velocity_error=warm_error,velocity_error=error,momentum_balance_error=momentum,internal_impulse_sum=internal_sum,boundary_reaction=np.asarray(reaction.sum(0)).tolist(),energy_before=before,energy_after=after,kinetic_loss=before-after,projection_loss=float(loss.sum()),closure_m_s=float(closure),complementarity_residual_m_s=float(residual),sweeps=int(sweeps)))
 (args.output/'analytic-cases.json').write_text(json.dumps(rows,indent=2)+'\n')
 x=np.array([[0.,0.,0.],[.0002,0.,0.]]);body=np.array([0,1],np.int32)
 grid=m.build_grid(x,body,dx=.001,field_physical_body=np.array([0,1]),contact_sweeps=(x,x));grid=replace(grid,clamped=j.broadcast_to((grid.node_body==1)[:,None],(grid.layout.node_count,3)))
 material=linear_metal();config=m.ImpactConfig(contact_iterations=128,contact_tolerance=1e-10)
 state=m.make_state(x,np.array([1e-9,1e-9]),body,material,grid,v=np.array([[3.,0,0],[0.,0,0]]));current,diagnostics=m.step(state,material,grid,config,j.asarray(1e-10));current.x.block_until_ready();np.savez_compressed(args.output/'integrated_state.npz',**{k:np.asarray(v) for k,v in current._asdict().items()});m.validate_state(current,material,grid)
 ledger=np.asarray(m.energy_ledger(current,grid));initial=np.asarray(m.energy_ledger(state,grid));temperature=np.asarray(m.thermodynamics(current,material).temperature)
 residual=np.sum(np.asarray(current.mass)[:,None]*(np.asarray(current.v)-np.asarray(state.v)),axis=0)-np.asarray(current.boundary_impulse)
 assert np.linalg.norm(residual)<1e-15 and abs(ledger.sum()-initial.sum())<1e-15 and np.max(np.abs(temperature-300.))<1e-9 and float(current.maximum_contact_closing_speed)<1e-10
 integrated=dict(valid=bool(current.valid),initial_ledger_J=initial.tolist(),final_ledger_J=ledger.tolist(),boundary_impulse_Ns=np.asarray(current.boundary_impulse).tolist(),momentum_residual_Ns=float(np.linalg.norm(residual)),temperature_K=temperature.tolist(),maximum_closure_m_s=float(current.maximum_contact_closing_speed))
 output=dict(cases=rows,integrated=integrated,solver_sha256=hashlib.sha256(Path('mpm_jax/fragment_multifield_v6_mpm.py').read_bytes()).hexdigest(),scope='Contact only analytic verification; not impact/fracture calibration')
 (args.output/'proof.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
