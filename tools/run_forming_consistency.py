"""Bounded candidate continuation of a verified original forming checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=1);a=parser.parse_args()
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from mpm_jax.fibre_mpm import FibreState,FibreConfig,build_layout
    from mpm_jax.materials.isotropic_brittle import BrittleHistory,isotropic_brittle
    from mpm_jax.fibre_forming_consistent import make_step
    from examples.three_strand_forming import geometry
    if a.steps<1 or a.steps>512:raise ValueError('Bounded1..512 step candidate only')
    a.output.mkdir(exist_ok=False)
    p=json.loads((a.parent/'plan.json').read_text());g=geometry();cpfile=a.parent/'checkpoint.npz'
    with np.load(cpfile,allow_pickle=False) as cp:
        state=FibreState(**{k[6:]:jnp.asarray(cp[k]) for k in cp.files if k.startswith('state_')},history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields)))
        start=int(cp['step'])
    material=isotropic_brittle(**p['material']);layout=build_layout(g['x'],g['ids'],dx=p['dx'],reference_bounds=np.broadcast_to(np.asarray(p['bounds']),(3,3,2)),padding_cells=3,max_nodes=1600000)
    step=make_step(material,layout,FibreConfig(p['dt']),tuple(jnp.asarray(m) for m in g['masks']),p['duration'])
    names=['fixture_work_J','initial_fixture_projection_loss_J','signed_contact_loss_J','kinetic_increment_J','elastic_increment_J','stress_midpoint_velocity_work_J','P2G_loss_J','G2P_loss_J','energy_remainder_J','grid_work_identity_residual_J','stress_virtual_work_residual_J','material_endpoint_remainder_J','closing_residual_m_per_s','contact_sweeps','minJ','momentum_step_residual_Ns','angular_step_residual_kg_m2_per_s','angular_transport_term_kg_m2_per_s','fixture_torque_impulse_kg_m2_per_s','angular_momentum_norm_kg_m2_per_s','fixture_torque_x','fixture_torque_y','fixture_torque_z','angular_residual_x','angular_residual_y','angular_residual_z','angular_momentum_x','angular_momentum_y','angular_momentum_z']
    from tools.review_forming_snapshot import inspect
    geometry_records=[inspect(state,p,g)]
    if not geometry_records[-1]['geometry_valid']:raise FloatingPointError('Parent geometry diagnostic rejected')
    (a.output/'plan.json').write_text(json.dumps(p,indent=2))
    def checkpoint(name,current_step):
        leaves={'state_'+key:np.asarray(value) for key,value in state._asdict().items() if key!='history'}
        leaves.update({'history_'+key:np.asarray(value) for key,value in state.history._asdict().items()})
        temporary=a.output/(name+'.tmp.npz')
        np.savez_compressed(temporary,**leaves,step=current_step,parent_checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),candidate=np.array('boundary-aware-midpoint-configuration-v1'))
        temporary.replace(a.output/(name+'.npz'))
    frame_steps=set(np.rint(np.linspace(0,p['steps'],61)).astype(int))
    def frame(current_step):
        np.savez_compressed(a.output/f'frame-{current_step:06d}.npz',x=np.asarray(state.x),fibre_id=np.asarray(state.fibre_id),time=np.asarray(state.time))
    frame(start)
    rows=[];begun=time.monotonic()
    for k in range(a.steps):
        state,metrics=step(state);metrics=np.asarray(metrics)
        if not bool(state.valid):
            checkpoint('rejected-state',start+k+1)
            raise FloatingPointError('Candidate state invalid; last accepted checkpoint and rejected diagnostic preserved')
        rows.append(dict(step=start+k+1,**dict(zip(names,map(float,metrics)))))
        if start+k+1 in frame_steps or k+1==a.steps:frame(start+k+1)
        if (k+1)%16==0 or k+1==a.steps:
            geometry_records.append(inspect(state,p,g))
            progress=dict(step=start+k+1,rows=rows,geometry_records=geometry_records,geometry_valid=geometry_records[-1]['geometry_valid'])
            (a.output/'progress.json').write_text(json.dumps(progress,indent=2))
            print(json.dumps(dict(step=start+k+1,time_s=float(state.time),geometry=geometry_records[-1],solver_seconds=time.monotonic()-begun)),flush=True)
            if not geometry_records[-1]['geometry_valid']:
                checkpoint('rejected-state',start+k+1)
                raise FloatingPointError('Actual particle geometry diagnostic rejected; original parent and last accepted checkpoint preserved')
            checkpoint('checkpoint',start+k+1)
    leaves={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
    np.savez_compressed(a.output/'checkpoint.npz',**leaves,step=start+a.steps,parent_checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),candidate=np.array('boundary-aware-midpoint-configuration-v1'))
    receipt=dict(parent_checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),original_plan=p,accepted_steps=[start,start+a.steps],solver_seconds=time.monotonic()-begun,rows=rows,geometry_records=geometry_records,production_eligible=False,scope='Isolated explicit candidate; initial boundary projection then midpoint configuration. Signed contact work and material endpoint defect remain explicit; no unconditional stability claim.')
    (a.output/'plan.json').write_text(json.dumps(p,indent=2))
    (a.output/'review.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
