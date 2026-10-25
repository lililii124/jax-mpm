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
    names=['fixture_work_J','initial_fixture_projection_loss_J','signed_contact_loss_J','kinetic_increment_J','elastic_increment_J','stress_midpoint_velocity_work_J','P2G_loss_J','G2P_loss_J','energy_remainder_J','grid_work_identity_residual_J','stress_virtual_work_residual_J','material_endpoint_remainder_J','closing_residual_m_per_s','contact_sweeps','minJ']
    rows=[];begun=time.monotonic()
    for k in range(a.steps):
        state,metrics=step(state);metrics=np.asarray(metrics)
        if not bool(state.valid):raise FloatingPointError('Candidate state invalid; no accepted checkpoint')
        rows.append(dict(step=start+k+1,**dict(zip(names,map(float,metrics)))))
    leaves={'state_'+k:np.asarray(v) for k,v in state._asdict().items() if k!='history'};leaves.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
    np.savez_compressed(a.output/'checkpoint.npz',**leaves,step=start+a.steps,parent_checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),candidate=np.array('boundary-aware-midpoint-configuration-v1'))
    receipt=dict(parent_checkpoint_sha256=hashlib.sha256(cpfile.read_bytes()).hexdigest(),original_plan=p,accepted_steps=[start,start+a.steps],solver_seconds=time.monotonic()-begun,rows=rows,production_eligible=False,scope='Isolated explicit candidate; initial boundary projection then midpoint configuration. Signed contact work and material endpoint defect remain explicit; no unconditional stability claim.')
    (a.output/'review.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
