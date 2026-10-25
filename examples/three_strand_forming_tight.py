"""Independent actual dry tight forming: close initial gaps and feed arc length.

No prescribed interior helix. This candidate is distinct from the published
partial baseline forming process and from the independent resin stage.
"""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np


METRIC_NAMES=['fixture_work_J', 'initial_fixture_projection_loss_J', 'signed_contact_loss_J', 'kinetic_increment_J', 'elastic_increment_J', 'stress_midpoint_velocity_work_J', 'P2G_loss_J', 'G2P_loss_J', 'energy_remainder_J', 'grid_work_identity_residual_J', 'stress_virtual_work_residual_J', 'material_endpoint_remainder_J', 'closing_residual_m_per_s', 'contact_sweeps', 'minJ', 'momentum_step_residual_Ns', 'angular_step_residual_kg_m2_per_s', 'angular_transport_term_kg_m2_per_s', 'fixture_torque_impulse_kg_m2_per_s', 'angular_momentum_norm_kg_m2_per_s', 'fixture_torque_x', 'fixture_torque_y', 'fixture_torque_z', 'angular_residual_x', 'angular_residual_y', 'angular_residual_z', 'angular_momentum_x', 'angular_momentum_y', 'angular_momentum_z', 'contact_exact_nodes', 'contact_exact_successes', 'contact_exact_capacity_valid', 'contact_rank_rejections', 'normal_contact_force_01_N', 'normal_contact_force_02_N', 'normal_contact_force_12_N', 'active_contact_nodes_01', 'active_contact_nodes_02', 'active_contact_nodes_12', 'contact_xmin_01_m', 'contact_xmin_02_m', 'contact_xmin_12_m', 'contact_xmax_01_m', 'contact_xmax_02_m', 'contact_xmax_12_m', 'contact_shared_nodes', 'contact_shared_capacity_valid']

def run(output,steps=64,parent=None,adopt_future_tension=False):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.three_strand_forming import geometry
    from mpm_jax.fibre_mpm import FibreConfig,FibreState,build_layout,make_state
    from mpm_jax.materials.isotropic_brittle import isotropic_brittle,BrittleHistory,brittle_energy
    from mpm_jax.fibre_forming_consistent import make_step
    from mpm_jax.fibre_forming_tight_boundary import prescribe,ORBIT,FREE_LENGTH,RAMP
    from tools.review_forming_snapshot import inspect
    from mpm_jax.fibre_forming_boundary import angle
    if not 1<=steps<=512:raise ValueError('Only bounded1..512 physical steps')
    folder=Path(output);folder.mkdir(exist_ok=False)
    design=json.loads((Path(__file__).resolve().parents[1]/'docs/three-strand-forming-tight-plan.json').read_text())
    g=geometry(spacing=.000125,orbit=ORBIT);dx=.00025;duration=.06;total=math.ceil(duration/(.15*dx/math.sqrt(60)))
    p=dict(case=design['case'],geometry=design['geometry'],particles=len(g['x']),dx=dx,dt=duration/total,duration=duration,steps=total,bounds=[[-.001,.025],[-.0035,.0035],[-.0035,.0035]],protocol=dict(rotation_ramp_s=RAMP,feed='arc-length symmetric axial collar feed',arc_tensile_strain=.03,tension_ramp_start_s=.005,tension_ramp_end_s=.015),material=dict(young=50000.,poisson=.25,density=1000.,tensile_strength=1e5,fracture_energy=100.,characteristic_length=dx),design={k:design[k] for k in ['geometry','material','protocol','contact','acceptance']},production_eligible=False)
    material=isotropic_brittle(**p['material']);layout=build_layout(g['x'],g['ids'],dx=dx,reference_bounds=np.broadcast_to(np.asarray(p['bounds']),(3,3,2)),padding_cells=3,max_nodes=1600000)
    state=make_state(g['x'],g['volume'],g['ids'],material,layout);start=0;parent_sha=None
    if parent:
        q=json.loads((parent/'plan.json').read_text())
        cpfile=parent/'checkpoint.npz';parent_sha=hashlib.sha256(cpfile.read_bytes()).hexdigest()
        with np.load(cpfile,allow_pickle=False) as cp:
            state=FibreState(**{k[6:]:jnp.asarray(cp[k]) for k in cp.files if k.startswith('state_')},history=BrittleHistory(*(jnp.asarray(cp['history_'+k]) for k in BrittleHistory._fields)));start=int(cp['step'])
        if q!=p:
            same_geometry=all(q[key]==p[key] for key in ['case','geometry','particles','dx','dt','duration','steps','bounds','material'])
            if not(adopt_future_tension and same_geometry and float(state.time)<.005 and 'arc_tensile_strain' not in q['protocol']):raise ValueError('Unapproved tight plan change; only explicit pre5ms adoption of future tensile ramp is allowed')
    if start+steps>total:raise ValueError('Cannot pass frozen end time')
    step=make_step(material,layout,FibreConfig(p['dt']),tuple(jnp.asarray(m) for m in g['masks']),duration,boundary=prescribe,contact_bins=64)
    energy=jax.jit(lambda s:jnp.sum(s.volume*brittle_energy(s.F,s.history,material)))(state);energy.block_until_ready()
    def collar_target(side,mask,t):
        theta=float(angle(t,duration,RAMP));c,s=np.cos(theta),np.sin(theta);Q=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        target=g['x'][mask]@(Q.T if side else np.eye(3));z=np.clip((t-.005)/.010,0,1);arc=FREE_LENGTH*(1+.03*z*z*(3-2*z));extension=math.sqrt(arc**2-(theta*ORBIT)**2)-FREE_LENGTH;target[:,0]+=(1 if side else -1)*extension/2;return target
    (folder/'plan.json').write_text(json.dumps(p,indent=2));rows=[];geometry_records=[];begun=time.monotonic()
    def save(k,name='checkpoint'):
        leaves={'state_'+key:np.asarray(value) for key,value in state._asdict().items() if key!='history'};leaves.update({'history_'+key:np.asarray(value) for key,value in state.history._asdict().items()});np.savez(folder/(name+'.tmp.npz'),**leaves,step=k,elastic_energy_J=np.asarray(energy));(folder/(name+'.tmp.npz')).replace(folder/(name+'.npz'))
    save(start)
    for k in range(start+1,start+steps+1):
        state,metrics,energy=step(state,energy);values=np.asarray(metrics)
        rows.append(dict(step=k,**dict(zip(METRIC_NAMES,map(float,values[:len(METRIC_NAMES)]))),contact_force_axial_bins_N=values[len(METRIC_NAMES):].reshape(3,64).tolist()))
        if not bool(state.valid):save(k,'rejected-state');raise FloatingPointError('Tight physical state rejected')
        if k%16==0 or k==start+steps:
            record=inspect(state,p,g,collar_target=collar_target);geometry_records.append(record)
            if not record['geometry_valid']:save(k,'rejected-state');raise FloatingPointError('Tight geometry rejected')
            save(k);np.savez_compressed(folder/f'frame-{k:06d}.npz',x=np.asarray(state.x),fibre_id=np.asarray(state.fibre_id),time=np.asarray(state.time));(folder/'progress.json').write_text(json.dumps(dict(accepted_step=k,rows=rows,geometry_records=geometry_records),indent=2));print(json.dumps(dict(step=k,time_s=float(state.time),geometry=record)),flush=True)
    receipt=dict(accepted_steps=[start,start+steps],parent_checkpoint_sha256=parent_sha,rows=rows,geometry_records=geometry_records,wall_seconds=time.monotonic()-begun,future_tension_adopted=bool(adopt_future_tension),full_turn_completed=start+steps==total,contact_coverage_qualified=False,production_eligible=False)
    (folder/'review.json').write_text(json.dumps(receipt,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=64);p.add_argument('--parent',type=Path);p.add_argument('--adopt-future-tension',action='store_true');a=p.parse_args();run(a.output,a.steps,a.parent,a.adopt_future_tension)

if __name__=='__main__':main()
