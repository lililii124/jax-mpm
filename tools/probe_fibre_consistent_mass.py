"""Actual original-size fibre prefix for the consistent-mass candidate."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from examples.single_fibre_tension import plan_case,motion,save_json,save_npz
from mpm_jax.fibre_mpm import build_layout,make_state,FibreConfig,validate_state
from mpm_jax.materials.isotropic_brittle import isotropic_brittle
from mpm_jax.fibre_consistent_mass import make_step


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100);a=p.parse_args()
    if not 10<=a.steps<=1000 or a.steps%10:p.error('Use10-step blocks up to1000')
    a.output.mkdir(exist_ok=False,parents=True)
    geometry,plan=plan_case();material=isotropic_brittle(**plan['material']);config=FibreConfig(plan['dt'])
    layout=build_layout(geometry['x'],geometry['fibre_id'],dx=plan['grid']['dx_m'],padding_cells=6,
        reference_bounds=plan['grid']['bounds_m'],max_nodes=500000)
    state=make_state(geometry['x'],geometry['volume'],geometry['fibre_id'],material,layout)
    def target(i,t):
        speed=(motion(t,plan['protocol'],jnp)-motion(t-config.dt,plan['protocol'],jnp))/config.dt
        return jnp.array([(-1 if i==0 else 1)*speed/2,0.,0.])
    step=make_step(material,layout,config,tuple(jnp.asarray(mask) for mask in geometry['masks']),target)
    def batch(carry):
        def body(c,_):
            following,warm,row=step(*c)
            return (following,warm),row
        return jax.lax.scan(body,carry,None,length=10)
    carry=(state,jnp.zeros((layout.node_count,3)))
    compiled=jax.jit(batch).lower(carry).compile();chunks=[];begun=time.monotonic()
    files=('mpm_jax/fibre_consistent_mass.py','mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py',
        'examples/single_fibre_tension.py','tools/probe_fibre_consistent_mass.py')
    save_json(a.output/'contract.json',dict(plan=plan,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        method='Consistent APIC kinetic mass; constrained Galerkin projection; explicit old stress kick; midpoint kinematics',
        scope='New numerical method, original material/geometry/physical collar displacement; not same-trajectory or full fracture admission',
        cg_relative_tolerance=1e-10,cg_max_iterations=256))
    for end in range(10,a.steps+1,10):
        carry,record=compiled(carry);jax.block_until_ready(carry)
        state,warm=carry;host=jax.tree.map(np.asarray,record);chunks.append(host)
        values={key:np.concatenate([chunk[key] for chunk in chunks]) for key in host}
        raw={k:np.asarray(v) for k,v in state._asdict().items() if k!='history'}
        raw.update({'history_'+k:np.asarray(v) for k,v in state.history._asdict().items()})
        save_npz(a.output/'candidate.npz',step=end,warm=np.asarray(warm),**raw)
        save_npz(a.output/'step-records.npz',**values)
        totals={k:float(values[k].sum()) for k in ('work_J','projection_loss_J','material_remainder_J','damage_J','residual_J')}
        scale=max(abs(totals['work_J']),float(host['kinetic_J'][-1]+host['elastic_J'][-1]),1e-30)
        stroke=float(motion(float(state.time),plan['protocol']))
        grip=max(float(np.max(np.linalg.norm(np.asarray(state.x)[mask]-geometry['x'][mask]-np.array([sign*stroke/2,0.,0.]),axis=1))) for mask,sign in zip(geometry['masks'],(-1,1)))
        checks=dict(state_valid=bool(values['valid'].all()),finite=all(np.isfinite(v).all() for v in values.values()),
            split_identity=float(np.max(abs(values['split_identity_error_J'])))<=1e-8*scale,
            force_adjoint=float(np.max(abs(values['force_adjoint_error_J'])))<=1e-10*scale,
            grips=grip<=1e-9,momentum=float(values['linear_momentum_error'].max())<=1e-10)
        review=dict(step=end,time_s=float(state.time),elapsed_s=time.monotonic()-begun,totals_J=totals,checks=checks,
            kinetic_J=float(host['kinetic_J'][-1]),elastic_J=float(host['elastic_J'][-1]),min_J=float(values['min_J'].min()),
            max_damage=float(values['max_damage'].max()),grip_error_m=grip,
            max_cg_iterations=int(max(values['projection_cg_iterations'].max(),values['force_cg_iterations'].max())),
            max_cg_relative_residual=float(max(values['projection_cg_relative_error'].max(),values['force_cg_relative_error'].max())),
            residual_over_work=totals['residual_J']/max(abs(totals['work_J']),1e-30),
            accepted_prefix=all(checks.values()),scope='New method bounded prefix, no fracture or convergence certification')
        save_json(a.output/'review.json',review)
        print(json.dumps(review),flush=True)
        if not all(checks.values()):raise ValueError('Candidate rejected; raw candidate and original gates retained')
        validate_state(state,layout)
        save_npz(a.output/'accepted.npz',step=end,warm=np.asarray(warm),**raw)


if __name__=='__main__':main()
