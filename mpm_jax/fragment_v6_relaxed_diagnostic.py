"""Bounded host orchestration for the conservative relaxation candidate."""
import jax.numpy as jnp
from . import fragment_multifield_v6_mpm as original
from . import fragment_v6_relaxed_candidate as candidate


def transfer_gates(d,state,grid):
    eps=jnp.finfo(state.x.dtype).eps
    angular=bool(jnp.all(jnp.linalg.norm(d.angular_defect_by_field,axis=-1)<=64*eps*jnp.maximum(d.angular_scale_by_field,1e-30)))
    energy_scale=jnp.maximum(jnp.abs(original.energy_ledger(state,grid)).sum(),1e-30)
    return dict(angular=angular,convex_identity=bool(jnp.abs(d.convex_energy_identity_residual_J)<=64*eps*energy_scale),
                scalar_blend=bool(jnp.isfinite(d.alpha)&(d.alpha>=0)&(d.alpha<=1)),nonnegative_blend_variance=bool(d.blending_variance_J>=0))


def compare_shared(state,material,grid,config,target_time,*,relaxation_time,max_steps=512):
    if not 1<=max_steps<=512 or not bool(jnp.isfinite(target_time)) or not float(target_time)>float(state.time): raise ValueError('Bounded finite future target required')
    baseline=changed=state;history=[];failure=None
    for index in range(max_steps):
        if float(baseline.time)>=float(target_time): break
        if not bool(baseline.valid) or not bool(changed.valid): break
        if float(baseline.time)!=float(changed.time): raise RuntimeError('Shared times diverged')
        dt=jnp.minimum(jnp.minimum(original.stable_timestep(baseline,material,grid,config),original.stable_timestep(changed,material,grid,config)),target_time-baseline.time)
        baseline,bd=original.step(baseline,material,grid,config,dt)
        changed,cd,extra=candidate.step(changed,material,grid,config,dt,relaxation_time=relaxation_time)
        gates=dict(baseline_valid=bool(baseline.valid),candidate_valid=bool(changed.valid),**transfer_gates(extra,changed,grid))
        history.append(dict(dt_s=dt,baseline=bd,candidate=cd,transfer=extra,gates=gates))
        if not all(gates.values()):
            failure=dict(first_rejected_step_1based=index+1,failed_gates=[k for k,v in gates.items() if not v]);break
    return baseline,changed,history,failure


def replay(state,material,grid,config,timesteps,*,relaxation_time):
    if not 1<=len(timesteps)<=512: raise ValueError('Bounded trace required')
    current=state
    for dt in timesteps:
        if not bool(current.valid): break
        current,_,extra=candidate.step(current,material,grid,config,dt,relaxation_time=relaxation_time)
        if not all(transfer_gates(extra,current,grid).values()):break
    return current
