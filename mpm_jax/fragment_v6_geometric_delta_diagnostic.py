"""Bounded own-CFL orchestration with dual carry and unchanged physical gates."""
import numpy as np
import jax.numpy as jnp
from . import fragment_v6_geometric_delta_candidate as candidate


def advance(carry,material,grid,config,target_time,*,energy_roundoff_J,max_steps=512,save=None):
    if not 1<=max_steps<=512 or not np.isfinite(target_time) or not float(target_time)>float(carry.state.time):
        raise ValueError('Finite future endpoint and1..512 step budget required')
    rows=[];failure=None
    for index in range(max_steps):
        if float(carry.state.time)>=float(target_time):break
        if not bool(carry.state.valid):break
        limit=candidate.geometric_timestep(carry,material,grid,config)
        dt=jnp.minimum(limit,target_time-carry.state.time)
        following,diagnostics,extra=candidate.step(carry,material,grid,config,dt)
        gates=dict(original_physical=bool(following.state.valid),
            **candidate.additional_gates(following,diagnostics,extra,grid,energy_roundoff_J))
        rows.append(dict(dt_s=float(dt),diagnostics=diagnostics,geometry=extra,gates=gates))
        carry=following
        if save is not None:save(index+1,carry,diagnostics,extra)
        if not all(gates.values()):
            failure=dict(first_rejected_step_1based=index+1,failed_gates=[k for k,v in gates.items() if not v]);break
    if failure is None and float(carry.state.time)!=float(target_time):
        failure=dict(endpoint_not_reached=True,steps=len(rows))
    return carry,rows,failure


def replay(carry,material,grid,config,timesteps,*,energy_roundoff_J):
    if not 1<=len(timesteps)<=512:raise ValueError('Bounded replay trace required')
    for dt in timesteps:
        carry,diagnostics,extra=candidate.step(carry,material,grid,config,jnp.asarray(dt))
        if not bool(carry.state.valid) or not all(candidate.additional_gates(carry,diagnostics,extra,grid,energy_roundoff_J).values()):
            raise RuntimeError('Replay rejected; never advance invalid carry')
    return carry
