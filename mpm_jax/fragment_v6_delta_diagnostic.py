"""Paired fixed-schedule bounded diagnostic; no production entrypoint."""
from functools import partial
import jax
import jax.numpy as jnp
from . import fragment_multifield_v6_mpm as original
from . import fragment_v6_delta_candidate as candidate


def candidate_fixed_schedule(state, material, grid, config, timesteps):
    """Independent replay through the exact same compiled candidate.step API.

    Discarding its diagnostics happens on the host, outside the executable.
    Never compile a wrapper that prunes diagnostics and changes fusion.
    """
    if timesteps.ndim!=1 or not 1<=len(timesteps)<=512: raise ValueError('Bounded1..512 steps required')
    current=state
    for dt in timesteps:
        if not bool(current.valid): break
        current,_=candidate.step(current,material,grid,config,dt)
    return current


@partial(jax.jit, static_argnames=('config',))
def original_fixed_schedule(state, material, grid, config, timesteps):
    def one(current, dt):
        following, diagnostics=original._step(current,material,grid,config,dt)
        return following,diagnostics
    return jax.lax.scan(one,state,timesteps)


def compare_shared_adaptive(state,material,grid,config,target_time,*,max_steps=512):
    """Each real step uses min(original CFL,candidate CFL,remaining time).

    Stops immediately after either state rejects an attempted update. No invalid
    carry is advanced again. Fixed caps and every original gate remain intact.
    """
    if not 1<=max_steps<=512: raise ValueError('Bounded1..512 steps required')
    old=delta=state
    timesteps=[];bt=[];ct=[]
    if not bool(jnp.isfinite(target_time)) or not float(target_time)>float(state.time): raise ValueError('Finite future target required')
    while bool(old.valid) and bool(delta.valid) and float(old.time)<float(target_time) and len(timesteps)<max_steps:
        if float(old.time)!=float(delta.time): raise RuntimeError('Shared real step times diverged')
        dt=jnp.minimum(jnp.minimum(original.stable_timestep(old,material,grid,config),original.stable_timestep(delta,material,grid,config)),target_time-old.time)
        # Both calls return full diagnostics. Host collection cannot alter the
        # numerical graph of either compiled production step executable.
        old,bd=original.step(old,material,grid,config,dt)
        delta,cd=candidate.step(delta,material,grid,config,dt)
        timesteps.append(dt);bt.append(bd);ct.append(cd)
    if not timesteps: raise ValueError('No valid initial pair to advance')
    stack=lambda values:jax.tree.map(lambda *rows:jnp.stack(rows),*values)
    return old,delta,len(timesteps),jnp.stack(timesteps),stack(bt),stack(ct)


@partial(jax.jit, static_argnames=('config',))
def compare_fixed_schedule(state, material, grid, config, timesteps):
    """Run both operators at the supplied original schedule, <=512 steps.

    A candidate CFL failure remains a failure; never shorten its timestep to
    make a paired cell pass. Each cell retains ALL original validity gates.
    Return final carries and both complete diagnostics at every scheduled step.
    The caller must pin/hash the original adaptive trace and validate its exact
    125--135ns endpoints before supplying its timestep array.
    """
    if timesteps.ndim != 1 or not 1 <= timesteps.shape[0] <= 512:
        raise ValueError('Require bounded 1..512 step original schedule')
    def one(pair, dt):
        old, delta = pair
        baseline, bd = original._step(old, material, grid, config, dt)
        changed, cd = candidate._step(delta, material, grid, config, dt)
        return (baseline, changed), (bd, cd)
    admitted=state._replace(valid=state.valid & jnp.all(jnp.isfinite(timesteps)) & jnp.all(timesteps>0))
    return jax.lax.scan(one, (admitted, admitted), timesteps)
