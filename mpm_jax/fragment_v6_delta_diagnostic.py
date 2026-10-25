"""Paired fixed-schedule bounded diagnostic; no production entrypoint."""
from functools import partial
import jax
import jax.numpy as jnp
from . import fragment_multifield_v6_mpm as original
from . import fragment_v6_delta_candidate as candidate


@partial(jax.jit, static_argnames=('config',))
def candidate_fixed_schedule(state, material, grid, config, timesteps):
    """Independent uninstrumented carry for bitwise instrumented equivalence."""
    def one(current, dt):
        following, _ = candidate._step(current, material, grid, config, dt)
        return following, None
    return jax.lax.scan(one, state, timesteps)[0]


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
