"""Bounded onset instrumentation. Frozen v6 mechanics, no velocity correction."""
from functools import partial
import jax
import jax.numpy as jnp
from . import fragment_multifield_v6_mpm as frozen
from .fragment_v6_energy_probe import CHANNELS


@partial(jax.jit, static_argnames=('config', 'max_steps'))
def advance_to_traced(state, material, grid, config, target_time, *, max_steps=512):
    """Retain every accepted adaptive dt and all nine signed energy channels.

    Unused rows are zero; count is the sole slice bound. No prescribed step count.
    Returns the same five probe outputs plus a (max_steps, 10) trace.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('Positive integer max_steps required')

    def condition(carry):
        current, count, *_ = carry
        return current.valid & (current.time < target_time) & (count < max_steps)

    def body(carry):
        current, count, totals, absolute, maximum, trace = carry
        dt = jnp.minimum(frozen.stable_timestep(current, material, grid, config), target_time-current.time)
        following, d = frozen._step(current, material, grid, config, dt)
        parts = d.p2g_dissipation+d.g2p_dissipation+d.contact_compatible_energy+d.viscosity_work+d.constitutive_work_defect
        identity = d.numerical_energy_increment-parts
        values = jnp.stack((d.p2g_dissipation, d.g2p_dissipation, d.contact_compatible_energy,
                            d.viscosity_work, d.constitutive_work_defect, d.numerical_energy_increment,
                            d.boundary_energy, d.energy_residual, identity))
        trace = trace.at[count].set(jnp.concatenate((dt[None], values)))
        return following, count+1, totals+values, absolute+jnp.abs(values), jnp.maximum(maximum, jnp.abs(identity)), trace

    admitted = state._replace(valid=state.valid & jnp.isfinite(target_time) & (target_time >= state.time))
    zero = jnp.zeros(len(CHANNELS), dtype=state.x.dtype)
    current, count, totals, absolute, maximum, trace = jax.lax.while_loop(condition, body,
        (admitted, jnp.asarray(0), zero, zero, jnp.asarray(0., dtype=state.x.dtype),
         jnp.zeros((max_steps, 1+len(CHANNELS)), dtype=state.x.dtype)))
    return current._replace(valid=current.valid & (current.time == target_time)), count, totals, absolute, maximum, trace
