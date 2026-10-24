"""Independent diagnostic wrapper around frozen v6 steps; no constitutive changes.

Importing this module does not change v6.advance_to. Use only in a separately
admitted diagnostic run, never as a silent replacement in a frozen trajectory.
"""
from functools import partial
import jax
import jax.numpy as jnp
from . import fragment_multifield_v6_mpm as frozen

CHANNELS = ('p2g', 'g2p', 'contact_allocation', 'viscosity', 'constitutive_defect',
            'numerical_increment', 'boundary', 'ledger_residual', 'identity_residual')


@partial(jax.jit, static_argnames=('config', 'max_steps'))
def advance_to_diagnosed(state, material, grid, config, target_time, *, max_steps=10000):
    """Same step/timestep sequence, plus signed and absolute diagnostic totals.

    Return state,count,totals,absolute_totals,max_absolute_step_identity_residual.
    Signed totals alone can hide cancellation; retain both for every epoch.
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('Positive integer max_steps required')

    def condition(carry):
        current, count, _, _, _ = carry
        return current.valid & (current.time < target_time) & (count < max_steps)

    def body(carry):
        current, count, totals, absolute, maximum = carry
        dt = jnp.minimum(frozen.stable_timestep(current, material, grid, config), target_time-current.time)
        following, d = frozen._step(current, material, grid, config, dt)
        parts = d.p2g_dissipation+d.g2p_dissipation+d.contact_compatible_energy+d.viscosity_work+d.constitutive_work_defect
        identity = d.numerical_energy_increment-parts
        values = jnp.stack((d.p2g_dissipation, d.g2p_dissipation, d.contact_compatible_energy,
                            d.viscosity_work, d.constitutive_work_defect, d.numerical_energy_increment,
                            d.boundary_energy, d.energy_residual, identity))
        return following, count+1, totals+values, absolute+jnp.abs(values), jnp.maximum(maximum, jnp.abs(identity))

    admitted = state._replace(valid=state.valid & jnp.isfinite(target_time) & (target_time >= state.time))
    zero = jnp.zeros(len(CHANNELS), dtype=state.x.dtype)
    current, count, totals, absolute, maximum = jax.lax.while_loop(condition, body,
        (admitted, jnp.asarray(0), zero, zero, jnp.asarray(0., dtype=state.x.dtype)))
    return current._replace(valid=current.valid & (current.time == target_time)), count, totals, absolute, maximum
