"""Passive ledger for the original zero-body-force single-fibre step.

Call after the unmodified solver step. Never returns or changes solver state.
The supplied boundary must be the original pure prescribed-velocity callback.
"""
from functools import partial
import jax
import jax.numpy as jnp
from mpm_jax.fibre_mpm import stencil, particle_to_grid, stored_energy
from mpm_jax.materials.isotropic_brittle import brittle_energy


def particle_kinetic(s, dx):
    return .5*jnp.sum(s.mass[:, None]*s.v*s.v)+dx**2/8*jnp.sum(s.mass[:, None, None]*s.C*s.C)


@partial(jax.jit, static_argnames=('config', 'boundary'))
def energy_terms(old, new, material, layout, config, boundary, target_velocities):
    """Return scalar per-step ledger; no body forces/gravity are supported.

    target_velocities has shape (number_of_grips,3), evaluated at new.time.
    The caller continues to apply the ORIGINAL 15% uncorrected residual gate.
    """
    if any(config.gravity):
        raise ValueError('This isolated audit is restricted to the original zero-gravity plan')
    support=stencil(old.x,old.fibre_id,layout)
    mass,momentum=particle_to_grid(old,layout,config,support)
    _,free_momentum=particle_to_grid(old._replace(tau=jnp.zeros_like(old.tau)),layout,config,support)
    denominator=jnp.where(mass>0,mass,1)[:,None]
    before=momentum/denominator
    free=free_momentum/denominator
    prescribed=boundary(before,mass,old,support,layout,new.time)
    after=prescribed.velocity
    grid_kinetic=lambda v:.5*jnp.sum(mass[:,None]*v*v)
    kp0=particle_kinetic(old,layout.dx);kp1=particle_kinetic(new,layout.dx)
    kg0=grid_kinetic(free);kgforce=grid_kinetic(before);kgafter=grid_kinetic(after)
    u0=stored_energy(old,material);u1=stored_energy(new,material)
    fixed=jnp.sum(new.volume*brittle_energy(new.F,old.history,material))
    undamaged=new.history._replace(damage=jnp.zeros_like(new.history.damage),failed=jnp.zeros_like(new.history.failed))
    original_damage=jnp.sum(new.volume*brittle_energy(new.F,undamaged,material)*(new.history.damage-old.history.damage))
    fixture=jnp.sum(prescribed.group_impulses*target_velocities)
    projection=.5*jnp.sum(mass[:,None]*(after-before)**2)
    result={'p2g_loss_J':kp0-kg0,'g2p_loss_J':kgafter-kp1,
            'grid_projection_J':projection,
            'force_elastic_integration_defect_J':-(kgforce-kg0+fixed-u0),
            'fixture_work_J':fixture,'grid_boundary_work_J':kgafter-kgforce,
            'damage_loss_J':fixed-u1,'original_damage_loss_J':original_damage,
            'kinetic_energy_J':kp1,'elastic_energy_J':u1,'kinetic_increment_J':kp1-kp0,
            'elastic_increment_J':u1-u0,
            'projection_work_identity_error_J':fixture-(kgafter-kgforce)-projection}
    residual=fixture-(kp1-kp0)-(u1-u0)-(fixed-u1)
    split=sum(result[k] for k in ('p2g_loss_J','g2p_loss_J','grid_projection_J','force_elastic_integration_defect_J'))
    result.update(original_residual_J=residual,split_sum_J=split,
                  identity_error_J=residual-split,
                  clock_error_s=new.time-old.time-config.dt)
    return result
