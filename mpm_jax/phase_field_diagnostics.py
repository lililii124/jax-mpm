"""Discrete energy audit for the small-strain AT2 MPM step."""
from typing import NamedTuple
import jax.numpy as jnp
from .phase_field import explicit_step, spectral_split, interpolate, scatter, energies


class StepAudit(NamedTuple):
    p2g_loss: object
    g2p_loss: object
    damping_loss: object
    boundary_work: object
    stress_work: object
    kinetic_balance_error: object
    constitutive_work_defect: object
    total_balance_error: object


def audited_step(state, volume, density, basis, law, dt, velocity_mask,
                 velocity_value, damage_mask, damage_value, damping=0., update_damage=True):
    """Return the ordinary step plus transfer, damping and work diagnostics.

    Boundary work is the exact kinetic-energy change during the constraint
    projection. The constitutive defect is reported separately from all losses.
    No defect is added to fracture energy or physical heat.
    """
    updated, result, reaction = explicit_step(state, volume, density, basis, law,
        dt, velocity_mask, velocity_value, damage_mask, damage_value, damping, update_damage)
    _, sp, sn = spectral_split(state.strain, law)
    d, _ = interpolate(updated.damage, basis)
    stress = ((1-law.residual)*(1-d)**2+law.residual)[:, None, None]*sp+sn
    n = len(state.damage)
    mass = scatter((density*volume)[:, None]*basis.weight, basis, n)
    momentum = scatter((density*volume)[:, None, None]*basis.weight[..., None]*state.velocity[:, None], basis, n)
    force = scatter(-volume[:, None, None]*jnp.einsum('pij,pkj->pki', stress, basis.gradient), basis, n)
    denominator = jnp.where(mass[:, None] > 0, mass[:, None], 1)
    before = momentum/denominator
    accelerated = jnp.where(mass[:, None] > 0, (momentum+dt*force)/denominator, 0)
    damped = accelerated/(1+dt*damping)
    constrained = jnp.where(velocity_mask, velocity_value, damped)
    grid_ke = lambda v: .5*jnp.sum(mass[:, None]*v*v)
    old_energy = energies(state, volume, basis, law, density)
    new_energy = energies(updated, volume, basis, law, density)
    p2g = old_energy[0]-grid_ke(before)
    g2p = grid_ke(constrained)-new_energy[0]
    damp = grid_ke(accelerated)-grid_ke(damped)
    boundary = grid_ke(constrained)-grid_ke(damped)
    stress_work = dt*jnp.sum(force*.5*(before+accelerated))
    kinetic_error = new_energy[0]-old_energy[0]+p2g+g2p+damp-boundary-stress_work
    defect = jnp.sum(new_energy[1:]-old_energy[1:])+stress_work
    total_error = jnp.sum(new_energy-old_energy)+p2g+g2p+damp-boundary-defect
    return updated, result, reaction, StepAudit(p2g, g2p, damp, boundary,
        stress_work, kinetic_error, defect, total_error)
