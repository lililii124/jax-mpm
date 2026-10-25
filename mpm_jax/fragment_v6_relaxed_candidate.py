"""Unadmitted conservative complete-map relaxation candidate.

Frozen original mechanics/gates; only G2P and extra diagnostic return differ.
The relaxation time is frozen from original125ns cell crossing, never fitted.
"""
from .fragment_multifield_v6_mpm import *
from .fragment_multifield_v6_mpm import _gather
from .fragment_v6_relaxed_transfer import transfer_relaxed

def _step(state, material, grid, config, dt, *, relaxation_time):
    support = stencil(state.x, state.body_id, grid.layout)
    # A swept candidate allocation is accepted only while every occupied shared
    # node pair is represented. Missing possible contact stops advancement.
    preliminary_mass,_,_=transfer_to_grid(state,support,grid)
    occupied=preliminary_mass>0
    count=jnp.zeros(grid.layout.node_count,dtype=jnp.int32).at[grid.global_node_id].add(occupied.astype(jnp.int32))
    a,b=grid.pairs[:,0],grid.pairs[:,1]
    represented=jnp.zeros(grid.layout.node_count,dtype=jnp.int32).at[a].add((occupied[a]&occupied[b]).astype(jnp.int32)).at[b].add((occupied[a]&occupied[b]).astype(jnp.int32))
    coverage=jnp.all(~occupied|(represented==count[grid.global_node_id]-1))
    old_thermo = thermodynamics(state, material)
    J = jnp.linalg.det(state.F)
    q = artificial_viscosity(old_thermo.density, old_thermo.sound_speed,
        jnp.trace(state.C, axis1=-2, axis2=-1), grid.layout.dx, material)
    effective_pressure = jnp.where(old_thermo.pressure < 0, (1-state.damage)*old_thermo.pressure, old_thermo.pressure)
    tau = state.tau-J[:, None, None]*(effective_pressure+q)[:, None, None]*jnp.eye(3)
    mass, old_velocity, p2g_loss = transfer_to_grid(state, support, grid)
    kicked, _ = stress_kick(old_velocity, mass, state.volume, tau, support, grid, dt)
    before_contact, boundary0, _ = clamp_grid(kicked, mass, grid.clamped)
    normals=contact_normals(state,mass,support,grid)
    projected, projection_loss, contact_impulses, contact_valid, maximum_closing, complementarity, contact_sweeps, contact_lambda = paired_contact(
        before_contact, mass, normals, grid.pairs, grid.clamped,config.contact_iterations,config.contact_tolerance,grid.contact_colors,grid.contact_color_active,state.contact_lambda*dt/jnp.where(state.contact_cache_dt>0,state.contact_cache_dt,dt))
    velocity, boundary1, _ = clamp_grid(projected, mass, grid.clamped)
    a,b=grid.pairs[:,0],grid.pairs[:,1]
    normal_unit=normals/jnp.maximum(jnp.linalg.norm(normals,axis=-1),1e-30)[:,None]
    closing=jnp.where((mass[a]>0)&(mass[b]>0),jnp.sum((velocity[a]-velocity[b])*normal_unit,axis=-1),0.)
    maximum_closing=jnp.maximum(jnp.max(closing,initial=0.),0.)
    contact_boundary_reaction=mass[:,None]*(projected-before_contact)-contact_impulses
    boundary_impulses = mass[:, None]*(before_contact-kicked+velocity-projected)+contact_boundary_reaction
    midpoint_velocity = .5*(old_velocity+velocity)
    boundary = -jnp.sum(boundary_impulses*midpoint_velocity)
    boundary_projection = boundary0+boundary1
    boundary_correction = boundary-boundary_projection
    impulse = boundary_impulses.sum(axis=0)
    allocation, physical_by_body, correction_by_body, body_work = compatible_contact_work(
        old_velocity, velocity, contact_impulses, mass, grid.pairs, projection_loss, grid.node_body)
    work = stress_work(old_velocity, velocity, state.volume, tau, support, grid, dt)
    specific_contact = jnp.sum(support.weights*_gather(allocation/jnp.where(mass > 0, mass, 1.), support), axis=-1)
    v, C, g2p_loss, transfer_diagnostics = transfer_relaxed(state, old_velocity, velocity, support, grid, dt=dt, relaxation_time=relaxation_time)
    F, rotation, strain = cayley_kinematics(state.F, C, dt)
    new_J = jnp.linalg.det(F)
    old_heat = state.internal_energy-state.cold_work-state.fracture_energy-elastic_energy(state.tau, material, state.damage)-damage_volume_energy(J, state.damage, material)
    if config.strength:
        factor=1-state.damage
        undamaged_tau=state.tau/jnp.maximum(factor,1e-12)[:,None,None]
        returned = deviatoric_return(undamaged_tau, state.plastic_strain, strain, rotation,
                                    jnp.where(factor>0,new_J,1.), dt, old_thermo.temperature, material)
        new_tau = returned.kirchhoff_stress*factor[:,None,None]
        ep=jnp.where(factor>0,returned.plastic_strain,state.plastic_strain)
        cold = state.cold_work+factor*returned.specific_cold_work
        plastic_heat = factor*returned.specific_plastic_heat
        material_valid = jnp.all(returned.valid|(factor==0))
    else:
        new_tau, ep, cold = jnp.zeros_like(state.tau), state.plastic_strain, state.cold_work
        plastic_heat = jnp.zeros_like(old_heat)
        material_valid = jnp.all(state.tau == 0) & jnp.all(state.cold_work == 0)
    # Irreversible strain-driven damage. Parameters are hypotheses, not calibration.
    proposed = jnp.clip((ep-config.damage_onset_strain)/(config.damage_failure_strain-config.damage_onset_strain), 0., 1.)
    damage = jnp.maximum(state.damage, proposed)
    retained = (1-damage)/jnp.maximum(1-state.damage, 1e-12)
    before_release = elastic_energy(new_tau, material, state.damage)
    new_tau = new_tau*retained[:, None, None]
    released = jnp.maximum(before_release-elastic_energy(new_tau, material, damage), 0.)
    released += jnp.maximum(damage_volume_energy(new_J,state.damage,material)-damage_volume_energy(new_J,damage,material),0.)
    fracture_energy = state.fracture_energy+released
    internal = (damage_volume_energy(new_J, damage, material)+old_heat+plastic_heat+cold+fracture_energy
                +elastic_energy(new_tau, material, damage))
    # Keep every noncalorimetric energy term visible as a signed numerical sink.
    compatible_internal = state.internal_energy+p2g_loss+g2p_loss+specific_contact+work/state.mass
    numerical_increment = jnp.sum(state.mass*(compatible_internal-internal))
    viscous_tau = -J[:, None, None]*q[:, None, None]*jnp.eye(3)
    viscosity_work = stress_work(old_velocity, velocity, state.volume,
                                 viscous_tau, support, grid, dt).sum()
    constitutive_defect = (work.sum()-viscosity_work
                          -jnp.sum(state.mass*(internal-state.internal_energy)))
    x = state.x+dt*v
    # x and F both use the same end-contact velocity field: first-order motion.
    current = ImpactState(x, v, F, C, state.mass, state.volume, state.body_id, internal,
        new_tau, ep, cold, state.time+dt, state.boundary_reservoir+boundary,
        state.numerical_reservoir+numerical_increment,
        state.boundary_impulse+impulse, state.boundary_projection_loss+boundary_projection,
        state.boundary_coupling_work+boundary_correction,
        state.contact_projection_loss+physical_by_body,
        state.contact_coupling_work+correction_by_body, state.initial_entropy,
        state.minimum_entropy, state.negative_entropy_increments,
        state.ever_phase_extrapolated, state.ever_tension_extrapolated,
        state.maximum_temperature, state.minimum_J, state.maximum_J, damage, fracture_energy, jnp.maximum(state.maximum_contact_closing_speed,maximum_closing), jnp.maximum(state.maximum_contact_complementarity_residual,complementarity), jnp.maximum(state.maximum_contact_iterations,contact_sweeps.astype(state.x.dtype)), contact_lambda, dt, state.valid)
    thermo = thermodynamics(current, material)
    entropy_increment = thermo.entropy-old_thermo.entropy
    current = current._replace(minimum_entropy=jnp.minimum(state.minimum_entropy, thermo.entropy),
        negative_entropy_increments=state.negative_entropy_increments+jnp.minimum(entropy_increment, 0.),
        ever_phase_extrapolated=state.ever_phase_extrapolated | thermo.phase_extrapolated,
        ever_tension_extrapolated=state.ever_tension_extrapolated | thermo.tension_extrapolated,
        maximum_temperature=jnp.maximum(state.maximum_temperature, thermo.temperature),
        minimum_J=jnp.minimum(state.minimum_J, new_J), maximum_J=jnp.maximum(state.maximum_J, new_J))
    limit = stable_timestep(state, material, grid, config)
    residual = energy_ledger(current, grid).sum()-energy_ledger(state, grid).sum()
    physical_residual = residual-numerical_increment
    momentum_residual = jnp.sum(state.mass[:, None]*(v-state.v), axis=0)-impulse
    energy_scale = jnp.maximum(jnp.abs(energy_ledger(state, grid)).sum()
                              +jnp.abs(energy_ledger(current, grid)).sum(), 1e-30)
    momentum_scale = jnp.maximum(jnp.sum(state.mass*(jnp.linalg.norm(state.v, axis=-1)
                              +jnp.linalg.norm(v, axis=-1)))+jnp.linalg.norm(impulse), 1e-30)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in current]))
    valid = (state.valid & support.valid & stencil(x, state.body_id, grid.layout).valid
             & jnp.all(old_thermo.valid) & jnp.all(thermo.valid) & contact_valid & material_valid
             & coverage & finite & (dt > 0) & (dt <= limit*(1+1e-12))
             & (jnp.abs(residual) <= 1e-10*energy_scale)
             & (jnp.linalg.norm(momentum_residual) <= 1e-10*momentum_scale))
    current = current._replace(valid=valid)
    diagnostics = ImpactDiagnostics(jnp.sum(state.mass*p2g_loss), jnp.sum(state.mass*g2p_loss),
        physical_by_body, correction_by_body, allocation.sum(), body_work,
        work.sum(), boundary, residual, physical_residual, numerical_increment, viscosity_work, constitutive_defect, momentum_residual, limit,
        q.max(), new_J.min(), thermo.temperature.min(), thermo.entropy.min(),
        entropy_increment.min(), jnp.mean(entropy_increment < 0),
        (current.minimum_entropy-current.initial_entropy).min(),
        current.negative_entropy_increments.min(), jnp.any(thermo.phase_extrapolated),
        jnp.any(thermo.tension_extrapolated), valid)
    return current, diagnostics, transfer_diagnostics



@partial(jax.jit, static_argnames=('config','relaxation_time'))
def step(state,material,grid,config,dt,*,relaxation_time):
    return _step(state,material,grid,config,dt,relaxation_time=relaxation_time)
