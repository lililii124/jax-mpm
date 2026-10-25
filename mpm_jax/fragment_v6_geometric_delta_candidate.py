"""Unadmitted executable DELTA candidate with a separate physical gradient.

Every baseline material/contact/coverage/energy/momentum gate remains in _step.
The extra L carry is not a hidden energy reservoir and does not replace C in
APIC momentum or kinetic energy. No impact run is admitted by importing this.
"""
from typing import NamedTuple
from .fragment_multifield_v6_mpm import *
from .fragment_multifield_v6_mpm import _gather
from . import fragment_multifield_v6_mpm as original
from .fragment_v6_delta_transfer import transfer_delta
from .fragment_v6_geometric_kinematics import geometric_kinematics
from .fibre_mpm import grid_coordinates


class GeometricCarry(NamedTuple):
    state: ImpactState
    L_physical: object


class GeometryDiagnostics(NamedTuple):
    valid: object
    minimum_map_J: object
    virtual_work_identity_J: object
    grid_physical_work_mismatch_J: object
    material_work_remainder_J: object
    work_decomposition_identity_J: object
    angular_momentum_residual: object
    transfer_angular_identity_residual: object
    angular_scale: object
    boundary_torque: object
    contact_torque: object
    stress_torque: object
    transfer_angular_identity_by_field: object
    angular_scale_by_field: object


def initialize(state):
    # This extension preserves all authenticated original arrays. The original
    # producer used C for its physical gradient, so the initial L uses those bytes.
    return GeometricCarry(state,state.C)


def geometric_timestep(carry,material,grid,config):
    state,L=carry
    thermo=thermodynamics(state,material)
    compression=jnp.maximum(-jnp.trace(L,axis1=-2,axis2=-1),0.)
    viscous=material.viscosity_linear*thermo.sound_speed+2*material.viscosity_quadratic*grid.layout.dx*compression
    affine=1.5*jnp.sqrt(3.)*grid.layout.dx*jnp.linalg.norm(state.C,axis=(-2,-1))
    limit=config.cfl*grid.layout.dx/jnp.max(thermo.longitudinal_speed+jnp.linalg.norm(state.v,axis=-1)+affine+viscous)
    return jnp.minimum(original.stable_timestep(state,material,grid,config),limit)


def _step(state, L_physical, material, grid, config, dt):
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
        jnp.trace(L_physical, axis1=-2, axis2=-1), grid.layout.dx, material)
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
    v, C, g2p_loss = transfer_delta(state, old_velocity, velocity, support, grid)
    _,increment_gradient=grid_to_particle(velocity-old_velocity,support,grid.layout)
    geometry=geometric_kinematics(state.F,L_physical,increment_gradient,dt)
    F,rotation,strain=geometry.F,geometry.rotation,geometry.strain
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
    limit = geometric_timestep(GeometricCarry(state,L_physical), material, grid, config)
    residual = energy_ledger(current, grid).sum()-energy_ledger(state, grid).sum()
    physical_residual = residual-numerical_increment
    momentum_residual = jnp.sum(state.mass[:, None]*(v-state.v), axis=0)-impulse
    energy_scale = jnp.maximum(jnp.abs(energy_ledger(state, grid)).sum()
                              +jnp.abs(energy_ledger(current, grid)).sum(), 1e-30)
    momentum_scale = jnp.maximum(jnp.sum(state.mass*(jnp.linalg.norm(state.v, axis=-1)
                              +jnp.linalg.norm(v, axis=-1)))+jnp.linalg.norm(impulse), 1e-30)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in current]))
    valid = (state.valid & geometry.valid & support.valid & stencil(x, state.body_id, grid.layout).valid
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
    coords,node_field=grid_coordinates(grid.layout,jnp.arange(grid.layout.node_count))
    gradient=4/grid.layout.dx**2*support.weights[...,None]*support.dpos
    local_force=-state.volume[:,None,None]*jnp.einsum('pij,pkj->pki',tau,gradient)
    grid_force=jnp.zeros_like(velocity).at[support.indices].add(local_force,mode='drop')
    virtual_identity=dt*jnp.sum(grid_force*midpoint_velocity)+work.sum()
    L_star=L_physical+increment_gradient
    midpoint_map=jnp.eye(3,dtype=state.x.dtype)+.5*dt*L_star
    midpoint_L=jnp.swapaxes(jnp.linalg.solve(jnp.swapaxes(midpoint_map,-1,-2),jnp.swapaxes(L_star,-1,-2)),-1,-2)
    physical_work=dt*jnp.sum(state.volume[:,None,None]*(tau-viscous_tau)*midpoint_L)
    grid_physical_mismatch=work.sum()-viscosity_work-physical_work
    material_remainder=physical_work-jnp.sum(state.mass*(internal-state.internal_energy))
    decomposition=constitutive_defect-grid_physical_mismatch-material_remainder
    spin=lambda cc:jnp.stack((cc[:,2,1]-cc[:,1,2],cc[:,0,2]-cc[:,2,0],cc[:,1,0]-cc[:,0,1]),axis=-1)
    angular_before=state.mass[:,None]*(jnp.cross(state.x,state.v)+grid.layout.dx**2/4*spin(state.C))
    angular_after=state.mass[:,None]*(jnp.cross(current.x,current.v)+grid.layout.dx**2/4*spin(current.C))
    boundary_torque=jnp.sum(jnp.cross(coords,boundary_impulses),axis=0)
    contact_torque=jnp.sum(jnp.cross(coords,contact_impulses),axis=0)
    stress_torque=dt*jnp.sum(jnp.cross(coords,grid_force),axis=0)
    angular_change=jnp.sum(angular_after-angular_before,axis=0)
    angular_scale=jnp.sum(jnp.linalg.norm(angular_before,axis=-1)+jnp.linalg.norm(angular_after,axis=-1))+jnp.sum(jnp.linalg.norm(jnp.cross(coords,boundary_impulses),axis=-1))+jnp.sum(jnp.linalg.norm(jnp.cross(coords,contact_impulses),axis=-1))+dt*jnp.sum(jnp.linalg.norm(jnp.cross(coords,grid_force),axis=-1))
    grid_angular_impulse=jnp.cross(coords,mass[:,None]*(velocity-old_velocity))
    particle_fields=lambda a:jnp.zeros((grid.layout.fibre_count,)+a.shape[1:],dtype=a.dtype).at[state.body_id].add(a)
    node_fields=lambda a:jnp.zeros((grid.layout.fibre_count,)+a.shape[1:],dtype=a.dtype).at[node_field].add(a)
    angular_by_field=particle_fields(angular_after-angular_before)-node_fields(grid_angular_impulse)
    scale_by_field=particle_fields(jnp.linalg.norm(angular_before,axis=-1)+jnp.linalg.norm(angular_after,axis=-1))+node_fields(jnp.linalg.norm(grid_angular_impulse,axis=-1))
    extra=GeometryDiagnostics(geometry.valid,jnp.min(jnp.linalg.det(jnp.eye(3)+dt*L_star)),virtual_identity,
        grid_physical_mismatch,material_remainder,decomposition,angular_change-boundary_torque,
        angular_change-boundary_torque-contact_torque-stress_torque,angular_scale,boundary_torque,contact_torque,stress_torque,angular_by_field,scale_by_field)
    return GeometricCarry(current,geometry.L),diagnostics,extra


@partial(jax.jit,static_argnames=('config',))
def step(carry,material,grid,config,dt):
    return _step(carry.state,carry.L_physical,material,grid,config,dt)


def save_checkpoint(path,carry):
    """Original C and explicit C_transfer alias must agree byte for byte."""
    import numpy as np
    values={name:np.asarray(value) for name,value in carry.state._asdict().items()}
    values['C_transfer']=values['C'].copy()
    values['L_physical']=np.asarray(carry.L_physical)
    np.savez(path,**values)


def load_checkpoint(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:
        required=set(ImpactState._fields)|{'C_transfer','L_physical'}
        if set(saved.files)!=required:raise ValueError('Exact dual-carry array closure required')
        if saved['C'].dtype!=saved['C_transfer'].dtype or saved['C'].tobytes()!=saved['C_transfer'].tobytes():
            raise ValueError('C_transfer alias differs')
        if saved['L_physical'].shape!=saved['C'].shape or not np.isfinite(saved['L_physical']).all():
            raise ValueError('Invalid physical gradient carry')
        state=ImpactState(*(jnp.asarray(saved[name]) for name in ImpactState._fields))
        return GeometricCarry(state,jnp.asarray(saved['L_physical']))


def additional_gates(carry,diagnostics,extra,grid,energy_roundoff_J):
    eps=jnp.finfo(carry.state.x.dtype).eps
    scale=jnp.maximum(jnp.sum(jnp.abs(energy_ledger(carry.state,grid))),1e-30)
    limit=jnp.minimum(energy_roundoff_J,64*eps*scale)
    angular_limit=64*eps*jnp.maximum(extra.angular_scale,1e-30)
    return dict(geometry=bool(extra.valid),
        virtual_work=bool(jnp.abs(extra.virtual_work_identity_J)<=limit),
        work_decomposition=bool(jnp.abs(extra.work_decomposition_identity_J)<=limit),
        angular_momentum=bool(jnp.linalg.norm(extra.angular_momentum_residual)<=angular_limit),
        transfer_angular_identity=bool(jnp.linalg.norm(extra.transfer_angular_identity_residual)<=angular_limit),
        per_field_transfer_angular_identity=bool(jnp.all(jnp.linalg.norm(extra.transfer_angular_identity_by_field,axis=-1)<=64*eps*jnp.maximum(extra.angular_scale_by_field,1e-30))),
        energy_identity=bool(jnp.abs(diagnostics.energy_residual)<=energy_roundoff_J))
