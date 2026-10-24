"""Two-body explicit APIC with compatible work; not an entropy-stable scheme."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .contact import contact_impulse
from .fibre_mpm import PackedLayout, build_layout, grid_to_particle, stencil
from .materials.shock_metal import artificial_viscosity, deviatoric_return, eos_state


@partial(jax.tree_util.register_dataclass,
         data_fields=('layout', 'pairs', 'clamped'), meta_fields=())
@dataclass(frozen=True)
class ImpactGrid:
    layout: PackedLayout
    pairs: jax.Array
    clamped: jax.Array


@dataclass(frozen=True)
class ImpactConfig:
    cfl: float = .1
    strength: bool = True

    def __post_init__(self):
        if not np.isfinite(self.cfl) or not 0 < self.cfl <= .2:
            raise ValueError('cfl must lie in (0,.2]')
        if not isinstance(self.strength, bool):
            raise ValueError('strength must be boolean')


class ImpactState(NamedTuple):
    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    mass: jax.Array
    volume: jax.Array
    body_id: jax.Array
    internal_energy: jax.Array
    tau: jax.Array
    plastic_strain: jax.Array
    cold_work: jax.Array
    time: jax.Array
    boundary_reservoir: jax.Array
    boundary_impulse: jax.Array
    boundary_projection_loss: jax.Array
    boundary_coupling_work: jax.Array
    contact_projection_loss: jax.Array
    contact_coupling_work: jax.Array
    initial_entropy: jax.Array
    minimum_entropy: jax.Array
    negative_entropy_increments: jax.Array
    ever_phase_extrapolated: jax.Array
    ever_tension_extrapolated: jax.Array
    maximum_temperature: jax.Array
    minimum_J: jax.Array
    maximum_J: jax.Array
    valid: jax.Array


class ImpactDiagnostics(NamedTuple):
    p2g_heat: jax.Array
    g2p_heat: jax.Array
    contact_projection_loss: jax.Array
    contact_coupling_work: jax.Array
    contact_compatible_energy: jax.Array
    contact_body_work: jax.Array
    stress_work: jax.Array
    boundary_energy: jax.Array
    energy_residual: jax.Array
    momentum_residual: jax.Array
    dt_limit: jax.Array
    maximum_viscosity: jax.Array
    minimum_J: jax.Array
    minimum_temperature: jax.Array
    minimum_entropy: jax.Array
    minimum_entropy_increment: jax.Array
    entropy_decrease_fraction: jax.Array
    minimum_entropy_change: jax.Array
    minimum_cumulative_negative_entropy: jax.Array
    phase_extrapolated: jax.Array
    tension_extrapolated: jax.Array
    valid: jax.Array


def matching_nodes(layout):
    """One pair for each shared global node of the two dense packed grids."""
    if layout.fibre_count != 2 or layout.node_map is not None:
        raise ValueError('Impact contact requires exactly two dense packed grids')
    lower, shape, stride, offsets = map(np.asarray,
        (layout.lower, layout.shape, layout.strides, layout.offsets))
    first = lower.max(axis=0)
    last = (lower+shape-1).min(axis=0)
    if np.any(last < first):
        return jnp.empty((0, 2), dtype=jnp.int32)
    indices = np.indices(tuple(last-first+1), dtype=np.int64).reshape(3, -1).T+first
    pairs = np.stack([offsets[b]+np.sum((indices-lower[b])*stride[b], axis=1)
                      for b in range(2)], axis=1)
    return jnp.asarray(pairs, dtype=jnp.int32)


def build_grid(x, body_id, *, dx, padding_cells=6, reference_bounds=None,
               origin=(0., 0., 0.), max_nodes=30_000_000, clamped=None):
    layout = build_layout(x, body_id, dx=dx, padding_cells=padding_cells,
                          reference_bounds=reference_bounds, origin=origin,
                          max_nodes=max_nodes)
    pairs = matching_nodes(layout)
    mask = np.zeros((layout.node_count, 3), dtype=bool) if clamped is None else np.asarray(clamped)
    if mask.shape != (layout.node_count, 3) or mask.dtype != np.dtype(bool):
        raise ValueError('clamped must be a boolean (packed node count,3) array')
    return ImpactGrid(layout, pairs, jnp.asarray(mask))


def elastic_energy(tau, material):
    return jnp.sum(tau*tau, axis=(-2, -1))/(4*material.shear_modulus*material.density)


def thermodynamics(state, material):
    J = jnp.linalg.det(state.F)
    eos_energy = state.internal_energy-state.cold_work-elastic_energy(state.tau, material)
    return eos_state(J, eos_energy, material)


def energy_ledger(state, grid):
    """Joules: translation, APIC affine motion, internal energy, boundary reservoir."""
    translation = .5*jnp.sum(state.mass*jnp.sum(state.v**2, axis=-1))
    affine = grid.layout.dx**2/8*jnp.sum(state.mass*jnp.sum(state.C**2, axis=(-2, -1)))
    internal = jnp.sum(state.mass*state.internal_energy)
    return jnp.stack((translation, affine, internal, state.boundary_reservoir))


def make_state(x, volume, body_id, material, grid, *, v=None, F=None, C=None,
               eos_energy=None, tau=None, plastic_strain=None, cold_work=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if x.shape != (n, 3) or n == 0:
        raise ValueError('x must be a nonempty (N,3) array')
    volume = np.asarray(volume, dtype=np.float64)
    ids = np.asarray(body_id)
    if volume.shape != (n,) or np.any(volume <= 0) or not np.isfinite(volume).all():
        raise ValueError('Reference volume must be finite and positive')
    if ids.shape != (n,) or ids.dtype.kind not in 'iu' or not np.array_equal(np.unique(ids), [0, 1]):
        raise ValueError('Both body IDs 0 and 1 must be present')
    if grid.layout.fibre_count != 2:
        raise ValueError('State and grid must have two bodies')

    def array(value, shape, default=0.):
        result = np.full(shape, default) if value is None else np.asarray(value, dtype=np.float64)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f'Initial field requires finite shape {shape}')
        return jnp.asarray(result, dtype=jnp.float64)

    x = array(x, (n, 3))
    if x.dtype != jnp.float64:
        raise ValueError('Enable jax_enable_x64 before initializing impact MPM')
    deformation = array(np.broadcast_to(np.eye(3), (n, 3, 3)) if F is None else F, (n, 3, 3))
    stress = array(tau, (n, 3, 3))
    cold = array(cold_work, (n,))
    ep = array(plastic_strain, (n,))
    if np.any(np.asarray(cold) < 0) or np.any(np.asarray(ep) < 0):
        raise ValueError('Plastic history and stored cold work must be nonnegative')
    internal = array(eos_energy, (n,))+cold+elastic_energy(stress, material)
    initial_J = jnp.linalg.det(deformation)
    initial_thermo = eos_state(initial_J, internal-cold-elastic_energy(stress, material), material)
    initial_entropy = initial_thermo.entropy
    state = ImpactState(x, array(v, (n, 3)), deformation, array(C, (n, 3, 3)),
        material.density*jnp.asarray(volume), jnp.asarray(volume), jnp.asarray(ids, dtype=jnp.int32),
        internal, stress, ep, cold, jnp.asarray(0.), jnp.asarray(0.), jnp.zeros(3),
        jnp.asarray(0.), jnp.asarray(0.), jnp.zeros(2), jnp.zeros(2),
        initial_entropy, initial_entropy, jnp.zeros(n),
        initial_thermo.phase_extrapolated, initial_thermo.tension_extrapolated,
        initial_thermo.temperature, initial_J, initial_J, jnp.asarray(True))
    validate_state(state, material, grid)
    return state


def _gather(field, support):
    return jnp.take(field, support.indices, axis=0, mode='fill', fill_value=0.)


def transfer_to_grid(state, support, grid):
    """Stress-free APIC transfer and its nonnegative per-particle variance loss."""
    layout = grid.layout
    weighted_mass = state.mass[:, None]*support.weights
    affine_velocity = state.v[:, None, :]+jnp.einsum('pij,pkj->pki', state.C, support.dpos)
    mass = jnp.zeros(layout.node_count, dtype=state.x.dtype).at[support.indices].add(weighted_mass, mode='drop')
    momentum = jnp.zeros((layout.node_count, 3), dtype=state.x.dtype).at[support.indices].add(
        weighted_mass[..., None]*affine_velocity, mode='drop')
    velocity = momentum/jnp.where(mass > 0, mass, 1.)[:, None]
    difference = affine_velocity-_gather(velocity, support)
    loss = .5*jnp.sum(support.weights*jnp.sum(difference*difference, axis=-1), axis=-1)
    return mass, velocity, loss


def transfer_to_particles(velocity, support, grid):
    v, C = grid_to_particle(velocity, support, grid.layout)
    fitted = v[:, None, :]+jnp.einsum('pij,pkj->pki', C, support.dpos)
    residual = _gather(velocity, support)-fitted
    loss = .5*jnp.sum(support.weights*jnp.sum(residual*residual, axis=-1), axis=-1)
    return v, C, loss


def stress_kick(velocity, mass, volume, tau, support, grid, dt):
    """Use the same MLS gradient for force and midpoint stress power."""
    gradient = 4/grid.layout.dx**2*support.weights[..., None]*support.dpos
    forces = -volume[:, None, None]*jnp.einsum('pij,pkj->pki', tau, gradient)
    force = jnp.zeros_like(velocity).at[support.indices].add(forces, mode='drop')
    updated = velocity+dt*force/jnp.where(mass > 0, mass, 1.)[:, None]
    updated = jnp.where((mass > 0)[:, None], updated, 0.)
    _, midpoint_C = grid_to_particle(.5*(velocity+updated), support, grid.layout)
    work = dt*volume*jnp.sum(tau*midpoint_C, axis=(-2, -1))
    return updated, work


def stress_work(old_velocity, final_velocity, volume, tau, support, grid, dt):
    """Particle work conjugate to the actual complete-step grid velocity path."""
    _, midpoint_C = grid_to_particle(.5*(old_velocity+final_velocity), support, grid.layout)
    return dt*volume*jnp.sum(tau*midpoint_C, axis=(-2, -1))


def clamp_grid(velocity, mass, mask):
    corrected = jnp.where(mask, 0., velocity)
    reservoir = .5*jnp.sum(mass*jnp.sum(velocity**2-corrected**2, axis=-1))
    impulse = jnp.sum(mass[:, None]*(corrected-velocity), axis=0)
    return corrected, reservoir, impulse


def paired_contact(velocity, mass, normals, pairs, clamped=None):
    """Frictionless projection and its nonnegative instantaneous kinetic loss."""
    a, b = pairs[:, 0], pairs[:, 1]
    va, vb, impulse = contact_impulse(mass[a], mass[b], velocity[a], velocity[b], normals)
    da, db = va-velocity[a], vb-velocity[b]
    loss_a = .5*mass[a]*jnp.sum(da*da, axis=-1)
    loss_b = .5*mass[b]*jnp.sum(db*db, axis=-1)
    result = velocity.at[a].set(va).at[b].set(vb)
    heat = jnp.zeros_like(mass).at[a].add(loss_a).at[b].add(loss_b)
    occupied = (mass[a] > 0) & (mass[b] > 0)
    valid = jnp.all(~occupied | (jnp.linalg.norm(normals, axis=-1) > 0))
    if clamped is not None:
        closing = jnp.sum((velocity[a]-velocity[b])*normals, axis=-1) > 0
        direction_floor = 64*jnp.finfo(velocity.dtype).eps*jnp.linalg.norm(normals, axis=-1)
        constrained_normal = jnp.any((clamped[a] | clamped[b])
            & (jnp.abs(normals) > direction_floor[:, None]), axis=-1)
        # A free two-mass projection does not solve a simultaneous fixed-DOF
        # contact constraint. Reject this unsupported intersection explicitly.
        valid = valid & jnp.all(~(occupied & closing & constrained_normal))
    return result, heat, impulse, valid


def compatible_contact_work(old_velocity, final_velocity, contact_impulses, mass,
                            pairs, projection_loss):
    """Use a common mass-weighted midpoint velocity for interface power.

    The returned compatible allocation may be signed. Its difference from the
    instantaneous projection loss is numerical coupling work, not physical heat.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    midpoint = .5*(old_velocity+final_velocity)
    mechanical = jnp.sum(contact_impulses*midpoint, axis=-1)
    total = -(mechanical[a]+mechanical[b])
    denominator = jnp.where(mass[a]+mass[b] > 0, mass[a]+mass[b], 1.)
    allocation_a = total*mass[b]/denominator
    allocation_b = total*mass[a]/denominator
    allocation = jnp.zeros_like(mass).at[a].add(allocation_a).at[b].add(allocation_b)
    correction = allocation-projection_loss
    body_work = jnp.array([jnp.sum(mechanical[a]+allocation_a),
                           jnp.sum(mechanical[b]+allocation_b)])
    by_body_projection = jnp.array([projection_loss[a].sum(), projection_loss[b].sum()])
    by_body_correction = jnp.array([correction[a].sum(), correction[b].sum()])
    return allocation, by_body_projection, by_body_correction, body_work


def contact_normals(state, mass, support, grid):
    # Weighted offsets avoid subtracting large world coordinates near contact.
    moment = jnp.zeros((grid.layout.node_count, 3), dtype=state.x.dtype).at[support.indices].add(
        -state.mass[:, None, None]*support.weights[..., None]*support.dpos, mode='drop')
    center = moment/jnp.where(mass > 0, mass, 1.)[:, None]
    a, b = grid.pairs[:, 0], grid.pairs[:, 1]
    normal = center[b]-center[a]
    # The MLS mass-gradient difference is proportional to this centroid normal.
    # If both vanish, the interface is unresolved; do not invent a direction.
    return jnp.where((jnp.linalg.norm(normal, axis=-1) > 64*jnp.finfo(state.x.dtype).eps*grid.layout.dx)[:, None], normal, 0.)


def cayley_kinematics(F, gradient, dt):
    """Midpoint deformation update and a proper spin rotation for the return map."""
    eye = jnp.eye(3, dtype=F.dtype)
    spin = .5*(gradient-jnp.swapaxes(gradient, -1, -2))
    rate = .5*(gradient+jnp.swapaxes(gradient, -1, -2))
    rotation = jnp.linalg.solve(eye-.5*dt*spin, eye+.5*dt*spin)
    half_rotation = jnp.linalg.solve(eye-.25*dt*spin, eye+.25*dt*spin)
    strain = dt*jnp.swapaxes(half_rotation, -1, -2)@rate@half_rotation
    update = jnp.linalg.solve(eye-.5*dt*gradient, eye+.5*dt*gradient)
    return update@F, rotation, strain


def stable_timestep(state, material, grid, config):
    thermo = thermodynamics(state, material)
    compression = jnp.maximum(-jnp.trace(state.C, axis1=-2, axis2=-1), 0.)
    viscous_signal = (material.viscosity_linear*thermo.sound_speed
                      +2*material.viscosity_quadratic*grid.layout.dx*compression)
    affine_transport = 1.5*jnp.sqrt(3.)*grid.layout.dx*jnp.linalg.norm(state.C, axis=(-2, -1))
    signal = thermo.longitudinal_speed+jnp.linalg.norm(state.v, axis=-1)+affine_transport+viscous_signal
    return config.cfl*grid.layout.dx/jnp.max(signal)


def _step(state, material, grid, config, dt):
    support = stencil(state.x, state.body_id, grid.layout)
    old_thermo = thermodynamics(state, material)
    J = jnp.linalg.det(state.F)
    q = artificial_viscosity(old_thermo.density, old_thermo.sound_speed,
        jnp.trace(state.C, axis1=-2, axis2=-1), grid.layout.dx, material)
    tau = state.tau-J[:, None, None]*(old_thermo.pressure+q)[:, None, None]*jnp.eye(3)
    mass, old_velocity, p2g_loss = transfer_to_grid(state, support, grid)
    kicked, _ = stress_kick(old_velocity, mass, state.volume, tau, support, grid, dt)
    before_contact, boundary0, _ = clamp_grid(kicked, mass, grid.clamped)
    projected, projection_loss, _, contact_valid = paired_contact(
        before_contact, mass, contact_normals(state, mass, support, grid), grid.pairs, grid.clamped)
    velocity, boundary1, _ = clamp_grid(projected, mass, grid.clamped)
    contact_impulses = mass[:, None]*(projected-before_contact)
    boundary_impulses = mass[:, None]*(before_contact-kicked+velocity-projected)
    midpoint_velocity = .5*(old_velocity+velocity)
    boundary = -jnp.sum(boundary_impulses*midpoint_velocity)
    boundary_projection = boundary0+boundary1
    boundary_correction = boundary-boundary_projection
    impulse = boundary_impulses.sum(axis=0)
    allocation, physical_by_body, correction_by_body, body_work = compatible_contact_work(
        old_velocity, velocity, contact_impulses, mass, grid.pairs, projection_loss)
    work = stress_work(old_velocity, velocity, state.volume, tau, support, grid, dt)
    specific_contact = jnp.sum(support.weights*_gather(allocation/jnp.where(mass > 0, mass, 1.), support), axis=-1)
    v, C, g2p_loss = transfer_to_particles(velocity, support, grid)
    F, rotation, strain = cayley_kinematics(state.F, C, dt)
    new_J = jnp.linalg.det(F)
    internal = state.internal_energy+p2g_loss+g2p_loss+specific_contact+work/state.mass
    if config.strength:
        returned = deviatoric_return(state.tau, state.plastic_strain, strain, rotation,
                                    new_J, dt, old_thermo.temperature, material)
        new_tau, ep = returned.kirchhoff_stress, returned.plastic_strain
        cold = state.cold_work+returned.specific_cold_work
        material_valid = jnp.all(returned.valid)
    else:
        new_tau, ep, cold = jnp.zeros_like(state.tau), state.plastic_strain, state.cold_work
        material_valid = jnp.all(state.tau == 0) & jnp.all(state.cold_work == 0)
    x = state.x+dt*v
    # x and F both use the same end-contact velocity field: first-order motion.
    current = ImpactState(x, v, F, C, state.mass, state.volume, state.body_id, internal,
        new_tau, ep, cold, state.time+dt, state.boundary_reservoir+boundary,
        state.boundary_impulse+impulse, state.boundary_projection_loss+boundary_projection,
        state.boundary_coupling_work+boundary_correction,
        state.contact_projection_loss+physical_by_body,
        state.contact_coupling_work+correction_by_body, state.initial_entropy,
        state.minimum_entropy, state.negative_entropy_increments,
        state.ever_phase_extrapolated, state.ever_tension_extrapolated,
        state.maximum_temperature, state.minimum_J, state.maximum_J, state.valid)
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
    momentum_residual = jnp.sum(state.mass[:, None]*(v-state.v), axis=0)-impulse
    energy_scale = jnp.maximum(jnp.abs(energy_ledger(state, grid)).sum()
                              +jnp.abs(energy_ledger(current, grid)).sum(), 1e-30)
    momentum_scale = jnp.maximum(jnp.sum(state.mass*(jnp.linalg.norm(state.v, axis=-1)
                              +jnp.linalg.norm(v, axis=-1)))+jnp.linalg.norm(impulse), 1e-30)
    finite = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in current]))
    valid = (state.valid & support.valid & stencil(x, state.body_id, grid.layout).valid
             & jnp.all(old_thermo.valid) & jnp.all(thermo.valid) & contact_valid & material_valid
             & finite & (dt > 0) & (dt <= limit*(1+1e-12))
             & (jnp.abs(residual) <= 1e-10*energy_scale)
             & (jnp.linalg.norm(momentum_residual) <= 1e-10*momentum_scale))
    current = current._replace(valid=valid)
    diagnostics = ImpactDiagnostics(jnp.sum(state.mass*p2g_loss), jnp.sum(state.mass*g2p_loss),
        physical_by_body, correction_by_body, allocation.sum(), body_work,
        work.sum(), boundary, residual, momentum_residual, limit,
        q.max(), new_J.min(), thermo.temperature.min(), thermo.entropy.min(),
        entropy_increment.min(), jnp.mean(entropy_increment < 0),
        (current.minimum_entropy-current.initial_entropy).min(),
        current.negative_entropy_increments.min(), jnp.any(thermo.phase_extrapolated),
        jnp.any(thermo.tension_extrapolated), valid)
    return current, diagnostics


@partial(jax.jit, static_argnames=('config',))
def step(state, material, grid, config, dt):
    return _step(state, material, grid, config, dt)


@partial(jax.jit, static_argnames=('config', 'max_steps'))
def advance_to(state, material, grid, config, target_time, *, max_steps=10000):
    """Stop at the requested output time or first invalid state; never delete mass."""
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')

    def condition(carry):
        current, count = carry
        return current.valid & (current.time < target_time) & (count < max_steps)

    def body(carry):
        current, count = carry
        dt = jnp.minimum(stable_timestep(current, material, grid, config), target_time-current.time)
        return _step(current, material, grid, config, dt)[0], count+1

    admitted = state._replace(valid=state.valid & jnp.isfinite(target_time) & (target_time >= state.time))
    current, count = jax.lax.while_loop(condition, body, (admitted, jnp.asarray(0)))
    return current._replace(valid=current.valid & (current.time == target_time)), count


def validate_state(state, material, grid):
    n = np.asarray(state.x).shape[0] if np.asarray(state.x).ndim == 2 else 0
    shapes = {'x': (n, 3), 'v': (n, 3), 'F': (n, 3, 3), 'C': (n, 3, 3),
        'tau': (n, 3, 3), 'mass': (n,), 'volume': (n,), 'body_id': (n,),
        'internal_energy': (n,), 'plastic_strain': (n,), 'cold_work': (n,),
        'initial_entropy': (n,), 'minimum_entropy': (n,), 'negative_entropy_increments': (n,),
        'ever_phase_extrapolated': (n,), 'ever_tension_extrapolated': (n,),
        'maximum_temperature': (n,), 'minimum_J': (n,), 'maximum_J': (n,),
        'time': (), 'boundary_reservoir': (), 'boundary_impulse': (3,),
        'boundary_projection_loss': (), 'boundary_coupling_work': (),
        'contact_projection_loss': (2,), 'contact_coupling_work': (2,), 'valid': ()}
    if n == 0 or set(shapes) != set(state._fields):
        raise FloatingPointError('Impact carry requires a nonempty registered state')
    for field, shape in shapes.items():
        value = np.asarray(getattr(state, field))
        dtype_ok = (value.dtype == np.dtype(bool) if field in
                    ('valid', 'ever_phase_extrapolated', 'ever_tension_extrapolated') else
                    value.dtype.kind in 'iu' if field == 'body_id' else value.dtype == np.dtype('float64'))
        if value.shape != shape or not dtype_ok:
            raise FloatingPointError(f'Impact carry field {field} has an invalid shape or dtype')
    if not np.array_equal(np.unique(np.asarray(state.body_id)), [0, 1]):
        raise FloatingPointError('Impact carry must retain both body IDs 0 and 1')
    if not bool(state.valid) or not bool(stencil(state.x, state.body_id, grid.layout).valid):
        raise FloatingPointError('Impact state is invalid or outside the allocated grid')
    thermo = thermodynamics(state, material)
    if not np.asarray(thermo.valid).all():
        raise FloatingPointError('Impact state lies outside the declared EOS domain')
    if any(not np.isfinite(np.asarray(a)).all() for a in state):
        raise FloatingPointError('Impact state contains nonfinite data')
    if np.any(np.asarray(state.mass) <= 0) or np.any(np.asarray(state.volume) <= 0):
        raise FloatingPointError('Impact particle mass and volume must remain positive')
    if not np.allclose(np.asarray(state.mass), material.density*np.asarray(state.volume),
                       rtol=32*np.finfo(np.asarray(state.mass).dtype).eps, atol=0.):
        raise FloatingPointError('Impact reference mass must equal rho0 times reference volume')
    if (np.any(np.asarray(state.plastic_strain) < 0) or np.any(np.asarray(state.cold_work) < 0)
            or float(state.boundary_projection_loss) < 0
            or np.any(np.asarray(state.contact_projection_loss) < 0) or float(state.time) < 0):
        raise FloatingPointError('Plastic history, projection losses and time must remain nonnegative')
    reservoir = float(state.boundary_projection_loss+state.boundary_coupling_work)
    ledger_scale = max(abs(reservoir)+abs(float(state.boundary_projection_loss))
                       +abs(float(state.boundary_coupling_work)), 1e-30)
    if abs(float(state.boundary_reservoir)-reservoir) > 1e-12*ledger_scale:
        raise FloatingPointError('Boundary ledger differs from its projection and coupling terms')
    if (np.any(np.asarray(state.minimum_entropy) > np.asarray(state.initial_entropy))
            or np.any(np.asarray(state.negative_entropy_increments) > 0)):
        raise FloatingPointError('Entropy-history extrema and negative increments are inconsistent')
    current_J = np.linalg.det(np.asarray(state.F))
    roundoff = 64*np.finfo(np.float64).eps
    temperature_floor = roundoff*np.maximum(np.abs(np.asarray(thermo.temperature)), 1.)
    J_floor = roundoff*np.maximum(np.abs(current_J), 1.)
    if (np.any(np.asarray(state.maximum_temperature)+temperature_floor < np.asarray(thermo.temperature))
            or np.any(np.asarray(state.minimum_J)-J_floor > current_J)
            or np.any(np.asarray(state.maximum_J)+J_floor < current_J)
            or np.any(np.asarray(state.minimum_J) < material.minimum_J-J_floor)
            or np.any(np.asarray(state.maximum_J) > material.maximum_J+J_floor)
            or np.any((np.asarray(state.maximum_temperature) >= material.melt_temperature)
                      & ~np.asarray(state.ever_phase_extrapolated))
            or np.any(np.asarray(thermo.phase_extrapolated) & ~np.asarray(state.ever_phase_extrapolated))
            or np.any(np.asarray(thermo.tension_extrapolated) & ~np.asarray(state.ever_tension_extrapolated))):
        raise FloatingPointError('EOS domain history does not contain the current state')
    tau = np.asarray(state.tau)
    scale = max(1., float(np.abs(tau).max()))
    if (np.max(np.abs(tau-np.swapaxes(tau, -1, -2))) > 1e-12*scale
            or np.max(np.abs(np.trace(tau, axis1=-2, axis2=-1))) > 1e-12*scale):
        raise FloatingPointError('Deviatoric Kirchhoff stress must be symmetric and traceless')
    return True
