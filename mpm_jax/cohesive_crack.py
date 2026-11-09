"""Energy-controlled cohesive faces coupled to two-field MPM."""

from dataclasses import dataclass
from functools import partial
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import grid_boundaries, grid_to_particle, particle_stencil, stencil
from .crack_mpm import _cross, crack_fields
from .materials import kirchhoff_stress


@dataclass(frozen=True)
class CohesiveLaw:
    strength: float
    energy: float
    stiffness: float

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in (self.strength, self.energy, self.stiffness)):
            raise ValueError('Cohesive parameters must be finite and positive')
        if self.peak_opening >= self.final_opening:
            raise ValueError('The peak opening must precede complete separation')

    @property
    def peak_opening(self):
        return self.strength / self.stiffness

    @property
    def final_opening(self):
        return 2 * self.energy / self.strength


class Response(NamedTuple):
    traction: jax.Array
    maximum: jax.Array
    damage: jax.Array
    stored: jax.Array
    dissipated: jax.Array


def response(jump, maximum, normal, bonded, law):
    """Isotropic effective-opening law; compression remains an elastic penalty.

    Energies are per reference area. Normal opening and tangential separation
    share one stiffness, strength and fracture energy; unloading is secant.
    Geometrically pre-opened segments have bonded=False, not preset damage.
    """
    gap = jump @ normal
    compression = jnp.minimum(gap, 0.)
    opening = jump - compression[:, None] * normal
    effective = jnp.linalg.norm(opening, axis=-1)
    maximum = jnp.where(bonded, jnp.maximum(maximum, effective), 0.)
    d0, df = law.peak_opening, law.final_opening
    damage = jnp.clip(df * (maximum - d0) / (jnp.maximum(maximum, d0) * (df - d0)), 0., 1.)
    remaining = jnp.where(bonded, 1 - damage, 0.)
    traction = law.stiffness * (remaining[:, None] * opening + compression[:, None] * normal)
    stored = .5 * law.stiffness * (remaining * effective**2 + compression**2)
    dissipated = jnp.where(bonded, law.energy * jnp.clip((maximum - d0) / (df - d0), 0., 1.), 0.)
    return Response(traction, maximum, damage, stored, dissipated)


class Interface(NamedTuple):
    plus: jax.Array
    minus: jax.Array
    maximum: jax.Array
    area: jax.Array
    bonded: jax.Array
    valid: jax.Array


class Transfer(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    valid: jax.Array


class StepWork(NamedTuple):
    boundary: jax.Array
    gravity: jax.Array
    cohesive: jax.Array
    cohesive_error: jax.Array
    transfer_loss: jax.Array
    bulk_error: jax.Array
    cohesive_grid_error: jax.Array
    iterations: jax.Array
    iteration_residual: jax.Array
    positive_normal_impulse: jax.Array
    negative_normal_impulse: jax.Array
    tangent_impulse: jax.Array
    boundary_projection_loss: jax.Array


def kinetic_energy(state, config):
    return .5*jnp.sum(state.mass*(jnp.sum(state.v**2, axis=-1)
                                 + .25*config.dx**2*jnp.sum(state.C**2, axis=(1, 2))))


def elastic_energy(state, material, config):
    F = state.F
    if config.material == 'linear':
        strain = .5*(F+jnp.swapaxes(F, -1, -2))-jnp.eye(2)
        stress = kirchhoff_stress(F, material, 'linear')/jnp.linalg.det(F)[:, None, None]
        return .5*jnp.sum(state.volume*jnp.sum(stress*strain, axis=(1, 2)))
    logJ = jnp.log(jnp.linalg.det(F))
    density = .5*material.mu*(jnp.sum(F*F, axis=(1, 2))-2)-material.mu*logJ+.5*material.lam*logJ**2
    return jnp.sum(state.volume*density)


def face_transfer(position, grid_mass, config, segment, plus):
    """Use the same occupied-node partition for face velocity and face force."""
    support = stencil(position, config)
    nodes = position[:, None, :] + support.dpos
    tangent = segment[1] - segment[0]
    tolerance = 16 * jnp.finfo(position.dtype).eps * jnp.maximum(jnp.sum(tangent*tangent), config.dx**2)
    node_above = _cross(tangent, nodes - segment[0]) >= -tolerance
    field = jnp.where(plus, ~node_above, node_above).astype(jnp.int32)
    indices = support.indices + field * ((config.n_grid + 1) ** 2)
    weights = jnp.where(grid_mass[indices] > 0, support.weights, 0.)
    total = weights.sum(axis=1)
    weights = weights / jnp.where(total > 0, total, 1.)[:, None]
    return Transfer(indices, weights, support.valid & jnp.all(total > 0))


def face_velocity(velocity, transfer):
    return jnp.sum(transfer.weights[..., None] * velocity[transfer.indices], axis=1)


def scatter_face_force(force, transfer, grid_shape):
    return jnp.zeros(grid_shape, dtype=force.dtype).at[transfer.indices].add(transfer.weights[..., None] * force[:, None, :])


def discrete_traction(old_jump, new_jump, maximum, normal, bonded, law):
    before = response(old_jump, maximum, normal, bonded, law)
    after = response(new_jump, before.maximum, normal, bonded, law)
    change = new_jump-old_jump
    average = .5*(before.traction+after.traction)
    energy = after.stored-before.stored+after.dissipated-before.dissipated
    square = jnp.sum(change*change, axis=-1)
    correction = (energy-jnp.sum(average*change, axis=-1))/jnp.where(square > 0, square, 1.)
    return average+correction[:, None]*change, after, energy


def midpoint_bulk(F, gradient, dt, material):
    """2D Cayley update with a symmetric, energy-conjugate neo-Hookean stress."""
    half = .5*dt*gradient
    left, right = jnp.eye(2)-half, jnp.eye(2)+half
    determinant = jnp.linalg.det(left)
    inverse = jnp.stack((left[:, 1, 1], -left[:, 0, 1], -left[:, 1, 0], left[:, 0, 0]), axis=-1).reshape(-1, 2, 2)
    inverse = inverse/jnp.where(determinant != 0, determinant, 1.)[:, None, None]
    following = inverse @ right @ F
    middle = .5*(F+following)
    a = 1+jnp.linalg.det(half)
    z = jnp.trace(half, axis1=-2, axis2=-1)/jnp.where(a != 0, a, 1.)
    safe_z = jnp.where((jnp.abs(z) >= 1e-5) & (jnp.abs(z) < 1), z, .5)
    quotient = jnp.where(jnp.abs(z) < 1e-5, 1+z*z/3+z**4/5, jnp.arctanh(safe_z)/safe_z)
    phi = quotient/jnp.where(a != 0, a, 1.)
    J0, J1 = jnp.linalg.det(F), jnp.linalg.det(following)
    mean_logJ = .5*(jnp.log(jnp.maximum(J0, 1e-300))+jnp.log(jnp.maximum(J1, 1e-300)))
    tau = material.mu*(middle @ jnp.swapaxes(middle, -1, -2))
    tau = tau + ((-material.mu+material.lam*mean_logJ)*phi)[:, None, None]*jnp.eye(2)
    valid = jnp.all((J0 > 0) & (J1 > 0) & (a > 0) & (jnp.abs(z) < 1) & (determinant > 0))
    return following, tau, valid


@partial(jax.jit, static_argnames=('config', 'law', 'midpoint_boundaries'))
def step(state, interface, material, config, segment, law, midpoint_boundaries=False):
    """Coupled midpoint solve on a straight symmetry plane, with an energy ledger."""
    if config.dim != 2 or config.material not in ('elastic', 'neo-hookean'):
        raise ValueError('Cohesive midpoint MPM currently requires 2D neo-Hookean elasticity')
    tangent = segment[1] - segment[0]
    length = jnp.linalg.norm(tangent)
    normal = jnp.stack((-tangent[1], tangent[0])) / jnp.maximum(length, jnp.finfo(state.x.dtype).tiny)
    support = particle_stencil(state, config)
    fields = crack_fields(state.x, support, segment)
    size = (config.n_grid+1)**2
    remapped = support._replace(indices=support.indices+fields*size)
    mass = jnp.zeros(2*size, dtype=state.x.dtype).at[remapped.indices].add(support.weights*state.mass[:, None])
    affine = jnp.einsum('pij,pkj->pki', state.C, support.dpos)
    old_contribution = support.weights[..., None]*state.mass[:, None, None]*(state.v[:, None, :]+affine)
    momentum = jnp.zeros((2*size, 2), dtype=state.x.dtype).at[remapped.indices].add(old_contribution)
    occupied = mass > 0
    denominator = jnp.where(occupied, mass, 1.)[:, None]
    transferred_velocity = momentum/denominator
    plus = face_transfer(interface.plus, mass, config, segment, True)
    minus = face_transfer(interface.minus, mass, config, segment, False)
    def constrain(v, time):
        return jnp.concatenate([grid_boundaries(v[i*size:(i+1)*size], config,
                                                time, mass[i*size:(i+1)*size]) for i in (0, 1)])

    initial_velocity = constrain(transferred_velocity, state.time) if midpoint_boundaries else transferred_velocity
    initial_impulse = mass[:, None]*(initial_velocity-transferred_velocity)
    projection_loss = .5*jnp.sum(mass[:, None]*(initial_velocity-transferred_velocity)**2)

    def evaluate(velocity):
        middle = .5*(initial_velocity+velocity)
        vp, gradient = grid_to_particle(middle, remapped, config)
        F, tau, valid = midpoint_bulk(state.F, gradient, config.dt, material)
        bulk_contribution = -(4/config.dx**2)*state.volume[:, None, None]*support.weights[..., None]*jnp.einsum('pij,pkj->pki', tau, support.dpos)
        bulk_force = jnp.zeros_like(momentum).at[remapped.indices].add(bulk_contribution)
        vplus, vminus = face_velocity(middle, plus), face_velocity(middle, minus)
        next_plus, next_minus = interface.plus+config.dt*vplus, interface.minus+config.dt*vminus
        traction, after, energy = discrete_traction(interface.plus-interface.minus, next_plus-next_minus,
                                                     interface.maximum, normal, interface.bonded, law)
        force = traction*interface.area[:, None]
        cohesive_force = scatter_face_force(-force, plus, momentum.shape)+scatter_face_force(force, minus, momentum.shape)
        raw = jnp.where(occupied[:, None], initial_velocity+config.dt*((bulk_force+cohesive_force)/denominator+jnp.asarray(config.gravity)), 0.)
        if midpoint_boundaries:
            bounded = 2*constrain(.5*(initial_velocity+raw), state.time+.5*config.dt)-initial_velocity
        else:
            bounded = constrain(raw, state.time+config.dt)
        return bounded, (middle, vp, F, next_plus, next_minus, after, force, energy, bulk_force, cohesive_force, raw, valid)

    def iterate(carry):
        velocity, count, _ = carry
        following, _ = evaluate(velocity)
        residual = jnp.max(jnp.abs(following-velocity))/(1+jnp.max(jnp.abs(following)))
        return following, count+1, residual

    velocity, iterations, residual = jax.lax.while_loop(
        lambda c: (c[1] < 24) & (c[2] > 1e-9), iterate,
        (constrain(initial_velocity, state.time+config.dt), jnp.asarray(0), jnp.asarray(jnp.inf)))
    bounded, data = evaluate(velocity)
    middle, vp, F, next_plus, next_minus, after, force, energy, bulk_force, cohesive_force, raw, constitutive_valid = data
    v, C = grid_to_particle(velocity, remapped, config)
    x = state.x+config.dt*vp
    valid = state.valid & interface.valid & plus.valid & minus.valid & support.valid & constitutive_valid & (residual <= 1e-8)
    valid = valid & (length > 0) & jnp.all(jnp.isfinite(segment))
    for array in (x, v, C, F, next_plus, next_minus):
        valid = valid & jnp.all(jnp.isfinite(array))
    valid = valid & jnp.all((x >= .5*config.dx) & (x < config.domain_size-.5*config.dx))
    following = state._replace(x=x, v=v, C=C, F=F, time=state.time+config.dt, valid=valid)
    next_interface = interface._replace(plus=next_plus, minus=next_minus, maximum=after.maximum, valid=valid)
    boundary_impulse = mass[:, None]*(bounded-raw)
    boundary_work = jnp.sum(boundary_impulse*middle)+jnp.sum(initial_impulse*initial_velocity)
    boundary_impulse = boundary_impulse+initial_impulse
    gravity_work = config.dt*jnp.sum(mass[:, None]*jnp.asarray(config.gravity)*middle)
    interface_energy_change = jnp.sum(interface.area*energy)
    work = config.dt*jnp.sum(force*(face_velocity(middle, plus)-face_velocity(middle, minus)))
    old_grid_energy = .5*jnp.sum(mass[:, None]*transferred_velocity**2)
    final_grid_energy = .5*jnp.sum(mass[:, None]*velocity**2)
    transfer_loss = kinetic_energy(state, config)-old_grid_energy+final_grid_energy-kinetic_energy(following, config)
    bulk_error = config.dt*jnp.sum(bulk_force*middle)+elastic_energy(following, material, config)-elastic_energy(state, material, config)
    cohesive_grid_error = config.dt*jnp.sum(cohesive_force*middle)+interface_energy_change
    node = jnp.arange(size)
    coordinates = config.dx*jnp.stack((node//(config.n_grid+1), node%(config.n_grid+1)), axis=-1)
    positive = jnp.tile((coordinates-segment[0]) @ normal >= 0, 2)
    normal_impulse = boundary_impulse @ normal
    positive_impulse = jnp.sum(jnp.where(positive, normal_impulse, 0.))
    negative_impulse = jnp.sum(jnp.where(~positive, normal_impulse, 0.))
    tangent_impulse = jnp.sum(boundary_impulse @ (tangent/jnp.maximum(length, jnp.finfo(state.x.dtype).tiny)))
    return following, next_interface, StepWork(boundary_work, gravity_work, work, work-interface_energy_change,
                                               transfer_loss, bulk_error, cohesive_grid_error, iterations, residual,
                                               positive_impulse, negative_impulse, tangent_impulse, projection_loss)
