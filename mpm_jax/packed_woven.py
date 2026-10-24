"""Rectangular-grid APIC for clamped plates and translating rigid impactors."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .fibre_mpm import build_layout, grid_coordinates, grid_to_particle, particle_to_grid, stencil
from .materials import initial_history, update_history
from .materials.woven import woven_energy, woven_kirchhoff_stress
from .rigid_sphere_contact import SphereState, project_sphere_contact


class PlateGrid(NamedTuple):
    layout: object
    clamp_owner: jax.Array
    contact_indices: jax.Array
    contact_coordinates: jax.Array
    grid_y_max: jax.Array
    contact_half_width: jax.Array


class WovenState(NamedTuple):
    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    mass: jax.Array
    volume: jax.Array
    fibre_id: jax.Array
    tau: jax.Array
    history: object
    time: jax.Array
    valid: jax.Array
    boundary_impulse: jax.Array
    contact_impulse: jax.Array
    external_impulse: jax.Array
    tool_work: jax.Array
    contact_loss: jax.Array
    grid_projection_work: jax.Array
    group_impulses: jax.Array


class PlateDiagnostics(NamedTuple):
    clamp_impulse: jax.Array
    contact_impulse: jax.Array
    external_impulse: jax.Array
    group_impulses: jax.Array
    tool_work: jax.Array
    contact_loss: jax.Array
    grid_projection_work: jax.Array
    contact_impulse_residual: jax.Array
    contact_velocity_violation: jax.Array
    contact_iterations: jax.Array
    contact_nodes: jax.Array
    contact_overflow: jax.Array
    contact_window_valid: jax.Array
    valid: jax.Array


@dataclass(frozen=True)
class PackedWovenConfig:
    dt: float
    mode: str
    material_model: str = 'woven'
    loading_duration: float = 150e-6
    stroke: float = .004
    initial_center: tuple = (0., .051, 0.)
    contact_half_width: float = .045
    contact_capacity: int = 65536

    def __post_init__(self):
        if self.mode not in ('sphere-impact', 'hemisphere-indentation', 'pressure-bending'):
            raise ValueError('Unknown plate loading')
        if self.material_model not in ('woven', 'johnson-cook'):
            raise ValueError('Unknown plate material')
        if (not np.isfinite((self.dt, self.loading_duration, self.stroke, self.contact_half_width)).all()
                or min(self.dt, self.loading_duration, self.contact_half_width) <= 0 or self.stroke < 0):
            raise ValueError('Invalid timestep, loading duration, stroke or contact window')
        if len(self.initial_center) != 3 or not np.isfinite(self.initial_center).all():
            raise ValueError('The initial tool center must have three finite components')
        if type(self.contact_capacity) is not int or self.contact_capacity < 1:
            raise ValueError('Contact capacity must be a positive integer')


def make_plate_grid(reference, *, size=(.16, .16), dx=.0005, band=.005,
                    y_bounds=(-.008, .004), padding_cells=3, max_nodes=8_000_000,
                    contact_half_width=.045):
    reference = np.asarray(reference, dtype=np.float64)
    size = np.asarray(size, dtype=float)
    if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0)
            or not np.isfinite(band) or not 0 < 2 * band < size.min()
            or not np.isfinite(y_bounds).all() or len(y_bounds) != 2 or y_bounds[0] >= y_bounds[1]):
        raise ValueError('Positive plate dimensions, nonoverlapping edge bands and ordered y bounds are required')
    bounds = np.array([[[-size[0]/2, size[0]/2], list(y_bounds), [-size[1]/2, size[1]/2]]])
    layout = build_layout(reference, np.zeros(len(reference), dtype=np.int32), dx=dx,
        padding_cells=padding_cells, reference_bounds=bounds, max_nodes=max_nodes)
    lower, shape = np.asarray(layout.lower[0]), np.asarray(layout.shape[0])
    axes = [(lower[axis] + np.arange(shape[axis])) * dx for axis in range(3)]
    # Coordinate masks preserve reflection even when an outer B-spline weight is zero.
    margin = band + 1.5 * dx
    eps = 32 * np.finfo(float).eps * max(float(size.max()), dx)
    x_low, x_high = axes[0] <= -size[0]/2 + margin + eps, axes[0] >= size[0]/2 - margin - eps
    z_low, z_high = axes[2] <= -size[1]/2 + margin + eps, axes[2] >= size[1]/2 - margin - eps
    owner = np.full(tuple(shape), -1, dtype=np.int8)
    owner[x_low, :, :] = 0
    owner[x_high, :, :] = 1
    owner[:, :, z_low] = np.where(owner[:, :, z_low] < 0, 2, owner[:, :, z_low])
    owner[:, :, z_high] = np.where(owner[:, :, z_high] < 0, 3, owner[:, :, z_high])
    union = owner >= 0
    if not np.array_equal(union, union[::-1]) or not np.array_equal(union, union[:, :, ::-1]):
        raise ValueError('The static four-edge node union is not reflection symmetric')
    selected_x = np.flatnonzero(np.abs(axes[0]) <= contact_half_width + eps)
    selected_z = np.flatnonzero(np.abs(axes[2]) <= contact_half_width + eps)
    contact = ((selected_x[:, None, None] * shape[1] + np.arange(shape[1])[None, :, None])
               * shape[2] + selected_z[None, None, :]).ravel().astype(np.int32)
    if not len(contact):
        raise ValueError('The contact window contains no grid nodes')
    coordinates = np.column_stack((axes[0][contact // (shape[1] * shape[2])],
        axes[1][(contact // shape[2]) % shape[1]], axes[2][contact % shape[2]]))
    return PlateGrid(layout, jnp.asarray(owner.ravel()), jnp.asarray(contact),
                     jnp.asarray(coordinates), jnp.asarray(axes[1][-1]),
                     jnp.asarray(contact_half_width))


def make_state(x, volume, mass, material, *, kind='woven', v=None):
    x, volume, mass = map(np.asarray, (x, volume, mass))
    if (x.ndim != 2 or x.shape[1] != 3 or not len(x) or volume.shape != (len(x),)
            or mass.shape != (len(x),) or not all(np.isfinite(a).all() for a in (x, volume, mass))
            or np.any(volume <= 0) or np.any(mass <= 0)):
        raise ValueError('Finite particles with positive mass and reference volume are required')
    if kind not in ('woven', 'johnson-cook'):
        raise ValueError('Unknown plate material')
    count = len(x)
    position = jnp.asarray(x, dtype=jnp.float64)
    velocity = jnp.zeros_like(position) if v is None else jnp.asarray(v, dtype=jnp.float64)
    if velocity.shape != position.shape or not np.isfinite(np.asarray(velocity)).all():
        raise ValueError('Initial velocity must have one finite vector per particle')
    F = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (count, 3, 3))
    history = None if kind == 'woven' else initial_history(count, jnp.float64, material, density=material.density)
    tau = woven_kirchhoff_stress(F, material) if kind == 'woven' else jnp.zeros_like(F)
    zero = jnp.asarray(0., dtype=jnp.float64)
    return WovenState(position, velocity, F, jnp.zeros_like(F), jnp.asarray(mass, dtype=jnp.float64),
        jnp.asarray(volume, dtype=jnp.float64), jnp.zeros(count, dtype=jnp.int32), tau,
        history, zero, jnp.asarray(True), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3),
        zero, zero, zero, jnp.zeros((4, 3)))


def loading_fraction(time, duration):
    phase = jnp.clip(time / duration, 0., 1.)
    return .5 * (1. - jnp.cos(jnp.pi * phase))


def _compact_contact(mass, velocity, grid, sphere, capacity):
    ids, points = grid.contact_indices, grid.contact_coordinates
    mask = (mass[ids] > 0) & (jnp.sum((points-sphere.center)**2, axis=-1) < sphere.radius**2)
    count = jnp.sum(mask)
    local = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
    slots = jnp.arange(capacity) < count
    selected = ids[local]
    weights = jnp.where(slots, mass[selected], 0.)
    return selected, points[local], weights, slots, count


@partial(jax.jit, static_argnames=('config',))
def step_with_diagnostics(state, material, grid, config, sphere, force=None):
    layout = grid.layout
    force = jnp.zeros_like(state.x) if force is None else force
    if force.shape != state.x.shape:
        raise ValueError('Particle forces must have shape (N,3)')
    if config.mode == 'pressure-bending':
        force = force * loading_fraction(state.time + config.dt, config.loading_duration)
    support = stencil(state.x, state.fibre_id, layout)
    mass, momentum = particle_to_grid(state, layout, config, support, force)
    occupied = mass > 0
    free_velocity = momentum / jnp.where(occupied, mass, 1.)[:, None]
    fixed = grid.clamp_owner >= 0
    velocity = jnp.where(fixed[:, None], 0., free_velocity)
    clamp_change = mass[:, None] * (velocity - free_velocity)
    group_impulses = jnp.stack([jnp.sum(jnp.where((grid.clamp_owner == group)[:, None], clamp_change, 0.), axis=0)
                                for group in range(4)])
    clamp_impulse = group_impulses.sum(0)
    grid_work = .5 * jnp.sum(mass * jnp.sum(velocity**2 - free_velocity**2, axis=-1))
    contact_impulse, loss, tool_work = jnp.zeros(3), jnp.asarray(0.), jnp.asarray(0.)
    residual, violation, iterations, contact_count = jnp.asarray(0.), jnp.asarray(0.), jnp.asarray(0), jnp.asarray(0)
    overflow, window_valid, contact_valid = jnp.asarray(False), jnp.asarray(True), jnp.asarray(True)
    if config.mode != 'pressure-bending':
        if config.mode == 'hemisphere-indentation':
            t = state.time + config.dt
            phase = jnp.clip(t / config.loading_duration, 0., 1.)
            speed = -.5 * config.stroke * jnp.pi / config.loading_duration * jnp.sin(jnp.pi * phase)
            speed = jnp.where(t < config.loading_duration, speed, 0.)
            center = jnp.asarray(config.initial_center) + jnp.array([0., -config.stroke * loading_fraction(t, config.loading_duration), 0.])
            sphere = SphereState(center, jnp.array([0., speed, 0.]), sphere.mass, sphere.radius)
        # This cap bound proves that no occupied grid node outside the window can intersect the sphere.
        vertical = jnp.maximum(sphere.center[1] - grid.grid_y_max, 0.)
        cap = jnp.sqrt(jnp.maximum(sphere.radius**2 - vertical**2, 0.))
        window_valid = ((jnp.max(jnp.abs(sphere.center[jnp.array([0, 2])])) + cap <= grid.contact_half_width)
                        & (grid.contact_half_width == config.contact_half_width))
        selected, coordinates, weights, slots, contact_count = _compact_contact(mass, velocity, grid, sphere, config.contact_capacity)
        overflow = contact_count > config.contact_capacity
        values = velocity[selected]
        if config.mode == 'sphere-impact':
            result = project_sphere_contact(weights, values, coordinates, sphere,
                fixed=(grid.clamp_owner[selected] >= 0) & slots)
            delta = jnp.where(slots[:, None], result.velocity - values, 0.)
            contact_impulse, loss = result.impulse_on_body, result.kinetic_loss
            residual, violation, iterations = result.impulse_residual, result.normal_velocity_violation, result.iterations
            contact_valid = result.valid
            sphere = SphereState(sphere.center + config.dt * result.sphere_velocity,
                                  result.sphere_velocity, sphere.mass, sphere.radius)
        else:
            radial = coordinates - sphere.center
            normal = radial / jnp.maximum(jnp.linalg.norm(radial, axis=-1), 1e-30)[:, None]
            closing = jnp.maximum(jnp.sum((sphere.velocity - values) * normal, axis=-1), 0.)
            delta = jnp.where(slots[:, None], closing[:, None] * normal, 0.)
            contact_impulse = jnp.sum(weights[:, None] * delta, axis=0)
            loss = .5 * jnp.sum(weights * jnp.sum(delta**2, axis=-1))
            tool_work = jnp.dot(contact_impulse, sphere.velocity)
            contact_valid = ~jnp.any(slots & (grid.clamp_owner[selected] >= 0)) & (sphere.center[1] > grid.grid_y_max)
        velocity = velocity.at[selected].add(delta)
    v, C = grid_to_particle(velocity, support, layout)
    F = (jnp.eye(3) + config.dt * C) @ state.F
    x = state.x + config.dt * v
    if config.material_model == 'woven':
        history, tau = None, woven_kirchhoff_stress(F, material)
        material_valid = jnp.asarray(True)
    else:
        history = update_history(state.history, F, state.F, config.dt, material, 'johnson-cook')
        tau = jnp.linalg.det(F)[:, None, None] * history.stress
        material_valid = jnp.all(history.valid)
    valid = (state.valid & support.valid & stencil(x, state.fibre_id, layout).valid & material_valid
        & jnp.all(jnp.linalg.det(F) > 0) & jnp.all(jnp.isfinite(F)) & jnp.all(jnp.isfinite(x))
        & jnp.all(jnp.isfinite(v)) & jnp.all(jnp.isfinite(C)) & jnp.all(jnp.isfinite(tau))
        & jnp.all(jnp.isfinite(force)) & contact_valid & window_valid & ~overflow)
    external = config.dt * force.sum(0)
    current = WovenState(x, v, F, C, state.mass, state.volume, state.fibre_id, tau, history,
        state.time + config.dt, valid, state.boundary_impulse + clamp_impulse,
        state.contact_impulse + contact_impulse, state.external_impulse + external,
        state.tool_work + tool_work, state.contact_loss + loss,
        state.grid_projection_work + grid_work, state.group_impulses + group_impulses)
    diagnostics = PlateDiagnostics(clamp_impulse, contact_impulse, external, group_impulses,
        tool_work, loss, grid_work, residual, violation, iterations, contact_count, overflow, window_valid, valid)
    return current, sphere, diagnostics


def kinetic_energy(state, dx):
    translational = .5 * jnp.sum(state.mass * jnp.sum(state.v**2, axis=-1))
    affine = dx**2 / 8 * jnp.sum(state.mass * jnp.sum(state.C**2, axis=(-1, -2)))
    return translational, affine


def stored_energy(state, material, kind='woven'):
    if kind != 'woven':
        raise ValueError('JC energy diagnostics require the actual stress-work and thermal histories')
    return jnp.sum(state.volume * woven_energy(state.F, material))
