"""Two-field 2D crack transfer; method reference: github.com/titoadibaskoro/mpm-fracture."""

from functools import partial
from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .core import grid_boundaries, grid_update, particle_stencil, particle_to_grid, particle_update
from .materials import kirchhoff_stress


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def crack_fields(position, support, segment):
    """Assign the particle-node link to the field on its side of a crack."""
    a, b = segment[0], segment[1]
    tangent = b - a
    ray = support.dpos
    start = position[:, None, :]
    node = start + ray
    relative = a - start
    denominator = _cross(ray, tangent)
    scale = jnp.maximum(jnp.sum(tangent * tangent), jnp.sum(ray * ray, axis=-1))
    tiny = jnp.finfo(position.dtype).eps * jnp.maximum(scale, jnp.finfo(position.dtype).tiny) * 16
    safe = jnp.where(jnp.abs(denominator) > tiny, denominator, 1.0)
    t = _cross(relative, tangent) / safe
    u = _cross(relative, ray) / safe
    crossing = ((jnp.abs(denominator) > tiny) & (t > 0) & (t < 1)
                & (u >= 0) & (u <= 1))
    node_offset = node - a
    node_on_line = jnp.abs(_cross(tangent, node_offset)) <= tiny
    along = jnp.sum(node_offset * tangent, axis=-1)
    node_on_segment = node_on_line & (along >= 0) & (along <= jnp.sum(tangent * tangent))
    particle_side = _cross(tangent, start - a)
    return jnp.where(node_on_segment, particle_side < 0, crossing).astype(jnp.int32)


def frictionless_contact(mass0, mass1, velocity0, velocity1, normal0):
    """Inelastic normal projection; normal0 points from field 1 into field 0."""
    occupied = (mass0 > 0) & (mass1 > 0)
    total = jnp.where(occupied, mass0 + mass1, 1.)
    closing = jnp.minimum(jnp.sum((velocity0 - velocity1) * normal0, axis=-1), 0.)
    closing = jnp.where(occupied, closing, 0.)
    correction = closing[:, None] * normal0
    return (velocity0 - (mass1 / total)[:, None] * correction,
            velocity1 + (mass0 / total)[:, None] * correction)


@partial(jax.jit, static_argnames=('config', 'contact'))
def step(state, material, config, segment, force=None, contact=False, end_exited=False):
    """Straight spatial crack; optional contact acts when both nodal fields close.

    Nodal contact has grid-scale proximity, not an independently resolved gap.
    The segment is not advected: use symmetric, small-strain mode-I loading.
    """
    if config.dim != 2:
        raise ValueError('Explicit line cracks require a two-dimensional configuration')
    support = particle_stencil(state, config)
    tangent = segment[1] - segment[0]
    direction = tangent / jnp.maximum(jnp.linalg.norm(tangent), jnp.finfo(state.x.dtype).tiny)
    outside = segment[0] + direction * (jnp.max((state.x - segment[0]) @ direction) + 3 * config.dx)
    visibility = segment.at[1].set(jnp.where(end_exited, outside, segment[1]))
    fields = crack_fields(state.x, support, visibility)
    velocities = []
    masses = []
    for field in (0, 1):
        weights = jnp.where(fields == field, support.weights, 0)
        mass, momentum = particle_to_grid(state, material, config, support._replace(weights=weights), force)
        velocities.append(grid_update(mass, momentum, config, state.time + config.dt))
        masses.append(mass)
    if contact:
        index = jnp.arange((config.n_grid + 1) ** 2)
        nodes = jnp.stack((index // (config.n_grid + 1), index % (config.n_grid + 1)), axis=-1) * config.dx
        tangent = segment[1] - segment[0]
        normal = jnp.stack((-tangent[1], tangent[0])) / jnp.maximum(jnp.linalg.norm(tangent), jnp.finfo(state.x.dtype).tiny)
        side = jnp.where(_cross(tangent, nodes - segment[0]) >= 0, 1., -1.)
        velocities = frictionless_contact(*masses, *velocities, side[:, None] * normal)
        velocities = [grid_boundaries(v, config, state.time + config.dt, m)
                      for v, m in zip(velocities, masses)]
    velocity = jnp.concatenate(velocities, axis=0)
    size = (config.n_grid + 1) ** config.dim
    split_support = support._replace(indices=support.indices + fields * size)
    following = particle_update(state, material, config, split_support, velocity, force)
    valid_segment = jnp.all(jnp.isfinite(segment)) & (jnp.sum((segment[1] - segment[0]) ** 2) > 0)
    return following._replace(valid=following.valid & valid_segment)


@dataclass(frozen=True)
class TensileTipLaw:
    """Experimental strength criterion with a specified physical averaging radius.

    v = vmax * max(1 - strength / traction, 0). This kinetic law is a modelling
    choice, not a calibrated fracture-energy law or a port of the reference code.
    """

    strength: float
    radius: float
    max_speed: float
    max_length: float

    def __post_init__(self):
        if not all(isfinite(v) and v > 0 for v in
                   (self.strength, self.radius, self.max_speed, self.max_length)):
            raise ValueError('Tip-law parameters must be finite and positive')


class TipReport(NamedTuple):
    traction: jax.Array
    speed: jax.Array
    increment: jax.Array
    sample_volume: jax.Array
    exited: jax.Array


def advance_straight_tip(state, material, config, segment, law):
    """Extend the second tip along the existing direction under mode-I tension."""
    if config.dim != 2 or config.material not in ('linear', 'elastic', 'neo-hookean'):
        raise ValueError('Straight-tip propagation supports 2D isotropic elasticity')
    tangent = segment[1] - segment[0]
    length = jnp.linalg.norm(tangent)
    direction = tangent / jnp.maximum(length, jnp.finfo(state.x.dtype).tiny)
    normal = jnp.stack((-direction[1], direction[0]))
    offset = state.x - segment[1]
    r2 = jnp.sum(offset * offset, axis=-1) / law.radius ** 2
    kernel = jnp.where((offset @ direction >= 0) & (r2 < 1), (1 - r2) ** 2, 0.)
    weights = kernel * state.volume
    volume = jnp.sum(weights)
    stress = kirchhoff_stress(state.F, material, config.material) / jnp.linalg.det(state.F)[:, None, None]
    traction = jnp.sum(weights * jnp.einsum('i,pij,j->p', normal, stress, normal)) / jnp.where(volume > 0, volume, 1.)
    speed = law.max_speed * jnp.maximum(1. - law.strength / jnp.maximum(traction, law.strength), 0.)
    speed = jnp.where(state.valid & (volume > 0), speed, 0.)
    increment = jnp.minimum(jnp.minimum(speed * config.dt, .25 * config.dx), jnp.maximum(law.max_length - length, 0.))
    segment = segment.at[1].add(increment * direction)
    # A tip beyond every material point has no resolved ligament ahead. Its
    # visibility ray must continue through the exterior grid support, otherwise
    # empty-space nodes reconnect the separated pieces at the free surface.
    exited = jnp.max((state.x - segment[1]) @ direction) <= 0
    return segment, TipReport(traction, increment / config.dt, increment, volume, exited)
