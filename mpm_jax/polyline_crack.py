"""Advected, nonbranching 2D cracks with two nodal velocity fields."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .core import grid_update, particle_stencil, particle_to_grid, particle_update, stencil
from .crack_mpm import _cross
from .materials import kirchhoff_stress


class CrackPath(NamedTuple):
    plus: jax.Array
    minus: jax.Array
    count: jax.Array
    extension: jax.Array
    pending: jax.Array
    valid: jax.Array

    @property
    def vertices(self):
        return .5*(self.plus+self.minus)


def make_path(vertices, capacity=128):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not 2 <= len(vertices) <= capacity:
        raise ValueError('Require at least two 2D vertices within the allocated capacity')
    if not np.isfinite(vertices).all() or np.any(np.linalg.norm(np.diff(vertices, axis=0), axis=1) == 0):
        raise ValueError('Crack segments must be finite and nonzero')
    for i in range(len(vertices)-1):
        for j in range(i+2, len(vertices)-1):
            a, b, c, d = vertices[i], vertices[i+1], vertices[j], vertices[j+1]
            cross = lambda u, v: u[0]*v[1]-u[1]*v[0]
            if cross(b-a, c-a)*cross(b-a, d-a) <= 0 and cross(d-c, a-c)*cross(d-c, b-c) <= 0:
                if np.all(np.maximum(np.minimum(a,b),np.minimum(c,d)) <= np.minimum(np.maximum(a,b),np.maximum(c,d))):
                    raise ValueError('Self-intersecting crack paths require more than two fields')
    padded = jnp.asarray(np.concatenate((vertices, np.repeat(vertices[-1:], capacity-len(vertices), axis=0))))
    zero = jnp.asarray(0., dtype=padded.dtype)
    return CrackPath(padded, padded, jnp.asarray(len(vertices), dtype=jnp.int32), zero, zero, jnp.asarray(True))


def vertex_normals(vertices, count):
    index = jnp.arange(len(vertices))
    edges = vertices[jnp.minimum(index+1, count-1)]-vertices[jnp.minimum(index, count-2)]
    directions = edges/jnp.maximum(jnp.linalg.norm(edges, axis=-1, keepdims=True), jnp.finfo(vertices.dtype).tiny)
    average = directions+directions[jnp.maximum(index-1, 0)]
    average = average/jnp.maximum(jnp.linalg.norm(average, axis=-1, keepdims=True), jnp.finfo(vertices.dtype).tiny)
    return jnp.stack((-average[:, 1], average[:, 0]), axis=-1)


def shifted_nodes(nodes, vertices, count):
    """Give a node exactly on the path a consistent left-side tie break."""
    normals = vertex_normals(vertices, count)
    scale = jnp.maximum(jnp.max(jnp.abs(vertices)), jnp.max(jnp.abs(nodes)))
    tolerance = 64*jnp.finfo(nodes.dtype).eps*jnp.maximum(scale, jnp.finfo(nodes.dtype).tiny)

    def nearest(i, carry):
        distance, normal = carry
        edge = vertices[i+1]-vertices[i]
        length2 = jnp.sum(edge*edge)
        fraction = jnp.clip(jnp.sum((nodes-vertices[i])*edge, axis=-1)/jnp.maximum(length2, tolerance*tolerance), 0., 1.)
        delta = nodes-(vertices[i]+fraction[..., None]*edge)
        d2 = jnp.sum(delta*delta, axis=-1)
        n = jnp.array([-edge[1], edge[0]])/jnp.maximum(jnp.sqrt(length2), tolerance)
        n = jnp.where((fraction <= tolerance/jnp.maximum(jnp.sqrt(length2), tolerance))[..., None], normals[i], n)
        n = jnp.where((fraction >= 1-tolerance/jnp.maximum(jnp.sqrt(length2), tolerance))[..., None], normals[i+1], n)
        return jnp.minimum(distance, d2), jnp.where((d2 < distance)[..., None], n, normal)

    distance, normal = jax.lax.fori_loop(0, count-1, nearest,
                                       (jnp.full(nodes.shape[:-1], jnp.inf), jnp.zeros_like(nodes)))
    return nodes+jnp.where((distance <= tolerance*tolerance)[..., None], 8*tolerance*normal, 0.)


def link_fields(position, support, vertices, count, nodes=None, tip_support_dx=None):
    """Intersection parity with a half-open rule at shared polyline vertices."""
    if nodes is None:
        nodes = shifted_nodes(position[:, None, :]+support.dpos, vertices, count)
    start = position[:, None, :]
    ray = nodes-start

    def cross_edge(i, parity):
        a, b = vertices[i]-start, vertices[i+1]-start
        ya, yb = _cross(ray, a), _cross(ray, b)
        denominator = _cross(ray, b-a)
        parameter = _cross(a, b-a)/jnp.where(denominator != 0, denominator, 1.)
        crossing = ((ya > 0) != (yb > 0)) & (parameter > 0) & (parameter < 1) & (denominator != 0)
        return parity ^ crossing

    fields = jax.lax.fori_loop(0, count-1, cross_edge, jnp.zeros(support.indices.shape, dtype=bool)).astype(jnp.int32)
    if tip_support_dx is not None:
        # Partly cut supports have endpoint shadow jumps. Tie both copies there.
        endpoints = jnp.stack((vertices[0],vertices[count-1]))
        scale = jnp.maximum(jnp.max(jnp.abs(nodes)),jnp.max(jnp.abs(endpoints)))
        tolerance = 128*jnp.finfo(position.dtype).eps*jnp.maximum(scale,tip_support_dx)
        contains = jnp.all(jnp.abs(nodes[...,None,:]-endpoints)<=1.5*tip_support_dx+tolerance,axis=-1)
        fields = jnp.where(jnp.any(contains,axis=-1),0,fields)
    return fields


def visibility_path(path, dx):
    vertices = path.vertices
    direction = vertices[1]-vertices[0]
    direction /= jnp.maximum(jnp.linalg.norm(direction), jnp.finfo(vertices.dtype).tiny)
    return vertices.at[0].add(-3*dx*direction)


def face_advection(position, normal, side, velocity, mass, config, vertices, count, nodes):
    support = stencil(position, config)
    probe = position+side*1e-6*config.dx*normal
    fields = link_fields(probe, support, vertices, count, nodes[support.indices])
    indices = support.indices+fields*((config.n_grid+1)**2)
    weights = jnp.where(mass[indices] > 0, support.weights, 0.)
    total = jnp.sum(weights, axis=-1)
    weights /= jnp.where(total > 0, total, 1.)[:, None]
    v = jnp.sum(weights[..., None]*velocity[indices], axis=1)
    return position+config.dt*v, total > 0


@partial(jax.jit, static_argnames=('config',))
def step(state, path, material, config):
    """Open-face transfer and material advection; no contact or cohesive traction."""
    if config.dim != 2 or config.material not in ('linear', 'elastic', 'neo-hookean'):
        raise ValueError('Polyline cracks currently support 2D isotropic elasticity')
    support = particle_stencil(state, config)
    vertices = visibility_path(path, config.dx)
    size = (config.n_grid+1)**2
    index = jnp.arange(size)
    nodes = config.dx*jnp.stack((index//(config.n_grid+1), index%(config.n_grid+1)), axis=-1)
    nodes = shifted_nodes(nodes, vertices, path.count)
    fields = link_fields(state.x, support, vertices, path.count, nodes[support.indices])
    masses, velocities = [], []
    for field in (0, 1):
        masked = support._replace(weights=jnp.where(fields == field, support.weights, 0.))
        mass, momentum = particle_to_grid(state, material, config, masked)
        masses.append(mass)
        velocities.append(grid_update(mass, momentum, config, state.time+config.dt))
    mass, velocity = jnp.concatenate(masses), jnp.concatenate(velocities)
    following = particle_update(state, material, config, support._replace(indices=support.indices+fields*size), velocity)
    normals = vertex_normals(path.vertices, path.count)
    plus, vp = face_advection(path.plus, normals, 1., velocity, mass, config, vertices, path.count, nodes)
    minus, vm = face_advection(path.minus, normals, -1., velocity, mass, config, vertices, path.count, nodes)
    tip = .5*(plus[path.count-1]+minus[path.count-1])
    plus, minus = plus.at[path.count-1].set(tip), minus.at[path.count-1].set(tip)
    active = jnp.arange(len(plus)) < path.count
    valid = path.valid & following.valid & jnp.all(jnp.where(active, vp & vm, True))
    plus, minus = jnp.where(active[:, None], plus, path.plus), jnp.where(active[:, None], minus, path.minus)
    return following._replace(valid=valid), path._replace(plus=plus, minus=minus, valid=valid)


class GrowthReport(NamedTuple):
    traction: jax.Array
    direction: jax.Array
    increment: jax.Array
    sample_volume: jax.Array
    capacity_exceeded: jax.Array
    intersection: jax.Array


def advance_tip(state, path, material, config, law, segment_length, max_turn=jnp.pi/4):
    """Experimental principal-tension direction, bounded turn and strength kinetics.

    Tip travel accumulates until a full segment can be inserted. Sub-resolution
    segments would rotate under face-interpolation noise before they grow.
    This inserts traction-free segments, without prescribed fracture energy.
    """
    vertices = path.vertices
    tip = vertices[path.count-1]
    edge = tip-vertices[path.count-2]
    length = jnp.linalg.norm(edge)
    forward = edge/jnp.maximum(length, jnp.finfo(vertices.dtype).tiny)
    offset = state.x-tip
    radius2 = jnp.sum(offset*offset, axis=-1)/law.radius**2
    weight = state.volume*jnp.where((offset @ forward >= 0) & (radius2 < 1), (1-radius2)**2, 0.)
    volume = jnp.sum(weight)
    stress = kirchhoff_stress(state.F, material, config.material)/jnp.linalg.det(state.F)[:, None, None]
    average = jnp.sum(weight[:, None, None]*stress, axis=0)/jnp.where(volume > 0, volume, 1.)
    values, vectors = jnp.linalg.eigh(average)
    normal = vectors[:, -1]
    candidate = jnp.array([normal[1], -normal[0]])
    candidate *= jnp.where(candidate @ forward >= 0, 1., -1.)
    angle = jnp.clip(jnp.arctan2(_cross(forward, candidate), forward @ candidate), -max_turn, max_turn)
    direction = jnp.cos(angle)*forward+jnp.sin(angle)*jnp.array([-forward[1], forward[0]])
    direction = jnp.where(values[-1]-values[0] > 1e-10*law.strength, direction, forward)
    normal = jnp.array([-direction[1], direction[0]])
    traction = normal @ average @ normal
    speed = law.max_speed*jnp.maximum(1-law.strength/jnp.maximum(traction, law.strength), 0.)
    remaining = jnp.maximum(law.max_length-path.extension, 0.)
    travel = jnp.where(state.valid & path.valid & (volume > 0), jnp.minimum(speed*config.dt, .25*config.dx), 0.)
    pending = jnp.minimum(path.pending+travel, remaining)
    increment = jnp.minimum(segment_length, remaining)
    growing = (pending >= increment) & (increment > 0) & (travel > 0)
    capacity_exceeded = growing & (path.count >= len(vertices))
    target = jnp.minimum(path.count, len(vertices)-1)
    newtip = tip+increment*direction

    def crossed(i, hit):
        a, b = vertices[i], vertices[i+1]
        ray, tangent = newtip-tip, b-a
        denominator = _cross(ray, tangent)
        safe = jnp.where(denominator != 0, denominator, 1.)
        t, u = _cross(a-tip, tangent)/safe, _cross(a-tip, ray)/safe
        return hit | ((denominator != 0) & (t > 0) & (t <= 1) & (u >= 0) & (u <= 1))

    intersection = jax.lax.fori_loop(0, jnp.maximum(path.count-2, 0), crossed, jnp.asarray(False)) & growing
    accepted = growing & ~capacity_exceeded & ~intersection
    plus = path.plus.at[target].set(jnp.where(accepted, newtip, path.plus[target]))
    minus = path.minus.at[target].set(jnp.where(accepted, newtip, path.minus[target]))
    following = path._replace(plus=plus, minus=minus, count=path.count+accepted.astype(jnp.int32),
                              extension=path.extension+jnp.where(accepted, increment, 0.),
                              pending=pending-jnp.where(accepted, increment, 0.),
                              valid=path.valid & ~capacity_exceeded & ~intersection)
    return following, GrowthReport(traction, direction, jnp.where(accepted, increment, 0.), volume, capacity_exceeded, intersection)
