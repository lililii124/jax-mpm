"""Quadratic APIC on independent packed rectangular grids for explicit fibres."""

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .materials.isotropic_brittle import (
    BrittleHistory, brittle_energy, initial_brittle_history, update_brittle,
)


@partial(jax.tree_util.register_dataclass,
         data_fields=("lower", "shape", "strides", "offsets", "origin", "node_map", "dense_indices"),
         meta_fields=("dx", "node_count", "fibre_count"))
@dataclass(frozen=True)
class PackedLayout:
    lower: jax.Array
    shape: jax.Array
    strides: jax.Array
    offsets: jax.Array
    origin: jax.Array
    dx: float
    node_count: int
    fibre_count: int
    node_map: jax.Array | None = None
    dense_indices: jax.Array | None = None


@dataclass(frozen=True)
class FibreConfig:
    dt: float
    gravity: tuple = (0., 0., 0.)

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        if len(self.gravity) != 3 or not np.isfinite(self.gravity).all():
            raise ValueError("gravity must contain three finite components")
        object.__setattr__(self, "gravity", tuple(self.gravity))


class FibreState(NamedTuple):
    x: jax.Array
    v: jax.Array
    F: jax.Array
    C: jax.Array
    mass: jax.Array
    volume: jax.Array
    fibre_id: jax.Array
    history: BrittleHistory
    tau: jax.Array
    time: jax.Array
    valid: jax.Array
    boundary_impulse: jax.Array
    external_impulse: jax.Array
    boundary_work: jax.Array


class FibreStencil(NamedTuple):
    indices: jax.Array
    weights: jax.Array
    dpos: jax.Array
    valid: jax.Array


class BoundaryResult(NamedTuple):
    velocity: jax.Array
    group_impulses: jax.Array
    group_work: jax.Array
    valid: jax.Array


class StepDiagnostics(NamedTuple):
    boundary_impulse: jax.Array
    external_impulse: jax.Array
    boundary_work: jax.Array
    group_impulses: jax.Array
    group_work: jax.Array
    grid_mass: jax.Array
    valid: jax.Array


def build_layout(x, fibre_id, *, dx, padding_cells=6, reference_bounds=None,
                 origin=(0., 0., 0.), max_nodes=30_000_000):
    """Pack per-fibre grids aligned to one global lattice; no full-domain field copies."""
    x, ids = np.asarray(x, dtype=np.float64), np.asarray(fibre_id)
    origin = np.asarray(origin, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("x must be a nonempty finite (N,3) array")
    if ids.shape != (len(x),) or ids.dtype.kind not in 'iu' or ids.min() < 0:
        raise ValueError("fibre_id must contain nonnegative integer particle IDs")
    count = int(ids.max())+1
    if not np.array_equal(np.unique(ids), np.arange(count)):
        raise ValueError("fibre IDs must be dense and start at zero")
    if origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(dx) or dx <= 0:
        raise ValueError("The grid origin and positive dx must be finite")
    if not isinstance(padding_cells, int) or padding_cells < 2:
        raise ValueError("padding_cells must be an integer >= 2")
    if not isinstance(max_nodes, int) or not 0 < max_nodes < 2**31:
        raise ValueError("max_nodes must be a positive int32-compatible budget")
    lower, upper = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
    np.minimum.at(lower, ids, x)
    np.maximum.at(upper, ids, x)
    if reference_bounds is not None:
        bounds = np.asarray(reference_bounds, dtype=np.float64).reshape(count, 3, 2)
        if not np.isfinite(bounds).all() or np.any(bounds[..., 1] < bounds[..., 0]):
            raise ValueError("reference_bounds must contain finite ordered fibre bounds")
        lower, upper = np.minimum(lower, bounds[..., 0]), np.maximum(upper, bounds[..., 1])
    first = np.floor((lower-origin)/dx).astype(np.int64)-padding_cells
    last = np.ceil((upper-origin)/dx).astype(np.int64)+padding_cells
    shape = last-first+1
    sizes = np.prod(shape, axis=1, dtype=np.int64)
    total = int(sizes.sum())
    if total > max_nodes or np.any(np.abs(first) >= 2**30) or np.any(shape >= 2**30):
        raise ValueError(f"Packed grid requires {total} nodes, above the {max_nodes} node budget or index range")
    offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
    strides = np.column_stack((shape[:, 1]*shape[:, 2], shape[:, 2], np.ones(count, dtype=np.int64)))
    return PackedLayout(*(jnp.asarray(value, dtype=jnp.int32) for value in (first, shape, strides, offsets)),
                        jnp.asarray(origin), float(dx), total, count)


def build_tube_layout(x, fibre_id, *, dx, padding_cells=6, reference_bounds=None,
                      origin=(0., 0., 0.), max_nodes=30_000_000):
    """Compress empty nodes; retain a padded tube around every reference stencil.

    The three separable dilations use a Chebyshev radius of ``padding_cells``
    and are intersected with the same local AABB as ``build_layout``. Missing
    nodes invalidate a step; the static tube does not silently follow particles.
    """
    from scipy.ndimage import maximum_filter1d

    dense = build_layout(x, fibre_id, dx=dx, padding_cells=padding_cells,
                         reference_bounds=reference_bounds, origin=origin, max_nodes=max_nodes)
    x, ids = np.asarray(x, dtype=np.float64), np.asarray(fibre_id)
    lower, shape, offsets = map(np.asarray, (dense.lower, dense.shape, dense.offsets))
    node_map = np.full(dense.node_count, -1, dtype=np.int32)
    order = np.argsort(ids, kind='stable')
    counts = np.bincount(ids, minlength=dense.fibre_count)
    starts = np.concatenate(([0], np.cumsum(counts)))
    neighbours = np.array(list(product(range(3), repeat=3)), dtype=np.int64)
    active_nodes, total = [], 0
    for fibre in range(dense.fibre_count):
        selected = order[starts[fibre]:starts[fibre+1]]
        base = np.floor((x[selected]-np.asarray(origin))/dx-.5).astype(np.int64)-lower[fibre]
        local = (base[:, None, :]+neighbours).reshape(-1, 3)
        if np.any(local < 0) or np.any(local >= shape[fibre]):
            raise ValueError(f"Reference fibre {fibre} has a stencil outside its dense grid")
        occupied = np.zeros(tuple(shape[fibre]), dtype=bool)
        occupied[tuple(local.T)] = True
        for axis in range(3):
            occupied = maximum_filter1d(occupied, size=2*padding_cells+1, axis=axis, mode='constant', cval=False)
        if not occupied[tuple(local.T)].all():
            raise ValueError("Tube dilation lost a reference stencil node")
        dense_ids = (np.flatnonzero(occupied.ravel())+int(offsets[fibre])).astype(np.int32)
        node_map[dense_ids] = np.arange(total, total+len(dense_ids), dtype=np.int32)
        active_nodes.append(dense_ids)
        total += len(dense_ids)
    inverse = np.concatenate(active_nodes)
    if total == 0 or not np.array_equal(node_map[inverse], np.arange(total, dtype=np.int32)):
        raise ValueError("Tube node mapping is not a complete one-to-one packed layout")
    return PackedLayout(dense.lower, dense.shape, dense.strides, dense.offsets, dense.origin,
                        dense.dx, total, dense.fibre_count, jnp.asarray(node_map), jnp.asarray(inverse))


def make_state(x, volume, fibre_id, material, layout, *, v=None, F=None, C=None, history=None):
    """Initialize independent histories; the crack-band length is fixed to grid dx."""
    x = np.asarray(x, dtype=np.float64)
    volume, ids = np.asarray(volume, dtype=np.float64), np.asarray(fibre_id)
    if volume.shape != (len(x),) or not np.isfinite(volume).all() or np.any(volume <= 0):
        raise ValueError("Reference particle volumes must be finite and positive")
    if ids.shape != (len(x),) or ids.dtype.kind not in 'iu' or ids.min() < 0 or ids.max() >= layout.fibre_count:
        raise ValueError("Particle fibre IDs are inconsistent with the layout")
    if not np.allclose(np.asarray(material.characteristic_length), layout.dx, rtol=1e-12, atol=0):
        raise ValueError("Fibre MPM requires material.characteristic_length equal to grid dx, independent of particle density")
    n, dtype = len(x), jnp.float64
    x = jnp.asarray(x, dtype=dtype)
    if x.dtype != dtype:
        raise ValueError("Enable jax_enable_x64 before creating a fibre state")
    velocity = jnp.zeros_like(x) if v is None else jnp.asarray(v, dtype=dtype)
    deformation = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (n, 3, 3)) if F is None else jnp.asarray(F, dtype=dtype)
    affine = jnp.zeros((n, 3, 3), dtype=dtype) if C is None else jnp.asarray(C, dtype=dtype)
    if velocity.shape != x.shape or deformation.shape != (n, 3, 3) or affine.shape != (n, 3, 3):
        raise ValueError("Initial v/F/C arrays have incompatible shapes")
    history = initial_brittle_history(n, dtype) if history is None else history
    history, tau = update_brittle(history, deformation, material)
    state = FibreState(x, velocity, deformation, affine, material.density*jnp.asarray(volume),
        jnp.asarray(volume), jnp.asarray(ids, dtype=jnp.int32), history, tau,
        jnp.asarray(0., dtype=dtype), jnp.all(history.valid), jnp.zeros(3, dtype=dtype),
        jnp.zeros(3, dtype=dtype), jnp.asarray(0., dtype=dtype))
    validate_state(state, layout)
    return state


def stencil(x, fibre_id, layout):
    q = (x-layout.origin)/layout.dx
    base = jnp.floor(q-.5).astype(jnp.int32)
    fraction = q-base
    weights_axis = jnp.stack((.5*(1.5-fraction)**2, .75-(fraction-1)**2, .5*(fraction-.5)**2), axis=1)
    offsets = jnp.array(list(product(range(3), repeat=3)), dtype=jnp.int32)
    nodes = base[:, None, :]+offsets
    local = nodes-layout.lower[fibre_id, None, :]
    inside = jnp.all((local >= 0) & (local < layout.shape[fibre_id, None, :]), axis=-1)
    indices = layout.offsets[fibre_id, None]+jnp.sum(local*layout.strides[fibre_id, None, :], axis=-1)
    if layout.node_map is not None:
        mapped = jnp.take(layout.node_map, jnp.where(inside, indices, layout.node_map.shape[0]),
                          mode='fill', fill_value=-1)
        inside = inside & (mapped >= 0)
        indices = mapped
    indices = jnp.where(inside, indices, layout.node_count)
    weights = jnp.prod(weights_axis[:, offsets, jnp.arange(3)], axis=-1)
    dpos = (offsets-fraction[:, None, :])*layout.dx
    return FibreStencil(indices, weights, dpos, jnp.all(inside) & jnp.all(jnp.isfinite(x)))


def grid_coordinates(layout, packed_indices):
    """Recover coordinates and fibre IDs for selected packed nodes, without a full grid table."""
    indices = jnp.asarray(packed_indices, dtype=jnp.int32)
    if layout.dense_indices is not None:
        indices = jnp.take(layout.dense_indices, indices, mode='fill', fill_value=-1)
    fibres = jnp.searchsorted(layout.offsets, indices, side='right')-1
    local = indices-layout.offsets[fibres]
    nodes = (local[..., None]//layout.strides[fibres]) % layout.shape[fibres]
    return layout.origin+(nodes+layout.lower[fibres])*layout.dx, fibres


def particle_to_grid(state, layout, config, support=None, force=None):
    support = stencil(state.x, state.fibre_id, layout) if support is None else support
    affine = state.mass[:, None, None]*state.C-(4*config.dt/layout.dx**2)*state.volume[:, None, None]*state.tau
    momentum = state.mass[:, None, None]*state.v[:, None, :]+jnp.einsum('pij,pkj->pki', affine, support.dpos)
    if force is not None:
        momentum = momentum+config.dt*force[:, None, :]
    mass = jnp.zeros(layout.node_count, dtype=state.x.dtype).at[support.indices].add(support.weights*state.mass[:, None], mode='drop')
    momentum = jnp.zeros((layout.node_count, 3), dtype=state.x.dtype).at[support.indices].add(support.weights[..., None]*momentum, mode='drop')
    return mass, momentum


def grid_to_particle(velocity, support, layout):
    gathered = jnp.take(velocity, support.indices, axis=0, mode='fill', fill_value=0)
    v = jnp.sum(support.weights[..., None]*gathered, axis=1)
    C = (4/layout.dx**2)*jnp.einsum('pk,pki,pkj->pij', support.weights, gathered, support.dpos)
    return v, C


def reference_grips(particle_masks, velocities):
    """Follow tagged reference particles and clamp every node of their current stencil.

    Each velocity is a callable of physical time returning three components.
    Reactions are impulses on particles; fixture reaction forces have opposite sign.
    """
    masks = tuple(jnp.asarray(value, dtype=bool) for value in particle_masks)
    velocities = tuple(velocities)
    if not masks or len(masks) != len(velocities) or any(mask.ndim != 1 or mask.shape != masks[0].shape for mask in masks):
        raise ValueError("One equally sized reference particle mask is required per grip velocity")

    def boundary(velocity, mass, state, support, layout, time):
        if masks[0].shape != state.mass.shape:
            raise ValueError("Reference grip masks must match the particle count")
        assigned = jnp.zeros(layout.node_count, dtype=bool)
        impulses, work = [], []
        valid = jnp.asarray(True)
        for mask, prescribed in zip(masks, velocities):
            selected = jnp.zeros(layout.node_count, dtype=jnp.int32).at[support.indices].max(
                jnp.broadcast_to(mask[:, None], support.weights.shape).astype(jnp.int32), mode='drop') > 0
            selected = selected & (mass > 0)
            valid = valid & ~jnp.any(assigned & selected)
            target = jnp.asarray(prescribed(time), dtype=velocity.dtype)
            if target.shape != (3,):
                raise ValueError("A prescribed grip velocity must have three components")
            corrected = jnp.where(selected[:, None], target, velocity)
            impulses.append(jnp.sum(mass[:, None]*(corrected-velocity), axis=0))
            work.append(.5*jnp.sum(mass*jnp.sum((corrected-velocity)*(corrected+velocity), axis=1)))
            velocity, assigned = corrected, assigned | selected
        return BoundaryResult(velocity, jnp.stack(impulses), jnp.stack(work), valid)

    boundary.group_count = len(masks)
    return boundary


def _step(state, material, layout, config, force, force_fn, boundary):
    if force is not None and force_fn is not None:
        raise ValueError("Pass a constant particle force or a state-dependent force_fn, not both")
    if force_fn is not None:
        force = force_fn(state, state.time)
    force = jnp.zeros_like(state.x) if force is None else jnp.asarray(force, dtype=state.x.dtype)
    if force.shape != state.x.shape:
        raise ValueError("Particle force must have shape (N,3) and units of newtons")
    support = stencil(state.x, state.fibre_id, layout)
    mass, momentum = particle_to_grid(state, layout, config, support, force)
    occupied = mass > 0
    gravity = jnp.asarray(config.gravity, dtype=state.x.dtype)
    velocity = jnp.where(occupied[:, None], momentum/jnp.where(occupied, mass, 1)[:, None]+config.dt*gravity, 0.)
    if boundary is None:
        result = BoundaryResult(velocity, jnp.zeros((0, 3), dtype=velocity.dtype), jnp.zeros(0, dtype=velocity.dtype), jnp.asarray(True))
    else:
        corrected = boundary(velocity, mass, state, support, layout, state.time+config.dt)
        result = corrected if isinstance(corrected, BoundaryResult) else BoundaryResult(
            corrected, jnp.zeros((0, 3), dtype=velocity.dtype), jnp.zeros(0, dtype=velocity.dtype), jnp.asarray(True))
    corrected = jnp.where(occupied[:, None], result.velocity, 0.)
    if corrected.shape != velocity.shape:
        raise ValueError("The grid boundary callback must preserve the packed velocity shape")
    impulse = jnp.sum(mass[:, None]*(corrected-velocity), axis=0)
    work = .5*jnp.sum(mass*jnp.sum((corrected-velocity)*(corrected+velocity), axis=1))
    external = config.dt*(jnp.sum(force, axis=0)+jnp.sum(state.mass)*gravity)
    v, C = grid_to_particle(corrected, support, layout)
    F = (jnp.eye(3, dtype=state.x.dtype)+config.dt*C)@state.F
    x = state.x+config.dt*v
    history, tau = update_brittle(state.history, F, material)
    future_base = jnp.floor((x-layout.origin)/layout.dx-.5).astype(jnp.int32)-layout.lower[state.fibre_id]
    future_inside = jnp.all((future_base >= 0) & (future_base+2 < layout.shape[state.fibre_id]))
    if layout.node_map is not None:
        future_inside = future_inside & stencil(x, state.fibre_id, layout).valid
    valid = (state.valid & support.valid & future_inside & result.valid & jnp.all(history.valid)
             & jnp.all(jnp.isfinite(force)) & jnp.all(jnp.isfinite(v)) & jnp.all(jnp.isfinite(C))
             & jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(F)) & jnp.all(jnp.linalg.det(F) > 0)
             & jnp.all(jnp.isclose(material.characteristic_length, layout.dx, rtol=1e-12, atol=0)))
    current = FibreState(x, v, F, C, state.mass, state.volume, state.fibre_id, history, tau,
        state.time+config.dt, valid, state.boundary_impulse+impulse,
        state.external_impulse+external, state.boundary_work+work)
    diagnostics = StepDiagnostics(impulse, external, work, result.group_impulses, result.group_work, mass.sum(), valid)
    return current, diagnostics


@partial(jax.jit, static_argnames=('config', 'boundary', 'force_fn'))
def step_with_diagnostics(state, material, layout, config, force=None, *, force_fn=None, boundary=None):
    return _step(state, material, layout, config, force, force_fn, boundary)


@partial(jax.jit, static_argnames=('config', 'boundary', 'force_fn'))
def step(state, material, layout, config, force=None, *, force_fn=None, boundary=None):
    return _step(state, material, layout, config, force, force_fn, boundary)[0]


@partial(jax.jit, static_argnames=('config', 'steps', 'boundary', 'force_fn'))
def advance(state, material, layout, config, steps, force=None, *, force_fn=None, boundary=None):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    def body(_, current):
        return jax.lax.cond(current.valid,
            lambda value: _step(value, material, layout, config, force, force_fn, boundary)[0],
            lambda value: value, current)

    return jax.lax.fori_loop(0, steps, body, state)


@partial(jax.jit, static_argnames=('config', 'steps', 'boundary', 'force_fn'))
def advance_with_diagnostics(state, material, layout, config, steps, force=None, *, force_fn=None, boundary=None):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer when collecting step diagnostics")
    groups = getattr(boundary, 'group_count', 0)

    def body(current, _):
        empty = StepDiagnostics(jnp.zeros(3), jnp.zeros(3), jnp.asarray(0.),
            jnp.zeros((groups, 3)), jnp.zeros(groups), jnp.asarray(0.), jnp.asarray(False))
        return jax.lax.cond(current.valid,
            lambda value: _step(value, material, layout, config, force, force_fn, boundary),
            lambda value: (value, empty), current)

    current, history = jax.lax.scan(body, state, None, length=steps)
    totals = StepDiagnostics(history.boundary_impulse.sum(0), history.external_impulse.sum(0), history.boundary_work.sum(),
        history.group_impulses.sum(0), history.group_work.sum(0), history.grid_mass[-1], history.valid.all())
    return current, totals


def validate_state(state, layout):
    x, ids = np.asarray(state.x), np.asarray(state.fibre_id)
    base = np.floor((x-np.asarray(layout.origin))/layout.dx-.5).astype(np.int64)-np.asarray(layout.lower)[ids]
    inside = np.all((base >= 0) & (base+2 < np.asarray(layout.shape)[ids]), axis=1)
    if not inside.all():
        particle = int(np.flatnonzero(~inside)[0])
        raise FloatingPointError(f"Particle {particle} in fibre {int(ids[particle])} has a quadratic stencil outside its packed grid; enlarge or rebuild the layout")
    if layout.node_map is not None:
        node_map = np.asarray(layout.node_map)
        neighbours = np.array(list(product(range(3), repeat=3)), dtype=np.int64)
        strides, offsets = np.asarray(layout.strides), np.asarray(layout.offsets)
        for start in range(0, len(x), 65536):
            selected = slice(start, start+65536)
            local = base[selected, None, :]+neighbours
            dense_ids = offsets[ids[selected], None]+np.sum(local*strides[ids[selected], None, :], axis=-1)
            missing = np.any(node_map[dense_ids] < 0, axis=1)
            if missing.any():
                particle = start+int(np.flatnonzero(missing)[0])
                raise FloatingPointError(f"Particle {particle} in fibre {int(ids[particle])} has a quadratic stencil outside its active tube; rebuild the layout before further integration")
    if not bool(state.valid) or not np.asarray(state.history.valid).all():
        raise FloatingPointError("Fibre MPM state is invalid; inspect deformation, boundary overlap, material length and applied forces")
    if any(not np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(state)):
        raise FloatingPointError("Fibre MPM contains nonfinite state arrays")
    if np.any(np.asarray(state.mass) <= 0) or np.any(np.asarray(state.volume) <= 0) or np.any(np.linalg.det(np.asarray(state.F)) <= 0):
        raise FloatingPointError("Fibre MPM requires positive mass, reference volume and deformation determinant")
    return True


def stored_energy(state, material):
    return jnp.sum(state.volume*brittle_energy(state.F, state.history, material))
