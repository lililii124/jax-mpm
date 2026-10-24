"""Isolated contrast experiment coupling; original fibre coupling unchanged.

Copied explicit evaluator retains geometry/search/force/torque checks. Only the
SC nonlinear iteration ceiling is exposed (8192); the original1e-8 convergence,
compatibility and energy requirements remain unchanged. Physical equations and
hysteretic ring state are unchanged; unconverged states are still rejected.
"""
import numpy as np
import jax.numpy as jnp
from .fibre_contact import FibreContactParameters,update_fibre_contact
from .fibre_coupling import (FibreCouplingHistory,FibreCouplingMetrics,section_centres,
                            closest_segments,_curvature,_nonrigid_bound)


def evaluate_contact(state, history, data, law, *, max_sweeps=8192):
    """Return force, updated attachment/search history and explicit validity flags.

    A false metrics.valid requires rejecting the coupled step. The complete
    coverage list can locate a previously inactive contact outside its local
    basin; active histories cannot jump to a different basin. No force is
    silently accepted for an exhausted window or an additional close region.
    Physical end-cap contacts and contacts on failed sections are unsupported.
    """
    centres, section_mass = section_centres(state.x, state.mass, data)
    offset = jnp.arange(-data.search_window, data.search_window+1)
    lo = data.fibre_segment_bounds[data.seed_fibres, 0]
    hi = data.fibre_segment_bounds[data.seed_fibres, 1]-1
    ids = jnp.clip(history.segments[:, :, None]+offset, lo[:, :, None], hi[:, :, None])
    left, right = ids[:, 0, :, None], ids[:, 1, None, :]
    p, q = centres[data.segments[left, 0]], centres[data.segments[left, 1]]
    r, s = centres[data.segments[right, 0]], centres[data.segments[right, 1]]
    a, b, wa, wb, distance = closest_segments(p, q, r, s)
    combinations = (2*data.search_window+1)**2
    choice = jnp.argmin(distance.reshape(data.pair_count, combinations), axis=1)
    pick = lambda values: jnp.take_along_axis(values.reshape(data.pair_count, combinations), choice[:, None], axis=1)[:, 0]
    coordinates = lambda values: jnp.take_along_axis(values.reshape(data.pair_count, combinations, 3), choice[:, None, None], axis=1)[:, 0]
    first, second, wa, wb, distance = coordinates(a), coordinates(b), pick(wa), pick(wb), pick(distance)
    selected = jnp.stack((pick(jnp.broadcast_to(left, a.shape[:-1])), pick(jnp.broadcast_to(right, a.shape[:-1]))), axis=-1)
    window_lower, window_upper = ids[:, :, 0], ids[:, :, -1]
    at_edge = ((selected == window_lower) & (window_lower > lo)) | ((selected == window_upper) & (window_upper < hi))
    near = data.coverage_pairs
    cover_first, cover_second, cover_wa, cover_wb, cover_distance = closest_segments(
        centres[data.segments[near[:, 0], 0]], centres[data.segments[near[:, 0], 1]],
        centres[data.segments[near[:, 1], 0]], centres[data.segments[near[:, 1], 1]])
    use_global = jnp.zeros(data.pair_count, dtype=bool)
    if data.pair_count and near.shape[0]:
        seed = jnp.maximum(data.coverage_seed, 0)
        registered = data.coverage_seed >= 0
        minimum = jnp.full(data.pair_count, jnp.inf, dtype=distance.dtype).at[seed].min(
            jnp.where(registered, cover_distance, jnp.inf))
        indices = jnp.arange(near.shape[0], dtype=jnp.int32)
        winner = jnp.full(data.pair_count, near.shape[0], dtype=jnp.int32).at[seed].min(
            jnp.where(registered & (cover_distance == minimum[seed]), indices, near.shape[0]))
        available = winner < near.shape[0]
        winner = jnp.minimum(winner, near.shape[0]-1)
        use_global = available & (minimum <= distance)
        first = jnp.where(use_global[:, None], cover_first[winner], first)
        second = jnp.where(use_global[:, None], cover_second[winner], second)
        wa, wb = jnp.where(use_global, cover_wa[winner], wa), jnp.where(use_global, cover_wb[winner], wb)
        distance = jnp.where(use_global, minimum, distance)
        selected = jnp.where(use_global[:, None], near[winner], selected)
    relocated = jnp.any((selected < window_lower) | (selected > window_upper), axis=-1)
    invalid_history = jnp.any((history.segments < lo) | (history.segments > hi), axis=-1)
    boundary = invalid_history | (history.law.active & relocated) | (jnp.any(at_edge, axis=-1) & ~use_global)
    if not isinstance(law, FibreContactParameters):
        # Virgin finite-range contacts can attach before solid surfaces touch.
        # Previously active patches retain their own history beyond this range.
        approach_range = jnp.where(law.work_of_adhesion > 0, law.interaction_range, 0.)
        capture_valid = (jnp.all(jnp.isfinite(approach_range))
                         & jnp.all(approach_range >= 0)
                         & jnp.all(approach_range <= data.proximity_margin))
        boundary = boundary | ~capture_valid
    audit_lower, audit_upper = jnp.maximum(selected-data.search_window, lo), jnp.minimum(selected+data.search_window, hi)
    directions = centres[data.segments[selected, 1]]-centres[data.segments[selected, 0]]
    lengths = jnp.linalg.norm(directions, axis=-1)
    tangents = directions/jnp.maximum(lengths, 1e-30)[..., None]
    normal = (second-first)/jnp.maximum(distance, 1e-30)[:, None]
    radius, curvature, curvature_valid = _curvature(tangents[:, 0], tangents[:, 1], normal, data.fibre_radius[data.seed_fibres])
    gap = distance-jnp.sum(data.fibre_radius[data.seed_fibres], axis=-1)
    parameters = law._replace(effective_radius=radius)
    if isinstance(law, FibreContactParameters):
        new_law, response = update_fibre_contact(history.law, -gap, parameters)
    else:
        from .local_contact_models import update_local_contact
        new_law, response = update_local_contact(history.law, -gap, parameters,max_sweeps=max_sweeps,tolerance=1e-8)
    pair_force = response.normal_force[:, None]*normal
    weights = jnp.stack((jnp.stack((1-wa, wa), axis=-1), jnp.stack((1-wb, wb), axis=-1)), axis=1)
    endpoint_sections = data.segments[selected]
    signs = jnp.array([-1., 1.], dtype=state.x.dtype)
    endpoint_forces = weights[..., None]*signs[None, :, None, None]*pair_force[:, None, None, :]
    section_force = jnp.zeros_like(centres).at[endpoint_sections.reshape(-1)].add(endpoint_forces.reshape(-1, 3))
    forces = section_force[data.particle_section]*state.mass[:, None]/jnp.maximum(section_mass[data.particle_section], 1e-30)[:, None]
    pair_owners = data.segment_fibre[near]
    close = cover_distance <= jnp.sum(data.fibre_radius[pair_owners], axis=-1)+data.proximity_margin
    if data.pair_count:
        seed = jnp.maximum(data.coverage_seed, 0)
        covered = (data.coverage_seed >= 0) & jnp.all((near >= audit_lower[seed]) & (near <= audit_upper[seed]), axis=-1)
    else:
        covered = jnp.zeros(near.shape[0], dtype=bool)
    uncovered = close & ~covered
    same_family = data.fibre_family[pair_owners[:, 0]] == data.fibre_family[pair_owners[:, 1]]
    displacement = _nonrigid_bound(data.coverage_reference, centres)
    motion_exceeded = 2*displacement > data.coverage_skin
    near_contact = response.active | (gap <= data.proximity_margin)
    endpoint_tolerance = 64*np.finfo(float).eps
    endpoints = (((selected[:, 0] == lo[:, 0]) & (wa <= endpoint_tolerance)) | ((selected[:, 0] == hi[:, 0]) & (wa >= 1-endpoint_tolerance))
                 | ((selected[:, 1] == lo[:, 1]) & (wb <= endpoint_tolerance)) | ((selected[:, 1] == hi[:, 1]) & (wb >= 1-endpoint_tolerance)))
    failed_sections = jnp.zeros(data.section_count, dtype=jnp.int32).at[data.particle_section].add(state.history.failed.astype(jnp.int32)) > 0
    failed_contact = near_contact & jnp.any(failed_sections[endpoint_sections], axis=(1, 2))
    degenerate = (distance <= 1e-14) | jnp.any(lengths <= 1e-14, axis=-1) | (near_contact & ~curvature_valid)
    valid = (state.valid & history.valid & jnp.all(new_law.valid) & jnp.all(jnp.isfinite(centres)) & jnp.all(section_mass > 0)
             & jnp.all(jnp.isfinite(forces)) & jnp.all((history.segments >= lo) & (history.segments <= hi))
             & ~jnp.any(boundary) & ~jnp.any(uncovered) & ~motion_exceeded & ~jnp.any(degenerate)
             & ~jnp.any(endpoints & near_contact) & ~jnp.any(failed_contact))
    if not isinstance(law, FibreContactParameters):
        valid = valid & capture_valid
    valid = valid & jnp.all(response.radius <= .2*jnp.min(data.fibre_radius[data.seed_fibres],axis=1))
    updated = FibreCouplingHistory(new_law, response.radius, selected.astype(history.segments.dtype), valid)
    metrics = FibreCouplingMetrics(valid, boundary, jnp.sum(uncovered), jnp.sum(uncovered & same_family), motion_exceeded,
        displacement, degenerate, endpoints & near_contact, failed_contact, gap, normal, jnp.stack((first, second), axis=1),
        radius, curvature, response.normal_force, response.radius, response.active, response.potential, selected,
        jnp.sum(forces, axis=0), jnp.sum(jnp.cross(state.x, forces), axis=0))
    return forces, updated, metrics

