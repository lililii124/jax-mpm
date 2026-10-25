"""Stateless Hertz forces on separate discrete centreline contact regions.

Each connected active region contributes one circular equivalent-curvature
contact, not one force per coverage cell. This remains a local approximation;
large contact radii and overlapping footprints are rejected. Adhesive patch
history, coalescence and splitting are outside this module.
The reported fixed-curvature Hertz potential omits work from changing effective
curvature; force/torque balance is not a claim of global energy conservation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .contact_patches import contact_patches
from .fibre_contact import FibreContactHistory, FibreContactParameters
from .fibre_coupling import (
    FibreCouplingHistory, _curvature, _nonrigid_bound, closest_segments, section_centres,
)


def closed_footprint_contains(point, surface_points, normal, radius, starts, directions, cylinder_radii):
    """Closed circular footprint AND both finite cylindrical interiors; no tolerance."""
    centre = jnp.mean(surface_points, axis=-2)
    delta = point-centre
    height = jnp.sum(delta*normal, axis=-1)
    radial2 = jnp.sum(delta*delta, axis=-1)-height*height
    heights = jnp.sum((surface_points-centre[..., None, :])*normal[..., None, :], axis=-1)
    relative = point[..., None, :]-starts
    length2 = jnp.sum(directions*directions, axis=-1)
    axial = jnp.sum(relative*directions, axis=-1)/jnp.maximum(length2, 1e-30)
    off_axis = relative-axial[..., None]*directions
    cylinders = jnp.all((axial >= 0)&(axial <= 1)&
        (jnp.sum(off_axis*off_axis, axis=-1) <= cylinder_radii*cylinder_radii), axis=-1)
    return ((radial2 <= radius*radius)&(height >= jnp.min(heights, axis=-1))&
            (height <= jnp.max(heights, axis=-1))&cylinders)


class FibrePatchMetrics(NamedTuple):
    valid: jax.Array
    search_boundary: jax.Array
    uncovered_near_pairs: jax.Array
    same_family_near_pairs: jax.Array
    coverage_motion_exceeded: jax.Array
    maximum_nonrigid_displacement: jax.Array
    degenerate_geometry: jax.Array
    unsupported_end_contact: jax.Array
    failed_section_contact: jax.Array
    gap: jax.Array
    normal: jax.Array
    closest_points: jax.Array
    effective_radius: jax.Array
    principal_curvatures: jax.Array
    normal_force: jax.Array
    contact_radius: jax.Array
    active: jax.Array
    potential: jax.Array
    pair_segments: jax.Array
    force_residual: jax.Array
    torque_residual: jax.Array
    minimum_coverage_gap: jax.Array
    active_cell_count: jax.Array
    patch_count: jax.Array
    active_capacity_exceeded: jax.Array
    patch_capacity_exceeded: jax.Array
    component_iteration_exhausted: jax.Array
    component_iterations: jax.Array
    overlapping_patch_pairs: jax.Array
    large_contact_radius: jax.Array
    hertz_only: jax.Array


def initial_patch_history(data, max_patches=4096):
    if type(max_patches) is not int or max_patches < 1:
        raise ValueError("max_patches must be a positive integer")
    return FibreCouplingHistory(
        FibreContactHistory(jnp.zeros(max_patches, dtype=bool), jnp.ones(max_patches, dtype=bool)),
        jnp.zeros(max_patches, dtype=data.coverage_reference.dtype),
        jnp.full((max_patches, 2), -1, dtype=data.segments.dtype), jnp.asarray(True))


def _overlap_count(points, owners, radius, active, patch_count, block=32):
    """Count overlapping circular footprints on a shared fibre, in small blocks."""
    n = len(radius)
    rows = (n+block-1)//block
    padded_n = rows*block
    point_pad = jnp.pad(points, ((0, padded_n-n), (0, 0), (0, 0)))
    owner_pad = jnp.pad(owners, ((0, padded_n-n), (0, 0)), constant_values=-1)
    radius_pad = jnp.pad(radius, (0, padded_n-n))
    active_pad = jnp.pad(active, (0, padded_n-n))
    right_index = jnp.arange(n)

    def count_block(index, count):
        offset = index*block
        left_points = jax.lax.dynamic_slice_in_dim(point_pad, offset, block)
        left_owners = jax.lax.dynamic_slice_in_dim(owner_pad, offset, block)
        left_radius = jax.lax.dynamic_slice_in_dim(radius_pad, offset, block)
        left_active = jax.lax.dynamic_slice_in_dim(active_pad, offset, block)
        allowed = left_active[:, None]&active[None, :]&((offset+jnp.arange(block))[:, None] < right_index)
        overlap = jnp.zeros((block, n), dtype=bool)
        for side in range(2):
            for other in range(2):
                distance2 = jnp.sum((left_points[:, None, side]-points[None, :, other])**2, axis=-1)
                overlap |= (left_owners[:, None, side] == owners[None, :, other]) & (
                    distance2 <= (left_radius[:, None]+radius[None, :])**2)
        return count+jnp.sum(allowed&overlap)

    return jax.lax.fori_loop(0, jnp.minimum((patch_count+block-1)//block, rows), count_block, jnp.asarray(0))


def load_bearing_failed_endpoints(failed_endpoints,closest_weights):
    """Exact zero force interpolation is inactive; every positive weight counts.

    No epsilon or rounding tolerance is permitted here. Footprint/intactness
    review is still required before admitting a changed physical application.
    """
    weights=jnp.stack((1-closest_weights,closest_weights),axis=-1)
    return jnp.any(failed_endpoints&(weights>0),axis=(1,2))


def evaluate_hertz_patches(state, history, data, law, neighbours, *, capacity=16384,
                           max_patches=4096, max_iterations=128, max_a_over_radius=.2):
    """Return force/history/metrics; a false validity flag rejects the whole step.

    The static graph must be produced by ``patch_neighbours`` for this coverage
    list. Patch slots can change identity: only stateless, uniform-modulus Hertz
    contact is supported. Gap/normal/curvature come from the current centrelines.
    Padding segments are -1 and inactive. The finite minimum_coverage_gap is the
    full coverage minimum, or zero when the coverage list itself is empty.
    """
    if not isinstance(law, FibreContactParameters):
        raise ValueError("connected contact regions currently support Hertz parameters only")
    if np.prod(law.effective_modulus.shape) != 1:
        raise ValueError("connected Hertz regions require one uniform reduced modulus")
    if not np.isfinite(max_a_over_radius) or not 0 < max_a_over_radius < 1:
        raise ValueError("the local contact-radius ratio must lie strictly between zero and one")
    if (history.law.active.shape != (max_patches,) or history.law.valid.shape != (max_patches,)
            or history.radius.shape != (max_patches,) or history.segments.shape != (max_patches, 2)):
        raise ValueError("contact history shape differs from the declared patch capacity")
    centres, section_mass = section_centres(state.x, state.mass, data)
    near = data.coverage_pairs
    owners = data.segment_fibre[near]
    radii = data.fibre_radius[owners]
    endpoint_sections = data.segments[near]
    p, q = centres[endpoint_sections[:, 0, 0]], centres[endpoint_sections[:, 0, 1]]
    r, s = centres[endpoint_sections[:, 1, 0]], centres[endpoint_sections[:, 1, 1]]
    first, second, wa, wb, distance = closest_segments(p, q, r, s)
    gap = distance-jnp.sum(radii, axis=1)
    close = gap <= data.proximity_margin
    selected = contact_patches(gap, gap < 0, neighbours, capacity=capacity,
                              max_patches=max_patches, max_iterations=max_iterations)
    count = len(near)
    representatives = selected.representatives
    live = representatives < count
    gather = lambda value: jnp.concatenate((value, jnp.zeros_like(value[:1]) if count else
                                           jnp.zeros((1, *value.shape[1:]), dtype=value.dtype)))[representatives]
    first_patch, second_patch = gather(first), gather(second)
    weight_first, weight_second, distance_patch = gather(wa), gather(wb), gather(distance)
    pair_segments, pair_owners, pair_radii = gather(near), gather(owners), gather(radii)
    pair_sections = data.segments[pair_segments]
    directions = centres[pair_sections[:, :, 1]]-centres[pair_sections[:, :, 0]]
    lengths = jnp.linalg.norm(directions, axis=-1)
    tangents = directions/jnp.maximum(lengths, 1e-30)[..., None]
    normal = (second_patch-first_patch)/jnp.maximum(distance_patch, 1e-30)[:, None]
    effective_radius, curvatures, curvature_valid = _curvature(
        tangents[:, 0], tangents[:, 1], normal, jnp.where(live[:, None], pair_radii, 1.))
    indentation = jnp.where(live, -gather(gap), 0.)
    E = jnp.reshape(law.effective_modulus, ())
    parameters_valid = jnp.isfinite(E)&(E > 0)&jnp.all(jnp.isfinite(law.work_of_adhesion))&jnp.all(law.work_of_adhesion >= 0)
    hertz_only = ~jnp.asarray(law.adhesive, dtype=bool)
    a = jnp.sqrt(jnp.maximum(indentation, 0.)*jnp.where(live&curvature_valid, effective_radius, 1.))
    force_magnitude = jnp.where(live, (4/3)*jnp.where(parameters_valid, E, 1.)*a*indentation, 0.)
    potential = .4*force_magnitude*indentation
    large_radius = live&(a > max_a_over_radius*jnp.min(pair_radii, axis=1))
    contact_points = jnp.stack((first_patch+pair_radii[:, :1]*normal,
                                second_patch-pair_radii[:, 1:]*normal), axis=1)
    overlaps = _overlap_count(contact_points, pair_owners, a, live, selected.patch_count)

    pair_force = force_magnitude[:, None]*normal
    weights = jnp.stack((jnp.stack((1-weight_first, weight_first), axis=-1),
                         jnp.stack((1-weight_second, weight_second), axis=-1)), axis=1)
    endpoint_force = weights[..., None]*jnp.array([-1., 1.])[None, :, None, None]*pair_force[:, None, None, :]
    section_force = jnp.zeros_like(centres).at[pair_sections.reshape(-1)].add(endpoint_force.reshape(-1, 3))
    force = section_force[data.particle_section]*state.mass[:, None]/jnp.maximum(section_mass[data.particle_section], 1e-30)[:, None]

    lo = data.fibre_segment_bounds[owners, 0]
    hi = data.fibre_segment_bounds[owners, 1]-1
    weights_all = jnp.stack((wa, wb), axis=1)
    endpoint_tolerance = 64*np.finfo(float).eps
    end_contact = close&jnp.any(((near == lo)&(weights_all <= endpoint_tolerance))|
                              ((near == hi)&(weights_all >= 1-endpoint_tolerance)), axis=1)
    failed = jnp.zeros(data.section_count, dtype=jnp.int32).at[data.particle_section].add(state.history.failed.astype(jnp.int32)) > 0
    failed_contact = close&load_bearing_failed_endpoints(failed[endpoint_sections], weights_all)

    def audit_redundant_search_cells(base):
        # Search cells are not additional Hertz forces. Only a closed point already
        # inside the SAME healthy live component footprint may be redundant.
        labels = jnp.full(count+1, capacity, dtype=jnp.int32).at[selected.cells].set(selected.labels)
        roots = labels[representatives]
        patches = jnp.full(capacity+1, max_patches, dtype=jnp.int32).at[
            jnp.where(live, roots, capacity)].min(jnp.where(live, jnp.arange(max_patches), max_patches))
        cells = jnp.nonzero(base, size=capacity, fill_value=count)[0]
        patch = patches[labels[cells]]
        pad = lambda value: jnp.concatenate((value, jnp.zeros_like(value[:1])), axis=0)[patch]
        take = lambda value: jnp.concatenate((value, jnp.zeros_like(value[:1])), axis=0)[cells]
        n_search = (take(second)-take(first))/jnp.maximum(take(distance), 1e-30)[:, None]
        point = .5*(take(first)+take(radii)[:, :1]*n_search+
                    take(second)-take(radii)[:, 1:]*n_search)
        damage = jnp.zeros(data.section_count, dtype=state.x.dtype).at[data.particle_section].max(state.history.damage)
        healthy = live&jnp.all(jnp.where(weights > 0, damage[pair_sections] == 0, True), axis=(1, 2))
        represented = ((cells < count)&(patch < max_patches)&pad(healthy)&
            jnp.all(take(owners) == pad(pair_owners), axis=1)&
            (cells != jnp.concatenate((representatives, jnp.array([count])))[patch])&
            closed_footprint_contains(point, pad(contact_points), pad(normal), pad(a),
                pad(centres[pair_sections[:, :, 0]]), pad(directions), pad(pair_radii)))
        exempt = jnp.zeros(count+1, dtype=jnp.int32).at[cells].max(represented.astype(jnp.int32))[:count] > 0
        return base&~exempt

    failed_contact = jax.lax.cond(jnp.any(failed_contact), audit_redundant_search_cells,
                                 lambda value: value, failed_contact)
    lengths_all = jnp.stack((jnp.linalg.norm(q-p, axis=1), jnp.linalg.norm(s-r, axis=1)), axis=1)
    degenerate_all = close&((distance <= 1e-14)|jnp.any(lengths_all <= 1e-14, axis=1))
    degenerate_patch = live&~curvature_valid
    unseeded = close&(data.coverage_seed < 0)
    same_family = close&(data.fibre_family[owners[:, 0]] == data.fibre_family[owners[:, 1]])
    displacement = _nonrigid_bound(data.coverage_reference, centres)
    motion_exceeded = 2*displacement > data.coverage_skin
    previous_indices_valid = jnp.all(jnp.where(history.law.active[:, None],
                                      (history.segments >= 0)&(history.segments < len(data.segments)), True))
    valid = (state.valid&history.valid&jnp.all(history.law.valid)&previous_indices_valid
             &jnp.all(jnp.isfinite(history.radius))&jnp.all(history.radius >= 0)
             &selected.valid&parameters_valid&hertz_only&jnp.all(jnp.isfinite(centres))&jnp.all(section_mass > 0)
             &jnp.all(jnp.isfinite(force))&~jnp.any(unseeded)&~jnp.any(same_family)&~motion_exceeded
             &~jnp.any(end_contact)&~jnp.any(failed_contact)&~jnp.any(degenerate_all)&~jnp.any(degenerate_patch)
             &~jnp.any(large_radius)&(overlaps == 0))
    active_overflow, patch_overflow = selected.active_count > capacity, selected.patch_count > max_patches
    iteration_exhausted = ~selected.valid&~active_overflow&~patch_overflow&(selected.iterations >= max_iterations)
    stored_segments = jnp.where(live[:, None], pair_segments, -1)
    updated = FibreCouplingHistory(FibreContactHistory(live, jnp.full(max_patches, valid)), a, stored_segments, valid)
    minimum_gap = jnp.min(gap) if count else jnp.asarray(0., dtype=state.x.dtype)
    metrics = FibrePatchMetrics(valid, jnp.zeros(max_patches, dtype=bool), jnp.sum(unseeded), jnp.sum(same_family),
        motion_exceeded, displacement, jnp.concatenate((degenerate_all, degenerate_patch)), end_contact, failed_contact,
        jnp.where(live, gather(gap), jnp.inf), normal, jnp.stack((first_patch, second_patch), axis=1),
        jnp.where(live, effective_radius, 0.), jnp.where(live[:, None], curvatures, 0.), force_magnitude, a, live, potential,
        stored_segments, jnp.sum(force, axis=0), jnp.sum(jnp.cross(state.x, force), axis=0), minimum_gap,
        selected.active_count, selected.patch_count, active_overflow, patch_overflow, iteration_exhausted,
        selected.iterations, overlaps, large_radius, hertz_only)
    return force, updated, metrics
