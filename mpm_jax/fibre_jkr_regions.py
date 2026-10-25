"""Persistent, conservative connected JKR contact; original Hertz is untouched.

Tracking envelopes are numerical identity regions, not extra adhesive range.
Virgin positive-gap cells never attach. Merges, splits, active disappearance,
and envelopes hiding several compressed regions require step rejection.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .contact_patches import contact_patches
from .fibre_contact import (
    FibreContactHistory, FibreContactParameters, FibreContactResponse, update_fibre_contact,
)
from .fibre_coupling import _curvature, _nonrigid_bound, closest_segments, section_centres
from .fibre_patch_contact import FibrePatchMetrics, _overlap_count
from .fibre_region_identity import RegionIdentity, match_region_identity


class JKRRegionHistory(NamedTuple):
    law: FibreContactHistory
    radius: jax.Array
    segments: jax.Array
    valid: jax.Array
    cell_slots: jax.Array
    region_ids: jax.Array
    fibre_pairs: jax.Array
    next_id: jax.Array


class JKRRegionMetrics(NamedTuple):
    contact: FibrePatchMetrics
    identity: RegionIdentity
    multiple_compressed_cores: jax.Array
    retention_bound_exceeded: jax.Array


def initial_jkr_region_history(data, max_patches=4096):
    if type(max_patches) is not int or max_patches < 1:
        raise ValueError("positive integer patch capacity required")
    return JKRRegionHistory(
        FibreContactHistory(jnp.zeros(max_patches, dtype=bool), jnp.ones(max_patches, dtype=bool)),
        jnp.zeros(max_patches, dtype=data.coverage_reference.dtype),
        jnp.full((max_patches, 2), -1, dtype=jnp.int32), jnp.asarray(True),
        jnp.full(len(data.coverage_pairs), -1, dtype=jnp.int32),
        jnp.full(max_patches, -1, dtype=jnp.int32),
        jnp.full((max_patches, 2), -1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))


def _cell_slots(selection, count, capacity, max_patches):
    live = selection.cells < count
    roots = live&(selection.labels == jnp.arange(capacity))
    root_ids = jnp.nonzero(roots, size=max_patches, fill_value=capacity)[0]
    root_slots = jnp.full(capacity+1, -1, dtype=jnp.int32).at[root_ids].set(
        jnp.arange(max_patches, dtype=jnp.int32))
    slots = jnp.where(live, root_slots[selection.labels], -1)
    return jnp.full(count+1, -1, dtype=jnp.int32).at[selection.cells].set(slots)[:count]


def jkr_region_checkpoint(history):
    """Host-only complete checkpoint fields; never omit persistent ownership."""
    values = {'jkr_region_'+key: np.asarray(value) for key, value in history._asdict().items()
              if key != 'law'}
    values.update({'jkr_region_law_'+key: np.asarray(value)
                   for key, value in history.law._asdict().items()})
    return values


def restore_jkr_region_checkpoint(template, checkpoint, array_module=jnp):
    """Strict host restore; graph identity must additionally be checked by caller."""
    expected = jkr_region_checkpoint(template)
    values = {}
    for key, reference in expected.items():
        if key not in checkpoint:
            raise ValueError('missing complete JKR region history: '+key)
        value = np.asarray(checkpoint[key])
        if (value.shape != reference.shape or value.dtype != reference.dtype
                or not np.all(np.isfinite(value))):
            raise ValueError('invalid JKR region checkpoint leaf: '+key)
        values[key] = array_module.asarray(value)
    if not bool(np.asarray(values['jkr_region_valid'])) or not np.all(values['jkr_region_law_valid']):
        raise ValueError('rejected JKR region history is not resumable')
    law = FibreContactHistory(values.pop('jkr_region_law_active'), values.pop('jkr_region_law_valid'))
    return JKRRegionHistory(law, *(values['jkr_region_'+key] for key in JKRRegionHistory._fields[1:]))


def evaluate_jkr_regions(state, history, data, law, neighbours, *, capacity=16384,
                           max_patches=4096, max_iterations=128, max_a_over_radius=.2,
                           tracking_gap=0., _response_update=update_fibre_contact):
    """Evaluate conservative connected JKR regions, including tensile history.

    A positive tracking_gap is required within the audited proximity margin.
    Each tracking-envelope component may contain at most one compressed core.
    Any topology ambiguity or lost active attachment rejects the whole step.
    This is a circular equivalent-curvature approximation, not elliptical JKR.
    """
    if not isinstance(law, FibreContactParameters) or not law.adhesive:
        raise ValueError("this connected path requires JKR; MD/SC remain unsupported")
    if np.prod(law.effective_modulus.shape) != 1 or np.prod(law.work_of_adhesion.shape) != 1:
        raise ValueError("connected JKR regions require uniform reduced modulus and work")
    if not np.isfinite(max_a_over_radius) or not 0 < max_a_over_radius < 1:
        raise ValueError("the local contact-radius ratio must lie strictly between zero and one")
    if (history.law.active.shape != (max_patches,) or history.law.valid.shape != (max_patches,)
            or history.radius.shape != (max_patches,) or history.segments.shape != (max_patches, 2)):
        raise ValueError("contact history shape differs from the declared patch capacity")
    if (not np.isfinite(tracking_gap) or tracking_gap <= 0
            or tracking_gap > data.proximity_margin):
        raise ValueError("positive JKR tracking gap must fit the audited proximity margin")
    if (history.cell_slots.shape != (len(data.coverage_pairs),)
            or history.region_ids.shape != (max_patches,)
            or history.fibre_pairs.shape != (max_patches, 2)):
        raise ValueError("persistent region history differs from coverage/capacity")
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
    selected = contact_patches(gap, gap <= tracking_gap, neighbours, capacity=capacity,
                              max_patches=max_patches, max_iterations=max_iterations)
    count = len(near)
    representatives = selected.representatives
    live = representatives < count
    gather = lambda value: jnp.concatenate((value, jnp.zeros_like(value[:1]) if count else
                                           jnp.zeros((1, *value.shape[1:]), dtype=value.dtype)))[representatives]
    first_patch, second_patch = gather(first), gather(second)
    weight_first, weight_second, distance_patch = gather(wa), gather(wb), gather(distance)
    pair_segments, pair_owners, pair_radii = gather(near), gather(owners), gather(radii)
    cell_slots = _cell_slots(selected, count, capacity, max_patches)
    identity = match_region_identity(history.cell_slots, cell_slots, history.region_ids,
        history.law.active, history.fibre_pairs,
        jnp.where(live[:, None], pair_owners, -1).astype(jnp.int32),
        history.next_id, selected.patch_count.astype(jnp.int32))
    # One tracking envelope must not hide multiple separated solid contacts.
    solid = contact_patches(gap, gap <= 0, neighbours, capacity=capacity,
                           max_patches=max_patches, max_iterations=max_iterations)
    padded_slots = jnp.concatenate((cell_slots, jnp.array([-1], dtype=jnp.int32)))
    core_slots = padded_slots[solid.representatives]
    core_count = jnp.bincount(jnp.where(core_slots >= 0, core_slots, max_patches),
                             length=max_patches+1)[:max_patches]
    multiple_cores = jnp.any(core_count > 1)
    previous_slot = jnp.maximum(identity.previous_slot, 0)
    previous_law = FibreContactHistory(
        (identity.previous_slot >= 0)&history.law.active[previous_slot],
        jnp.where(identity.previous_slot >= 0, history.law.valid[previous_slot], True))
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
    parameters = law._replace(effective_radius=jnp.where(live&curvature_valid, effective_radius, 1.))
    topology_valid = identity.valid&selected.valid&solid.valid&~multiple_cores

    def reject_topology(_):
        zero = jnp.zeros_like(indentation)
        false = jnp.zeros_like(live)
        return FibreContactHistory(false, false), FibreContactResponse(
            zero, zero, zero, zero, false, false, false, zero, zero, false)

    new_law, response = jax.lax.cond(topology_valid,
        lambda _: _response_update(previous_law, indentation, parameters),
        reject_topology, operand=None)
    active = live&response.active
    a = jnp.where(live, response.radius, 0.)
    force_magnitude = jnp.where(live, response.normal_force, 0.)
    potential = jnp.where(live, response.potential, 0.)
    # Detachment must be evaluated before an attachment exits the envelope.
    # This is a strict bound; an oversized step that loses overlap still rejects.
    retention_exceeded = jnp.any(live&(-response.critical_indentation >= tracking_gap))
    law_valid = jnp.all(jnp.where(live, response.valid, True))
    hertz_only = jnp.asarray(False)
    large_radius = live&(a > max_a_over_radius*jnp.min(pair_radii, axis=1))
    contact_points = jnp.stack((first_patch+pair_radii[:, :1]*normal,
                                second_patch-pair_radii[:, 1:]*normal), axis=1)
    overlaps = _overlap_count(contact_points, pair_owners, a, active, selected.patch_count)

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
    failed_contact = close&jnp.any(failed[endpoint_sections], axis=(1, 2))
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
             &selected.valid&solid.valid&parameters_valid&identity.valid&law_valid
             &~multiple_cores&~retention_exceeded&jnp.all(jnp.isfinite(centres))&jnp.all(section_mass > 0)
             &jnp.all(jnp.isfinite(force))&~jnp.any(unseeded)&~jnp.any(same_family)&~motion_exceeded
             &~jnp.any(end_contact)&~jnp.any(failed_contact)&~jnp.any(degenerate_all)&~jnp.any(degenerate_patch)
             &~jnp.any(large_radius)&(overlaps == 0))
    active_overflow, patch_overflow = selected.active_count > capacity, selected.patch_count > max_patches
    iteration_exhausted = ~selected.valid&~active_overflow&~patch_overflow&(selected.iterations >= max_iterations)
    stored_segments = jnp.where(live[:, None], pair_segments, -1)
    updated = JKRRegionHistory(
        FibreContactHistory(active, jnp.full(max_patches, valid)), a, stored_segments, valid,
        cell_slots, identity.region_ids,
        jnp.where(live[:, None], pair_owners, -1).astype(jnp.int32), identity.next_id)
    minimum_gap = jnp.min(gap) if count else jnp.asarray(0., dtype=state.x.dtype)
    metrics = FibrePatchMetrics(valid, jnp.zeros(max_patches, dtype=bool), jnp.sum(unseeded), jnp.sum(same_family),
        motion_exceeded, displacement, jnp.concatenate((degenerate_all, degenerate_patch)), end_contact, failed_contact,
        jnp.where(live, gather(gap), jnp.inf), normal, jnp.stack((first_patch, second_patch), axis=1),
        jnp.where(live, effective_radius, 0.), jnp.where(live[:, None], curvatures, 0.), force_magnitude, a, active, potential,
        stored_segments, jnp.sum(force, axis=0), jnp.sum(jnp.cross(state.x, force), axis=0), minimum_gap,
        selected.active_count, selected.patch_count, active_overflow, patch_overflow, iteration_exhausted,
        selected.iterations, overlaps, large_radius, hertz_only)
    return force, updated, JKRRegionMetrics(metrics, identity, multiple_cores, retention_exceeded)
