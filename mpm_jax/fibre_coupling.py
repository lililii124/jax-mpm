"""Deformed centreline contact between independently moving circular fibres.

The circular equivalent-curvature law is not an exact elliptical JKR solution.
Radii remain fixed under the small cross-sectional-strain approximation. Updating
curvature omits its contribution to the geometric potential gradient; pair force
and torque balance do not imply conservation of the full contact energy.
"""

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .fibre_contact import initial_fibre_contact_history, update_fibre_contact


@partial(jax.tree_util.register_dataclass,
         data_fields=("particle_section", "section_fibre", "segments", "segment_fibre",
                      "fibre_segment_bounds", "fibre_radius", "fibre_family", "seed_fibres",
                      "seed_segments", "coverage_pairs", "coverage_seed", "coverage_reference"),
         meta_fields=("section_count", "particle_count", "pair_count", "search_window",
                      "coverage_skin", "proximity_margin"))
@dataclass(frozen=True)
class FibreContactData:
    particle_section: jax.Array
    section_fibre: jax.Array
    segments: jax.Array
    segment_fibre: jax.Array
    fibre_segment_bounds: jax.Array
    fibre_radius: jax.Array
    fibre_family: jax.Array
    seed_fibres: jax.Array
    seed_segments: jax.Array
    coverage_pairs: jax.Array
    coverage_seed: jax.Array
    coverage_reference: jax.Array
    section_count: int
    particle_count: int
    pair_count: int
    search_window: int
    coverage_skin: float
    proximity_margin: float


class FibreCouplingHistory(NamedTuple):
    law: object
    radius: jax.Array
    segments: jax.Array
    valid: jax.Array


class FibreCouplingMetrics(NamedTuple):
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


def _closest_segments(p, q, r, s, xp):
    """Exact finite-segment distance, including all four endpoint constraints."""
    u, v, w = q-p, s-r, p-r
    dot = lambda a, b: xp.sum(a*b, axis=-1)
    a, b, c, d, e = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    a, b, c, d, e = xp.broadcast_arrays(a, b, c, d, e)
    aa, cc = xp.where(a > 0, a, 1.), xp.where(c > 0, c, 1.)
    determinant = a*c-b*b
    interior = determinant > 32*np.finfo(float).eps*a*c
    den = xp.where(interior, determinant, 1.)
    si, ti = (b*e-c*d)/den, (a*e-b*d)/den
    ss = xp.stack((si, xp.zeros_like(a), xp.ones_like(a), xp.clip(-d/aa, 0., 1.), xp.clip((b-d)/aa, 0., 1.)), axis=-1)
    tt = xp.stack((ti, xp.clip(e/cc, 0., 1.), xp.clip((e+b)/cc, 0., 1.), xp.zeros_like(a), xp.ones_like(a)), axis=-1)
    first = p[..., None, :]+ss[..., None]*u[..., None, :]
    second = r[..., None, :]+tt[..., None]*v[..., None, :]
    distances = xp.sum((second-first)**2, axis=-1)
    valid_interior = interior & (si >= 0) & (si <= 1) & (ti >= 0) & (ti <= 1)
    distances = xp.concatenate((xp.where(valid_interior, distances[..., 0], xp.inf)[..., None], distances[..., 1:]), axis=-1)
    index = xp.argmin(distances, axis=-1)
    pick = lambda value: xp.take_along_axis(value, index[..., None], axis=-1)[..., 0]
    aweight, bweight = pick(ss), pick(tt)
    first, second = p+aweight[..., None]*u, r+bweight[..., None]*v
    return first, second, aweight, bweight, xp.sqrt(xp.maximum(pick(distances), 0.))


def closest_segments(first_start, first_end, second_start, second_end):
    return _closest_segments(first_start, first_end, second_start, second_end, jnp)


def section_centres(x, mass, data):
    totals = jnp.zeros(data.section_count, dtype=mass.dtype).at[data.particle_section].add(mass)
    moment = jnp.zeros((data.section_count, 3), dtype=x.dtype).at[data.particle_section].add(mass[:, None]*x)
    return moment/jnp.where(totals > 0, totals, 1.)[:, None], totals


def _host_centres(x, mass, sections, count):
    totals = np.bincount(sections, weights=mass, minlength=count)
    centres = np.column_stack([np.bincount(sections, weights=mass*x[:, axis], minlength=count)/totals for axis in range(3)])
    return centres


def _coverage_neighbours(centres, segments, owners, radii, skin, margin, max_pairs):
    """A reference capsule neighbour list; the runtime displacement bound is mandatory."""
    cell = 2*float(np.max(radii))+skin+margin
    lower = np.minimum(centres[segments[:, 0]], centres[segments[:, 1]])
    upper = np.maximum(centres[segments[:, 0]], centres[segments[:, 1]])
    tolerance = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(centres))))
    expansion = radii[owners]+.5*(skin+margin)+tolerance
    first = np.floor((lower-expansion[:, None])/cell).astype(np.int64)
    last = np.floor((upper+expansion[:, None])/cell).astype(np.int64)
    bins, candidates = {}, []
    for index, (lo, hi) in enumerate(zip(first, last)):
        if np.prod(hi-lo+1) > 4096:
            raise ValueError("A centreline segment spans too many coverage cells")
        keys = list(product(*(range(int(a), int(b)+1) for a, b in zip(lo, hi))))
        prior = set()
        for key in keys:
            prior.update(bins.get(key, ()))
        prior = np.asarray(sorted(other for other in prior if owners[other] != owners[index]), dtype=np.int32)
        if len(prior):
            p, q = centres[segments[prior, 0]], centres[segments[prior, 1]]
            r, s = centres[segments[index]]
            distance = _closest_segments(p, q, r, s, np)[-1]
            close = prior[distance <= radii[owners[prior]]+radii[owners[index]]+skin+margin+tolerance]
            candidates.extend((int(other), index) for other in close)
            if len(candidates) > max_pairs:
                raise ValueError(f"Contact coverage exceeds max_coverage_pairs={max_pairs}")
        for key in keys:
            bins.setdefault(key, []).append(index)
    return np.asarray(candidates, dtype=np.int32).reshape(-1, 2)


def build_contact_data(geometry, *, search_window=4, coverage_skin=None, proximity_margin=0.,
                       coverage_positions=None, max_coverage_pairs=2_000_000):
    """Build section membership and audited search coverage without a shared grid.

    ``coverage_positions`` can refresh the skin reference after a controlled host
    rebuild. Preserve the existing coupling history when refreshing identical
    fibre/seed IDs. New candidate pairs need explicitly initialized histories.
    """
    x = np.asarray(geometry['x'], dtype=np.float64)
    mass = np.asarray(geometry['mass'], dtype=np.float64)
    ids = np.asarray(geometry['fibre_id'])
    fibres = geometry['metadata']['fibres']
    cross_count = geometry['metadata']['cross_section_particles']
    if not fibres or not isinstance(cross_count, int) or cross_count < 1:
        raise ValueError("Nonempty fibre sections with a positive cross-section particle count are required")
    if (x.ndim != 2 or x.shape[1] != 3 or mass.shape != (len(x),) or ids.shape != mass.shape or ids.dtype.kind not in 'iu'
            or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass <= 0)):
        raise ValueError("Finite particle coordinates and positive masses are required")
    if not isinstance(search_window, int) or isinstance(search_window, bool) or search_window < 1:
        raise ValueError("search_window must be a positive integer")
    if not isinstance(max_coverage_pairs, int) or max_coverage_pairs < 1:
        raise ValueError("max_coverage_pairs must be a positive integer")
    radii = np.array([item['radius_m'] for item in fibres], dtype=float)
    skin = float(np.max(radii)) if coverage_skin is None else float(coverage_skin)
    if np.any(radii <= 0) or not np.isfinite(radii).all() or not np.isfinite([skin, proximity_margin]).all() or skin <= 0 or proximity_margin < 0:
        raise ValueError("Radii/coverage_skin must be positive and proximity_margin nonnegative")
    sections, section_owners, segments, segment_owners, bounds = [], [], [], [], []
    particle_offset, section_offset, segment_offset = 0, 0, 0
    for index, fibre in enumerate(fibres):
        n = fibre['longitudinal_sections']
        count = fibre['particle_count']
        declared_slice = fibre.get('particle_slice', [particle_offset, particle_offset+count])
        if (fibre['id'] != index or n < 2 or count != n*cross_count
                or declared_slice != [particle_offset, particle_offset+count]
                or not np.all(ids[particle_offset:particle_offset+count] == index)):
            raise ValueError("Fibre sections must be contiguous and agree with particle IDs/counts")
        sections.extend(np.repeat(section_offset+np.arange(n), cross_count))
        section_owners.extend([index]*n)
        segments.extend(zip(section_offset+np.arange(n-1), section_offset+np.arange(1, n)))
        segment_owners.extend([index]*(n-1))
        bounds.append((segment_offset, segment_offset+n-1))
        particle_offset, section_offset, segment_offset = particle_offset+count, section_offset+n, segment_offset+n-1
    if particle_offset != len(x):
        raise ValueError("Section counts do not cover every particle")
    sections, section_owners = np.asarray(sections, dtype=np.int32), np.asarray(section_owners, dtype=np.int32)
    segments, owners, bounds = np.asarray(segments, dtype=np.int32), np.asarray(segment_owners, dtype=np.int32), np.asarray(bounds, dtype=np.int32)
    centres = _host_centres(x, mass, sections, section_offset)
    seeds, seed_segments, lookup = [], [], {}
    for index, candidate in enumerate(geometry['contact_candidates']):
        pair = tuple(candidate['fibre_ids'])
        if len(pair) != 2 or not 0 <= pair[0] < pair[1] < len(fibres) or pair in lookup:
            raise ValueError("Each ordered fibre pair must have one independent contact seed")
        lookup[pair] = index
        selected = []
        for side, fibre in enumerate(pair):
            ids_here = np.arange(*bounds[fibre])
            point = np.asarray(candidate['reference_centreline_points_m'][side])
            distances = _closest_segments(centres[segments[ids_here, 0]], centres[segments[ids_here, 1]], point, point, np)[-1]
            selected.append(int(ids_here[np.argmin(distances)]))
        seeds.append(pair)
        seed_segments.append(selected)
    reference_x = x if coverage_positions is None else np.asarray(coverage_positions, dtype=np.float64)
    if reference_x.shape != x.shape or not np.isfinite(reference_x).all():
        raise ValueError("coverage_positions must match the finite particle coordinates")
    reference = _host_centres(reference_x, mass, sections, section_offset)
    near = _coverage_neighbours(reference, segments, owners, radii, skin, proximity_margin, max_coverage_pairs)
    coverage_seed = np.array([lookup.get(tuple(owners[pair]), -1) for pair in near], dtype=np.int32)
    arrays = (sections, section_owners, segments, owners, bounds, radii,
              np.array([item['family_id'] for item in fibres], dtype=np.int32),
              np.asarray(seeds, dtype=np.int32).reshape(-1, 2), np.asarray(seed_segments, dtype=np.int32).reshape(-1, 2),
              near, coverage_seed, reference)
    converted = tuple(jnp.asarray(value) for value in arrays)
    if converted[-1].dtype != jnp.float64:
        raise ValueError("Enable jax_enable_x64 before building fibre contact data")
    return FibreContactData(*converted, section_offset, len(x), len(seeds), search_window, skin, float(proximity_margin))


def initial_contact_history(data):
    return FibreCouplingHistory(initial_fibre_contact_history((data.pair_count,)),
                                jnp.zeros(data.pair_count, dtype=data.coverage_reference.dtype),
                                data.seed_segments, jnp.asarray(True))


def contact_data_summary(data):
    return {
        "particles": data.particle_count, "sections": data.section_count,
        "segments": int(data.segments.shape[0]), "candidate_pairs": data.pair_count,
        "coverage_segment_pairs": int(data.coverage_pairs.shape[0]),
        "search_window_segments_each_side": data.search_window,
        "coverage_skin_m": data.coverage_skin, "proximity_margin_m": data.proximity_margin,
        "centreline": "Current mass-weighted cross-section centroids with piecewise linear interpolation.",
        "radius": "Fixed reference circular radii; only small cross-sectional strain is represented.",
        "curvature": "Two projected cylinder curvature tensors; R_eff is the inverse square root of their summed determinant. Orthogonal equal cylinders give R_eff=R, not R/2. This circular replacement is not an exact elliptical JKR solution.",
        "energy": "Normal force is the fixed-curvature indentation derivative. Curvature changes do not include their geometric work term, so global contact-energy conservation is not asserted.",
        "coverage": "All reference capsules within radii sum plus skin and proximity margin are enumerated. After removing one global proper rigid motion, twice the maximum section displacement must remain within skin; otherwise rebuild before accepting a step. Newly close unseeded or out-of-window pairs are invalid, including same-family pairs.",
        "limits": "Independent fibres only; no self-contact within one fibre. Physical end-cap contacts, active contact on failed sections, degenerate curvature and exhausted search windows require rejection. The returned validity flag must be checked before accepting forces or new attachment history.",
    }


def _curvature(first_tangent, second_tangent, normal, radii):
    projected = first_tangent-jnp.sum(first_tangent*normal, axis=-1)[:, None]*normal
    length = jnp.linalg.norm(projected, axis=-1)
    axis = jax.nn.one_hot(jnp.argmin(jnp.abs(normal), axis=-1), 3)
    fallback = jnp.cross(normal, axis)
    fallback /= jnp.maximum(jnp.linalg.norm(fallback, axis=-1), 1e-30)[:, None]
    one = jnp.where((length > 1e-12)[:, None], projected/jnp.maximum(length, 1e-30)[:, None], fallback)
    two = jnp.cross(normal, one)
    basis = jnp.stack((one, two), axis=-1)
    tensors = ((jnp.eye(3)-first_tangent[:, :, None]*first_tangent[:, None, :])/radii[:, 0, None, None]
               +(jnp.eye(3)-second_tangent[:, :, None]*second_tangent[:, None, :])/radii[:, 1, None, None])
    matrix = jnp.einsum('pij,pik,pkl->pjl', basis, tensors, basis)
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    valid = jnp.all(jnp.isfinite(eigenvalues), axis=-1) & (eigenvalues[:, 0] > 1e-10*eigenvalues[:, 1])
    radius = 1/jnp.sqrt(jnp.where(valid, jnp.prod(eigenvalues, axis=-1), 1.))
    return radius, eigenvalues, valid


def _nonrigid_bound(reference, current):
    ref, cur = reference-jnp.mean(reference, axis=0), current-jnp.mean(current, axis=0)
    u, _, vt = jnp.linalg.svd(ref.T@cur, full_matrices=False)
    correction = jnp.diag(jnp.array([1., 1., jnp.linalg.det(u@vt)], dtype=reference.dtype))
    rotation = u@correction@vt
    return jnp.max(jnp.linalg.norm(cur-ref@rotation, axis=-1))


def evaluate_contact(state, history, data, law):
    """Return force, updated attachment/search history and explicit validity flags.

    A false metrics.valid requires rejecting the coupled step. No force is
    silently accepted for an exhausted window or an uncovered close pair.
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
    boundary = jnp.any(at_edge, axis=-1)
    directions = centres[data.segments[selected, 1]]-centres[data.segments[selected, 0]]
    lengths = jnp.linalg.norm(directions, axis=-1)
    tangents = directions/jnp.maximum(lengths, 1e-30)[..., None]
    normal = (second-first)/jnp.maximum(distance, 1e-30)[:, None]
    radius, curvature, curvature_valid = _curvature(tangents[:, 0], tangents[:, 1], normal, data.fibre_radius[data.seed_fibres])
    gap = distance-jnp.sum(data.fibre_radius[data.seed_fibres], axis=-1)
    new_law, response = update_fibre_contact(history.law, -gap, law._replace(effective_radius=radius))
    pair_force = response.normal_force[:, None]*normal
    weights = jnp.stack((jnp.stack((1-wa, wa), axis=-1), jnp.stack((1-wb, wb), axis=-1)), axis=1)
    endpoint_sections = data.segments[selected]
    signs = jnp.array([-1., 1.], dtype=state.x.dtype)
    endpoint_forces = weights[..., None]*signs[None, :, None, None]*pair_force[:, None, None, :]
    section_force = jnp.zeros_like(centres).at[endpoint_sections.reshape(-1)].add(endpoint_forces.reshape(-1, 3))
    forces = section_force[data.particle_section]*state.mass[:, None]/jnp.maximum(section_mass[data.particle_section], 1e-30)[:, None]
    near = data.coverage_pairs
    cover_distance = closest_segments(centres[data.segments[near[:, 0], 0]], centres[data.segments[near[:, 0], 1]],
                                      centres[data.segments[near[:, 1], 0]], centres[data.segments[near[:, 1], 1]])[-1]
    pair_owners = data.segment_fibre[near]
    close = cover_distance <= jnp.sum(data.fibre_radius[pair_owners], axis=-1)+data.proximity_margin
    if data.pair_count:
        seed = jnp.maximum(data.coverage_seed, 0)
        covered = (data.coverage_seed >= 0) & jnp.all((near >= window_lower[seed]) & (near <= window_upper[seed]), axis=-1)
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
    updated = FibreCouplingHistory(new_law, response.radius, selected, valid)
    metrics = FibreCouplingMetrics(valid, boundary, jnp.sum(uncovered), jnp.sum(uncovered & same_family), motion_exceeded,
        displacement, degenerate, endpoints & near_contact, failed_contact, gap, normal, jnp.stack((first, second), axis=1),
        radius, curvature, response.normal_force, response.radius, response.active, response.potential, selected,
        jnp.sum(forces, axis=0), jnp.sum(jnp.cross(state.x, forces), axis=0))
    return forces, updated, metrics


def validate_contact(metrics):
    """Host-side refusal with enough detail to distinguish search and model limits."""
    if not bool(np.asarray(metrics.valid)):
        raise ValueError("Invalid fibre contact: " + ", ".join(f"{name}={np.asarray(getattr(metrics, name)).tolist()}" for name in
            ("search_boundary", "uncovered_near_pairs", "same_family_near_pairs", "coverage_motion_exceeded",
             "degenerate_geometry", "unsupported_end_contact", "failed_section_contact")))
    return True
