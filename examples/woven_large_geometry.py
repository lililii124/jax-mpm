"""Rectangular material-point sampling for large, fully infiltrated woven plates."""

from copy import deepcopy
from functools import lru_cache

import numpy as np

from examples.woven_geometry import (
    ARCHITECTURES,
    WeaveSample,
    _coordinates,
    audit_yarn_surfaces,
    build_paths,
)


def unit_cell(architecture, pitch):
    """Translations preserve family and geometry, with yarn IDs relabelled."""
    if architecture not in ARCHITECTURES or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("a known architecture and positive finite pitch are required")
    p = float(pitch)
    if architecture == "plain":
        primitive, lengths = ((p, p), (p, -p)), (2*p, 2*p)
    elif architecture == "twill-2x2":
        primitive, lengths = ((p, -p), (2*p, 2*p)), (4*p, 4*p)
    else:
        primitive = ((2*p/np.sqrt(3), 0.), (p/np.sqrt(3), p))
        lengths = (2*p/np.sqrt(3), 2*p)
    return {
        "primitive_translation_basis_xz_m": np.asarray(primitive).tolist(),
        "primitive_cell_area_m2": float(abs(np.linalg.det(primitive))),
        "orthogonal_repeating_cell_lengths_xz_m": list(lengths),
        "diameter_rule": "diameter >= 10 times the longer complete orthogonal repeating-cell edge",
        "scope": "geometric repeat with family identity; individual yarn IDs relabel under translation",
    }


def check_translation(paths, translation, pitch, amplitude):
    """Check the centreline height and tangent across an interior cell window."""
    translation = np.asarray(translation, dtype=float)
    errors = [0., 0.]
    samples, missing = 0, 0
    for path in paths:
        if abs(path.offset) > 1.5*pitch:
            continue
        direction = np.asarray(path.direction)
        normal = np.array((-direction[1], direction[0]))
        expected = path.offset + normal @ translation
        matches = [other for other in paths if other.family_id == path.family_id
                   and abs(other.offset-expected) < 1e-10*pitch]
        if len(matches) != 1:
            missing += 1
            continue
        coordinate = np.linspace(-2*pitch, 2*pitch, 101)+.031*pitch
        first = path.profile(coordinate)
        second = matches[0].profile(coordinate+direction @ translation)
        for i in range(2):
            errors[i] = max(errors[i], float(np.max(np.abs(first[i]-second[i]))))
        samples += len(coordinate)
    return {
        "translation_xz_m": translation.tolist(),
        "height_error_m": errors[0], "slope_error": errors[1],
        "samples": samples, "unmatched_paths": missing,
        "passed": bool(samples > 0 and missing == 0 and errors[0] <= 1e-10*amplitude
                       and errors[1] <= 1e-10*max(1., amplitude/pitch)),
    }


@lru_cache(maxsize=32)
def _periodic_audit(architecture, pitch, yarn_width, yarn_height, gap):
    cell = unit_cell(architecture, pitch)
    amplitude = (yarn_height+gap)/2
    paths, _, _ = build_paths(architecture, size=(16*pitch, 16*pitch),
                             pitch=pitch, amplitude=amplitude)
    translations = list(cell["primitive_translation_basis_xz_m"])
    translations += np.diag(cell["orthogonal_repeating_cell_lengths_xz_m"]).tolist()
    periodic = [check_translation(paths, t, pitch, amplitude) for t in translations]
    cell_size = tuple(2*np.asarray(cell["orthogonal_repeating_cell_lengths_xz_m"]))
    surfaces = audit_yarn_surfaces(
        architecture, size=cell_size, pitch=pitch, yarn_width=yarn_width,
        yarn_height=yarn_height, gap=gap,
    )
    return {
        "passed": bool(all(item["passed"] for item in periodic) and surfaces["passed"]),
        "translations": periodic,
        "surface_audit_window_xz_m": list(cell_size),
        "surface_audit": surfaces,
        "scope": "sampled periodic-cell surface exclusion and actual path-translation checks; "
                 "not an exhaustive continuous nonintersection proof; all plate particles have exact ownership checks",
    }


def audit_periodic_geometry(architecture, *, pitch=.0025, yarn_width=.000625,
                            yarn_height=.0005, gap=.0005):
    return deepcopy(_periodic_audit(architecture, float(pitch), float(yarn_width),
                                    float(yarn_height), float(gap)))


def sample_large_weave(
    architecture, *, size=(.16, .16), spacing=(.0005, .00025, .0005),
    pitch=.0025, yarn_width=.000625, yarn_height=.0005, gap=.0005,
    matrix_cover=.00025, origin=(0., 0., 0.), chunk_size=4096,
    max_particles=1_000_000,
):
    """Use an explicitly anisotropic lattice without changing the yarn geometry.

    ``size`` is (x,z), ``spacing`` is (x,y,z), and origin is the plate centre.
    Matrix and yarn occupy the complete specimen volume. The grid spacing is
    independent of this sampling interface.
    """
    size, spacing, origin = (np.asarray(v, dtype=np.float64) for v in (size, spacing, origin))
    dimensions = np.array((pitch, yarn_width, yarn_height, gap, matrix_cover), dtype=float)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    if size.shape != (2,) or spacing.shape != (3,) or origin.shape != (3,):
        raise ValueError("size, spacing and origin must have 2, 3 and 3 components")
    if (not all(np.isfinite(v).all() for v in (size, spacing, origin, dimensions))
            or np.any(size <= 0) or np.any(spacing <= 0) or np.any(dimensions <= 0)
            or yarn_width >= pitch):
        raise ValueError("finite positive dimensions, finite origin and yarn_width < pitch are required")
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1
            or not isinstance(max_particles, int) or isinstance(max_particles, bool)
            or max_particles < 1):
        raise ValueError("chunk_size and max_particles must be positive integers")
    amplitude = (yarn_height+gap)/2
    thickness = 2*amplitude+yarn_height+2*matrix_cover
    lengths = np.array((size[0], thickness, size[1]))
    ratios = lengths/spacing
    if np.any(ratios > max_particles):
        raise ValueError("particle budget exceeded before allocation")
    counts = np.rint(ratios).astype(np.int64)
    if np.any(counts < 1) or not np.allclose(counts*spacing, lengths, rtol=0, atol=1e-10*spacing):
        raise ValueError("each explicit spacing must divide its specimen length; no lattice adjustment")
    count = int(np.prod(counts, dtype=object))
    if count > max_particles:
        raise ValueError(f"particle count {count} exceeds budget {max_particles}")
    audit = audit_periodic_geometry(architecture, pitch=pitch, yarn_width=yarn_width,
                                    yarn_height=yarn_height, gap=gap)
    if not audit["passed"]:
        raise ValueError("periodic yarn geometry failed its surface or translation audit")
    paths, crossings, angles = build_paths(architecture, size=size, pitch=pitch, amplitude=amplitude)
    axes = [-lengths[d]/2+(np.arange(counts[d])+.5)*spacing[d] for d in range(3)]
    nx, ny, nz = map(int, counts)
    local = np.empty((count, 3), dtype=np.float64)
    local[:, 0] = np.repeat(axes[0], ny*nz)
    local[:, 1] = np.tile(np.repeat(axes[1], nz), nx)
    local[:, 2] = np.tile(axes[2], nx*ny)
    phase = np.zeros(count, dtype=np.uint8)
    yarn_ids = np.full(count, -1, dtype=np.int32)
    family_ids = np.full(count, -1, dtype=np.int8)
    orientation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    overlap_count, invalid_projection, candidate_count = 0, 0, 0
    for path in paths:
        dx, dz = path.direction
        # Prune by the exact normal-plane strip on the xz lattice before
        # expanding through the thickness. No geometric decimation is used.
        pruning_roundoff = 32*np.finfo(float).eps*(max(lengths)+abs(path.offset)+yarn_width)
        planar_x, planar_z = np.nonzero(
            np.abs(-dz*axes[0][:, None]+dx*axes[2][None, :]-path.offset)
            <= yarn_width/2+pruning_roundoff
        )
        candidates = (((planar_x[:, None]*ny+np.arange(ny))*nz)+planar_z[:, None]).ravel()
        candidate_count += len(candidates)
        for start in range(0, len(candidates), chunk_size):
            indices = candidates[start:start+chunk_size]
            width, height, residual, q = _coordinates(local[indices], path, pitch)
            inside = (2*width/yarn_width)**2+(2*height/yarn_height)**2 < 1
            invalid_projection += int(np.count_nonzero(inside & (np.abs(residual) > 1e-8*pitch)))
            selected = indices[inside]
            overlap_count += int(np.count_nonzero(phase[selected]))
            phase[selected], yarn_ids[selected], family_ids[selected] = 1, path.yarn_id, path.family_id
            orientation[selected] = q[inside]
    if overlap_count or invalid_projection:
        raise ValueError(f"invalid weave geometry: {overlap_count} sampled yarn overlaps, "
                         f"{invalid_projection} unresolved normal projections")
    volume = np.full(count, float(np.prod(spacing)))
    total_volume = float(volume.sum())
    yarn_count = int(phase.sum())
    family_volumes = {str(i): float(volume[family_ids == i].sum()) for i in range(len(angles))}
    if not yarn_count or any(value == 0 for value in family_volumes.values()):
        raise ValueError("particle spacing misses a yarn family")
    metadata = {
        "architecture": architecture, "geometry_units": "m", "size_xz_m": size.tolist(),
        "thickness_m": float(thickness), "origin_m": origin.tolist(), "pitch_m": float(pitch),
        "yarn_width_m": float(yarn_width), "yarn_height_m": float(yarn_height),
        "crossing_gap_m": float(gap), "matrix_cover_m": float(matrix_cover),
        "particle_spacing_m": spacing.tolist(), "sampling": "explicit anisotropic Cartesian volume lattice",
        "particle_spacing_equivalent_m": float(np.prod(spacing)**(1/3)),
        "lattice_counts": counts.tolist(), "particle_count": count,
        "particle_budget": max_particles, "yarn_particles": yarn_count,
        "matrix_particles": count-yarn_count, "yarn_reference_volume_m3": float(volume[phase == 1].sum()),
        "matrix_reference_volume_m3": float(volume[phase == 0].sum()),
        "reference_volume_m3": total_volume,
        "yarn_volume_fraction": float(yarn_count/count), "family_reference_volume_m3": family_volumes,
        "family_angles_deg": angles, "sampled_yarn_intersections": overlap_count,
        "unresolved_normal_projections": invalid_projection, "normal_projection_candidates": candidate_count,
        "phase_definition": "0 matrix, 1 homogenized yarn; fully infiltrated, no void phase",
        "orientation_definition": "columns: reference centreline tangent, in-plane transverse, their right-handed cross product",
        "projection_scope": "all material particles checked against exact yarn strips and normal-plane ellipse ownership",
        "unit_cell": unit_cell(architecture, pitch), "periodic_geometry_audit": audit,
        "crossings": crossings,
        "paths": [{"yarn_id": p.yarn_id, "family_id": p.family_id, "index": p.index,
                   "direction_xz": list(p.direction), "offset_m": p.offset,
                   "knots_m": p.knots.tolist(), "heights_m": p.heights.tolist()} for p in paths],
    }
    return WeaveSample(local+origin, volume, phase, yarn_ids, family_ids, orientation, metadata)
