"""Uniform material points in continuous, interlaced yarn and matrix geometry."""

from dataclasses import dataclass

import numpy as np


ARCHITECTURES = ("plain", "triaxial", "twill-2x2")


@dataclass(frozen=True)
class YarnPath:
    yarn_id: int
    family_id: int
    index: int
    direction: tuple[float, float]
    offset: float
    knots: np.ndarray
    heights: np.ndarray

    def profile(self, coordinate):
        interval = np.clip(
            np.searchsorted(self.knots, coordinate, side="right") - 1,
            0,
            len(self.knots) - 2,
        )
        length = self.knots[interval + 1] - self.knots[interval]
        fraction = np.clip((coordinate - self.knots[interval]) / length, 0, 1)
        first, second = self.heights[interval], self.heights[interval + 1]
        angle = np.pi * fraction
        height = 0.5 * (first + second) + 0.5 * (first - second) * np.cos(angle)
        slope = -0.5 * (first - second) * np.pi / length * np.sin(angle)
        curvature = -0.5 * (first - second) * (np.pi / length) ** 2 * np.cos(angle)
        return height, slope, curvature


@dataclass
class WeaveSample:
    positions: np.ndarray
    volume: np.ndarray
    phase: np.ndarray
    yarn_id: np.ndarray
    family_id: np.ndarray
    orientation: np.ndarray
    metadata: dict


def build_paths(architecture, *, size, pitch, amplitude):
    """C1 yarn paths with horizontal tangents at every prescribed crossing."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    size = np.asarray(size, dtype=float)
    if size.shape != (2,) or np.any(size <= 0) or not np.all(np.isfinite(size)):
        raise ValueError("size must contain two positive in-plane lengths")
    if (
        not np.isfinite(pitch)
        or pitch <= 0
        or not np.isfinite(amplitude)
        or amplitude <= 0
    ):
        raise ValueError("pitch and amplitude must be positive")
    angles = (
        (0.0, np.pi / 3, -np.pi / 3) if architecture == "triaxial" else (0.0, np.pi / 2)
    )
    phases = (0.0, 0.0, pitch / 2) if architecture == "triaxial" else (0.0, 0.0)
    margin = 2 * pitch
    low, high = -size / 2 - margin, size / 2 + margin
    corners = np.array([(x, z) for x in (low[0], high[0]) for z in (low[1], high[1])])
    definitions = []
    for family, (angle, phase) in enumerate(zip(angles, phases)):
        direction = np.array((np.cos(angle), np.sin(angle)))
        normal = np.array((-direction[1], direction[0]))
        projected = corners @ normal
        first = int(np.floor((projected.min() - phase) / pitch)) - 1
        last = int(np.ceil((projected.max() - phase) / pitch)) + 1
        for index in range(first, last + 1):
            definitions.append(
                (family, index, direction, normal, index * pitch + phase)
            )
    nodes = [[] for _ in definitions]
    crossings = []
    for i, (a, ai, ad, an, ao) in enumerate(definitions):
        for j in range(i + 1, len(definitions)):
            b, bi, bd, bn, bo = definitions[j]
            if a == b:
                continue
            point = np.linalg.solve(np.stack((an, bn)), np.array((ao, bo)))
            if np.any(point < low) or np.any(point > high):
                continue
            if architecture == "triaxial":
                above = (b - a) % 3 == 1
            elif architecture == "plain":
                above = (ai + bi) % 2 == 0
            else:
                above = (bi - ai) % 4 < 2
            sign = 1.0 if above else -1.0
            nodes[i].append((float(point @ ad), sign * amplitude))
            nodes[j].append((float(point @ bd), -sign * amplitude))
            if np.all(np.abs(point) <= size / 2):
                crossings.append(
                    {
                        "yarns": [i, j],
                        "above_yarn": i if above else j,
                        "point_xz": point.tolist(),
                    }
                )
    paths = []
    for yarn_id, ((family, index, direction, normal, offset), raw_nodes) in enumerate(
        zip(definitions, nodes)
    ):
        if len(raw_nodes) < 2:
            continue
        ordered = np.array(sorted(raw_nodes))
        if np.any(np.diff(ordered[:, 0]) < 1e-9 * pitch):
            raise ValueError(
                "coincident crossings require an explicit three-yarn topology"
            )
        corner_offsets = (
            np.array(
                [
                    (x, z)
                    for x in (-size[0] / 2, size[0] / 2)
                    for z in (-size[1] / 2, size[1] / 2)
                ]
            )
            @ normal
        )
        if (
            offset < corner_offsets.min() - pitch / 2
            or offset > corner_offsets.max() + pitch / 2
        ):
            continue
        paths.append(
            YarnPath(
                yarn_id,
                family,
                index,
                tuple(direction),
                offset,
                ordered[:, 0],
                ordered[:, 1],
            )
        )
    return paths, crossings, np.degrees(angles).tolist()


def _coordinates(points, path, pitch):
    direction = np.asarray(path.direction)
    normal = np.array((-direction[1], direction[0]))
    planar = points[:, (0, 2)]
    projection = planar @ direction
    coordinate = np.clip(projection, path.knots[0], path.knots[-1])
    for _ in range(16):
        height, slope, curvature = path.profile(coordinate)
        residual = coordinate - projection - (points[:, 1] - height) * slope
        derivative = 1 + slope**2 - (points[:, 1] - height) * curvature
        safe = np.where(np.abs(derivative) > 1e-8, derivative, 1.0)
        increment = np.clip(residual / safe, -pitch / 4, pitch / 4)
        coordinate = np.clip(coordinate - increment, path.knots[0], path.knots[-1])
    height, slope, _ = path.profile(coordinate)
    residual = coordinate - projection - (points[:, 1] - height) * slope
    scale = np.sqrt(1 + slope**2)
    width = planar @ normal - path.offset
    thickness = (points[:, 1] - height - slope * (projection - coordinate)) / scale
    tangent = (
        np.column_stack(
            (
                np.full(len(points), direction[0]),
                slope,
                np.full(len(points), direction[1]),
            )
        )
        / scale[:, None]
    )
    transverse = np.broadcast_to((direction[1], 0.0, -direction[0]), tangent.shape)
    upright = np.cross(tangent, transverse)
    orientation = np.stack((tangent, transverse, upright), axis=-1)
    return width, thickness, residual, orientation


def audit_yarn_surfaces(
    architecture,
    *,
    size,
    pitch,
    yarn_width,
    yarn_height,
    gap,
    samples_per_span=16,
    section_samples=32,
):
    """Independent surface sampling, separate from material-point occupancy."""
    if samples_per_span < 2 or section_samples < 8:
        raise ValueError("surface sampling is too sparse")
    paths, _, _ = build_paths(
        architecture, size=size, pitch=pitch, amplitude=(yarn_height + gap) / 2
    )
    angle = np.arange(section_samples) * 2 * np.pi / section_samples
    minimum, intersections, unresolved, sample_count, curvature_bound = (
        np.inf,
        0,
        0,
        0,
        0.0,
    )
    for path in paths:
        lengths = np.diff(path.knots)
        curvature_bound = max(
            curvature_bound,
            float(
                np.max(
                    yarn_height
                    / 2
                    * 0.5
                    * np.abs(np.diff(path.heights))
                    * (np.pi / lengths) ** 2
                )
            ),
        )
        coordinate = (
            path.knots[:-1, None]
            + lengths[:, None] * np.arange(samples_per_span) / samples_per_span
        ).ravel()
        height, slope, _ = path.profile(coordinate)
        dx, dz = path.direction
        center = np.column_stack(
            (
                dx * coordinate - dz * path.offset,
                height,
                dz * coordinate + dx * path.offset,
            )
        )
        transverse = np.array((dz, 0.0, -dx))
        upright = (
            np.column_stack((-slope * dx, np.ones(len(slope)), -slope * dz))
            / np.sqrt(1 + slope**2)[:, None]
        )
        points = (
            center[:, None, :]
            + yarn_width / 2 * np.cos(angle)[None, :, None] * transverse
            + yarn_height / 2 * np.sin(angle)[None, :, None] * upright[:, None, :]
        ).reshape(-1, 3)
        points = points[
            np.all(np.abs(points[:, (0, 2)]) <= np.asarray(size) / 2, axis=1)
        ]
        sample_count += len(points)
        for other in paths:
            if other.family_id == path.family_id:
                continue
            normal = np.array((-other.direction[1], other.direction[0]))
            near = np.abs(points[:, (0, 2)] @ normal - other.offset) <= yarn_width / 2
            candidate = points[near]
            if not len(candidate):
                continue
            width, depth, residual, _ = _coordinates(candidate, other, pitch)
            radius_squared = (2 * width / yarn_width) ** 2 + (
                2 * depth / yarn_height
            ) ** 2
            converged = np.abs(residual) <= 1e-8 * pitch
            if converged.any():
                minimum = min(minimum, float(radius_squared[converged].min()))
            intersections += int(
                np.count_nonzero(converged & (radius_squared < 1 - 1e-8))
            )
            unresolved += int(np.count_nonzero(~converged & (radius_squared <= 1)))
    return {
        "surface_samples": sample_count,
        "samples_per_crossing_span": samples_per_span,
        "section_samples": section_samples,
        "sampled_surface_intersections": intersections,
        "unresolved_surface_projections": unresolved,
        "minimum_other_yarn_ellipse_radius_squared": None
        if not np.isfinite(minimum)
        else minimum,
        "maximum_half_height_times_curvature_bound": curvature_bound,
        "passed": intersections == 0 and unresolved == 0 and curvature_bound < 1,
        "scope": "sampled surface exclusion and local curvature bound; not a continuous collision proof",
    }


def sample_weave(
    architecture,
    *,
    size,
    pitch,
    yarn_width,
    yarn_height,
    gap,
    matrix_cover,
    spacing,
    origin=(0.0, 0.0, 0.0),
    chunk_size=4096,
):
    """Fill a rectangular specimen with disjoint yarn and matrix particles.

    All dimensions are metres. This is a fully infiltrated composite, with no
    open textile pores. Sampled overlap is rejected rather than assigned by priority.
    """
    values = np.array((pitch, yarn_width, yarn_height, gap, matrix_cover, spacing))
    if not np.all(np.isfinite(values)) or np.any(values <= 0) or yarn_width >= pitch:
        raise ValueError(
            "positive finite dimensions and yarn_width < pitch are required"
        )
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain three finite coordinates")
    amplitude = (yarn_height + gap) / 2
    paths, crossings, angles = build_paths(
        architecture, size=size, pitch=pitch, amplitude=amplitude
    )
    thickness = 2 * amplitude + yarn_height + 2 * matrix_cover
    lengths = np.array((size[0], thickness, size[1]))
    counts = np.rint(lengths / spacing).astype(int)
    if np.any(counts < 1) or not np.allclose(
        counts * spacing, lengths, rtol=0, atol=1e-10 * spacing
    ):
        raise ValueError(
            "spacing must divide all specimen lengths; no anisotropic lattice adjustment"
        )
    axes = [-lengths[d] / 2 + (np.arange(counts[d]) + 0.5) * spacing for d in range(3)]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    count = len(local)
    phase = np.zeros(count, dtype=np.uint8)
    yarn_ids = np.full(count, -1, dtype=np.int32)
    family_ids = np.full(count, -1, dtype=np.int8)
    orientation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    overlap_count, invalid_projection = 0, 0
    for path in paths:
        direction = np.asarray(path.direction)
        normal = np.array((-direction[1], direction[0]))
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            near = (
                np.abs(local[start:end][:, (0, 2)] @ normal - path.offset)
                <= yarn_width / 2
            )
            indices = start + np.flatnonzero(near)
            if not len(indices):
                continue
            width, height, residual, q = _coordinates(local[indices], path, pitch)
            inside = (width / (yarn_width / 2)) ** 2 + (
                height / (yarn_height / 2)
            ) ** 2 < 1
            bad = inside & (np.abs(residual) > 1e-8 * pitch)
            invalid_projection += int(bad.sum())
            selected = indices[inside]
            overlap_count += int(np.count_nonzero(phase[selected]))
            phase[selected] = 1
            yarn_ids[selected] = path.yarn_id
            family_ids[selected] = path.family_id
            orientation[selected] = q[inside]
    if overlap_count or invalid_projection:
        raise ValueError(
            f"invalid weave geometry: {overlap_count} sampled yarn overlaps, {invalid_projection} unresolved normal projections"
        )
    volume = np.full(count, spacing**3)
    yarn_volume = float(volume[phase == 1].sum())
    total_volume = float(volume.sum())
    family_volumes = {
        str(i): float(volume[family_ids == i].sum()) for i in range(len(angles))
    }
    if not yarn_volume or any(value == 0 for value in family_volumes.values()):
        raise ValueError("particle spacing misses a yarn family")
    metadata = {
        "architecture": architecture,
        "geometry_units": "m",
        "size_xz_m": list(size),
        "thickness_m": thickness,
        "origin_m": origin.tolist(),
        "pitch_m": pitch,
        "yarn_width_m": yarn_width,
        "yarn_height_m": yarn_height,
        "crossing_gap_m": gap,
        "matrix_cover_m": matrix_cover,
        "particle_spacing_m": spacing,
        "lattice_counts": counts.tolist(),
        "particle_count": count,
        "yarn_particles": int(phase.sum()),
        "matrix_particles": int(count - phase.sum()),
        "yarn_reference_volume_m3": yarn_volume,
        "matrix_reference_volume_m3": total_volume - yarn_volume,
        "yarn_volume_fraction": yarn_volume / total_volume,
        "family_reference_volume_m3": family_volumes,
        "family_angles_deg": angles,
        "sampled_yarn_intersections": overlap_count,
        "unresolved_normal_projections": invalid_projection,
        "phase_definition": "0 matrix, 1 homogenized yarn; no fibre-scale strands or void phase",
        "orientation_definition": "columns: reference centreline tangent, in-plane transverse, their right-handed cross product",
        "topology": "cyclic pairwise over/under with staggered crossings"
        if architecture == "triaxial"
        else "alternating one-over-one"
        if architecture == "plain"
        else "two-over-two with one-pick diagonal shift",
        "projection_scope": "normal-plane Newton solve; sampled overlap check is not a proof of continuous surface clearance",
        "crossings": crossings,
        "paths": [
            {
                "yarn_id": p.yarn_id,
                "family_id": p.family_id,
                "index": p.index,
                "direction_xz": list(p.direction),
                "offset_m": p.offset,
                "knots_m": p.knots.tolist(),
                "heights_m": p.heights.tolist(),
            }
            for p in paths
        ],
    }
    return WeaveSample(
        local + origin, volume, phase, yarn_ids, family_ids, orientation, metadata
    )
