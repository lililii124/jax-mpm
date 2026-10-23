"""Explicit circular fibres in a periodic 0/60/120 degree interlacing."""

from functools import lru_cache

import numpy as np


DEFAULT_ANGLES_DEG = tuple(range(0, 31, 5))
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(8)


def _frame(family, theta_deg=0.0):
    angle = np.deg2rad(60.0 * family + theta_deg)
    direction = np.array([np.cos(angle), np.sin(angle)])
    return direction, np.array([-direction[1], direction[0]])


def centreline(family, index, parameter, *, pitch, crimp, theta_deg=0.0):
    """Return position, unit tangent and signed curvature versus planar arclength."""
    if family not in (0, 1, 2):
        raise ValueError("family must be 0, 1 or 2")
    t = np.asarray(parameter, dtype=np.float64)
    direction, normal = _frame(family, theta_deg)
    omega = np.pi * np.sqrt(3.0) / pitch
    phase = -1.0 if int(index) % 2 == 0 else 1.0
    z = phase * crimp * np.sin(omega * t)
    slope = phase * crimp * omega * np.cos(omega * t)
    second = -phase * crimp * omega**2 * np.sin(omega * t)
    xy = t[..., None] * direction + (index + 0.5) * pitch * normal
    position = np.stack((xy[..., 0], z, xy[..., 1]), axis=-1)
    tangent = np.stack((np.broadcast_to(direction[0], t.shape), slope, np.broadcast_to(direction[1], t.shape)), axis=-1)
    tangent /= np.sqrt(1.0 + slope**2)[..., None]
    curvature = second / (1.0 + slope**2) ** 1.5
    return position, tangent, curvature


def _segment_distance(first, second, *, closest=False):
    p, q = first[:-1, None, :], second[None, :-1, :]
    u, v = np.diff(first, axis=0)[:, None, :], np.diff(second, axis=0)[None, :, :]
    r = p - q
    a, b, c = np.sum(u*u, axis=-1), np.sum(u*v, axis=-1), np.sum(v*v, axis=-1)
    d, e = np.sum(u*r, axis=-1), np.sum(v*r, axis=-1)
    denominator = a*c-b*b
    safe = np.where(denominator > 1e-12*a*c, denominator, 1.0)
    s, t = (b*e-c*d)/safe, (a*e-b*d)/safe
    interior = np.sum((r+s[..., None]*u-t[..., None]*v)**2, axis=-1)
    interior = np.where((denominator > 1e-12*a*c) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), interior, np.inf)

    def endpoint_distance(delta, segment, squared_length):
        value = np.clip(np.sum(delta*segment, axis=-1)/squared_length, 0, 1)
        return np.sum((delta-value[..., None]*segment)**2, axis=-1)

    distances = [interior, endpoint_distance(r, v, c), endpoint_distance(r+u, v, c),
                 endpoint_distance(-r, u, a), endpoint_distance(v-r, u, a)]
    choice = int(np.argmin([value.min() for value in distances]))
    first_index, second_index = np.unravel_index(np.argmin(distances[choice]), distances[choice].shape)
    distance = float(np.sqrt(distances[choice][first_index, second_index]))
    if not closest:
        return distance
    fractions = [(s, t), (np.zeros_like(e), np.clip(e/c, 0, 1)),
                 (np.ones_like(e), np.clip((e+b)/c, 0, 1)),
                 (np.clip(-d/a, 0, 1), np.zeros_like(d)),
                 (np.clip((b-d)/a, 0, 1), np.ones_like(d))]
    first_fraction, second_fraction = fractions[choice]
    return distance, (first_index, float(first_fraction[first_index, second_index])), (second_index, float(second_fraction[first_index, second_index]))


@lru_cache(maxsize=128)
def _crossing_offsets(first_family, first_parity, second_family, second_parity, pitch, crimp, diameter):
    u, n = _frame(first_family)
    v, m = _frame(second_family)
    intersection = np.linalg.solve(np.stack((n, m)), pitch*np.array([first_parity+.5, second_parity+.5]))
    crossing = np.array([intersection@u, intersection@v])
    span = 2*crimp/np.sin(np.pi/3)
    count = int(np.ceil(2*span/min(pitch/128, diameter/16)))+1
    parameters = [np.linspace(value-span, value+span, count) for value in crossing]
    curves = [centreline(family, parity, parameter, pitch=pitch, crimp=crimp)[0]
              for family, parity, parameter in zip((first_family, second_family), (first_parity, second_parity), parameters)]
    _, first, second = _segment_distance(*curves, closest=True)
    value = np.array([parameters[k][item[0]]+item[1]*(parameters[k][item[0]+1]-parameters[k][item[0]]) for k, item in enumerate((first, second))])
    omega = np.pi*np.sqrt(3.0)/pitch
    for _ in range(5):
        positions, first_derivatives, second_derivatives = [], [], []
        for family, parity, parameter in zip((first_family, second_family), (first_parity, second_parity), value):
            direction, _ = _frame(family)
            sign = -1. if parity == 0 else 1.
            positions.append(centreline(family, parity, parameter, pitch=pitch, crimp=crimp)[0])
            first_derivatives.append(np.array([direction[0], sign*crimp*omega*np.cos(omega*parameter), direction[1]]))
            second_derivatives.append(np.array([0., -sign*crimp*omega**2*np.sin(omega*parameter), 0.]))
        delta = positions[0]-positions[1]
        left, right = first_derivatives
        gradient = np.array([delta@left, -delta@right])
        hessian = np.array([[left@left+delta@second_derivatives[0], -left@right],
                            [-left@right, right@right-delta@second_derivatives[1]]])
        if np.linalg.det(hessian) <= 1e-12:
            break
        change = np.linalg.solve(hessian, gradient)
        value = np.clip(value-change, crossing-span, crossing+span)
    return tuple(float(item) for item in value-crossing)


@lru_cache(maxsize=32)
def clearance_report(pitch, diameter, crimp, clearance):
    """Bound infinite-weave tube clearance using line segments and interpolation error."""
    if any(not np.isfinite(value) or value <= 0 for value in (pitch, diameter, crimp)):
        raise ValueError("pitch, diameter and crimp must be finite and positive")
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("clearance must be finite and nonnegative")
    omega = np.pi*np.sqrt(3.0)/pitch
    curvature_radius_product = diameter/2*crimp*omega**2
    if curvature_radius_product >= 0.9:
        raise ValueError("Tube radius is too close to the centreline curvature radius")
    step = min(pitch/128, diameter/16)
    span = 2*crimp/np.sin(np.pi/3)
    minimum, maximum_error = np.inf, 0.0
    for first_family, second_family in ((0, 1), (0, 2), (1, 2)):
        u, n = _frame(first_family)
        v, m = _frame(second_family)
        for first_index in (0, 1):
            for second_index in (0, 1):
                intersection = np.linalg.solve(np.stack((n, m)), pitch*np.array([first_index+.5, second_index+.5]))
                parameters = []
                for direction in (u, v):
                    crossing = float(intersection@direction)
                    parameters.append(np.linspace(crossing-span, crossing+span, int(np.ceil(2*span/step))+1))
                first = centreline(first_family, first_index, parameters[0], pitch=pitch, crimp=crimp)[0]
                second = centreline(second_family, second_index, parameters[1], pitch=pitch, crimp=crimp)[0]
                minimum = min(minimum, _segment_distance(first, second))
                maximum_error = max(maximum_error, crimp*omega**2*max(np.diff(value).max()**2 for value in parameters)/8)
    bound = min(minimum-2*maximum_error, pitch)
    return {
        "passed": bool(bound >= diameter+clearance),
        "minimum_centreline_distance_lower_bound_m": float(bound),
        "minimum_surface_clearance_lower_bound_m": float(bound-diameter),
        "required_surface_clearance_m": float(clearance),
        "polyline_minimum_distance_m": float(minimum),
        "single_curve_interpolation_error_bound_m": float(maximum_error),
        "maximum_radius_times_curvature": float(curvature_radius_product),
        "self_tube_regular": True,
        "basis": "All three crossing-family pairs and both line-index parities. At a planar crossing the separation is 2*crimp, so any closer pair lies within 2*crimp/sin(60deg) of that crossing in each line parameter. Segment distances are exact; two analytic sinusoid interpolation bounds are subtracted. Parallel-family centreline distance is at least pitch. A fibre is a sinusoidal graph with a constant binormal; radius*maximum_curvature<1 keeps its offset graphs monotone and ordered, avoiding self-intersection. The periodic weave translates by (2*pitch/sqrt(3),0) and (pitch/sqrt(3),pitch).",
    }


def symmetry_report(pitch, crimp):
    t = np.linspace(-2*pitch, 2*pitch, 101)
    rotation = np.array([[.5, 0, -np.sqrt(3)/2], [0, 1, 0], [np.sqrt(3)/2, 0, .5]])
    turnover = np.diag([1., -1., -1.])
    rotation_error, turnover_error = 0.0, 0.0
    for family in range(3):
        for index in (-2, -1, 0, 1, 2):
            points = centreline(family, index, t, pitch=pitch, crimp=crimp)[0]
            next_family, next_index, next_t = ((family+1, index, t) if family < 2 else (0, -index-1, -t))
            rotated = centreline(next_family, next_index, next_t, pitch=pitch, crimp=crimp)[0]
            mirror_family, mirror_index, mirror_t = ((0, -index-1, t) if family == 0 else (3-family, index, -t))
            mirrored = centreline(mirror_family, mirror_index, mirror_t, pitch=pitch, crimp=crimp)[0]
            rotation_error = max(rotation_error, float(np.abs(points@rotation.T-rotated).max()))
            turnover_error = max(turnover_error, float(np.abs(points@turnover.T-mirrored).max()))
    return {
        "rotation_60_max_coordinate_error_m": rotation_error,
        "in_plane_reflection_with_thickness_turnover_max_coordinate_error_m": turnover_error,
        "passed": bool(max(rotation_error, turnover_error) < 1e-12*pitch),
        "angle_interval_deg": [0., 30.],
        "angle_interval_condition": "Only for identical fibres and contact laws with loading, grips and external fields invariant under thickness turnover. XZ is the weave plane and Y is thickness. The reflection used is (x,y,z)->(x,-y,-z); a planar reflection without reversing over/under is not asserted. One-sided contact or a substrate invalidates this reduction.",
    }


def _integral(a, b, omega, amplitude):
    nodes = (a+b)[..., None]/2+(b-a)[..., None]/2*_GAUSS_X
    return (b-a)/2*np.sum(_GAUSS_W*np.sqrt(1+(amplitude*omega*np.cos(omega*nodes))**2), axis=-1)


def _arc_samples(start, end, pitch, crimp, spacing):
    omega = np.pi*np.sqrt(3.0)/pitch
    knots = np.linspace(start, end, int(np.ceil((end-start)/(pitch/32)))+1)
    cumulative = np.concatenate(([0.], np.cumsum(_integral(knots[:-1], knots[1:], omega, crimp))))
    length = float(cumulative[-1])
    count = max(1, int(np.ceil(length/spacing)))
    requested = (np.arange(count)+.5)*length/count
    cells = np.clip(np.searchsorted(cumulative, requested)-1, 0, len(knots)-2)
    left, right = knots[cells], knots[cells+1]
    parameter = np.interp(requested, cumulative, knots)
    for _ in range(3):
        residual = cumulative[cells]+_integral(left, parameter, omega, crimp)-requested
        parameter -= residual/np.sqrt(1+(crimp*omega*np.cos(omega*parameter))**2)
        parameter = np.clip(parameter, left, right)
    return parameter, length


def _disk_sampling(radius, spacing):
    rings = max(1, int(round(radius/spacing)))
    nominal_radii = radius*(np.arange(rings)+.5)/rings
    counts = np.maximum(4, 4*np.rint(2*np.pi*nominal_radii/spacing/4).astype(int))
    boundaries = np.concatenate(([0.], np.cumsum(counts)/counts.sum()))
    radii, angles = [], []
    for index, count in enumerate(counts):
        radii.extend([radius*np.sqrt((boundaries[index]+boundaries[index+1])/2)]*count)
        angles.extend(2*np.pi*(np.arange(count)+.5)/count)
    return np.array(radii), np.array(angles), counts.tolist()


def make_geometry(*, theta_deg=0.0, dimensions=(.024, .016), diameter=.0002,
                  pitch=.001, crimp=.00014, clearance=.000025,
                  particle_spacing=.000045, packing=1.0, density=1800.0,
                  grip_length=.003, max_particles=1_000_000):
    """Sample a finite coupon; packing multiplies sampling density, not solid fraction.

    Tube centreline endpoints are clipped to the rectangle inset by one radius.
    End caps are normal to the centreline, so every circular section remains
    inside the nominal rectangle. Each particle has constant reference volume
    within its fibre; no matrix particles or inter-fibre bonding are introduced.
    """
    dimensions = np.asarray(dimensions, dtype=np.float64)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= diameter):
        raise ValueError("dimensions must contain two positive lengths greater than diameter")
    if not np.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    if not isinstance(max_particles, int) or isinstance(max_particles, bool) or max_particles < 1:
        raise ValueError("max_particles must be a positive integer")
    if any(not np.isfinite(value) or value <= 0 for value in (particle_spacing, packing, density, grip_length)):
        raise ValueError("sampling, density and grip lengths must be finite and positive")
    if 2*grip_length >= dimensions[0]:
        raise ValueError("Two grips must leave a positive gauge length")
    validation = clearance_report(float(pitch), float(diameter), float(crimp), float(clearance))
    if not validation["passed"]:
        raise ValueError("Initial fibre tubes do not satisfy the requested clearance")
    symmetry = symmetry_report(pitch, crimp)
    if not symmetry["passed"]:
        raise ValueError("The periodic geometry failed its symmetry check")
    radius, spacing = diameter/2, particle_spacing/packing**(1/3)
    inset = dimensions/2-radius
    origin = np.array([dimensions[0]/2, crimp+radius+clearance, dimensions[1]/2])
    radii, angular_quantiles, ring_counts = _disk_sampling(radius, spacing)
    cross_count = len(radii)
    positions, volumes, tangents, ids, families, fibres = [], [], [], [], [], []
    line_positions, line_parameters, line_ids, line_offsets = [], [], [], [0]
    total = 0
    for family in range(3):
        direction, normal = _frame(family, theta_deg)
        max_offset = float(np.abs(normal)@inset)
        first = int(np.ceil(-max_offset/pitch-.5))
        last = int(np.floor(max_offset/pitch-.5))
        for index in range(first, last+1):
            base = (index+.5)*pitch*normal
            start, end = -np.inf, np.inf
            for axis in range(2):
                if abs(direction[axis]) < 1e-14:
                    if abs(base[axis]) > inset[axis]:
                        end = -np.inf
                        break
                else:
                    limits = ((-inset[axis]-base[axis])/direction[axis], (inset[axis]-base[axis])/direction[axis])
                    start, end = max(start, min(limits)), min(end, max(limits))
            if end-start < 4*diameter:
                continue
            parameter, length = _arc_samples(start, end, pitch, crimp, spacing)
            count = len(parameter)*cross_count
            if total+count > max_particles:
                raise ValueError(f"Sampling would exceed max_particles={max_particles}; choose a coarser spacing or smaller coupon")
            path, tangent, curvature = centreline(family, index, parameter, pitch=pitch, crimp=crimp, theta_deg=theta_deg)
            normal_3d = np.array([normal[0], 0., normal[1]])
            bending_normal = np.cross(normal_3d, tangent)
            angles = np.broadcast_to(angular_quantiles, (len(parameter), cross_count)).copy()
            product = curvature[:, None]*radii[None, :]
            for _ in range(7):
                angles -= (angles-product*np.sin(angles)-angular_quantiles)/(1-product*np.cos(angles))
            residual = np.max(np.abs(angles-product*np.sin(angles)-angular_quantiles))
            if residual > 1e-12:
                raise ValueError("Equal-volume cross-section sampling did not converge")
            offsets = (radii[None, :, None]*np.cos(angles)[..., None]*bending_normal[:, None, :]
                       +radii[None, :, None]*np.sin(angles)[..., None]*normal_3d)
            points = (path[:, None, :]+offsets+origin).reshape(-1, 3)
            volume = np.pi*radius**2*length/count
            identifier = len(fibres)
            positions.append(points)
            volumes.append(np.full(count, volume, dtype=np.float64))
            tangents.append(np.repeat(tangent, cross_count, axis=0))
            ids.append(np.full(count, identifier, dtype=np.int32))
            families.append(np.full(count, family, dtype=np.int16))
            sample_t = np.linspace(start, end, int(np.ceil((end-start)/(pitch/32)))+1)
            samples = centreline(family, index, sample_t, pitch=pitch, crimp=crimp, theta_deg=theta_deg)[0]+origin
            line_parameters.append(sample_t)
            line_positions.append(samples)
            line_ids.append(np.full(len(samples), identifier, dtype=np.int32))
            line_offsets.append(line_offsets[-1]+len(samples))
            planar_ends = np.array([start, end])[:, None]*direction+base
            conservative_lower = np.array([planar_ends[:, 0].min()-radius, -crimp-radius, planar_ends[:, 1].min()-radius])+origin
            conservative_upper = np.array([planar_ends[:, 0].max()+radius, crimp+radius, planar_ends[:, 1].max()+radius])+origin
            fibres.append({"id": identifier, "family_id": family, "periodic_line_index": index,
                "parameter_range_m": [float(start), float(end)], "reference_arclength_m": length,
                "radius_m": float(radius), "centreline_sample_slice": line_offsets[-2:],
                "reference_volume_m3": float(np.pi*radius**2*length), "mass_kg": float(np.pi*radius**2*length*density),
                "particle_count": count, "longitudinal_sections": len(parameter),
                "reference_section_spacing_m": length/len(parameter),
                "bounds_m": np.column_stack((points.min(axis=0), points.max(axis=0))).ravel().tolist(),
                "reference_tube_bounds_m": np.column_stack((conservative_lower, conservative_upper)).ravel().tolist()})
            total += count
    if not positions:
        raise ValueError("The coupon contains no fibres")
    x, volume = np.concatenate(positions), np.concatenate(volumes)
    tangent, fibre_id, family_id = np.concatenate(tangents), np.concatenate(ids), np.concatenate(families)
    if np.any(x[:, [0, 2]] < -1e-13) or np.any(x[:, [0, 2]] > dimensions+1e-13):
        raise ValueError("A fibre sample lies outside the prescribed coupon")
    candidates = []
    for left in fibres:
        u, n = _frame(left['family_id'], theta_deg)
        for right in fibres[left['id']+1:]:
            if left['family_id'] == right['family_id']:
                continue
            v, m = _frame(right['family_id'], theta_deg)
            crossing = np.linalg.solve(np.stack((n, m)), pitch*np.array([left['periodic_line_index']+.5, right['periodic_line_index']+.5]))
            parameters = np.array([crossing@u, crossing@v])
            pair = (left, right)
            if not all(fibre['parameter_range_m'][0] <= value <= fibre['parameter_range_m'][1] for fibre, value in zip(pair, parameters)):
                continue
            offsets = _crossing_offsets(left['family_id'], left['periodic_line_index'] % 2,
                                        right['family_id'], right['periodic_line_index'] % 2, pitch, crimp, diameter)
            closest = parameters+offsets
            interior = all(fibre['parameter_range_m'][0] <= value <= fibre['parameter_range_m'][1] for fibre, value in zip(pair, closest))
            if not interior:
                closest = parameters.copy()
            points = np.array([centreline(fibre['family_id'], fibre['periodic_line_index'], value,
                pitch=pitch, crimp=crimp, theta_deg=theta_deg)[0]+origin for fibre, value in zip(pair, closest)])
            distance = np.linalg.norm(points[1]-points[0])
            normal = (points[1]-points[0])/distance
            candidates.append({"fibre_ids": [left['id'], right['id']], "family_ids": [left['family_id'], right['family_id']],
                "crossing_parameters_m": parameters.tolist(), "reference_parameters_m": closest.tolist(),
                "seed_kind": "interior-closest-approach" if interior else "crossing-near-trimmed-end",
                "reference_centreline_points_m": points.tolist(), "normal_from_first_to_second": normal.tolist(),
                "reference_surface_points_m": [list(points[0]+radius*normal), list(points[1]-radius*normal)],
                "reference_midpoint_m": ((points[0]+points[1])/2).tolist(), "reference_gap_m": float(distance-diameter),
                "radius_m": float(radius)})
    metadata = {
        "geometry": "triaxial-circular-fibres-v1", "units": "m, kg", "matrix_present": False,
        "weave_plane": "XZ", "thickness_axis": "Y", "loading_axis": "X",
        "theta_deg": float(theta_deg), "theta_definition": "Weave rotation relative to the fixed x uniaxial loading axis",
        "nominal_family_angles_deg": [float(theta_deg+60*family) for family in range(3)],
        "dimensions_m": dimensions.tolist(), "origin_m": origin.tolist(), "diameter_m": float(diameter),
        "pitch_m": float(pitch), "crimp_amplitude_m": float(crimp), "clearance_m": float(clearance),
        "density_kg_m3": float(density), "requested_particle_spacing_m": float(particle_spacing),
        "sampling_density_multiplier": float(packing), "cross_section_ring_counts": ring_counts,
        "cross_section_particles": cross_count, "particle_count": total, "fibre_count": len(fibres),
        "reference_volume_m3": float(volume.sum()), "mass_kg": float(density*volume.sum()),
        "nominal_envelope_solid_fraction": float(volume.sum()/(np.prod(dimensions)*2*(crimp+radius))),
        "gauge_length_m": float(dimensions[0]-2*grip_length), "gauge_periods": float((dimensions[0]-2*grip_length)/pitch),
        "width_periods": float(dimensions[1]/pitch), "grip_length_m": float(grip_length),
        "grip_x_intervals_m": [[0., float(grip_length)], [float(dimensions[0]-grip_length), float(dimensions[0])]],
        "geometric_reference_length_m": float(diameter),
        "fracture_length_note": "The fixed geometric length is the fibre diameter. particle_spacing_equivalent_m is only a sampling diagnostic. Neither is assigned to a constitutive crack-band length; the solver must choose and report that length separately, for example from grid support.",
        "sampling": "Uniform centreline arclength and equal-area circular rings. Angular quantiles use the tube Jacobian 1-curvature*r*cos(angle); each particle carries an equal share of the analytic circular-tube volume within its fibre. Arclength uses composite eight-point Gauss quadrature.",
        "end_caps": "Centreline clipped to the nominal rectangle inset by radius, with circular caps normal to the local tangent. Corner remnants shorter than four diameters in planar line parameter are omitted.",
        "coupling_requirement": "Independent per-fibre velocity fields and explicit contact. Scattering these IDs onto one shared velocity field would impose bonding and is not the intended model.",
        "centreline_sampling": "Per-fibre endpoint-inclusive samples with planar parameter spacing no larger than pitch/32; centreline() is the analytic audit reference.",
        "contact_candidate_count": len(candidates),
        "contact_candidate_scope": "One reference search seed per retained planar crossing, with analytic curved-centreline refinement and circular surface offsets. If the refined location lies past a trimmed endpoint, the exact over/under crossing is retained as the seed instead. These are not initially active contacts or a complete candidate list after finite sliding. Update or expand the search from deformed positions; do not preserve normals or gaps as kinematic constraints.",
        "clearance_validation": dict(validation), "symmetry_validation": symmetry, "fibres": fibres,
    }
    return {"x": x, "volume": volume, "mass": density*volume, "fibre_id": fibre_id,
            "family_id": family_id, "tangent": tangent, "particle_spacing_equivalent_m": np.cbrt(volume),
            "reference_centrelines": {"x": np.concatenate(line_positions), "parameter_m": np.concatenate(line_parameters),
                "fibre_id": np.concatenate(line_ids), "offsets": np.array(line_offsets, dtype=np.int64)},
            "contact_candidates": candidates, "metadata": metadata}
