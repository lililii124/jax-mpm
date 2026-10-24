"""Independent array checks for large woven plates and four-edge fixtures."""

from itertools import product
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.woven_large_geometry import sample_large_weave, unit_cell


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _near(first, second, *, rtol=2e-10, atol=0.):
    return bool(np.isfinite(first) and np.isfinite(second) and np.isclose(first, second, rtol=rtol, atol=atol))


def _numpy_elastic(F, volume, phase, orientation, parameters, chunk=4096):
    E, nu, shear = [np.asarray(parameters[name]) for name in ("yarn_young", "yarn_poisson", "yarn_shear")]
    compliance = np.diag(np.r_[1/E, 1/shear[[2, 1, 0]]])
    for i, j, k in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
        compliance[i, j] = compliance[j, i] = -nu[k]/E[i]
    stiffness = np.linalg.inv(compliance)
    mu = parameters["matrix_young"]/(2*(1+parameters["matrix_poisson"]))
    lam = parameters["matrix_young"]*parameters["matrix_poisson"]/((1+parameters["matrix_poisson"])*(1-2*parameters["matrix_poisson"]))
    energy, strain_max, tau_error_scale = 0., 0., 0.
    tau = np.empty_like(F)
    for start in range(0, len(F), chunk):
        sl = slice(start, start+chunk)
        f, q = F[sl], orientation[sl]
        J = np.linalg.det(f)
        if not np.all(J > 0) or not np.isfinite(f).all():
            raise ValueError("nonpositive or nonfinite final deformation")
        green = .5*(f.swapaxes(1, 2)@f-np.eye(3))
        local = q.swapaxes(1, 2)@green@q
        strain = np.stack((local[:, 0, 0], local[:, 1, 1], local[:, 2, 2],
                           2*local[:, 1, 2], 2*local[:, 0, 2], 2*local[:, 0, 1]), axis=1)
        stress = strain@stiffness.T
        S = np.zeros_like(f)
        S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = stress[:, 0], stress[:, 1], stress[:, 2]
        S[:, 1, 2] = S[:, 2, 1] = stress[:, 3]
        S[:, 0, 2] = S[:, 2, 0] = stress[:, 4]
        S[:, 0, 1] = S[:, 1, 0] = stress[:, 5]
        yarn_tau = f@q@S@q.swapaxes(1, 2)@f.swapaxes(1, 2)
        matrix_tau = mu*(f@f.swapaxes(1, 2)-np.eye(3))+lam*np.log(J)[:, None, None]*np.eye(3)
        chosen = phase[sl] == 1
        tau[sl] = np.where(chosen[:, None, None], yarn_tau, matrix_tau)
        yarn_energy = .5*np.sum(strain*stress, axis=1)
        matrix_energy = .5*mu*(np.sum(f*f, axis=(1, 2))-3)-mu*np.log(J)+.5*lam*np.log(J)**2
        energy += float(np.sum(volume[sl]*np.where(chosen, yarn_energy, matrix_energy)))
        if np.any(chosen):
            strain_max = max(strain_max, float(np.max(np.linalg.norm(local[chosen], axis=(1, 2)))))
        tau_error_scale = max(tau_error_scale, float(np.max(np.abs(tau[sl]))))
    return energy, strain_max, tau, tau_error_scale


def saved_energy_gate(rows, reference_volume, maximum_young):
    """One-sided 5% creation check; additional transfer dissipation is allowed."""
    initial = None
    history, passed = [], True
    for row in rows:
        elastic = float(row["stored_elastic_energy_J"])
        kinetic = float(row["kinetic_translational_J"])+float(row["kinetic_apic_affine_J"])
        ball = float(row["ball_kinetic_energy_J"]) if row.get("ball_kinetic_energy_J") not in (None, "") else 0.
        energy = elastic+kinetic+ball
        if initial is None:
            initial = energy
        work = float(row["prescribed_tool_work_J"])+float(row["particle_external_work_J"])
        loss = float(row["contact_kinetic_loss_J"])
        projection = float(row["grid_projection_work_J"])
        supplied = initial+work
        roundoff = 64*np.finfo(float).eps*max(initial+abs(work), maximum_young*reference_volume, 1e-24)
        limit = .05*max(initial+max(work, 0.), 1e-24)+roundoff
        residual = energy+loss-projection-supplied
        valid = bool(np.isfinite([energy, work, loss, projection, residual, limit]).all()
                     and elastic >= -roundoff and kinetic >= 0 and ball >= 0
                     and loss >= -roundoff and projection <= roundoff and residual <= limit)
        passed &= valid
        history.append({"time": float(row["time"]), "energy_J": energy,
                        "supplied_energy_J": supplied, "creation_residual_J": residual,
                        "one_sided_limit_J": limit, "roundoff_allowance_J": roundoff, "passed": valid})
    return {"passed": bool(passed and history), "history": history, "relative_creation_limit": .05,
            "roundoff_rule": "64 eps times max(initial energy + abs(actual work), Emax*reference volume, 1e-24 J)",
            "scope": "saved source-reported energies; final constitutive energy separately recomputed from F; negative residual is additional numerical dissipation"}


def sphere_and_initial_energy_checks(rows, final, plan):
    """Bind the energy baseline and saved tool ledger to actual vector states."""
    if not rows:
        return {"initial_energy_ledger_matches_quiescent_plate": False}
    impact = plan["loading"] == "sphere-impact"
    mass = float(plan["tool"]["mass_kg"])
    initial_center = np.asarray(plan["tool"]["initial_center_m"], dtype=float)
    initial_velocity = np.asarray(plan["tool"]["initial_velocity_m_s"], dtype=float)
    centers = np.asarray([[float(row[f"ball_center_{axis}_m"]) for axis in "xyz"] for row in rows])
    velocities = np.asarray([[float(row[f"ball_velocity_{axis}_m_s"]) for axis in "xyz"] for row in rows])
    energy_names = ("stored_elastic_energy_J", "kinetic_translational_J", "kinetic_apic_affine_J",
                    "prescribed_tool_work_J", "particle_external_work_J", "contact_kinetic_loss_J", "grid_projection_work_J")
    checks = {
        "initial_energy_ledger_matches_quiescent_plate": all(float(rows[0][name]) == 0. for name in energy_names),
        "saved_tool_vector_states_finite": bool(np.isfinite(centers).all() and np.isfinite(velocities).all()),
        "saved_initial_tool_state_matches_plan": bool(np.array_equal(centers[0], initial_center) and np.array_equal(velocities[0], initial_velocity)),
        "saved_final_tool_state_matches_checkpoint": bool(np.array_equal(centers[-1], final["sphere_center"])
            and np.array_equal(velocities[-1], final["sphere_velocity"])),
    }
    if impact:
        expected = .5*mass*np.sum(velocities*velocities, axis=1)
        checks["saved_ball_kinetic_energy_matches_vector_velocity"] = all(
            _near(float(row["ball_kinetic_energy_J"]), value, rtol=2e-13, atol=1e-24)
            for row, value in zip(rows, expected))
    else:
        checks["prescribed_or_absent_tool_has_no_free_ball_energy"] = all(
            row.get("ball_kinetic_energy_J") in (None, "") for row in rows)
    return checks


def _same_blocks(first, second, *, rtol=0., atol=0., chunk=4096):
    if first.shape != second.shape:
        return False
    return all(np.allclose(first[start:start+chunk], second[start:start+chunk],
                           rtol=rtol, atol=atol, equal_nan=False)
               for start in range(0, len(first), chunk))


def review_reference_geometry(
    reference, volume, mass, phase, yarn_id, family_id, orientation, geometry,
    *, dx, tool_diameter=None, matrix_density=1100., yarn_density=1400.,
    expected_size=(.16, .16), expected_spacing=(.0005, .00025, .0005),
):
    """Reconstruct initial arrays; this does not validate any time integration."""
    inputs = [np.asarray(value) for value in (reference, volume, mass, phase, yarn_id, family_id, orientation)]
    reference, volume, mass, phase, yarn_id, family_id, orientation = inputs
    if (not np.isfinite(dx) or dx <= 0 or not np.isfinite(matrix_density)
            or matrix_density <= 0 or not np.isfinite(yarn_density) or yarn_density <= 0):
        raise ValueError("positive finite dx and phase densities are required")
    n = len(reference)
    shapes = ((n, 3), (n,), (n,), (n,), (n,), (n,), (n, 3, 3))
    checks = {"array_shapes": n > 0 and all(a.shape == shape for a, shape in zip(inputs, shapes))}
    if not checks["array_shapes"]:
        return {"passed": False, "checks": checks, "production_eligible": False,
                "scope": "reference geometry only"}
    parameters = {
        "size": geometry["size_xz_m"], "spacing": geometry["particle_spacing_m"],
        "pitch": geometry["pitch_m"], "yarn_width": geometry["yarn_width_m"],
        "yarn_height": geometry["yarn_height_m"], "gap": geometry["crossing_gap_m"],
        "matrix_cover": geometry["matrix_cover_m"], "origin": geometry["origin_m"],
    }
    expected = sample_large_weave(geometry["architecture"], **parameters)
    checks.update({
        "float64_reference_arrays": all(a.dtype == np.dtype("float64") for a in (reference, volume, mass, orientation)),
        "integer_static_labels": all(a.dtype.kind in "iu" for a in (phase, yarn_id, family_id)),
        "finite_reference_arrays": all(np.isfinite(a).all() for a in inputs),
        "requested_plate_dimensions": np.array_equal(parameters["size"], expected_size),
        "requested_particle_spacing": np.array_equal(parameters["spacing"], expected_spacing),
        "fixed_two_mm_thickness": bool(np.isclose(expected.metadata["thickness_m"], .002, rtol=0, atol=2e-18)),
        "actual_counts_match_metadata": n == geometry["particle_count"] == expected.metadata["particle_count"],
        "reference_coordinates_reconstructed": _same_blocks(reference, expected.positions, atol=2e-15),
        "reference_volume_reconstructed": _same_blocks(volume, expected.volume, rtol=2e-15),
        "phase_reconstructed": np.array_equal(phase, expected.phase),
        "yarn_id_reconstructed": np.array_equal(yarn_id, expected.yarn_id),
        "family_id_reconstructed": np.array_equal(family_id, expected.family_id),
        "orientation_reconstructed": _same_blocks(orientation, expected.orientation, atol=2e-13),
        "phase_mass_reconstructed": _same_blocks(mass, expected.volume*np.where(expected.phase == 1, yarn_density, matrix_density), rtol=2e-14),
        "positive_mass_and_volume": bool(np.all(mass > 0) and np.all(volume > 0)),
        "periodic_geometry_rechecked": bool(expected.metadata["periodic_geometry_audit"]["passed"]),
        "unit_cell_metadata_matches": geometry["unit_cell"] == expected.metadata["unit_cell"],
    })
    cell = unit_cell(geometry["architecture"], geometry["pitch_m"])
    diameter_ratios = None
    if tool_diameter is not None:
        diameter_ratios = (float(tool_diameter)/np.asarray(cell["orthogonal_repeating_cell_lengths_xz_m"])).tolist()
        checks["tool_diameter_at_least_ten_true_cells"] = bool(
            np.isfinite(tool_diameter) and tool_diameter > 0 and min(diameter_ratios) >= 10.-2e-13)
    resolution = {
        "three_cells_across_yarn_thickness": bool(geometry["yarn_height_m"]/dx >= 3.-1e-12),
        "eight_particle_layers_through_plate": bool(expected.metadata["lattice_counts"][1] >= 8),
    }
    return {
        "passed": all(bool(value) for value in checks.values()),
        "checks": {name: bool(value) for name, value in checks.items()},
        "resolution_checks": resolution,
        "production_eligible": False,
        "scope": "reference arrays and declared coarse sampling only; no dynamic, contact or convergence acceptance",
        "metrics": {"particles": n, "reference_volume_m3": float(volume.sum()),
                    "mass_kg": float(mass.sum()), "particle_spacing_m": parameters["spacing"],
                    "thickness_cells": float(.002/dx), "yarn_thickness_cells": float(geometry["yarn_height_m"]/dx),
                    "diameter_to_orthogonal_cell_edge_ratios": diameter_ratios,
                    "unit_cell": cell},
    }


def review_clamp_owner(owner, reference, *, dx, grid_origin, lower, shape,
                       plate_origin=(0., 0., 0.), size=(.16, .16), band=.005):
    """Check actual grid ownership, full reference support and reflection symmetry.

    Owners 0..3 denote x-min, x-max, z-min, z-max; -1 is free. Corner
    attribution may break label reflection, but its union must stay symmetric.
    The 1.5 dx band extension is the quadratic stencil radius.
    """
    owner, reference = np.asarray(owner), np.asarray(reference, dtype=float)
    lower, shape = np.asarray(lower), np.asarray(shape)
    grid_origin, plate_origin, size = (np.asarray(a, dtype=float) for a in (grid_origin, plate_origin, size))
    if (lower.shape != (3,) or shape.shape != (3,) or lower.dtype.kind not in "iu"
            or shape.dtype.kind not in "iu" or np.any(shape < 1)
            or owner.shape != (int(np.prod(shape)),) or owner.dtype.kind not in "iu"
            or reference.ndim != 2 or reference.shape[1] != 3 or not len(reference)
            or grid_origin.shape != (3,) or plate_origin.shape != (3,) or size.shape != (2,)
            or not np.isfinite(reference).all() or not np.isfinite([dx, band]).all()
            or dx <= 0 or band <= 0 or 2*band >= np.min(size)
            or not all(np.isfinite(a).all() for a in (grid_origin, plate_origin, size))):
        raise ValueError("invalid rectangular layout, ownership or plate dimensions")
    lo, hi = plate_origin[[0, 2]]-size/2, plate_origin[[0, 2]]+size/2
    edge_particles = (reference[:, 0] <= lo[0]+band, reference[:, 0] >= hi[0]-band,
                      reference[:, 2] <= lo[1]+band, reference[:, 2] >= hi[1]-band)
    strides = np.array((shape[1]*shape[2], shape[2], 1), dtype=np.int64)
    checks = {
        "valid_owner_labels": bool(np.all((owner >= -1) & (owner <= 3))),
        "four_nonempty_edge_groups": all(np.any(owner == group) for group in range(4)),
        "four_nonempty_reference_edge_bands": all(np.any(mask) for mask in edge_particles),
    }
    owner_in_band = True
    for start in range(0, len(owner), 16384):
        ids = np.arange(start, min(start+16384, len(owner)), dtype=np.int64)
        nodes = grid_origin+(lower+(ids[:, None]//strides)%shape)*dx
        tolerance = 64*np.finfo(float).eps*max(np.max(size), np.max(np.abs(plate_origin)), dx)
        allowed = np.column_stack((nodes[:, 0] <= lo[0]+band+1.5*dx+tolerance,
                                   nodes[:, 0] >= hi[0]-band-1.5*dx-tolerance,
                                   nodes[:, 2] <= lo[1]+band+1.5*dx+tolerance,
                                   nodes[:, 2] >= hi[1]-band-1.5*dx-tolerance))
        groups = owner[start:start+len(ids)]
        selected = groups >= 0
        owner_in_band &= bool(np.all((groups[selected] <= 3)))
        if np.all(groups[selected] <= 3):
            owner_in_band &= bool(np.all(allowed[selected, groups[selected]]))
    checks["no_clamped_nodes_outside_four_edge_bands"] = owner_in_band
    offsets = np.asarray(list(product(range(3), repeat=3)), dtype=np.int64)
    edge_union = np.flatnonzero(np.logical_or.reduce(edge_particles))
    all_inside, all_clamped = True, True
    for start in range(0, len(edge_union), 4096):
        points = reference[edge_union[start:start+4096]]
        scaled = (points-grid_origin)/dx
        base = np.floor(scaled-.5).astype(np.int64)
        fraction = scaled-base
        axis_weights = np.stack((.5*(1.5-fraction)**2, .75-(fraction-1)**2,
                                 .5*(fraction-.5)**2), axis=1)
        weights = np.prod(axis_weights[:, offsets, np.arange(3)], axis=-1)
        nodes = base[:, None, :]+offsets-lower
        inside = np.all((nodes >= 0) & (nodes < shape), axis=-1)
        required = weights > 0
        all_inside &= bool(np.all(inside[required]))
        valid = required & inside
        indices = np.sum(nodes*strides, axis=-1)[valid]
        all_clamped &= bool(np.all(owner[indices] >= 0))
    checks["all_positive_edge_support_inside_grid"] = all_inside
    checks["all_positive_edge_support_clamped"] = all_clamped
    centre_nodes = (plate_origin-grid_origin)/dx
    reflection_plane = np.rint(2*centre_nodes).astype(np.int64)
    for axis, label in ((0, "x"), (2, "z")):
        layout_symmetric = bool(abs(2*centre_nodes[axis]-reflection_plane[axis]) <= 1e-10
                                and 2*lower[axis]+shape[axis]-1 == reflection_plane[axis])
        checks[f"grid_{label}_reflection_matches_plate"] = layout_symmetric
        checks[f"clamp_union_{label}_reflection_equivalent"] = bool(layout_symmetric and np.array_equal(
            (owner >= 0).reshape(tuple(shape)), np.flip((owner >= 0).reshape(tuple(shape)), axis=axis)))
    return {"passed": all(checks.values()), "checks": checks,
            "metrics": {"clamped_grid_nodes": int(np.count_nonzero(owner >= 0)),
                        "edge_owner_counts": [int(np.count_nonzero(owner == group)) for group in range(4)],
                        "reference_edge_counts": [int(np.count_nonzero(mask)) for mask in edge_particles],
                        "band_width_m": float(band)},
            "scope": "static grid mask only; all-component application and absence of other constraints require runtime/source checks"}


def accepted_output_steps(plan, summary, status, saved_rows):
    """Keep the full allocation/protocol separate from a planned accepted prefix."""
    declared = summary.get('segment_complete') is True or status.get('segment_complete') is True
    if not declared:
        return list(plan['output_steps']), False
    end = summary.get('segment_end_frame')
    start = summary.get('segment_start_frame')
    if (type(end) is not int or type(start) is not int or not 0 <= start <= end < len(plan['output_steps'])-1
            or end != saved_rows-1 or plan['probe_steps'] != 0
            or summary.get('segment_complete') is not True or status.get('segment_complete') is not True
            or any(item.get('completed') is not False or item.get('probe_complete') is not False
                   or item.get('phase') != 'checkpoint-ready' for item in (summary, status))
            or summary.get('runtime_stop_after_frame') != end or status.get('runtime_stop_after_frame') != end
            or status.get('segment_start_frame') != start or status.get('segment_end_frame') != end
            or status.get('frame') != end
            or summary.get('full_steps') != plan['full_steps'] or status.get('full_steps') != plan['full_steps']
            or summary.get('target_simulation_time') != plan['duration_s']
            or summary.get('steps') != plan['output_steps'][end]
            or status.get('steps') != plan['output_steps'][end]):
        raise ValueError('Segment metadata does not identify a planned complete saved prefix of the full protocol')
    return list(plan['output_steps'][:end+1]), True


def accepted_clock_matches(final_time, summary_time, status_time, saved_time, steps, dt):
    """Bind the actual state and metadata to the same accepted physical instant."""
    try:
        actual = np.asarray([final_time, summary_time, status_time, saved_time], dtype=float)
        return bool(actual.shape == (4,) and np.isfinite(actual).all()
            and np.all(actual == actual[-1])
            and np.isclose(actual[-1], steps*dt, rtol=5e-12, atol=0.))
    except (ValueError, TypeError):
        return False


def review_folder(folder):
    folder = Path(folder)
    plan, summary, status, layout = [json.loads((folder/name).read_text(encoding="utf-8"))
                                    for name in ("plan.json", "summary.json", "status.json", "layout.json")]
    rows, diagnostics = _csv_rows(folder/"observables.csv"), _csv_rows(folder/"diagnostics.csv")
    with np.load(folder/"reference.npz", allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    with np.load(folder/"final_state_0.npz", allow_pickle=False) as data:
        final = {key: data[key] for key in data.files}
    with np.load(folder/"layout_arrays.npz", allow_pickle=False) as data:
        owner, contact_indices = data["clamp_owner"], data["contact_indices"]
    n = plan["particles"]
    static = review_reference_geometry(reference["x"], reference["volume"], reference["mass"],
        reference["phase"], reference["yarn_id"], reference["family_id"], reference["orientation"],
        plan["geometry"], dx=plan["grid"]["dx_m"],
        tool_diameter=None if plan["loading"] == "pressure-bending" else plan["tool"]["diameter_m"])
    clamp = review_clamp_owner(owner, reference["x"], dx=layout["dx_m"], grid_origin=layout["origin"],
        lower=np.asarray(layout["lower"], dtype=np.int64), shape=np.asarray(layout["shape"], dtype=np.int64),
        plate_origin=plan["geometry"]["origin_m"], size=plan["geometry"]["size_xz_m"], band=plan["clamp"]["band_width_m"])
    from examples.woven_large_plate import WOVEN_MATERIAL, acoustic_speed, array_descriptor, digest, source_record
    with np.load(folder/"particle_fields.npz", allow_pickle=False) as data:
        render_fields = {key: data[key] for key in data.files}
    checks = {
        "reference_geometry": static["passed"], "actual_four_edge_node_mask": clamp["passed"],
        "material_parameters_unchanged": plan["material_parameters"] == WOVEN_MATERIAL,
        "elastic_woven_material_only": plan["material_kind"] == "woven",
        "four_edge_full_components_no_backing": plan["clamp"] == {
            "profile": "four-edge-full-clamp", "band_width_m": .005, "components": [0, 1, 2],
            "owner_order": ["xmin", "xmax", "zmin", "zmax"],
            "corner_assignment": "first matching owner; one impulse per node", "backing": False,
            "support_margin_m": 1.5*plan["grid"]["dx_m"]},
        "no_gravity_or_implicit_domain_walls": summary["configs"][0]["gravity"] == [0., 0., 0.] and summary["configs"][0]["grid_type"] == "packed-rectangular",
        "numerical_source_matches_current_files": plan["source"] == source_record(),
        "summary_source_matches_plan": summary["source"] == plan["source"],
        "status_signature_matches_plan": status["signature"] == digest(plan),
        "runtime_valid": status["coupling_valid"] is True and bool(final["valid"]),
        "target_time_and_loading_unchanged": plan["duration_s"] == 200e-6 and plan["loading_duration_s"] == 150e-6 and plan["CFL"] == .15,
        "declared_coarse_grid_and_sampling_budget": plan["grid"]["dx_m"] == .0005
            and plan["grid"]["y_bounds_m"] == [-.008, .004] and plan["grid"]["padding_cells"] == 3
            and plan["grid"]["max_nodes"] == 8_000_000 and plan["grid"]["contact_capacity"] == 65536
            and plan["grid"]["contact_half_width_m"] == .045 and layout["nodes"] <= 8_000_000,
        "tool_geometry_mass_and_initial_gap_fixed": plan["tool"]["diameter_m"] == .1
            and plan["tool"]["mass_kg"] == .01 and np.allclose(plan["tool"]["initial_center_m"], [0., .052, 0.], rtol=0, atol=1e-16)
            and plan["tool"]["initial_nodal_gap_m"] == .0005,
        "finite_final_arrays": all(np.isfinite(value).all() for value in final.values()),
        "complete_final_array_shapes": all(final[key].shape == shape for key, shape in
            (("x", (n, 3)), ("v", (n, 3)), ("F", (n, 3, 3)), ("C", (n, 3, 3)),
             ("tau", (n, 3, 3)), ("mass", (n,)), ("volume", (n,)), ("group_impulses", (4, 3)),
             ("sphere_center", (3,)), ("sphere_velocity", (3,)))),
        "mass_and_reference_volume_unchanged": np.array_equal(final["mass"], reference["mass"]) and np.array_equal(final["volume"], reference["volume"]),
        "static_single_bonded_field": np.all(final["fibre_id"] == 0),
        "rendered_static_labels_match_reference": set(render_fields) == {"phase", "family_id", "yarn_id"}
            and all(np.array_equal(value, reference[key]) for key, value in render_fields.items()),
        "static_label_descriptors_match_render_file": summary["particle_fields"] == {
            "file": "particle_fields.npz", "fields": {key: array_descriptor(value) for key, value in render_fields.items()}},
        "summary_and_status_agree": all(summary[key] == status[key] for key in ("completed", "probe_complete", "steps", "full_steps")),
        "fixed_fixture_physical_work_zero": all(float(row["actual_fixed_clamp_work_J"]) == 0. for row in rows),
        "no_contact_capacity_or_search_failure": all(float(row["contact_overflow_events"]) == 0. and float(row["contact_invalid_window_events"]) == 0. for row in rows),
        "all_saved_diagnostics_valid": all(row["state_valid"] == "True" and row["coupling_valid"] == "True" and float(row["min_J"]) > 0. and float(row["minimum_step_J"]) > 0. for row in diagnostics),
        "small_stepwise_yarn_strain_scope": all(0 <= float(row["peak_step_yarn_green_strain_norm"]) <= .1 for row in rows),
    }
    with np.load(folder/"checkpoint.npz", allow_pickle=False) as checkpoint:
        expected_fields = {key if key.startswith("sphere_") else "state_"+key for key in final}
        expected_fields |= {"signature", "aggregate", "frame", "solver_seconds", "jax_version", "tables"}
        checks["complete_checkpoint_fields"] = set(checkpoint.files) == expected_fields
        checks["checkpoint_final_state_same_arrays"] = all(np.array_equal(value, checkpoint[key if key.startswith("sphere_") else "state_"+key]) for key, value in final.items())
        checks["checkpoint_signature_and_runtime_match"] = (str(checkpoint["signature"]) == digest(plan)
            and str(checkpoint["jax_version"]) == summary["jax_version"]
            and float(checkpoint["solver_seconds"]) == summary["solver_seconds"])
        aggregate = checkpoint["aggregate"]
        checks["checkpoint_valid_accepted_prefix"] = bool(aggregate.shape == (11,) and np.isfinite(aggregate).all()
            and aggregate[1] == 1. and int(aggregate[0]) == summary["steps"]
            and int(checkpoint["frame"]) == len(rows)-1)
        tables = json.loads(str(checkpoint["tables"]))
        checks["checkpoint_tables_match_csv"] = all(
            len(actual) == len(saved) and all(set(row) == set(original) and all(
                row[key] == ("" if original[key] is None else str(original[key])) for key in row)
                for row, original in zip(actual, saved))
            for actual, saved in ((rows, tables["rows"]), (diagnostics, tables["diagnostics"])))
    for key, value in reference.items():
        checks[f"reference_descriptor_{key}"] = array_descriptor(value) == plan["reference_arrays"][key]
    checks["clamp_owner_descriptor"] = array_descriptor(owner) == layout["clamp_owner"]
    checks["contact_indices_descriptor"] = array_descriptor(contact_indices) == layout["contact_indices"]
    checks["actual_grid_contact_half_width_matches_configuration"] = layout.get("actual_contact_half_width_m") == plan["grid"]["contact_half_width_m"]
    dt, dx = plan["dt_s"], plan["grid"]["dx_m"]
    speed = acoustic_speed(plan["material_parameters"])
    substeps = int(np.ceil(200e-6/(60*.15*dx/speed)))
    checks["timestep_and_full_protocol_independently_aligned"] = (plan["frames"] == 60
        and plan["full_steps"] == 60*substeps and dt == 200e-6/(60*substeps)
        and plan["frame_steps"] == list(range(0, 60*substeps+1, substeps))
        and _near(plan["initial_acoustic_bound_m_s"], speed, rtol=2e-14))
    lower, shape = np.asarray(layout["lower"]), np.asarray(layout["shape"])
    bounds = np.array([[-.08, .08], [-.008, .004], [-.08, .08]])
    first = np.floor(bounds[:, 0]/dx).astype(int)-3
    last = np.ceil(bounds[:, 1]/dx).astype(int)+3
    checks["rectangular_grid_has_no_cubic_air_extension"] = (np.array_equal(lower, first)
        and np.array_equal(shape, last-first+1) and layout["nodes"] == int(np.prod(shape))
        and layout["origin"] == [0., 0., 0.])
    axes = [(lower[axis]+np.arange(shape[axis]))*dx for axis in range(3)]
    tolerance = 32*np.finfo(float).eps*.16
    selected_x = np.flatnonzero(np.abs(axes[0]) <= .045+tolerance)
    selected_z = np.flatnonzero(np.abs(axes[2]) <= .045+tolerance)
    expected_contact = ((selected_x[:, None, None]*shape[1]+np.arange(shape[1])[None, :, None])
                        *shape[2]+selected_z[None, None, :]).ravel().astype(np.int32)
    checks["contact_window_nodes_independently_reconstructed"] = np.array_equal(contact_indices, expected_contact)
    output_steps, segment = accepted_output_steps(plan, summary, status, len(rows))
    with np.load(folder/"trajectory.npz", allow_pickle=False) as data:
        times, body = data["time"], data["body"]
    checks["saved_time_prefix_matches_plan"] = bool(len(rows) == len(diagnostics) == len(times) == len(output_steps)
        and np.allclose(times, np.asarray(output_steps)*dt, rtol=5e-12, atol=0.)
        and all(int(float(row["accepted_steps"])) == step for row, step in zip(rows, output_steps))
        and all(float(row["time"]) == t for row, t in zip(rows, times))
        and body.shape == (n,) and np.all(body == 0))
    checks['accepted_state_clock_matches_saved_frame'] = bool(len(times) > 0 and accepted_clock_matches(
        float(final['time']), summary.get('simulation_time'), status.get('time'),
        float(times[-1]), summary['steps'], dt))
    full = bool(summary["completed"] is True and status["completed"] is True and plan["probe_steps"] == 0
                and summary["steps"] == plan["full_steps"] and len(times) == 61
                and _near(float(final["time"]), 200e-6, rtol=5e-12))
    probe = bool(summary["completed"] is False and summary["probe_complete"] is True
                 and status["probe_complete"] is True and 0 < plan["probe_steps"] < plan["full_steps"]
                 and summary["steps"] == plan["probe_steps"])
    arrays = {key: np.load(folder/f"trajectory_{key}.npy", mmap_mode="r") for key in ("x", "speed", "damage", "temperature")}
    try:
        checks["mapped_history_shapes_and_dtypes"] = bool(
            arrays["x"].shape == (len(plan['output_steps']), n, 3) and arrays["x"].dtype == np.dtype("float64")
            and all(arrays[key].shape == (len(plan['output_steps']), n) and arrays[key].dtype == np.dtype("float32") for key in ("speed", "damage", "temperature")))
        checks["first_and_last_coordinates_match_state"] = (np.array_equal(arrays["x"][0], reference["x"])
            and np.array_equal(arrays["x"][len(times)-1], final["x"]))
        checks["final_speed_matches_state"] = np.array_equal(arrays["speed"][len(times)-1], np.linalg.norm(final["v"], axis=1).astype(np.float32))
        checks["all_history_finite_and_inside_rectangular_support"] = True
        checks["elastic_damage_and_temperature_fields_are_constant"] = True
        checks["saved_displacements_match_raw_coordinates"] = True
        checks["clamped_reference_particles_stationary"] = True
        checks["all_saved_frames_match_checkpoint_descriptors"] = len(tables.get("trajectory_frames", [])) == len(times)
        size = np.asarray(plan["geometry"]["size_xz_m"])
        edge = (np.abs(reference["x"][:, 0]) >= size[0]/2-.005) | (np.abs(reference["x"][:, 2]) >= size[1]/2-.005)
        roi = np.linalg.norm(reference["x"][:, (0, 2)], axis=1) <= .005
        lower, shape, origin = np.asarray(layout["lower"]), np.asarray(layout["shape"]), np.asarray(layout["origin"])
        for frame, row in enumerate(rows):
            if checks["all_saved_frames_match_checkpoint_descriptors"]:
                described = tables["trajectory_frames"][frame]
                checks["all_saved_frames_match_checkpoint_descriptors"] &= bool(set(described) == set(arrays)
                    and all(array_descriptor(array[frame]) == described[name] for name, array in arrays.items()))
            maximum, minimum_y, edge_error, center_sum, center_count = 0., np.inf, 0., 0., 0
            for start in range(0, n, 8192):
                sl = slice(start, start+8192)
                x = arrays["x"][frame, sl]
                displacement = x-reference["x"][sl]
                base = np.floor((x-origin)/dx-.5)-lower
                checks["all_history_finite_and_inside_rectangular_support"] &= bool(np.isfinite(x).all()
                    and np.isfinite(arrays["speed"][frame, sl]).all() and np.all(arrays["speed"][frame, sl] >= 0)
                    and np.all(base >= 0) and np.all(base+2 < shape))
                checks["elastic_damage_and_temperature_fields_are_constant"] &= bool(np.all(arrays["damage"][frame, sl] == 0)
                    and np.all(arrays["temperature"][frame, sl] == np.float32(293.15)))
                norms = np.linalg.norm(displacement, axis=1)
                maximum = max(maximum, float(norms.max()))
                minimum_y = min(minimum_y, float(displacement[:, 1].min()))
                selected = edge[sl]
                if selected.any():
                    edge_error = max(edge_error, float(norms[selected].max()))
                selected = roi[sl]
                center_sum += float(displacement[selected, 1].sum())
                center_count += int(selected.sum())
            checks["saved_displacements_match_raw_coordinates"] &= bool(
                _near(maximum, float(row["maximum_displacement_m"]), atol=1e-14)
                and _near(minimum_y, float(row["minimum_displacement_y_m"]), atol=1e-14)
                and center_count > 0 and _near(center_sum/center_count, float(row["center_mean_displacement_y_m"]), atol=1e-14)
                and _near(edge_error, float(row["clamp_position_error_m"]), atol=1e-14))
            checks["clamped_reference_particles_stationary"] &= bool(edge_error <= 1e-10*dx
                and float(row["clamp_speed_error_m_s"]) <= 1e-9)
    finally:
        for array in arrays.values():
            array._mmap.close()
    energy, strain_max, tau, stress_scale = _numpy_elastic(final["F"], reference["volume"], reference["phase"], reference["orientation"], plan["material_parameters"])
    volume = float(reference["volume"].sum())
    energy_roundoff = 64*np.finfo(float).eps*40e9*volume
    checks["final_energy_recomputed"] = _near(energy, float(rows[-1]["stored_elastic_energy_J"]), rtol=2e-9, atol=energy_roundoff)
    checks["final_stress_recomputed"] = bool(np.allclose(final["tau"], tau, rtol=2e-9, atol=64*np.finfo(float).eps*40e9))
    checks["final_yarn_strain_independent"] = bool(strain_max <= .1 and strain_max <= float(rows[-1]["peak_step_yarn_green_strain_norm"])+1e-12)
    mass = reference["mass"]
    plate_momentum = np.sum(mass[:, None]*final["v"], axis=0)
    plate_balance = plate_momentum-final["boundary_impulse"]-final["contact_impulse"]-final["external_impulse"]
    scale = max(np.linalg.norm(final["boundary_impulse"])+np.linalg.norm(final["contact_impulse"])+np.linalg.norm(final["external_impulse"]),
                plan["tool"]["mass_kg"]*np.linalg.norm(plan["tool"]["initial_velocity_m_s"]), 1e-20)
    momentum_limit = 2e-9*scale
    checks["plate_linear_momentum_balance"] = bool(np.linalg.norm(plate_balance) <= momentum_limit)
    checks["edge_impulses_sum_to_actual_clamp"] = bool(np.allclose(final["group_impulses"].sum(0), final["boundary_impulse"], rtol=2e-10, atol=momentum_limit))
    checks["finite_positive_tool_inertia_and_radius"] = bool(float(final["sphere_mass"]) == plan["tool"]["mass_kg"] > 0
        and float(final["sphere_radius"]) == plan["tool"]["diameter_m"]/2 > 0)
    expected_external = np.zeros(3)
    pressure_evaluation_roundoff = 0.
    if plan["loading"] == "pressure-bending":
        size = np.asarray(plan["geometry"]["size_xz_m"])
        spacing = np.asarray(plan["geometry"]["particle_spacing_m"])
        top = ((reference["x"][:, 1] == reference["x"][:, 1].max())
               & (np.abs(reference["x"][:, 0]) < size[0]/2-.005) & (np.abs(reference["x"][:, 2]) < size[1]/2-.005))
        instants = np.arange(1, summary["steps"]+1)*dt
        fraction = .5*(1-np.cos(np.pi*np.clip(instants/plan["loading_duration_s"], 0, 1)))
        peak_force = plan["pressure"]["peak_Pa"]*spacing[0]*spacing[2]*top.sum()
        expected_external[1] = -peak_force*dt*float(fraction.sum())
        # The source evaluates 0.5*(1-cos), whose early-time cancellation
        # has an absolute floating-point budget, not a relative-only one.
        pressure_evaluation_roundoff = 16*np.finfo(float).eps*summary["steps"]*dt*peak_force
    checks["actual_pressure_or_zero_external_impulse"] = bool(np.allclose(final["external_impulse"], expected_external,
        rtol=2e-10, atol=max(momentum_limit, pressure_evaluation_roundoff)))
    checks["saved_vector_impulses_match_final_state"] = True
    for stem in ("boundary_impulse", "contact_impulse", "external_impulse"):
        recorded = np.array([float(rows[-1][f"{stem}_{axis}_Ns"]) for axis in "xyz"])
        checks["saved_vector_impulses_match_final_state"] &= bool(np.allclose(recorded, final[stem], rtol=2e-10, atol=momentum_limit))
    if plan["loading"] == "sphere-impact":
        checks["free_impact_initial_conditions_fixed"] = (plan["tool"]["initial_velocity_m_s"] == [0., -20., 0.]
            and plan["tool"]["motion"] == "free finite mass" and plan["tool"]["stroke_m"] == 0.)
        ball_change = float(final["sphere_mass"])*(final["sphere_velocity"]-np.asarray(plan["tool"]["initial_velocity_m_s"]))
        checks["free_sphere_equal_opposite_contact_impulse"] = bool(np.linalg.norm(ball_change+final["contact_impulse"]) <= momentum_limit)
        checks["coupled_system_linear_momentum_balance"] = bool(np.linalg.norm(plate_momentum+ball_change-final["boundary_impulse"]-final["external_impulse"]) <= momentum_limit)
        checks["impact_has_no_prescribed_tool_work"] = bool(float(final["tool_work"]) == 0.)
        if full:
            checks["complete_impact_exercised_contact"] = bool(np.linalg.norm(final["contact_impulse"]) > 0. and max(float(row["maximum_displacement_m"]) for row in rows) > 0.)
    elif plan["loading"] == "hemisphere-indentation":
        checks["hemisphere_loading_conditions_fixed"] = plan["tool"]["stroke_m"] == .004 and plan["tool"]["initial_velocity_m_s"] == [0., 0., 0.]
        phase = np.clip(float(final["time"])/plan["loading_duration_s"], 0, 1)
        expected_center = np.asarray(plan["tool"]["initial_center_m"])+[0., -.004*.5*(1-np.cos(np.pi*phase)), 0.]
        checks["hemisphere_actual_prescribed_center"] = bool(np.allclose(final["sphere_center"], expected_center, rtol=0, atol=1e-13))
        if full:
            checks["hemisphere_hold_velocity_zero"] = bool(np.array_equal(final["sphere_velocity"], np.zeros(3)))
            checks["complete_indentation_exercised_contact"] = bool(np.linalg.norm(final["contact_impulse"]) > 0.)
    else:
        checks["pressure_loading_conditions_fixed"] = plan["pressure"]["peak_Pa"] == .25e6
        checks["pressure_has_no_tool_contact_or_work"] = bool(np.array_equal(final["contact_impulse"], np.zeros(3)) and float(final["tool_work"]) == 0.)
    translational = .5*float(np.sum(mass*np.sum(final["v"]**2, axis=1)))
    affine = dx**2/8*float(np.sum(mass*np.sum(final["C"]**2, axis=(1, 2))))
    checks["final_apic_kinetic_energy_recomputed"] = _near(translational, float(rows[-1]["kinetic_translational_J"]), atol=1e-24) and _near(affine, float(rows[-1]["kinetic_apic_affine_J"]), atol=1e-24)
    energies = saved_energy_gate(rows, volume, 40e9)
    checks["saved_one_sided_energy_creation_limit"] = energies["passed"]
    checks.update(sphere_and_initial_energy_checks(rows, final, plan))
    checks = {name: bool(value) for name, value in checks.items()}
    files = ("plan.json", "summary.json", "status.json", "layout.json", "reference.npz", "final_state_0.npz", "checkpoint.npz",
             "layout_arrays.npz", "particle_fields.npz", "trajectory.npz", "observables.csv", "diagnostics.csv",
             "trajectory_x.npy", "trajectory_speed.npy", "trajectory_damage.npy", "trajectory_temperature.npy")
    return {"passed": bool(full and all(checks.values())), "probe_passed": bool(probe and all(checks.values())),
            "segment_passed": bool(segment and all(checks.values())),
            "segment_end_frame": len(times)-1, "allocated_states": len(plan['output_steps']),
            "accepted_saved_states": len(times),
            "full_protocol_completed": full, "full_saved_history_checked": True,
            "coarse_preview_eligible": bool(full and all(checks.values())), "production_eligible": False,
            "checks": checks, "reference_review": static, "clamp_review": clamp, "energy_review": energies,
            "metrics": {"independent_final_energy_J": energy, "independent_final_yarn_green_norm": strain_max,
                        "independent_final_max_stress_Pa": stress_scale, "final_plate_momentum_residual_Ns": float(np.linalg.norm(plate_balance)),
                        "momentum_tolerance_Ns": momentum_limit, "pressure_wave_evaluation_roundoff_Ns": pressure_evaluation_roundoff,
                        "steps": summary["steps"], "full_steps": plan["full_steps"]},
            "scope": "coarse fully bonded elastic woven dynamics; no damage, separation, material validation or resolved yarn-thickness claim. All saved coordinates checked; constitutive F and energy independently reconstructed only for the final state.",
            "producer_source": plan["source"], "review_source_sha256": file_sha256(__file__),
            "input_sha256": {name: file_sha256(folder/name) for name in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-probe", action="store_true")
    parser.add_argument("--allow-segment", action="store_true")
    args = parser.parse_args()
    output = args.output or args.folder/"interpretation.json"
    try:
        result = review_folder(args.folder)
    except Exception as error:
        result = {"passed": False, "probe_passed": False, "segment_passed": False, "production_eligible": False,
                  "full_saved_history_checked": False, "error_type": type(error).__name__, "error": str(error),
                  "review_source_sha256": file_sha256(__file__)}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "probe_passed", "production_eligible")}))
    raise SystemExit(0 if result["passed"] or (args.allow_probe and result["probe_passed"])
                    or (args.allow_segment and result["segment_passed"]) else 1)


if __name__ == "__main__":
    main()
