"""Independently review saved finite-range adhesion runs with NumPy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {key: None if value in ("", "None") else float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def motion(path, time):
    times = np.asarray(path["times"])
    centers = np.asarray(path["centers"])
    duration = np.diff(times)
    phase = np.clip((time - times[:-1]) / duration, 0.0, 1.0)
    rate = np.pi / (2 * duration) * np.sin(np.pi * phase)
    rate[(time <= times[:-1]) | (time >= times[1:])] = 0.0
    increments = np.diff(centers, axis=0)
    position = centers[0] + ((1 - np.cos(np.pi * phase))[:, None] / 2 * increments).sum(0)
    return position, (rate[:, None] * increments).sum(0)


def surface_result(points, area, center, radius, law):
    radial = points - center
    distance = np.linalg.norm(radial, axis=1)
    if np.any(distance == 0):
        raise ValueError("A surface point reached a sphere center")
    gap = distance - radius
    work, reach, stiffness = (law[key] for key in (
        "work_of_adhesion", "interaction_range", "penalty_stiffness"
    ))
    compressed = gap < 0
    adhesive = (gap >= 0) & (gap < reach) & (work > 0)
    traction = np.where(compressed, -stiffness * gap, np.where(adhesive, -work / reach, 0.0))
    forces = (area * traction / distance)[:, None] * radial
    penalty = float((0.5 * stiffness * np.minimum(gap, 0.0) ** 2 * area).sum())
    attraction = float((work * (np.clip(gap / reach, 0.0, 1.0) - 1) * area).sum())
    return {
        "reaction": -forces.sum(0),
        "moment": -np.cross(radial, forces).sum(0),
        "force_moment_at_origin": np.cross(points, forces).sum(0),
        "forces": forces,
        "contact_reference_area_m2": int(compressed.sum()) * area,
        "adhesive_reference_area_m2": int(adhesive.sum()) * area,
        "compressed_points": int(compressed.sum()),
        "adhesive_points": int(adhesive.sum()),
        "min_gap_m": float(gap.min()),
        "max_penetration_m": float(max(0, -gap.min())),
        "potential_J": penalty + attraction,
        "penalty_potential_J": penalty,
        "adhesive_potential_J": attraction,
    }


def path_clearance(paths, duration):
    times = np.linspace(0.0, duration, 1001)
    centers = [np.array([motion(path, time)[0] for time in times]) for path in paths]
    speeds = [float(np.max(
        np.linalg.norm(np.diff(np.asarray(path["centers"]), axis=0), axis=1)
        * np.pi / (2 * np.diff(path["times"]))
    )) for path in paths]
    lower_bounds = [
        float(np.linalg.norm(centers[i] - centers[j], axis=1).min())
        - paths[i]["radius"] - paths[j]["radius"]
        - (speeds[i] + speeds[j]) * (times[1] - times[0]) / 2
        for i in range(len(paths)) for j in range(i)
    ]
    return min(lower_bounds) if lower_bounds else None


def _review(folder, trajectory):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError("The numerical run is incomplete")
    definition, config = summary["definition"], summary["configs"][0]
    law, paths = definition["adhesion"], definition["rigid_spheres"]
    observations = read_rows(folder / "observables.csv")
    diagnostics = read_rows(folder / "diagnostics.csv")
    times = np.array([row["time"] for row in observations])
    if len(trajectory) != len(times) or len(diagnostics) != len(times):
        raise ValueError("Saved positions and tables have different frame counts")
    if trajectory.shape[1:] != (sum(summary["particles"]), 3):
        raise ValueError("Saved positions disagree with the particle count or dimension")
    reference = trajectory[0]
    surface = np.flatnonzero(reference[:, 1] == reference[:, 1].max())
    surface_y_matches = bool(np.isclose(
        reference[surface[0], 1], definition["contact_sampling_plane_y_m"], rtol=1e-12, atol=1e-14
    ))
    area = definition["reference_area_per_surface_point_m2"]
    dx = config["domain_size"] / config["n_grid"]
    spacing = np.asarray(definition["particle_spacings_m"])
    maximum_errors, comparisons = {}, {}
    recomputed = [[] for _ in paths]
    undeformed_reactions = [[] for _ in paths]
    surface_displacement = []
    trajectories_finite = True

    def compare(name, actual, expected, atol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected)))
        maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
        matches = bool(np.allclose(actual, expected, rtol=1e-9, atol=atol))
        comparisons[name] = comparisons.get(name, True) and matches

    for frame, (time, observed) in enumerate(zip(times, observations)):
        all_points = trajectory[frame]
        trajectories_finite &= bool(np.isfinite(all_points).all())
        points = all_points[surface]
        surface_displacement.append(points[:, 1] - reference[surface, 1])
        compare("surface_displacement_y_min_m", observed["surface_displacement_y_min_m"],
                np.min(surface_displacement[-1]))
        compare("surface_displacement_y_max_m", observed["surface_displacement_y_max_m"],
                np.max(surface_displacement[-1]))
        compare("surface_reference_area_m2", observed["surface_reference_area_m2"],
                len(surface) * area)
        sums = {"contact_potential_J": 0.0, "penalty_potential_J": 0.0,
                "adhesive_potential_J": 0.0, "actuator_power_W": 0.0}
        reaction = np.zeros(3)
        particle_force = np.zeros(3)
        origin_moment = np.zeros(3)
        for index, path in enumerate(paths):
            center, velocity = motion(path, time)
            result = surface_result(points, area, center, path["radius"], law)
            undeformed_reactions[index].append(
                surface_result(reference[surface], area, center, path["radius"], law)["reaction"]
            )
            prefix = f"sphere_{index}_"
            for axis, name in enumerate("xyz"):
                column = prefix + f"undeformed_surface_reaction_{name}_N"
                if column in observed or definition.get("series") == "adhesion-suite-v1":
                    compare(column, observed[column], undeformed_reactions[index][-1][axis])
            column = prefix + "deformation_feedback_force_difference_N"
            if column in observed or definition.get("series") == "adhesion-suite-v1":
                compare(column, observed[column], np.linalg.norm(
                    result["reaction"] - undeformed_reactions[index][-1]
                ))
            for axis, name in enumerate("xyz"):
                for key, expected in (
                    (f"center_{name}_m", center[axis]),
                    (f"velocity_{name}_m_per_s", velocity[axis]),
                    (f"reaction_{name}_N", result["reaction"][axis]),
                    (f"reaction_moment_{name}_Nm", result["moment"][axis]),
                ):
                    compare(prefix + key, observed[prefix + key], expected)
            result["max_bulk_penetration_m"] = float(max(
                0, path["radius"] - np.linalg.norm(all_points - center, axis=1).min()
            ))
            result["actuator_power_W"] = float(-result["reaction"] @ velocity)
            for key in ("contact_reference_area_m2", "adhesive_reference_area_m2",
                        "compressed_points", "adhesive_points", "min_gap_m",
                        "max_penetration_m", "max_bulk_penetration_m", "potential_J",
                        "actuator_power_W"):
                compare(prefix + key, observed[prefix + key], result[key])
            sums["contact_potential_J"] += result["potential_J"]
            for key in ("penalty_potential_J", "adhesive_potential_J", "actuator_power_W"):
                sums[key] += result[key]
            reaction += result["reaction"]
            particle_force += result["forces"].sum(0)
            origin_moment += result["force_moment_at_origin"] + np.cross(center, result["reaction"]) + result["moment"]
            result.pop("forces")
            result.pop("force_moment_at_origin")
            recomputed[index].append(result)
        for key, expected in sums.items():
            compare(key, observed[key], expected)
        compare("force_balance_error_N", observed["force_balance_error_N"], np.linalg.norm(particle_force + reaction))
        compare("moment_balance_error_Nm", observed["moment_balance_error_Nm"], np.linalg.norm(origin_moment))
        for axis, name in enumerate("xyz"):
            compare(f"sphere_total_reaction_{name}_N", observed[f"sphere_total_reaction_{name}_N"], reaction[axis])
        compare("force_sample_time_s", observed["force_sample_time_s"], time)
        compare("base_boundary_sample_time_s", observed["base_boundary_sample_time_s"], time + config["dt"])

    with np.load(folder / "final_state_0.npz") as final:
        final_finite = all(bool(np.isfinite(final[key]).all()) for key in final.files)
        final_valid = bool(final["valid"] and final["history_valid"].all())
        final_position_matches = bool(np.array_equal(final["x"], trajectory[-1]))
        final_J = float(np.linalg.det(final["F"]).min())
        positive_weights = bool((final["mass"] > 0).all() and (final["volume"] > 0).all())
        step_number = float(np.max(law["penalty_stiffness"] * area * config["dt"]**2 / final["mass"][surface]))
    maximum_penetration = max(row["max_bulk_penetration_m"] for rows in recomputed for row in rows)
    penetration_limit = min(dx, law["interaction_range"]) * 0.25
    interior_excess = max(
        row["max_bulk_penetration_m"] - row["max_penetration_m"]
        for rows in recomputed for row in rows
    )
    force_scale = max(1.0, max(np.linalg.norm(row["reaction"]) for rows in recomputed for row in rows))
    clearance = path_clearance(paths, summary["simulation_time"])
    mass = np.array([row["mass"] for row in diagnostics])
    checks = {
        "initial_frame_at_zero_time": bool(np.isclose(times[0], 0.0, rtol=0, atol=1e-14)),
        "complete_output_frame_count": len(times) == int(status["frames"]) + 1,
        "completion_status_matches": bool(status.get("completed") and status["steps_completed"] == summary["steps"] and np.isclose(status["time"], summary["simulation_time"], rtol=1e-12)),
        "full_duration": bool(np.isclose(times[-1], summary["simulation_time"], rtol=1e-12, atol=1e-14)),
        "step_count_matches_duration": bool(np.isclose(
            summary["steps"] * config["dt"], times[-1], rtol=1e-12, atol=1e-14
        )),
        "all_prescribed_paths_completed": all(times[-1] >= path["times"][-1] - 1e-12 for path in paths),
        "uniform_output_times": bool(np.allclose(np.diff(times), times[-1] / status["frames"], rtol=1e-12, atol=1e-14)),
        "strictly_increasing_times": bool((np.diff(times) > 0).all()),
        "matching_table_times": bool(np.array_equal(times, [row["time"] for row in diagnostics])),
        "finite_tables": all(value is None or np.isfinite(value) for row in observations + diagnostics for value in row.values()),
        "finite_saved_positions_and_final_state": trajectories_finite and final_finite,
        "valid_final_state": final_valid,
        "final_positions_match_trajectory": final_position_matches,
        "positive_mass_volume_and_J": positive_weights and final_J > 0 and min(row["min_J"] for row in diagnostics) > 0,
        "constant_saved_mass": bool(np.allclose(mass, mass[0], rtol=1e-13, atol=0)),
        "reference_surface_matches_metadata": surface_y_matches and len(surface) == definition["surface_particles"],
        "cohesive_nominal_stress_matches_law": bool(np.isclose(
            definition["cohesive_nominal_stress_Pa"],
            law["work_of_adhesion"] / law["interaction_range"], rtol=1e-12, atol=1e-14
        )),
        "reference_area_covers_initial_top": bool(np.isclose(len(surface) * area, np.prod((np.asarray(definition["upper"]) - definition["lower"])[[0, 2]]), rtol=1e-12)),
        "independent_observables_match": all(comparisons.values()),
        "adhesive_range_covers_three_surface_spacings": law["interaction_range"] / max(spacing[[0, 2]]) >= 3,
        "local_penalty_step_number": step_number <= 0.09 * (1 + 1e-12),
        "limited_penalty_overlap": maximum_penetration <= penetration_limit,
        "no_deeper_unsampled_contact": interior_excess <= 0.05 * spacing[1],
        "prescribed_spheres_do_not_intersect": clearance is None or clearance > 0,
        "force_reciprocity": max(row["force_balance_error_N"] for row in observations) < 1e-9 * force_scale,
        "moment_reciprocity": max(row["moment_balance_error_Nm"] for row in observations) < 1e-9 * force_scale * config["domain_size"],
        "all_spheres_compressed_layer": all(any(row["compressed_points"] > 0 for row in rows) for rows in recomputed),
        "all_spheres_fully_separated_at_end": all(rows[-1]["min_gap_m"] > law["interaction_range"] + max(spacing) and np.linalg.norm(rows[-1]["reaction"]) == 0 for rows in recomputed),
    }
    if law["work_of_adhesion"] > 0:
        checks.update({
            "all_spheres_had_tensile_resultant": all(
                any(row["reaction"][1] < -1e-8 for row in rows) for rows in recomputed
            ),
            "all_spheres_had_tensile_resultant_during_retraction": all(
                any(row["reaction"][1] < -1e-8 and motion(path, time)[1][1] > 0
                    for time, row in zip(times, rows))
                for path, rows in zip(paths, recomputed)
            ),
        })
    else:
        checks.update({
            "no_adhesive_zone_without_adhesion": all(
                row["adhesive_points"] == 0 and row["adhesive_reference_area_m2"] == 0
                for rows in recomputed for row in rows
            ),
            "no_adhesive_potential_without_adhesion": all(
                row["adhesive_potential_J"] == 0 for rows in recomputed for row in rows
            ),
            "no_tensile_normal_resultant_without_adhesion": all(
                row["reaction"][1] >= -1e-8 for rows in recomputed for row in rows
            ),
        })
    spheres = []
    for index, rows in enumerate(recomputed):
        normal = np.array([row["reaction"][1] for row in rows])
        actual_reactions = np.array([row["reaction"] for row in rows])
        undeformed = np.asarray(undeformed_reactions[index])
        active = [i for i, row in enumerate(rows) if row["compressed_points"] + row["adhesive_points"]]
        retracting = np.array([motion(paths[index], time)[1][1] > 0 for time in times])
        spheres.append({
            "sphere": index,
            "peak_compressive_reaction_y_N": float(normal.max()),
            "peak_tensile_reaction_y_N": float(normal.min()),
            "peak_compressive_saved_time_s": float(times[normal.argmax()]),
            "minimum_reaction_saved_time_s": float(times[normal.argmin()]),
            "minimum_reaction_during_retraction_N": float(normal[retracting].min()) if retracting.any() else None,
            "peak_contact_reference_area_m2": max(row["contact_reference_area_m2"] for row in rows),
            "peak_adhesive_reference_area_m2": max(row["adhesive_reference_area_m2"] for row in rows),
            "maximum_penetration_m": max(row["max_bulk_penetration_m"] for row in rows),
            "last_active_saved_time_s": float(times[active[-1]]) if active else None,
            "final_minimum_gap_m": rows[-1]["min_gap_m"],
            "final_reaction_N": rows[-1]["reaction"].tolist(),
            "maximum_reaction_difference_from_undeformed_surface_N": float(
                np.linalg.norm(actual_reactions - undeformed, axis=1).max()
            ),
            "sampled_time_s": times.tolist(),
            "reaction_y_N": normal.tolist(),
            "undeformed_surface_reaction_y_N": undeformed[:, 1].tolist(),
        })
    return {
        "case": summary["case"],
        "scope": "Independent NumPy reconstruction from saved particle positions and prescribed sphere paths; sampled-frame checks, no mesh-convergence or experimental-accuracy claim.",
        "sources": {name: hashlib.sha256((folder / name).read_text(encoding="utf-8").encode()).hexdigest() for name in ("summary.json", "observables.csv", "diagnostics.csv")},
        "checks": {name: bool(value) for name, value in checks.items()},
        "passed": all(bool(value) for value in checks.values()),
        "saved_frames": len(times),
        "expected_saved_frames": int(status["frames"]) + 1,
        "surface_points": len(surface),
        "work_of_adhesion_J_per_m2": law["work_of_adhesion"],
        "surface_displacement_y_min_m": float(np.min(surface_displacement)),
        "surface_displacement_y_max_m": float(np.max(surface_displacement)),
        "min_J": min(row["min_J"] for row in diagnostics),
        "maximum_relative_mass_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "adhesive_range_per_surface_spacing": float(law["interaction_range"] / max(spacing[[0, 2]])),
        "adhesive_range_per_grid_cell": float(law["interaction_range"] / dx),
        "maximum_penalty_step_number": step_number,
        "maximum_penetration_m": maximum_penetration,
        "penetration_review_limit_m": penetration_limit,
        "maximum_unsampled_interior_penetration_excess_m": interior_excess,
        "rigid_sphere_clearance_lower_bound_m": clearance,
        "maximum_observable_reconstruction_error": maximum_errors,
        "spheres": spheres,
        "interpretation": "Finite penalty overlap and reversible reference-area adhesion. The finite layer is dynamic; no analytical Maugis solution, irreversible bond fracture, or tangential friction is inferred.",
        "undeformed_surface_comparison": "Diagnostic forces from the same tool path and interface law evaluated on the initial surface, without bulk deformation. They are not a JKR reference or a second numerical simulation.",
    }


def review(folder):
    folder = Path(folder)
    trajectory = np.load(folder / "trajectory_x.npy", mmap_mode="r")
    try:
        return _review(folder, trajectory)
    finally:
        trajectory._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=Path("outputs/adhesion/dugdale-array"))
    args = parser.parse_args()
    result = review(args.folder)
    (args.folder / "interpretation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failed = [name for name, value in result["checks"].items() if not value]
    print(json.dumps({"case": result["case"], "passed": result["passed"], "failed_checks": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
