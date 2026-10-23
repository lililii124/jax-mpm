"""Compare matched MPM indentation runs with and without surface adhesion."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE_FILES = (
    "summary.json", "observables.csv", "diagnostics.csv", "interpretation.json", "status.json"
)
CASES = {"nonadhesive": "nonadhesive-pull-off", "adhesive": "adhesive-pull-off"}


def source_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def read_run(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    interpretation = json.loads((folder / "interpretation.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if summary.get("completed") is not True or status.get("completed") is not True:
        raise ValueError(f"Incomplete run: {folder}")
    checks = interpretation.get("checks", {})
    if interpretation.get("passed") is not True or not checks or not all(value is True for value in checks.values()):
        raise ValueError(f"Independent review did not pass: {folder}")
    hashes = {name: source_hash(folder / name) for name in SOURCE_FILES}
    for name in SOURCE_FILES[:3]:
        if interpretation.get("sources", {}).get(name) != hashes[name]:
            raise ValueError(f"Stale independent review: {folder / name}")
    with (folder / "observables.csv").open(encoding="utf-8", newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows or not all(np.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Missing or nonfinite observations: {folder}")
    return {
        "folder": folder, "summary": summary, "review": interpretation,
        "status": status, "rows": rows, "hashes": hashes,
    }


def common_definition(definition):
    common = copy.deepcopy(definition)
    for field in ("purpose", "suite_case", "comparison_role", "cohesive_nominal_stress_Pa"):
        common.pop(field, None)
    common["adhesion"].pop("work_of_adhesion")
    return common


def curve(run, field):
    return np.array([row[field] for row in run["rows"]])


def lift_hold_profile(definitions):
    profiles = {
        "single-sphere-lift-hold-v1": {"preset": "lift-hold", "clearance": .055, "revision": None, "work": 72.0},
        "single-sphere-lift-hold-v3": {"preset": "lift-hold-v3", "clearance": .035, "revision": 3, "work": 72.0},
        "single-sphere-lift-hold-v4": {"preset": "lift-hold-v4", "clearance": .035, "revision": 4, "work": 144.0},
    }
    named = any("lift-hold" in str(item.get(key, "")) for item in definitions for key in ("preset", "comparison_pair"))
    if not named:
        return None
    identifier = definitions[0].get("comparison_pair")
    if identifier not in profiles or any(item.get("comparison_pair") != identifier for item in definitions):
        raise ValueError("Unknown or mismatched lift-hold comparison profile")
    return profiles[identifier]


def lift_hold_path_matches(definition, profile):
    try:
        path = definition["rigid_spheres"][0]
        centers, times = np.asarray(path["centers"]), np.asarray(path["times"])
        target_times = [0., .08, .28, .42, .72, .94, 1.12, 1.3]
        heights = [.045, .045, -.035, -.035, profile["clearance"], profile["clearance"], .295, .295]
        if centers.shape != (8, 3) or times.shape != (8,):
            return False
        return bool(
            path["radius"] == .16
            and np.allclose(times, target_times, rtol=0, atol=1e-12)
            and np.allclose(centers[:, (0, 2)], .5, rtol=0, atol=1e-12)
            and np.allclose(centers[:, 1] - path["radius"] - definition["contact_sampling_plane_y_m"], heights, rtol=0, atol=1e-12)
        )
    except (KeyError, ValueError, TypeError):
        return False


def review(root):
    root = Path(root)
    runs = {role: read_run(root / name) for role, name in CASES.items()}
    base, adhesive = runs["nonadhesive"], runs["adhesive"]
    summaries = [run["summary"] for run in (base, adhesive)]
    definitions = [summary["definition"] for summary in summaries]
    d0, d1 = definitions
    checks = {
        "expected_suite_and_cases": all(
            run["summary"]["definition"].get("series") == "adhesion-suite-v1"
            and run["summary"]["definition"].get("suite_case") == CASES[role]
            and run["summary"]["definition"].get("comparison_role") == role
            for role, run in runs.items()
        ),
        "same_nonempty_pair_identifier": bool(d0.get("comparison_pair")) and d0.get("comparison_pair") == d1.get("comparison_pair"),
        "same_geometry_loading_and_interface_except_work": common_definition(d0) == common_definition(d1),
        "one_sphere_per_run": all(len(definition["rigid_spheres"]) == 1 for definition in definitions),
        "zero_and_positive_adhesion": d0["adhesion"]["work_of_adhesion"] == 0 and d1["adhesion"]["work_of_adhesion"] > 0,
        "true_geometry_scale": all(definition.get("geometry_scale") == 1.0 for definition in definitions),
    }
    for field in ("configs", "materials", "steps", "simulation_time", "particles", "dtype", "jax_version", "initial"):
        checks["same_" + field] = summaries[0][field] == summaries[1][field]

    time = curve(base, "time")
    other_time = curve(adhesive, "time")
    checks["same_saved_times_without_interpolation"] = np.array_equal(time, other_time)
    checks["saved_times_cover_full_paths"] = all(
        len(run["rows"]) == run["status"]["frames"] + 1
        and run["status"]["frame"] == run["status"]["frames"]
        and run["status"]["steps_completed"] == run["summary"]["steps"]
        and np.isclose(run["status"]["time"], run["summary"]["simulation_time"], rtol=1e-12, atol=1e-14)
        and np.isclose(run["rows"][-1]["time"], run["summary"]["simulation_time"], rtol=1e-12, atol=1e-14)
        and run["rows"][-1]["time"] >= run["summary"]["definition"]["rigid_spheres"][0]["times"][-1] - 1e-12
        for run in runs.values()
    )
    trajectories = []
    try:
        for run in (base, adhesive):
            trajectories.append(np.load(run["folder"] / "trajectory_x.npy", mmap_mode="r"))
        checks["same_saved_shape_and_dtype"] = trajectories[0].shape == trajectories[1].shape and trajectories[0].dtype == trajectories[1].dtype
        checks["identical_initial_particle_positions"] = np.array_equal(trajectories[0][0], trajectories[1][0])
        initial_hashes = [hashlib.sha256(np.ascontiguousarray(trajectory[0]).tobytes()).hexdigest() for trajectory in trajectories]
    finally:
        for trajectory in trajectories:
            trajectory._mmap.close()

    prefix = "sphere_0_"
    if checks["same_saved_times_without_interpolation"]:
        checks["identical_sampled_sphere_motion"] = all(
            np.array_equal(curve(base, prefix + field), curve(adhesive, prefix + field))
            for field in ("center_x_m", "center_y_m", "center_z_m", "velocity_x_m_per_s", "velocity_y_m_per_s", "velocity_z_m_per_s")
        )
    else:
        checks["identical_sampled_sphere_motion"] = False
    if not all(checks.values()):
        raise ValueError("The runs are not a controlled pair: " + ", ".join(name for name, passed in checks.items() if not passed))

    metrics = {}
    for role, run in runs.items():
        reaction = curve(run, prefix + "reaction_y_N")
        retracting = curve(run, prefix + "velocity_y_m_per_s") > 0
        active = (curve(run, prefix + "compressed_points") + curve(run, prefix + "adhesive_points")) > 0
        active_indices = np.flatnonzero(active)
        last = int(active_indices[-1]) if len(active_indices) else None
        first_separated = last + 1 if last is not None and last + 1 < len(time) else None
        power = curve(run, prefix + "actuator_power_W")
        metrics[role] = {
            "work_of_adhesion_J_per_m2": run["summary"]["definition"]["adhesion"]["work_of_adhesion"],
            "peak_compressive_reaction_y_N": float(reaction.max()),
            "minimum_reaction_y_N": float(reaction.min()),
            "minimum_retraction_reaction_y_N": float(reaction[retracting].min()) if retracting.any() else None,
            "tensile_retraction_saved_frames": int(((reaction < -1e-8) & retracting).sum()),
            "peak_contact_reference_area_m2": float(curve(run, prefix + "contact_reference_area_m2").max()),
            "peak_adhesive_reference_area_m2": float(curve(run, prefix + "adhesive_reference_area_m2").max()),
            "surface_displacement_y_min_m": float(curve(run, "surface_displacement_y_min_m").min()),
            "surface_displacement_y_max_m": float(curve(run, "surface_displacement_y_max_m").max()),
            "last_active_saved_time_s": float(time[last]) if last is not None else None,
            "first_permanently_separated_saved_time_s": float(time[first_separated]) if first_separated is not None else None,
            "final_minimum_gap_m": float(curve(run, prefix + "min_gap_m")[-1]),
            "sampled_actuator_work_trapezoidal_J": float(np.sum((power[1:] + power[:-1]) * np.diff(time) / 2)),
            "maximum_force_difference_from_undeformed_surface_N": float(curve(run, prefix + "deformation_feedback_force_difference_N").max()),
        }

    table = []
    quantities = (
        "reaction_y_N", "contact_reference_area_m2", "adhesive_reference_area_m2",
        "min_gap_m", "potential_J", "deformation_feedback_force_difference_N",
    )
    for index, instant in enumerate(time):
        row = {"time_s": float(instant), "sphere_center_y_m": base["rows"][index][prefix + "center_y_m"]}
        for role, run in runs.items():
            for quantity in quantities:
                row[role + "_" + quantity] = run["rows"][index][prefix + quantity]
            for quantity in ("surface_displacement_y_min_m", "surface_displacement_y_max_m"):
                row[role + "_" + quantity] = run["rows"][index][quantity]
        row["adhesive_minus_nonadhesive_reaction_y_N"] = row["adhesive_reaction_y_N"] - row["nonadhesive_reaction_y_N"]
        table.append(row)

    contrast = None
    profile = lift_hold_profile(definitions)
    if profile is not None:
        base_active = curve(base, prefix + "compressed_points") + curve(base, prefix + "adhesive_points")
        adhesion_active = curve(adhesive, prefix + "adhesive_points")
        reaction0 = curve(base, prefix + "reaction_y_N")
        reaction1 = curve(adhesive, prefix + "reaction_y_N")
        uplift_difference = curve(adhesive, "surface_displacement_y_max_m") - curve(base, "surface_displacement_y_max_m")
        holding = (time >= 0.72 - 1e-12) & (time <= 0.94 + 1e-12)
        checks["lift_hold_preset_identified"] = d0.get("preset") == d1.get("preset") == profile["preset"]
        checks["lift_hold_fixed_work_of_adhesion"] = (
            d0["adhesion"]["work_of_adhesion"] == 0.0
            and d1["adhesion"]["work_of_adhesion"] == profile["work"]
        )
        if profile["revision"] is not None:
            checks["lift_hold_loading_revision_identified"] = all(type(item.get("loading_revision")) is int and item["loading_revision"] == profile["revision"] and item.get("comparison_hold_bottom_clearance_m") == profile["clearance"] for item in definitions)
        checks["lift_hold_fixed_physical_path"] = all(lift_hold_path_matches(item, profile) for item in definitions)
        checks["lift_hold_sphere_stationary"] = holding.any() and all(
            bool(np.max(np.abs(curve(run, prefix + "velocity_" + axis + "_m_per_s")[holding])) < 1e-12)
            and bool(np.ptp(curve(run, prefix + "center_" + axis + "_m")[holding]) < 1e-12)
            for run in (base, adhesive) for axis in "xyz"
        )
        visible = holding & (base_active == 0) & (np.abs(reaction0) < 1e-8) & (adhesion_active > 0) & (reaction1 < -1e-8) & (uplift_difference >= 0.025)
        intervals, start = [], None
        for index, qualifying in enumerate(visible):
            if qualifying and start is None:
                start = index
            if start is not None and (not qualifying or index == len(visible) - 1):
                last = index if qualifying else index - 1
                intervals.append({"first_saved_time_s": float(time[start]), "last_saved_time_s": float(time[last]),
                                  "saved_span_s": float(time[last] - time[start]), "saved_samples": last - start + 1})
                start = None
        duration = max((interval["saved_span_s"] for interval in intervals), default=0.0)
        checks["lift_hold_has_visible_deformation_contrast"] = duration >= 0.10
        contrast = {
            "preset": profile["preset"], "hold_bottom_clearance_m": profile["clearance"],
            "hold_window_s": [0.72, 0.94], "minimum_uplift_difference_m": 0.025,
            "required_saved_span_s": 0.10, "longest_qualifying_saved_span_s": duration,
            "qualifying_saved_intervals": intervals,
            "saved_adhesive_minus_nonadhesive_surface_uplift_m": uplift_difference.tolist(),
            "maximum_hold_uplift_difference_m": float(uplift_difference[holding].max()) if holding.any() else None,
            "criterion": "At consecutive saved states in the lift hold, the nonadhesive sphere is force-free with no interacting points, while the adhesive sphere has tensile reaction and at least 25 mm greater surface uplift.",
            "sampling_scope": "The interval spans qualifying output samples; no unsaved-time or quasi-static-equilibrium claim.",
        }

    return {
        "passed": all(bool(value) for value in checks.values()),
        "comparison": "matched dynamic indentation and complete withdrawal",
        "checks": {name: bool(value) for name, value in checks.items()},
        "sources": {
            role: {"source_directory": run["folder"].as_posix(), "files": run["hashes"],
                   "initial_particle_position_bytes_sha256": initial_hashes[index]}
            for index, (role, run) in enumerate(runs.items())
        },
        "metrics": metrics,
        **({"deformation_contrast": contrast} if contrast is not None else {}),
        "sampled_time_s": time.tolist(),
        "interpretation": (
            "Both runs use the same dynamic MPM body, imposed sphere trajectory, penalty stiffness and time step; only work of adhesion changes. "
            "The reaction is computed from the deformed surface. The finite layer and resolved interaction range are not an analytical JKR or Maugis benchmark. "
            "Release times are saved-frame brackets. Sampled actuator work includes dynamic loading and unloading; it is not irreversible adhesion energy or a global energy-conservation test."
        ),
    }, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("outputs/adhesion"))
    args = parser.parse_args()
    result, rows = review(args.root)
    destination = args.root / "adhesion_comparison.json"
    csv_path = args.root / "adhesion_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result["csv"] = {"filename": csv_path.name, "lf_sha256": source_hash(csv_path), "rows": len(rows)}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "report": destination.as_posix()}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
