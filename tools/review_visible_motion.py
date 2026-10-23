"""Review finite-deformation examples from saved coordinates and diagnostics."""

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


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def cycle(time, definition):
    phase = np.clip((time - definition.get("start", 0.0)) / definition["period"], 0, 1)
    return 0.5 * (1 - np.cos(2 * np.pi * phase))


def translation(time, definition):
    if "period" in definition:
        return np.asarray(definition["displacement"]) * cycle(time, definition)
    elapsed = max(0.0, time - definition.get("start", 0.0))
    if definition.get("end") is not None:
        elapsed = min(elapsed, definition["end"] - definition.get("start", 0.0))
    ramp = definition.get("ramp_time", 0.0)
    ramp_elapsed = min(elapsed, ramp)
    distance = (
        0.5 * (ramp_elapsed - ramp / np.pi * np.sin(np.pi * ramp_elapsed / ramp))
        + max(0, elapsed - ramp)
        if ramp else elapsed
    )
    return np.asarray(definition["velocity"]) * distance


def pressure(time, definition):
    rise = definition["rise_s"]
    fall_start = rise + definition["hold_s"]
    if time < rise:
        factor = 0.5 * (1 - np.cos(np.pi * max(0, time) / rise))
    elif time <= fall_start:
        factor = 1.0
    elif time < fall_start + definition["fall_s"]:
        factor = 0.5 * (1 + np.cos(np.pi * (time - fall_start) / definition["fall_s"]))
    else:
        factor = 0.0
    return definition["peak_Pa"] * factor


def review(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError("The numerical run is incomplete")
    definition, config = summary["definition"], summary["configs"][0]
    if definition["series"] != "visible-motion-v1":
        raise ValueError("Unexpected loading-series definition")
    scene = definition["purpose"].removeprefix("visible-")
    if scene not in ("wave", "cantilever", "surface-load", "shear", "torsion"):
        raise ValueError("Unknown finite-deformation loading case")
    rows = read_rows(folder / "observables.csv")
    diagnostics = read_rows(folder / "diagnostics.csv")
    times = np.array([row["time"] for row in rows])
    positions = np.load(folder / "trajectory_x.npy", mmap_mode="r")
    speeds = np.load(folder / "trajectory_speed.npy", mmap_mode="r")
    if positions.shape != (len(times), sum(summary["particles"]), 3):
        raise ValueError("Saved-coordinate shape does not match the run")
    if len(diagnostics) != len(times) or speeds.shape != positions.shape[:2]:
        raise ValueError("Saved fields and diagnostic tables have different shapes")
    reference = positions[0]
    dx = config["domain_size"] / config["n_grid"]
    nodes_lower = np.floor(reference / dx - 0.5).astype(int)
    nodes_upper = nodes_lower + 2
    edge, axis = (0.175, 0) if scene == "wave" else (0.28, 1) if scene == "shear" else (0.2, 0)
    anchor = nodes_upper[:, axis] <= int(np.floor(edge / dx))
    if scene == "surface-load":
        anchor = np.any(
            (nodes_upper[:, (0, 2)] <= int(np.floor(0.25 / dx)))
            | (nodes_lower[:, (0, 2)] >= int(np.ceil(0.75 / dx))), axis=1
        )
    tip = (
        reference[:, 0] > definition["tip_reference_x_min_m"]
        if scene in ("wave", "cantilever", "torsion") else np.zeros(len(reference), dtype=bool)
    )
    top = reference[:, 1] > 0.72
    center = (reference[:, 0] - 0.5)**2 + (reference[:, 2] - 0.5)**2 < 0.06**2
    checks, errors, matches = {}, {}, {}
    response, anchor_motion, anchor_speed, constrained_motion = [], [], [], []
    finite = True

    def compare(name, actual, expected, atol=1e-10):
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        errors[name] = max(errors.get(name, 0.0), error)
        matches[name] = matches.get(name, True) and bool(np.allclose(actual, expected, rtol=1e-9, atol=atol))

    for index, (time, row, diagnostic) in enumerate(zip(times, rows, diagnostics)):
        current = positions[index]
        change = current - reference
        magnitude = np.linalg.norm(change, axis=1)
        finite &= bool(np.isfinite(current).all() and np.isfinite(speeds[index]).all())
        compare("maximum_displacement_m", row["maximum_displacement_m"], magnitude.max())
        compare("rms_displacement_m", row["rms_displacement_m"], np.sqrt(np.mean(magnitude**2)))
        compare("sample_min_J", row["sample_min_J"], diagnostic["min_J"])
        compare("sample_max_J", row["sample_max_J"], diagnostic["max_J"])
        if anchor.any():
            anchor_motion.append(float(magnitude[anchor].max()))
            anchor_speed.append(float(speeds[index, anchor].max()))
            compare("anchor_max_displacement_m", row["anchor_max_displacement_m"], anchor_motion[-1])
            compare("anchor_max_speed_m_per_s", row["anchor_max_speed_m_per_s"], anchor_speed[-1])
        if scene in ("wave", "cantilever", "torsion"):
            for component, name in enumerate("xyz"):
                compare(f"tip_mean_displacement_{name}_m", row[f"tip_mean_displacement_{name}_m"], change[tip, component].mean())
            compare("tip_max_displacement_m", row["tip_max_displacement_m"], magnitude[tip].max())
        if scene == "wave":
            response.append(float(change[tip, 1].mean()))
            constrained_motion.append(float(np.abs(change[:, (0, 2)]).max()))
            compare("constrained_xz_max_displacement_m", row["constrained_xz_max_displacement_m"], constrained_motion[-1])
        elif scene == "shear":
            response.append(float(change[top, 0].mean()))
            target = translation(time, definition["driven_boundary"])[0]
            height = definition["geometry"]["upper"][1] - definition["geometry"]["lower"][1]
            compare("target_top_displacement_x_m", row["target_top_displacement_x_m"], target)
            compare("top_mean_displacement_x_m", row["top_mean_displacement_x_m"], response[-1])
            compare("nominal_engineering_shear", row["nominal_engineering_shear"], target / height)
        elif scene == "cantilever":
            response.append(float(change[tip, 1].mean()))
            target = translation(time, definition["driven_boundary"])[1]
            compare("target_grip_displacement_y_m", row["target_grip_displacement_y_m"], target)
            compare("tip_displacement_tracking_error_y_m", row["tip_displacement_tracking_error_y_m"], response[-1] - target)
        elif scene == "torsion":
            initial = reference[tip, 1:3] - [0.5, 0.5]
            section = current[tip, 1:3] - [0.5, 0.5]
            angle = float(np.arctan2(
                (initial[:, 0] * section[:, 1] - initial[:, 1] * section[:, 0]).sum(),
                (initial * section).sum(),
            ))
            response.append(angle)
            target = definition["driven_boundary"]["angle"] * cycle(time, definition["driven_boundary"])
            compare("target_tip_rotation_x_rad", row["target_tip_rotation_x_rad"], target)
            compare("tip_rotation_x_rad", row["tip_rotation_x_rad"], angle)
            compare("tip_rotation_tracking_error_rad", row["tip_rotation_tracking_error_rad"], angle - target)
        else:
            response.append(float(change[center, 1].mean()))
            compare("center_mean_displacement_y_m", row["center_mean_displacement_y_m"], response[-1])
            compare("center_min_displacement_y_m", row["center_min_displacement_y_m"], change[center, 1].min())
            pressure_definition = definition["surface_pressure"]
            imposed_pressure = pressure(time, pressure_definition)
            compare("surface_pressure_Pa", row["surface_pressure_Pa"], imposed_pressure)
            compare("prescribed_surface_force_y_N", row["prescribed_surface_force_y_N"], -imposed_pressure * pressure_definition["area_m2"])
            compare("surface_force_assignment_error_N", row["surface_force_assignment_error_N"], 0.0, atol=1e-9)

    with np.load(folder / "final_state_0.npz") as final:
        final_finite = all(bool(np.isfinite(final[key]).all()) for key in final.files)
        final_valid = bool(final["valid"] and final["history_valid"].all())
        final_matches = bool(np.array_equal(final["x"], positions[-1]))
        positive_weights = bool((final["mass"] > 0).all() and (final["volume"] > 0).all())
        final_positive_J = bool((np.linalg.det(final["F"]) > 0).all())
        if scene == "wave":
            checks["final_transverse_guides_hold"] = bool(np.abs(final["v"][:, (0, 2)]).max() < 1e-10)
    positions._mmap.close()
    speeds._mmap.close()
    response = np.asarray(response)
    mass = np.array([row["mass"] for row in diagnostics])
    checks.update({
        "initial_frame_at_zero_time": bool(np.isclose(times[0], 0, rtol=0, atol=1e-14)),
        "complete_output_frame_count": len(times) == status["frames"] + 1,
        "completion_status_matches": bool(status.get("completed") and status["steps_completed"] == summary["steps"]),
        "full_duration": bool(np.isclose(times[-1], summary["simulation_time"], rtol=1e-12, atol=1e-14) and np.isclose(status["time"], times[-1], rtol=1e-12)),
        "time_step_count_matches_duration": bool(np.isclose(summary["steps"] * config["dt"], times[-1], rtol=1e-12, atol=1e-14)),
        "matching_table_times": bool(np.array_equal(times, [row["time"] for row in diagnostics])),
        "uniform_increasing_times": bool((np.diff(times) > 0).all() and np.allclose(np.diff(times), times[-1] / status["frames"], rtol=1e-12, atol=1e-14)),
        "finite_tables": all(value is None or np.isfinite(value) for row in rows + diagnostics for value in row.values()),
        "finite_saved_arrays": finite and final_finite,
        "valid_final_state": final_valid,
        "final_positions_match_saved_trajectory": final_matches,
        "positive_mass_volume_and_J": positive_weights and final_positive_J and min(row["min_J"] for row in diagnostics) > 0,
        "constant_saved_mass": bool(np.allclose(mass, mass[0], rtol=1e-13, atol=0)),
        "sampled_anchor_exists": bool(anchor.any()),
        "sampled_anchor_matches_definition": int(anchor.sum()) == definition["anchor_particles_with_fully_constrained_support"],
        "sampled_anchor_stays_fixed": bool(anchor_motion and max(anchor_motion) < 1e-9),
        "sampled_anchor_has_zero_velocity": bool(anchor_speed and max(anchor_speed) < 1e-9),
        "independent_observables_match": all(matches.values()),
        "geometry_is_unscaled": definition["geometry_scale"] == 1.0,
    })
    if scene == "wave":
        checks["visible_positive_and_negative_tip_motion"] = bool(response.max() >= 0.05 and response.min() <= -0.04)
        checks["transverse_guides_hold_all_saved_positions"] = max(constrained_motion) < 1e-10
        checks["reported_transverse_guide_velocities_zero"] = max(row["constrained_xz_max_speed_m_per_s"] for row in rows) < 1e-10
        checks["complete_nominal_wave_period"] = bool(np.isclose(times[-1], 4 * definition["free_span_m"] / definition["initial_shear_speed_m_per_s"], rtol=1e-12))
    elif scene == "shear":
        checks["visible_top_shear_displacement"] = bool(response.max() >= 0.12)
        checks["top_follows_prescribed_translation"] = max(abs(row["top_mean_displacement_x_m"] - row["target_top_displacement_x_m"]) for row in rows) <= 0.015
    elif scene == "cantilever":
        checks["visible_tip_bending"] = bool(response.max() >= 0.08)
        checks["tip_follows_prescribed_cycle"] = max(abs(row["tip_displacement_tracking_error_y_m"]) for row in rows) <= 0.015
        checks["complete_loading_and_return_cycle"] = bool(times[-1] >= definition["driven_boundary"]["start"] + definition["driven_boundary"]["period"] and abs(rows[-1]["target_grip_displacement_y_m"]) < 1e-14)
    elif scene == "torsion":
        checks["visible_tip_rotation"] = bool(response.max() >= 1.0)
        checks["tip_follows_prescribed_rotation"] = max(abs(row["tip_rotation_tracking_error_rad"]) for row in rows) <= 0.12
        checks["complete_loading_and_return_cycle"] = bool(times[-1] >= definition["driven_boundary"]["start"] + definition["driven_boundary"]["period"] and abs(rows[-1]["target_tip_rotation_x_rad"]) < 1e-14)
    else:
        checks["visible_plate_deflection"] = bool(response.min() <= -0.02)
        checks["complete_pressure_unloading"] = rows[-1]["surface_pressure_Pa"] == 0
    return {
        "case": summary["case"],
        "scope": "Independent NumPy reconstruction of saved positions, imposed loading and response observables; sampled-frame checks, not experimental validation or an inherited axial-wave accuracy result.",
        "sources": {name: digest(folder / name) for name in ("summary.json", "observables.csv", "diagnostics.csv")},
        "checks": {name: bool(value) for name, value in checks.items()},
        "passed": all(bool(value) for value in checks.values()),
        "saved_frames": len(times),
        "min_J": min(row["min_J"] for row in diagnostics),
        "maximum_relative_mass_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "peak_displacement_m": max(row["maximum_displacement_m"] for row in rows),
        "response": {
            "quantity": "tip rotation about x" if scene == "torsion" else "plate center y displacement" if scene == "surface-load" else "top x displacement" if scene == "shear" else "tip y displacement",
            "units": "rad" if scene == "torsion" else "m",
            "minimum": float(response.min()), "minimum_time_s": float(times[response.argmin()]),
            "maximum": float(response.max()), "maximum_time_s": float(times[response.argmax()]),
            "final": float(response[-1]),
        },
        "maximum_observable_reconstruction_error": errors,
        "interpretation": "Displacements and rotations come from the numerical trajectory at physical scale. Visibility thresholds were selected before the production runs.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = review(args.folder)
    (args.folder / "interpretation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    failed = [name for name, value in result["checks"].items() if not value]
    print(json.dumps({"case": result["case"], "passed": result["passed"], "failed_checks": failed}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
