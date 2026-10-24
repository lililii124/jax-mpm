"""Check saved JH-2 failure, debris motion and complete output independently."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def review(folder, *, allow_prefix=False):
    folder = Path(folder)
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((folder / "checkpoint.json").read_text(encoding="utf-8"))
    frame, total = int(status["frame"]), int(status["frames"])
    complete = status["completed"] is True
    rows = checkpoint["observations"]
    saved = {name: np.load(folder / f"trajectory_{name}.npy", mmap_mode="r")
             for name in ("x", "damage", "failed")}
    with np.load(folder / checkpoint["states"][0]) as state:
        mass = np.asarray(state["mass"])
        volume = np.asarray(state["volume"])
        final_x = np.asarray(state["x"])
        final_failed = np.asarray(state["history_failed"])
        final_damage = np.asarray(state["history_damage"])
        final_plastic_strain = np.asarray(state["history_plastic_strain"])
        final_time = float(state["time"])
        final_valid = bool(state["valid"]) and bool(np.asarray(state["history_valid"]).all())
        min_j = float(np.linalg.det(state["F"]).min())
    particles = len(mass)
    substeps = status["steps_completed"] // frame if frame else 0
    nominal_dt = 3e-4 / (total * substeps) if substeps else np.nan
    clock_tolerance = 32 * np.finfo(np.float64).eps * max(status["steps_completed"], 1) * 3e-4
    checks = {
        "planned_full_frame_count": total == 60,
        "original_scale": particles == 386420,
        "valid_segment_or_complete": complete or (allow_prefix and status.get("phase") == "checkpoint-ready" and status.get("segment_complete") is True),
        "saved_frame_matches_checkpoint": checkpoint["frame"] == frame,
        "complete_only_at_final_frame": complete == (frame == total),
        "full_plan_not_shortened": (complete or
                                    np.isclose(status["target_time"], 3e-4,
                                                   rtol=0, atol=1e-14)),
        "checkpoint_signature_bound": (complete or
                                       checkpoint["signature"] == status["signature"]),
        "time_and_step_count": (len(rows) == frame + 1 and frame > 0
                                and status["steps_completed"] % frame == 0
                                and np.isclose(rows[-1]["time"], status["time"], rtol=1e-12, atol=1e-14)
                                and np.isclose(status["time"], 3e-4 * frame / total,
                                               rtol=1e-12, atol=1e-14)
                                and (complete or status["target_steps"] == total * substeps)
                                and all(np.isclose(row["time"], index * substeps * nominal_dt,
                                                    rtol=0, atol=clock_tolerance)
                                        for index, row in enumerate(rows))
                                and abs(final_time - status["time"]) <= clock_tolerance),
        "saved_array_shapes": (saved["x"].shape == (total + 1, particles, 3)
                               and saved["damage"].shape == (total + 1, particles)
                               and saved["failed"].shape == (total + 1, particles)),
        "valid_final_state": final_valid and min_j > 0,
        "failure_matches_plastic_threshold": bool(np.array_equal(
            final_failed, final_plastic_strain >= .05)),
        "positive_constant_mass": bool(np.isfinite(mass).all() and np.isfinite(volume).all()
                                       and np.all(mass > 0) and np.all(volume > 0)
                                       and np.allclose(mass, 2500. * volume, rtol=1e-13, atol=0)
                                       and np.isclose(volume.sum(), .18 * .025 * .18,
                                                      rtol=1e-12, atol=0)),
        "saved_final_fields_match_checkpoint": bool(
            np.array_equal(saved["x"][frame], final_x)
            and np.array_equal(saved["damage"][frame], final_damage)
            and np.array_equal(saved["failed"][frame], final_failed)),
        "initially_intact": bool(np.count_nonzero(saved["failed"][0]) == 0
                                 and np.count_nonzero(saved["damage"][0]) == 0),
    }
    failed_fractions, damaged_fractions = [], []
    free_flight_count, free_flight_error = 0, 0.0
    maximum_active_sphere_overlap = 0.0
    for index in range(frame + 1):
        x, damage, failed = (saved[name][index] for name in ("x", "damage", "failed"))
        failed_fractions.append(float(volume[failed.astype(bool)].sum() / volume.sum()))
        damaged_fractions.append(float(volume[damage >= .5].sum() / volume.sum()))
        if not (np.isfinite(x).all() and np.isfinite(damage).all()
                and np.all((0 <= damage) & (damage <= 1))
                and np.all((failed == 0) | (failed == 1))):
            checks["finite_bounded_saved_fields"] = False
        if index and (np.any(failed < saved["failed"][index - 1])
                      or np.any(damage < saved["damage"][index - 1] - 1e-12)):
            checks["irreversible_damage_and_failure"] = False
        if (not np.isclose(rows[index]["failed_reference_volume_m3"],
                           volume[failed.astype(bool)].sum(), rtol=1e-10, atol=1e-15)
                or rows[index]["failed_particle_count"] != int(failed.sum())):
            checks["observer_matches_saved_failure"] = False
        phase = index / total
        stroke = .006 * (1 - np.cos(2 * np.pi * phase))
        center = np.array((.125, .1375 - stroke, .125))
        active = ~failed.astype(bool)
        if active.any():
            maximum_active_sphere_overlap = max(
                maximum_active_sphere_overlap,
                float(np.maximum(.0375 - np.linalg.norm(x[active] - center, axis=1), 0.).max()),
            )
        if not np.isclose(rows[index]["nominal_indentation_m"], stroke,
                          rtol=1e-10, atol=1e-12):
            checks["prescribed_sphere_motion_matches_observer"] = False
        if index >= 2:
            prior_failed = saved["failed"][index - 2].astype(bool)
            if prior_failed.any():
                second = x[prior_failed] - saved["x"][index - 1, prior_failed]
                first = (saved["x"][index - 1, prior_failed]
                         - saved["x"][index - 2, prior_failed])
                free_flight_count += int(prior_failed.sum())
                free_flight_error = max(free_flight_error, float(np.max(np.abs(second - first))))
    checks.setdefault("finite_bounded_saved_fields", True)
    checks.setdefault("irreversible_damage_and_failure", True)
    checks.setdefault("observer_matches_saved_failure", True)
    checks.setdefault("prescribed_sphere_motion_matches_observer", True)
    initial_y = saved["x"][0, :, 1]
    spacing = .00125
    escaped = final_failed & ((final_x[:, 1] < initial_y.min() - spacing)
                              | (final_x[:, 1] > initial_y.max() + spacing))
    report = {
        "schema": "ceramic-fragmentation-review-v1",
        "completed": complete,
        "accepted_saved_frame": frame,
        "target_saved_frame": total,
        "particles": particles,
        "failed_reference_volume_fraction_history": failed_fractions,
        "damaged_reference_volume_fraction_history": damaged_fractions,
        "failed_particles_outside_initial_plate_thickness_by_one_spacing": int(escaped.sum()),
        "failed_particle_free_flight_triplets": free_flight_count,
        "maximum_saved_free_flight_increment_error_m": free_flight_error,
        "maximum_active_particle_sphere_overlap_m": maximum_active_sphere_overlap,
        "minimum_final_J": min_j,
        "checks": checks,
        "sources": {name: digest(folder / name) for name in (
            "status.json", "checkpoint.json", checkpoint["states"][0],
            "trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy")},
        "scope": "Ballistic failed points are noncontacting debris; no resolved cohesive chunks or fragment contact.",
    }
    if complete:
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        config = summary["configs"][0]
        definition = summary["definition"]
        material = summary["materials"][0]
        expected_clamps = (
            ((0., 0., 0.), (.043, .25, .25)),
            ((.207, 0., 0.), (.25, .25, .25)),
            ((0., 0., 0.), (.25, .25, .043)),
            ((0., 0., .207), (.25, .25, .25)),
        )
        clamps_match = len(config["boundaries"]) == 5 and all(
            np.allclose(boundary["lower"], lower, rtol=0, atol=1e-14)
            and np.allclose(boundary["upper"], upper, rtol=0, atol=1e-14)
            and np.allclose(boundary["velocity"], 0., rtol=0, atol=0)
            for boundary, (lower, upper) in zip(config["boundaries"][:4], expected_clamps)
        )
        signature_data = {
            "configs": summary["configs"], "materials": summary["materials"],
            "frames": total, "substeps": summary["steps"] // total,
            "label": summary["case"], "definition": definition,
        }
        expected_signature = hashlib.sha256(
            json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        with np.load(folder / "final_state_0.npz") as final_state, np.load(
            folder / checkpoint["states"][0]) as checkpoint_state:
            final_matches_checkpoint = (set(final_state.files) == set(checkpoint_state.files)
                                        and all(np.array_equal(final_state[name], checkpoint_state[name])
                                                for name in final_state.files))
        with (folder / "observables.csv").open(newline="", encoding="utf-8") as stream:
            table = list(csv.DictReader(stream))
        table_matches_checkpoint = len(table) == len(rows) and all(
            set(record) == set(observation)
            and all(record[key] == ("" if value is None else str(value))
                    for key, value in observation.items())
            for record, observation in zip(table, rows)
        )
        checks.update({
            "complete_summaries_exist": (folder / "summary.json").is_file()
                                        and (folder / "final_state_0.npz").is_file(),
            "observer_saved_rows_complete": len(table) == total + 1 and table_matches_checkpoint,
            "final_state_matches_checkpoint": final_matches_checkpoint,
            "full_plan_signature_matches_checkpoint": expected_signature == checkpoint["signature"],
            "full_duration_and_step_count": (summary["steps"] == status["steps_completed"]
                                             and np.isclose(summary["simulation_time"], 3e-4,
                                                            rtol=0, atol=1e-14)),
            "intended_jh2_failure_mode": (config["dim"] == 3 and config["material"] == "jh2"
                                          and config["failure_mode"] == "ballistic"
                                          and config["n_grid"] == 96
                                          and np.isclose(config["domain_size"], .25)
                                          and material["failure_strain"] == definition["failure_strain"] == .05),
            "four_edge_clamps_and_sphere": (clamps_match
                                           and config["boundaries"][4] == definition["rigid_sphere"]
                                           and np.isclose(definition["edge_clamp_width_m"], .008)),
            "visible_failed_volume": max(failed_fractions) >= .005,
            "visible_damage_volume": max(damaged_fractions) >= .05,
            "debris_left_plate_thickness": int(escaped.sum()) >= 100,
            "ballistic_free_flight_observed": free_flight_count > 0 and free_flight_error <= 1e-9,
            "active_sphere_overlap_within_grid_support": maximum_active_sphere_overlap <= 1.5 * .25 / 96,
        })
        report["sources"].update({name: digest(folder / name) for name in (
            "summary.json", "observables.csv", "final_state_0.npz")})
    report["passed"] = all(checks.values())
    for array in saved.values():
        array._mmap.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--allow-prefix", action="store_true")
    args = parser.parse_args()
    report = review(args.folder, allow_prefix=args.allow_prefix)
    (args.folder / "fragmentation-review.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "frame": report["accepted_saved_frame"],
                      "failed_fraction": report["failed_reference_volume_fraction_history"][-1],
                      "escaped": report["failed_particles_outside_initial_plate_thickness_by_one_spacing"]}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
