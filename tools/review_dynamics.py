"""Review saved dynamics tables without loading particle trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


BENCHMARKS = ("rubber-ball", "large-bending", "large-twist", "ball-beam", "floating-block")
TANKS = ("baffled-tank", "unbaffled-tank")


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def column(rows, key):
    return np.asarray([row[key] for row in rows])


def extrema(rows, key):
    values, times = column(rows, key), column(rows, "time")
    low, high = int(np.argmin(values)), int(np.argmax(values))
    return {
        "initial": float(values[0]),
        "final": float(values[-1]),
        "min": float(values[low]),
        "min_time_s": float(times[low]),
        "max": float(values[high]),
        "max_time_s": float(times[high]),
    }


def momentum(row):
    return np.array([row[f"momentum_{axis}"] for axis in "xyz"])


def source_record(path):
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "lf_text_sha256": hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
    }


def body_review(rows):
    mass = column(rows, "mass")
    return {
        "body": int(rows[0]["body"]),
        "mass_kg": float(mass[0]),
        "max_relative_mass_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "min_J": float(column(rows, "min_J").min()),
        "max_J": float(column(rows, "max_J").max()),
        "current_volume_m3": extrema(rows, "current_volume"),
        "max_speed_m_s": float(column(rows, "max_speed").max()),
        "mean_vertical_velocity_m_s": extrema(
            [{"time": row["time"], "value": row["momentum_y"] / row["mass"]} for row in rows],
            "value",
        ),
        "positive_saved_volume_and_J": bool(
            np.all(column(rows, "current_volume") > 0) and np.all(column(rows, "min_J") > 0)
        ),
    }


def tank_acceleration(time, definition):
    ramp_time = definition["ramp_s"]
    ramp = 0.5 * (1 - np.cos(np.pi * np.clip(time / ramp_time, 0, 1)))
    return (
        definition["acceleration_peak_m_s2"]
        * ramp
        * np.sin(2 * np.pi * definition["frequency_Hz"] * time)
    )


def tank_acceleration_integral(time, definition):
    amplitude = definition["acceleration_peak_m_s2"]
    omega = 2 * np.pi * definition["frequency_Hz"]
    ramp_time = definition["ramp_s"]
    beta = np.pi / ramp_time
    ramp_end = min(time, ramp_time)

    def sine_integral(frequency):
        if abs(frequency) < 1e-12:
            return 0.0
        return (1 - np.cos(frequency * ramp_end)) / frequency

    result = amplitude * (
        0.5 * sine_integral(omega)
        - 0.25 * (sine_integral(omega + beta) + sine_integral(omega - beta))
    )
    if time > ramp_time:
        result += amplitude * (np.cos(omega * ramp_time) - np.cos(omega * time)) / omega
    return float(result)


def review_case(name, folder):
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError(f"Run is incomplete: {folder}")
    diagnostics = read_rows(folder / "diagnostics.csv")
    observations = read_rows(folder / "observables.csv")
    grouped = [
        [row for row in diagnostics if int(row["body"]) == body]
        for body in range(len(summary["particles"]))
    ]
    time = column(observations, "time")
    finite = all(np.isfinite(value) for row in diagnostics + observations for value in row.values())
    final_time = float(time[-1])
    result = {
        "case": summary["case"],
        "sources": [
            source_record(folder / filename)
            for filename in ("summary.json", "diagnostics.csv", "observables.csv")
        ],
        "particles": summary["particles"],
        "steps": summary["steps"],
        "duration_s": final_time,
        "saved_frames": len(time),
        "all_saved_numbers_finite": bool(finite),
        "tables_have_matching_times": all(
            np.array_equal(column(rows, "time"), time) for rows in grouped
        ),
        "bodies": [body_review(rows) for rows in grouped],
        "observables": {
            key: extrema(observations, key)
            for key in observations[0]
            if key not in ("time", "external_work_trapezoidal")
            and not key.startswith("boundary_impulse")
        },
    }

    external = np.asarray(summary["external_impulse"])
    masses = np.array([rows[0]["mass"] for rows in grouped])
    contact = np.asarray(summary["contact_impulse_on_body_1"])
    supports = []
    for body, rows in enumerate(grouped):
        gravity = np.asarray(summary["configs"][body]["gravity"])
        body_contact = np.zeros(3) if len(grouped) == 1 else (-contact if body == 0 else contact)
        body_external = external * masses[body] / masses.sum() if name in TANKS else np.zeros(3)
        if name not in TANKS and np.any(external):
            raise ValueError(f"Unspecified per-body applied-force partition: {name}")
        residual = momentum(rows[-1]) - momentum(rows[0]) - masses[body] * gravity * final_time
        residual -= body_contact + body_external
        supports.append(residual)
    result["impulse_diagnostics"] = {
        "contact_impulse_on_body_1_N_s": contact.tolist(),
        "nonzero_net_contact_impulse": bool(np.linalg.norm(contact) > 1e-6),
        "inferred_support_impulse_by_body_N_s": [support.tolist() for support in supports],
        "inferred_support_sum_N_s": np.sum(supports, axis=0).tolist(),
        "reported_boundary_impulse_N_s": summary["boundary_impulse"],
        "interpretation": "Support impulses are residuals of particle momentum, gravity, applied force and the separately accumulated grid contact impulse. They combine every support on each body; this is a diagnostic, not an independent momentum-conservation test.",
    }

    if name in ("rubber-ball", "ball-beam"):
        center = column(observations, "ball_center_y")
        lowest = column(observations, "ball_lowest_y")
        mean_speed = column(grouped[0], "momentum_y") / masses[0]
        bottom = int(np.argmin(center))
        rebound = np.flatnonzero((np.arange(len(time)) > bottom) & (mean_speed > 1e-6))
        floor = summary["definition"]["floor_y"]
        result["ball_response"] = {
            "minimum_center_height_m": float(center[bottom]),
            "minimum_center_time_s": float(time[bottom]),
            "minimum_particle_height_m": float(lowest.min()),
            "final_particle_clearance_above_floor_m": float(lowest[-1] - floor),
            "rebound_observed": bool(len(rebound) > 0 and center[-1] > center[bottom]),
            "first_upward_saved_frame_after_lowest_center_s": float(time[rebound[0]])
            if len(rebound)
            else None,
            "height_recovered_after_lowest_center_m": float(center[-1] - center[bottom]),
            "final_mean_vertical_velocity_m_s": float(mean_speed[-1]),
        }
        if name == "rubber-ball":
            impulse = column(observations, "boundary_impulse_y")
            changed = np.flatnonzero(np.diff(impulse) > 1e-6)
            active = np.flatnonzero(impulse > 1e-6)
            result["ball_response"].update(
                first_saved_nonzero_support_time_s=float(time[active[0]]) if len(active) else None,
                last_saved_support_increment_time_s=float(time[changed[-1] + 1])
                if len(changed)
                else None,
                min_support_impulse_increment_N_s=float(np.diff(impulse).min()),
                support_timing_note="Output-interval observations; grid contact starts within the reported interval and has grid-scale location error.",
            )
            before = time < time[active[0]] if len(active) else np.ones(len(time), dtype=bool)
            dt, gravity_y = summary["configs"][0]["dt"], summary["configs"][0]["gravity"][1]
            predicted = center[0] + 0.5 * gravity_y * (time * time + dt * time)
            result["ball_response"][
                "max_precontact_semiimplicit_freefall_position_error_m"
            ] = float(np.max(np.abs(center[before] - predicted[before])))
    if name == "floating-block":
        center = column(observations, "floating_center_y")
        velocity = column(observations, "floating_mean_velocity_y")
        bottom = int(np.argmin(center))
        result["floating_response"] = {
            "maximum_downward_heave_m": float(center[0] - center[bottom]),
            "minimum_center_time_s": float(time[bottom]),
            "height_recovered_after_lowest_center_m": float(center[-1] - center[bottom]),
            "final_mean_vertical_velocity_m_s": float(velocity[-1]),
            "upward_reversal_observed": bool(bottom < len(time) - 1 and velocity[-1] > 0),
            "interpretation": "Release from an initial heave imbalance. The observed reversal is not evidence of a complete oscillation cycle or a settled equilibrium.",
        }
    if name == "large-bending":
        displacement = column(observations, "tracked_displacement_y")
        peak = int(np.argmax(np.abs(displacement)))
        result["bending_response"] = {
            "maximum_absolute_tip_mean_displacement_y_m": float(abs(displacement[peak])),
            "peak_time_s": float(time[peak]),
            "peak_is_final_saved_frame": bool(peak == len(time) - 1),
            "interpretation": "Mean displacement of the tracked free-end particles; not the largest individual particle displacement or a complete oscillation envelope.",
        }
    if name == "large-twist":
        angle = column(observations, "tip_rotation_x_rad")
        peak = int(np.argmax(np.abs(angle)))
        result["twist_response"] = {
            "maximum_absolute_tip_section_rotation_rad": float(abs(angle[peak])),
            "maximum_absolute_tip_section_rotation_deg": float(np.degrees(abs(angle[peak]))),
            "peak_time_s": float(time[peak]),
            "final_tip_section_rotation_rad": float(angle[-1]),
            "largest_saved_angle_increment_rad": float(np.max(np.abs(np.diff(angle)))),
            "interpretation": "Principal least-squares yz-plane rotation of the x > 0.72 m particles; section warping and deformation are not separated from rigid rotation.",
        }
    if name in TANKS:
        definition = summary["definition"]
        dt = summary["configs"][0]["dt"]
        steps = summary["steps"]
        prescribed = tank_acceleration(time, definition)
        discrete_impulse = (
            -masses.sum() * dt * tank_acceleration(np.arange(steps) * dt, definition).sum()
        )
        continuous_impulse = -masses.sum() * tank_acceleration_integral(final_time, definition)
        result["base_acceleration_check"] = {
            "max_observable_acceleration_error_m_s2": float(
                np.max(np.abs(prescribed - column(observations, "tank_acceleration_x")))
            ),
            "reported_applied_impulse_x_N_s": float(external[0]),
            "reconstructed_left_rule_applied_impulse_x_N_s": float(discrete_impulse),
            "reported_minus_reconstructed_impulse_x_N_s": float(external[0] - discrete_impulse),
            "continuous_prescribed_impulse_x_N_s": float(continuous_impulse),
            "reported_minus_continuous_impulse_x_N_s": float(external[0] - continuous_impulse),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--engineering", type=Path, default=Path("outputs/engineering"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/dynamics-review.json")
    )
    args = parser.parse_args()
    result = {
        "scope": "Read-only review of completed saved CSV/JSON diagnostics; no trajectory loading or additional solve.",
        "limitations": [
            "Saved-frame minima and maxima need not capture extrema between frames.",
            "Positive saved J and volume do not establish spatial or temporal convergence.",
            "Boundary impulse is derived from total momentum and is not an independent conservation test.",
            "Reported energies omit APIC affine kinetic energy; no energy-conservation or calibrated damping claim is made.",
        ],
        "cases": {},
        "pending_cases": [],
    }
    for name in BENCHMARKS + TANKS:
        base = args.benchmarks if name in BENCHMARKS else args.engineering
        folder = base / name / "neo-hookean"
        summary_path = folder / "summary.json"
        if not summary_path.exists() or not json.loads(
            summary_path.read_text(encoding="utf-8")
        ).get("completed"):
            result["pending_cases"].append(name)
            continue
        result["cases"][name] = review_case(name, folder)
    if all(name in result["cases"] for name in TANKS):
        baffled, plain = (result["cases"][name] for name in TANKS)
        result["tank_comparison"] = {
            "baffled_water_mass_kg": baffled["bodies"][0]["mass_kg"],
            "unbaffled_water_mass_kg": plain["bodies"][0]["mass_kg"],
            "baffled_peak_surface_y_m": baffled["observables"]["free_surface_y"]["max"],
            "unbaffled_peak_surface_y_m": plain["observables"]["free_surface_y"]["max"],
            "baffled_water_center_x_m": baffled["observables"]["water_center_x"],
            "unbaffled_water_center_x_m": plain["observables"]["water_center_x"],
            "interpretation": "Descriptive comparison under the same tank acceleration and initial surface elevation. Baffles displace water, so liquid masses differ. No baffle-efficiency or damping conclusion is supported by this pair alone.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
    print("Reviewed: " + ", ".join(result["cases"]))
    if result["pending_cases"]:
        print("Pending: " + ", ".join(result["pending_cases"]))


if __name__ == "__main__":
    main()
