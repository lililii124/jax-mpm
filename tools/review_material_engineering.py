"""Check completed material-engineering runs without importing the solver."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CASES = {
    "jc-tension": "johnson-cook",
    "jc-necking": "johnson-cook",
    "concrete-bending": "hjc",
    "concrete-bending-large": "hjc",
    "ceramic-indentation": "jh2",
}


def rows(path):
    def number(value):
        if value in ("", "None"):
            return None
        if value in ("True", "False"):
            return value == "True"
        return float(value)

    with path.open(encoding="utf-8", newline="") as stream:
        return [{key: number(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def extrema(table, key):
    values = np.array([row[key] for row in table])
    low, high = int(values.argmin()), int(values.argmax())
    return {
        "min": float(values[low]), "min_time_s": table[low]["time"],
        "max": float(values[high]), "max_time_s": table[high]["time"],
        "final": float(values[-1]),
    }


def review(folder, name):
    family = {"jc-necking": "jc-tension", "concrete-bending-large": "concrete-bending"}.get(name, name)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("completed"):
        raise ValueError(f"Incomplete run: {folder}")
    diagnostics = rows(folder / "diagnostics.csv")
    observations = rows(folder / "observables.csv")
    with np.load(folder / "final_state_0.npz") as archive:
        state = {key: archive[key] for key in archive.files}
    time = np.array([row["time"] for row in observations])
    mass = np.array([row["mass"] for row in diagnostics])
    checks = {
        "final_arrays_finite": all(bool(np.isfinite(value).all()) for value in state.values()),
        "final_state_valid": bool(state["valid"] and state["history_valid"].all()),
        "positive_particle_mass_and_volume": bool(
            (state["mass"] > 0).all() and (state["volume"] > 0).all()
        ),
        "positive_saved_J": all(row["min_J"] > 0 for row in diagnostics),
        "finite_saved_observables": all(
            value is None or np.isfinite(value)
            for row in observations + diagnostics for value in row.values()
        ),
        "matching_table_times": np.array_equal(time, [row["time"] for row in diagnostics]),
        "strictly_increasing_times": bool((np.diff(time) > 0).all()),
        "full_duration": bool(np.isclose(time[-1], summary["simulation_time"], rtol=1e-12)),
        "constant_saved_mass": bool(np.allclose(mass, mass[0], rtol=1e-13, atol=0)),
    }
    columns = {
        "jc-tension": (
            "mean_grip_tensile_force_N", "central_material_section_force_N",
            "engineering_strain", "load_bearing_axial_gap_m", "max_damage",
        ),
        "concrete-bending": (
            "loading_reaction_y_N", "midspan_bottom_deflection_m", "max_damage",
            "damaged_reference_volume_fraction",
        ),
        "ceramic-indentation": (
            "indenter_reaction_y_N", "nominal_indentation_m", "max_damage",
            "damaged_reference_volume_fraction",
        ),
    }
    result = {
        "case": summary["case"],
        "scope": "Saved-frame diagnostics and final particle state; no mesh-convergence or experimental-calibration claim.",
        "sources": {
            filename: hashlib.sha256((folder / filename).read_text(encoding="utf-8").encode()).hexdigest()
            for filename in ("summary.json", "diagnostics.csv", "observables.csv")
        },
        "checks": checks,
        "saved_frames": len(time),
        "min_J": min(row["min_J"] for row in diagnostics),
        "maximum_relative_mass_drift": float(np.max(np.abs(mass / mass[0] - 1))),
        "failed_mass_fraction": float(state["mass"][state["history_failed"]].sum() / mass[0]),
        "observables": {key: extrema(observations, key) for key in columns[family]},
    }
    if family == "jc-tension":
        positions = np.load(folder / "trajectory_x.npy", mmap_mode="r")
        failed = np.load(folder / "trajectory_failed.npy", mmap_mode="r")
        dx = summary["configs"][0]["domain_size"] / summary["configs"][0]["n_grid"]
        active_x = np.sort(state["x"][~state["history_failed"], 0])
        split = np.flatnonzero(np.diff(active_x) > 3 * dx) + 1
        groups = np.split(active_x, split) if len(active_x) else []
        levels, sections = np.unique(positions[0, :, 0], return_inverse=True)
        active_count = np.bincount(sections, weights=~state["history_failed"], minlength=len(levels))
        spacing = float(np.diff(levels).mean())
        middle = np.abs(levels - np.mean(summary["definition"]["bar_ends_x_m"])) < 0.51 * spacing
        F = state["F"]
        normal = np.linalg.det(F)[:, None] * np.linalg.inv(F).transpose(0, 2, 1)[:, :, 0]
        traction = np.einsum("pi,pi->p", state["history_stress"][:, 0, :], normal)
        force = np.bincount(
            sections,
            weights=traction * state["volume"] * (~state["history_failed"]) / spacing,
            minlength=len(levels),
        )
        central_force = float(force[middle].mean())
        checks.update(
            final_failure_field_matches=bool(np.array_equal(failed[-1], state["history_failed"])),
            failure_is_irreversible=all(bool((failed[i] >= failed[i - 1]).all()) for i in range(1, len(failed))),
            central_reference_section_has_no_active_particles=bool((active_count[middle] == 0).all()),
            separated_grid_support=bool(len(groups) >= 2),
            zero_final_central_section_force=bool(abs(central_force) < 1e-8),
            section_force_matches_observer=bool(np.isclose(
                central_force, observations[-1]["central_material_section_force_N"], atol=1e-8
            )),
        )
        first = next((i for i, frame in enumerate(failed) if frame.any()), None)
        result["separation"] = {
            "first_saved_failure_time_s": None if first is None else float(time[first]),
            "support_separation_threshold_m": 3 * dx,
            "active_groups": [
                {"particles": len(group), "x_min_m": float(group[0]), "x_max_m": float(group[-1])}
                for group in groups
            ],
            "failed_reference_layer_centres_m": levels[active_count == 0].tolist(),
            "final_central_section_force_N": central_force,
            "interpretation": "Mass remains in noncontacting debris. Grip reactions include vibration of the separated pieces; localization is not fracture-energy regularized.",
        }
        positions._mmap.close()
        failed._mmap.close()
    else:
        result["interpretation"] = (
            "Constitutive damage with retained grid participation. D=1 is not a resolved crack surface or a detached fragment."
        )
    if name in ("jc-necking", "concrete-bending-large"):
        review_revision(result, folder, name, summary, observations, diagnostics, state)
    result["passed"] = all(bool(value) for value in checks.values())
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    return result


def review_revision(result, folder, name, summary, observations, diagnostics, state):
    checks = result["checks"]
    definition, config = summary["definition"], summary["configs"][0]
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    time = np.asarray([row["time"] for row in observations])
    dx = config["domain_size"] / config["n_grid"]
    total_volume = float(state["volume"].sum())
    checks.update(
        complete_status=bool(status.get("completed")),
        complete_saved_frame_count=len(time) == status["frames"] + 1 and status["frame"] == status["frames"],
        step_count_matches_duration=bool(np.isclose(summary["steps"] * config["dt"], time[-1], rtol=1e-12, atol=1e-14)),
        complete_step_count=status["steps_completed"] == summary["steps"],
        duration_matches_definition=bool(np.isclose(time[-1], definition["duration_s"], rtol=1e-12, atol=1e-14)),
        at_least_one_hundred_thousand_particles=len(state["x"]) >= 100000,
        initial_time_is_zero=bool(np.isclose(time[0], 0.0, rtol=0, atol=1e-14)),
        uniform_saved_times=bool(np.allclose(np.diff(time), time[-1] / status["frames"], rtol=1e-12, atol=1e-14)),
        status_time_matches=bool(np.isclose(status["time"], time[-1], rtol=1e-12, atol=1e-14)),
    )
    mapped = []
    try:
        for field in ("x", "damage"):
            mapped.append(np.load(folder / f"trajectory_{field}.npy", mmap_mode="r"))
        positions, damage = mapped
        checks["saved_shapes_match"] = positions.shape == (len(time), len(state["x"]), 3) and damage.shape == (len(time), len(state["x"]))
        checks["final_fields_match"] = bool(np.array_equal(positions[-1], state["x"]) and np.array_equal(damage[-1], state["history_damage"]))
        checks["finite_saved_positions_and_damage"] = all(bool(np.isfinite(x).all() and np.isfinite(d).all()) for x, d in zip(positions, damage))
        checks["bounded_irreversible_damage"] = bool(np.min(damage) >= 0 and np.max(damage) <= 1) and all(bool((damage[i] >= damage[i - 1]).all()) for i in range(1, len(time)))
        checks["initially_undamaged"] = bool(np.count_nonzero(damage[0]) == 0)
        if name == "jc-necking":
            mapped.append(np.load(folder / "trajectory_failed.npy", mmap_mode="r"))
            check_necking(result, definition, config, time, observations, state, positions, mapped[-1])
        else:
            reference = positions[0]
            loading = definition["loading_boundary"]
            center_x = 0.5 * (loading["lower"][0] + loading["upper"][0])
            bottom = np.isclose(reference[:, 1], reference[:, 1].min(), atol=1e-12, rtol=0)
            middle = bottom & (np.abs(reference[:, 0] - center_x) <= max(dx, 0.0125))
            deflection, damaged_fraction, maximum = [], [], []
            for x, d in zip(positions, damage):
                displacement = x[:, 1] - reference[:, 1]
                deflection.append(float(-displacement[middle].mean()))
                maximum.append(float(-displacement.min()))
                damaged_fraction.append(float(state["volume"][d >= 0.5].sum() / total_volume))
            checks.update(
                explicit_strain_floor_profile=summary["materials"][0].get("legacy_v3") is False,
                retained_compressive_contact=config["failure_mode"] == "retain" and not bool(state["history_failed"].any()),
                deflection_matches_positions=bool(np.allclose(deflection, [row["midspan_bottom_deflection_m"] for row in observations], rtol=1e-11, atol=1e-12)),
                maximum_displacement_matches_positions=bool(np.allclose(maximum, [row["maximum_downward_displacement_m"] for row in observations], rtol=1e-11, atol=1e-12)),
                damaged_volume_matches_particles=bool(np.allclose(damaged_fraction, [row["damaged_reference_volume_fraction"] for row in observations], rtol=1e-11, atol=1e-13)),
                visible_actual_midspan_deflection=max(deflection) >= 0.008,
                substantial_damaged_volume=max(damaged_fraction) >= 0.05,
                prescribed_twelve_mm_stroke=bool(np.isclose(observations[-1]["prescribed_displacement_y_m"], -0.012, rtol=0, atol=1e-12)),
                stationary_final_loading_patch=abs(observations[-1]["prescribed_speed_y_m_per_s"]) < 1e-12,
            )
            result["deformation_and_damage"] = {
                "maximum_actual_midspan_deflection_m": max(deflection),
                "maximum_downward_displacement_m": max(maximum),
                "maximum_reference_volume_fraction_with_damage_at_least_half": max(damaged_fraction),
                "saved_midspan_deflection_m": deflection,
                "saved_damaged_volume_fraction": damaged_fraction,
                "display_requirement": "At least 8 mm actual midspan deflection and 5% reference volume at D >= 0.5; geometric scale remains one.",
            }
    finally:
        for value in mapped:
            value._mmap.close()


def check_necking(result, definition, config, time, observations, state, positions, failures):
    checks = result["checks"]
    reference = positions[0]
    levels, layers = np.unique(reference[:, 0], return_inverse=True)
    spacing = definition["particle_spacing_m"]
    center = np.mean(definition["bar_ends_x_m"])
    central = np.abs(levels - center) <= 0.51 * spacing
    counts = np.bincount(layers)

    def radii(x):
        cy = np.bincount(layers, weights=x[:, 1]) / counts
        cz = np.bincount(layers, weights=x[:, 2]) / counts
        squared = (x[:, 1] - cy[layers])**2 + (x[:, 2] - cz[layers])**2
        return np.sqrt(np.bincount(layers, weights=squared) / counts)

    initial_radius = radii(reference)
    radius_ratio = [float((radii(x)[central] / initial_radius[central]).mean()) for x in positions]
    ratio_observed = [row["central_geometric_rms_radius_ratio"] for row in observations]
    maximum_velocity_error = max(row["grip_inner_velocity_error_m_per_s"] for row in observations)
    maximum_position_error = max(row["grip_inner_axial_position_error_m"] for row in observations)
    maximum_transverse_motion = max(row["grip_inner_transverse_displacement_m"] for row in observations)
    speed = definition["grip_speed_m_per_s"]
    central_particles = central[layers]
    if failures.shape != (len(time), len(state["x"])):
        raise ValueError("The necking failure history has a different shape")
    active_counts = np.asarray([np.count_nonzero(frame[central_particles] == 0) for frame in failures])
    intact = active_counts == counts[central].sum()
    minimum_intact_radius = float(np.min(np.asarray(radius_ratio)[intact])) if intact.any() else None
    checks.update(
        binary_failure_history=all(bool(((frame == 0) | (frame == 1)).all()) for frame in failures),
        no_initial_failures=bool(np.count_nonzero(failures[0]) == 0),
        central_active_count_matches_observer=bool(np.array_equal(active_counts, [row["central_active_particles"] for row in observations])),
        specified_pr_revision=definition.get("source_commit") == "c3c1556ec86f9ebc6f0697149dff052702d20a75",
        full_reference_duration=bool(np.isclose(time[-1], 150e-6, rtol=0, atol=1e-14)),
        uniform_axial_lattice=bool(np.allclose(np.diff(levels), spacing, rtol=1e-12, atol=1e-15)),
        uniform_reference_particle_volumes=bool(np.allclose(state["volume"], spacing**3, rtol=1e-13, atol=0)),
        coordinate_radius_matches_observer=bool(np.allclose(radius_ratio, ratio_observed, rtol=1e-10, atol=1e-12)),
        resolved_initial_diameter=definition["grid_cells_across_nominal_diameter"] >= 12,
        measurable_neck_before_central_failure=minimum_intact_radius is not None and minimum_intact_radius <= 0.9,
        inner_grip_sample_exists=min(row["grip_inner_particles"] for row in observations) > 0,
        intact_grips=max(row["failed_grip_particles"] for row in observations) == 0,
        constrained_grip_velocity=maximum_velocity_error <= max(1e-8, 1e-8 * speed),
        constrained_grip_axial_motion=maximum_position_error <= 1.1 * speed * config["dt"],
        constrained_grip_transverse_motion=maximum_transverse_motion <= 1e-8 * config["domain_size"],
    )
    result["necking"] = {
        "saved_central_geometric_rms_radius_ratio": radius_ratio,
        "minimum_central_radius_ratio_before_any_central_failure": minimum_intact_radius,
        "maximum_inner_grip_velocity_error_m_per_s": maximum_velocity_error,
        "maximum_inner_grip_axial_position_error_m": maximum_position_error,
        "maximum_inner_grip_transverse_displacement_m": maximum_transverse_motion,
        "interpretation": "Coordinate radii are reference-layer RMS measures. Post-failure debris geometry is not inferred from its frozen deformation gradient. The PR is an implementation reference, not experimental validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/material-engineering"))
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    args = parser.parse_args()
    names = CASES if args.case == "all" else (args.case,)
    records = {}
    for name in names:
        folder = args.root / name / CASES[name]
        result = review(folder, name)
        (folder / "interpretation.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        records[name] = result
        print(f"{name}: {'passed' if result['passed'] else 'FAILED'}", flush=True)
    if not all(record["passed"] for record in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
