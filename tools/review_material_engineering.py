"""Check completed material-engineering runs without importing the solver."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CASES = {
    "jc-tension": "johnson-cook",
    "concrete-bending": "hjc",
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
        "observables": {key: extrema(observations, key) for key in columns[name]},
    }
    if name == "jc-tension":
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
        del positions, failed
    else:
        result["interpretation"] = (
            "Constitutive damage with retained grid participation. D=1 is not a resolved crack surface or a detached fragment."
        )
    result["passed"] = all(bool(value) for value in checks.values())
    result["checks"] = {key: bool(value) for key, value in checks.items()}
    return result


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
