"""Compare four reviewed transverse-wave runs without loading trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RUNS = {
    "wave-grid32": (32, 1.0),
    "wave-grid48": (48, 1.0),
    "wave-grid64": (64, 1.0),
    "wave-half-dt": (48, 0.5),
}
SOURCE_FILES = ("summary.json", "observables.csv", "diagnostics.csv")
TIME_ATOL = 1e-12
RESPONSE = "tip_mean_displacement_y_m"


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def columns(path, names):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(
            set(names).issubset(reader.fieldnames or ()), f"Missing columns: {path}"
        )
        values = np.asarray([[float(row[name]) for name in names] for row in reader])
    require(values.ndim == 2 and len(values) >= 3, f"Insufficient samples: {path}")
    require(np.isfinite(values).all(), f"Non-finite samples: {path}")
    return values.T


def without(mapping, keys):
    return {key: value for key, value in mapping.items() if key not in keys}


def read_run(root, name):
    folder = root / name / "neo-hookean"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    interpretation = json.loads(
        (folder / "interpretation.json").read_text(encoding="utf-8")
    )
    require(summary.get("completed") is True, f"{name}: incomplete summary")
    require(status.get("completed") is True, f"{name}: incomplete status")
    checks = interpretation.get("checks")
    require(
        interpretation.get("passed") is True
        and isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values()),
        f"{name}: missing or failed individual review",
    )
    require(
        interpretation.get("case") == summary["case"],
        f"{name}: review case label differs",
    )
    sources = {filename: digest(folder / filename) for filename in SOURCE_FILES}
    require(
        all(
            interpretation.get("sources", {}).get(key) == value
            for key, value in sources.items()
        ),
        f"{name}: individual review source SHA does not match the saved files",
    )
    require(
        len(summary["configs"]) == len(summary["materials"]) == 1,
        f"{name}: expected one body",
    )
    definition, config = summary["definition"], summary["configs"][0]
    grid, factor = RUNS[name]
    require(
        definition.get("series") == "visible-motion-v1"
        and definition.get("purpose") == "visible-wave"
        and definition.get("comparison_family") == "finite-transverse-shear-wave-v1",
        f"{name}: unexpected wave family",
    )
    require(config["n_grid"] == grid, f"{name}: incorrect grid")
    require(
        definition["time_step_factor"] == factor, f"{name}: incorrect time-step factor"
    )
    require(
        definition["bulk_CFL_limit"] == 0.12 * factor, f"{name}: incorrect CFL limit"
    )
    require(
        definition["particles_per_cell_per_axis"] == 3,
        f"{name}: expected three particles per cell per axis",
    )
    require(
        definition["model"] == config["material"] == "neo-hookean",
        f"{name}: incorrect material family",
    )
    for key in (
        "geometry",
        "initial_velocity_profile",
        "initial_velocity_amplitude_m_per_s",
        "constrained_components",
        "free_span_m",
        "initial_shear_speed_m_per_s",
        "initial_acoustic_speed_m_per_s",
        "density_kg_per_m3",
        "tip_reference_x_min_m",
    ):
        require(key in definition, f"{name}: missing physical definition {key}")
    require(bool(config.get("boundaries")), f"{name}: missing boundary definitions")
    time, response = columns(folder / "observables.csv", ("time", RESPONSE))
    diagnostic_time = columns(folder / "diagnostics.csv", ("time",))[0]
    require(np.array_equal(time, diagnostic_time), f"{name}: table times differ")
    require((np.diff(time) > 0).all(), f"{name}: output times are not increasing")
    require(len(time) == status["frames"] + 1, f"{name}: incomplete output frames")
    require(
        interpretation.get("saved_frames") == len(time),
        f"{name}: review frame count differs",
    )
    require(
        status["steps_completed"] == summary["steps"], f"{name}: incomplete step count"
    )
    require(
        summary["steps"] % status["frames"] == 0, f"{name}: nonuniform steps per frame"
    )
    duration = 4 * definition["free_span_m"] / definition["initial_shear_speed_m_per_s"]
    require(np.isfinite(duration) and duration > 0, f"{name}: invalid duration")
    require(
        np.allclose(time, np.linspace(0, duration, len(time)), rtol=0, atol=TIME_ATOL),
        f"{name}: saved times do not match the nominal common schedule",
    )
    require(
        np.allclose(
            [
                summary["simulation_time"],
                status["time"],
                summary["steps"] * config["dt"],
            ],
            duration,
            rtol=0,
            atol=TIME_ATOL,
        ),
        f"{name}: duration, status or time-step count does not match",
    )
    require(config["dt"] > 0, f"{name}: invalid time step")
    maximum_dt = (
        definition["bulk_CFL_limit"]
        * config["domain_size"]
        / grid
        / definition["initial_acoustic_speed_m_per_s"]
    )
    require(
        config["dt"] <= maximum_dt * (1 + 1e-12),
        f"{name}: time step exceeds the recorded CFL limit",
    )
    sources.update(
        {
            filename: digest(folder / filename)
            for filename in ("interpretation.json", "status.json")
        }
    )
    return {
        "folder": folder.as_posix(),
        "summary": summary,
        "definition": definition,
        "config": config,
        "duration": duration,
        "time": time,
        "response": response,
        "sources": sources,
    }


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def compare(root):
    root = Path(root)
    runs = {name: read_run(root, name) for name in RUNS}
    baseline = runs["wave-grid48"]
    permitted_definition_differences = {
        "anchor_particles_with_fully_constrained_support",
        "time_step_factor",
        "bulk_CFL_limit",
    }
    maximum_time_difference = 0.0
    times_identical = True
    for name, run in runs.items():
        for key in ("dtype", "jax_version"):
            require(
                run["summary"][key] == baseline["summary"][key],
                f"{name}: {key} differs",
            )
        require(
            run["summary"]["materials"] == baseline["summary"]["materials"],
            f"{name}: material parameters differ",
        )
        require(
            without(run["config"], {"n_grid", "dt"})
            == without(baseline["config"], {"n_grid", "dt"}),
            f"{name}: configuration differs beyond grid and time step",
        )
        require(
            without(run["definition"], permitted_definition_differences)
            == without(baseline["definition"], permitted_definition_differences),
            f"{name}: physical definitions differ",
        )
        require(
            run["duration"] == baseline["duration"], f"{name}: nominal durations differ"
        )
        require(
            len(run["time"]) == len(baseline["time"]),
            f"{name}: output frame counts differ",
        )
        difference = float(np.max(np.abs(run["time"] - baseline["time"])))
        require(
            difference <= TIME_ATOL,
            f"{name}: output times differ beyond roundoff tolerance",
        )
        maximum_time_difference = max(maximum_time_difference, difference)
        times_identical &= np.array_equal(run["time"], baseline["time"])
    half = runs["wave-half-dt"]
    require(
        half["summary"]["particles"] == baseline["summary"]["particles"],
        "Half-step run: particle counts differ from grid48",
    )
    half_ratio = half["config"]["dt"] / baseline["config"]["dt"]
    steps_per_frame = baseline["summary"]["steps"] // (len(baseline["time"]) - 1)
    half_steps_per_frame = half["summary"]["steps"] // (len(half["time"]) - 1)
    require(
        half_steps_per_frame in (2 * steps_per_frame - 1, 2 * steps_per_frame),
        "Half-step run: step ratio does not match the documented integer rounding",
    )
    require(half_ratio < 1, "Half-step run: actual dt is not smaller")

    histories = {}
    for name, run in runs.items():
        values, times = run["response"], run["time"]
        histories[name] = {
            "source_directory": run["folder"],
            "lf_text_sha256": run["sources"],
            "grid": run["config"]["n_grid"],
            "dt_s": run["config"]["dt"],
            "steps": run["summary"]["steps"],
            "particles": sum(run["summary"]["particles"]),
            "minimum_m": float(values.min()),
            "minimum_time_s": float(times[values.argmin()]),
            "maximum_m": float(values.max()),
            "maximum_time_s": float(times[values.argmax()]),
            "rms_m": rms(values),
            "final_m": float(values[-1]),
            "tip_mean_displacement_y_m": values.tolist(),
        }
    differences = {}
    for first, second in (
        ("wave-grid32", "wave-grid64"),
        ("wave-grid48", "wave-grid64"),
        ("wave-half-dt", "wave-grid48"),
    ):
        delta = runs[first]["response"] - runs[second]["response"]
        differences[f"{first}_minus_{second}"] = {
            "rms_difference_m": rms(delta),
            "maximum_absolute_difference_m": float(np.abs(delta).max()),
        }
    coarse = differences["wave-grid32_minus_wave-grid64"]["rms_difference_m"]
    medium = differences["wave-grid48_minus_wave-grid64"]["rms_difference_m"]
    return {
        "scope": "Same-problem numerical comparison of saved mean tip-y displacement histories; no analytic or experimental reference and no claimed convergence order.",
        "passed": True,
        "checks": {
            "individual_reviews_pass_and_source_hashes_match": True,
            "same_geometry_material_boundary_initial_velocity_and_particle_density": True,
            "same_config_except_grid_and_dt": True,
            "same_numeric_dtype_and_jax_version": True,
            "same_nominal_duration_and_output_schedule": True,
            "actual_output_times_match_within_roundoff": True,
            "expected_grids_and_time_step_factors": True,
            "half_step_particle_count_and_step_rounding_match": True,
        },
        "comparison_family": baseline["definition"]["comparison_family"],
        "common_definition": without(
            baseline["definition"], permitted_definition_differences
        ),
        "common_config": without(baseline["config"], {"n_grid", "dt"}),
        "materials": baseline["summary"]["materials"],
        "dtype": baseline["summary"]["dtype"],
        "jax_version": baseline["summary"]["jax_version"],
        "quantity": RESPONSE,
        "units": "m",
        "saved_frames": len(baseline["time"]),
        "nominal_duration_s": baseline["duration"],
        "time_s": baseline["time"].tolist(),
        "time_absolute_tolerance_s": TIME_ATOL,
        "actual_times_bitwise_identical": bool(times_identical),
        "maximum_time_difference_s": maximum_time_difference,
        "rms_definition": "sqrt(mean(value**2)) over all equally spaced saved frames, including both endpoints; no temporal interpolation",
        "runs": histories,
        "pairwise_differences": differences,
        "observations": {
            "grid48_is_closer_to_grid64_than_grid32_in_tip_history_rms": bool(
                medium < coarse
            ),
            "grid48_to_grid32_difference_ratio_against_grid64": medium / coarse
            if coarse > 0
            else None,
            "actual_half_to_base_dt_ratio": half_ratio,
            "interpretation": "The grid64 history is a numerical comparator, not an exact solution. A decreasing difference is an observation; a single half-step pair does not establish temporal convergence or order. Particle locations and the grid representation of the same prescribed boundary change with refinement.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path, default=Path("outputs/visible-motion")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Wave comparison rejected: {error}\n")
    output = args.output or args.root / "wave_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "passed": result["passed"],
                "observations": result["observations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
