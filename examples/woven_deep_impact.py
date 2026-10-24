"""Separate 40 J deep-impact experiment; no claim of calibrated material response.

Run prefixes before the full protocol: --stop-after-frame 2, then --resume
--stop-after-frame 10. Each completed segment is sealed in a separate snapshot.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from examples import woven_large_plate as base


def make_plan(*, mass=.05, speed=40., frames=60, dx=.0005):
    if not np.isfinite([mass, speed]).all() or mass <= 0 or speed <= 0:
        raise ValueError("mass and speed must be finite and positive")
    if speed > 40:
        raise ValueError("speed exceeds the bounded 8 mm ballistic travel design")
    sample, plan = base.make_plan(frames=frames, dx=dx)
    plan["series"] = "woven-deep-impact"
    plan["case_path"] = ["plain", "deep-sphere-impact"]
    plan["tool"]["mass_kg"] = float(mass)
    plan["tool"]["initial_velocity_m_s"] = [0., -float(speed), 0.]
    shape = np.ceil(np.array([*sample.metadata["size_xz_m"][:1], .020,
                              sample.metadata["size_xz_m"][1]])/dx).astype(int)+7
    if np.prod(shape) > plan["grid"]["max_nodes"]:
        raise ValueError("deep-impact grid exceeds the declared CPU node budget")
    plan["grid"].update(y_bounds_m=[-.016, .004], estimated_shape=shape.tolist(),
                         estimated_nodes=int(np.prod(shape)))
    plan["deep_impact"] = {
        "initial_energy_J": .5*mass*speed**2,
        "baseline_energy_J": 2., "ballistic_travel_m": speed*plan["duration_s"],
        "minimum_peak_center_indentation_m": .002,
        "energy_relative_tolerance": .1,
        "scope": "Exploratory higher-energy elastic response; indentation is an acceptance target, not a guaranteed outcome. No fracture or material calibration."}
    return sample, plan


def review(folder):
    """Evaluate actual saved center motion, stability, clamps and energy accounting."""
    from tools.review_woven_large import _numpy_elastic
    folder = Path(folder)
    plan = json.loads((folder/"plan.json").read_text())
    with (folder/"observables.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (folder/"diagnostics.csv").open(newline="") as stream:
        diagnostics = list(csv.DictReader(stream))
    def values(name):
        return np.array([float(row[name]) for row in rows])
    initial = plan["deep_impact"]["initial_energy_J"]
    accounted = (values("kinetic_energy_J") + values("stored_elastic_energy_J")
                 + values("ball_kinetic_energy_J") + values("contact_kinetic_loss_J")
                 - values("grid_projection_work_J") - values("particle_external_work_J"))
    finite = all(np.isfinite(float(value)) for row in rows for value in row.values() if value not in ("", "None"))
    stable = finite and all(row["state_valid"] == "True" and row["coupling_valid"] == "True"
                            and float(row["minimum_step_J"]) > 0 for row in diagnostics)
    clamp = bool(np.max(values("clamp_position_error_m")) <= 1e-12
                 and np.max(values("clamp_speed_error_m_s")) <= 1e-12)
    peak = float(np.max(-values("center_mean_displacement_y_m")))
    energy_error = float(np.max(np.abs(accounted-initial))/initial)
    with np.load(folder/"reference.npz", allow_pickle=False) as ref, np.load(folder/"final_state_0.npz", allow_pickle=False) as final:
        material_energy, strain, _, _ = _numpy_elastic(final["F"], ref["volume"], ref["phase"],
                                                      ref["orientation"], plan["material_parameters"])
    material_passed = bool(np.isfinite(material_energy) and material_energy >= -1e-8
                           and np.isclose(material_energy, values("stored_elastic_energy_J")[-1],
                                          rtol=1e-8, atol=1e-8))
    summary = json.loads((folder/"summary.json").read_text())
    return {"peak_center_indentation_m": peak, "target_m": .002,
            "stable": stable, "clamp_passed": clamp,
            "material_energy_passed": material_passed,
            "independent_final_material_energy_J": material_energy,
            "final_yarn_green_strain_norm": strain,
            "max_relative_energy_error": energy_error,
            "energy_passed": bool(energy_error <= plan["deep_impact"]["energy_relative_tolerance"]),
            "completed": summary["completed"],
            "accepted": bool(summary["completed"] and stable and clamp and material_passed and peak >= .002
                             and energy_error <= plan["deep_impact"]["energy_relative_tolerance"]),
            "scope": "Saved-frame peak center ROI displacement; deeper peaks between saved frames are not resolved. Elastic demonstration only."}


def seal(folder, target):
    """Preserve exact raw bytes, including rejected results, without overwriting history."""
    target = Path(target)
    shutil.copytree(folder, target)
    files = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(target.rglob("*")) if path.is_file()}
    base.save_json(target/"receipt.json", {"files": files,
        "source": base.source_record(), "scope": "Exact raw segment snapshot; failed outputs retained."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/woven-deep-impact"))
    parser.add_argument("--mass", type=float, default=.05)
    parser.add_argument("--speed", type=float, default=40.)
    parser.add_argument("--stop-after-frame", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    folder = args.output/"plain/deep-sphere-impact"
    snapshot = args.output/"segments"/f"f{args.stop_after_frame:02d}"
    if snapshot.exists():
        raise FileExistsError("segment snapshot exists; choose the next frame or a fresh output")
    if args.resume:
        sample, plan = base.load_saved_case(folder)
        if (plan["series"] != "woven-deep-impact" or plan["tool"]["mass_kg"] != args.mass
                or plan["tool"]["initial_velocity_m_s"][1] != -args.speed):
            raise ValueError("resume requires the same deep-impact configuration")
        status = json.loads((folder/"status.json").read_text())
        if args.stop_after_frame <= status["frame"]:
            raise ValueError("resume must advance the saved frame")
    else:
        sample, plan = make_plan(mass=args.mass, speed=args.speed)
        if folder.exists():
            raise FileExistsError("fresh deep-impact run requires a new output folder")
    base.runtime_stop_frame(plan, args.stop_after_frame)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    try:
        base.run_case(sample, plan, args.output, resume=args.resume,
                      stop_after_frame=args.stop_after_frame)
        base.save_json(folder/"deep-review.json", review(folder))
    finally:
        if folder.exists():
            seal(folder, snapshot)


if __name__ == "__main__":
    main()
