"""Verify integrated rigid-tool poses for ParaView without importing JAX."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REVIEW_INPUTS = (
    "plan.json", "summary.json", "status.json", "layout.json", "reference.npz", "final_state_0.npz",
    "checkpoint.npz", "layout_arrays.npz", "particle_fields.npz", "trajectory.npz", "observables.csv",
    "diagnostics.csv", "trajectory_x.npy", "trajectory_speed.npy", "trajectory_damage.npy", "trajectory_temperature.npy",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4*1024**2):
            digest.update(block)
    return digest.hexdigest()


def integrated_sphere_states(folder, definition, output_times):
    """Use exact saved states; never infer an impact trajectory from waypoints."""
    folder = Path(folder)
    if definition.get("motion_kind") != "integrated-saved-states" or definition.get("state_file") != "observables.csv":
        raise ValueError("integrated motion requires the declared local state ledger")
    names = (*REVIEW_INPUTS, "interpretation.json")
    for name in names:
        path = folder/name
        if not path.is_file() or path.is_symlink():
            raise ValueError("complete, local reviewed state files are required")
    hashes = {name: file_sha256(folder/name) for name in names}
    summary = json.loads((folder/"summary.json").read_text())
    review = json.loads((folder/"interpretation.json").read_text())
    checks = review.get("checks")
    source = summary.get("source")
    if (summary.get("completed") is not True or summary.get("probe_complete") is not False
            or summary.get("steps") != summary.get("full_steps")
            or not isinstance(source, dict) or not source
            or definition.get("producer_source_fingerprint") != source.get("fingerprint")
            or review.get("producer_source") != source or review.get("passed") is not True
            or review.get("review_source_sha256") != file_sha256(Path(__file__).resolve().parents[1]/"tools/review_woven_large.py")
            or review.get("full_saved_history_checked") is not True
            or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values())
            or set(review.get("input_sha256", {})) != set(REVIEW_INPUTS)
            or any(review["input_sha256"].get(name) != hashes[name] for name in REVIEW_INPUTS)):
        raise ValueError("full numerical acceptance and its raw input chain are required")
    declaration = summary.get("definition", {})
    declared = declaration.get("rigid_spheres", [])
    if (declaration.get("loading") != "sphere-impact" or declaration.get("probe_steps") != 0
            or declared != [definition] or definition.get("state_file_sha256") != hashes["observables.csv"]):
        raise ValueError("tool declaration does not match the reviewed integrated state ledger")
    plan = json.loads((folder/"plan.json").read_text())
    status = json.loads((folder/"status.json").read_text())
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if (plan.get("source") != source or plan.get("loading") != "sphere-impact" or plan.get("probe_steps") != 0
            or status.get("completed") is not True or status.get("coupling_valid") is not True
            or status.get("signature") != signature or status.get("steps") != plan.get("full_steps")):
        raise ValueError("complete status and original numerical plan differ")
    with np.load(folder/"checkpoint.npz", allow_pickle=False) as checkpoint:
        if (str(checkpoint["signature"]) != signature or not bool(checkpoint["state_valid"])
                or float(checkpoint["aggregate"][1]) != 1. or int(checkpoint["aggregate"][0]) != plan["full_steps"]):
            raise ValueError("complete checkpoint does not match the accepted plan")
    with (folder/"observables.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    centers = np.asarray([[float(row[f"ball_center_{axis}_m"]) for axis in "xyz"] for row in rows])
    velocities = np.asarray([[float(row[f"ball_velocity_{axis}_m_s"]) for axis in "xyz"] for row in rows])
    requested = np.asarray(output_times, dtype=np.float64)
    declared_times = np.asarray(definition["times"], dtype=np.float64)
    mass, radius = float(definition["mass"]), float(definition["radius"])
    if (len(times) < 2 or times.ndim != 1 or centers.shape != (len(times), 3) or velocities.shape != centers.shape
            or not all(np.isfinite(value).all() for value in (times, centers, velocities, requested))
            or not np.all(np.diff(times) > 0) or times[0] != 0.
            or not np.array_equal(times, declared_times) or not np.array_equal(times, requested)
            or not np.array_equal(centers, np.asarray(definition["centers"]))
            or not np.array_equal(velocities, np.asarray(definition["velocities"]))
            or not np.isfinite([mass, radius]).all() or mass <= 0 or radius <= 0):
        raise ValueError("export times and tool poses must exactly match every finite saved state")
    with np.load(folder/"trajectory.npz", allow_pickle=False) as index:
        if not np.array_equal(index["time"], times):
            raise ValueError("tool ledger times differ from particle frame times")
    tool = plan["tool"]
    if (mass != tool["mass_kg"] or radius != tool["diameter_m"]/2
            or declaration["tool"] != tool
            or not np.array_equal(centers[0], tool["initial_center_m"])
            or not np.array_equal(velocities[0], tool["initial_velocity_m_s"])):
        raise ValueError("tool mass, radius or initial state differs from its physical plan")
    with np.load(folder/"final_state_0.npz", allow_pickle=False) as final:
        if (not np.array_equal(centers[-1], final["sphere_center"])
                or not np.array_equal(velocities[-1], final["sphere_velocity"])
                or float(final["sphere_mass"]) != mass or float(final["sphere_radius"]) != radius
                or float(final["time"]) != times[-1]):
            raise ValueError("tool endpoint differs from the complete physical state")
    energy = np.asarray([float(row["ball_kinetic_energy_J"]) for row in rows])
    if not np.allclose(energy, .5*mass*np.sum(velocities**2, axis=1), rtol=2e-13, atol=1e-24):
        raise ValueError("saved tool energy is inconsistent with its integrated velocity")
    evidence = {"type": "integrated_finite_mass_rigid_sphere", "motion": "exact saved center at every output time",
                "state_file_interpolation": "linear only between saved output-time keyframes in the ParaView state file; PNG frames use exact saved states",
                "input_sha256": hashes, "producer_source": source,
                "review_source_sha256": review.get("review_source_sha256"),
                "motion_reader_source_sha256": file_sha256(__file__),
                "output_times": times.tolist(), "centers": centers.tolist(), "velocities": velocities.tolist(),
                "mass_kg": mass, "radius_m": radius, "included_in_particle_count": False,
                "interpretation": "Finite-mass rigid tool coupled to the deformable plate; motion is solved, not prescribed"}
    return centers, evidence
