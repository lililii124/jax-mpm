"""Read-only independent audit of the saved HJC 7000-step terminal state."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return value is None or isinstance(value, (str, bool))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.input
    case = json.loads((data / "case.json").read_text())
    cp = json.loads((data / "checkpoint.json").read_text())
    status = json.loads((data / "status.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    segment = json.loads((data / "segment.json").read_text())

    params = case["material_parameters_SI"]
    assert case["model"] == "hjc"
    assert params["legacy_v3"] is True
    assert params["mat111_compatible"] is False
    assert params["mat111_compacted_tension"] is False
    frames = int(case["frames"])
    steps = int(case["planned_steps"])
    substeps = int(case["substeps"])
    n = int(case["particle_count"])
    assert frames == 350 and steps == 7000 and substeps == 20
    assert steps == frames * substeps and n == 479232
    assert status["completed"] is True and summary["completed"] is True
    assert status["frame"] == cp["frame"] == frames
    assert status["steps_completed"] == summary["steps"] == steps
    assert abs(float(status["time"]) - float(case["duration_s"])) <= 1e-14
    assert abs(float(summary["simulation_time"]) - float(case["duration_s"])) <= 1e-14
    assert segment["completed"] is False and segment["frame"] == 341
    assert cp["signature"] == segment["signature"]
    assert len(cp["rows"]) == len(cp["observations"]) == frames + 1
    assert finite(cp["rows"]) and finite(cp["observations"])

    dt = float(case["time_step_s"]) * substeps
    time_tol = max(float(case["time_step_s"]) * 1e-5, 1e-15)
    rows, observations = cp["rows"], cp["observations"]
    masses = []
    min_j = math.inf
    max_j = -math.inf
    max_damage = 0.0
    max_plastic_strain = 0.0
    max_temperature = 0.0
    max_clamp_motion = 0.0
    max_failed_fraction = 0.0
    for i, (row, obs) in enumerate(zip(rows, observations)):
        expected = i * dt
        assert row["body"] == 0
        assert abs(float(row["time"]) - expected) <= time_tol
        assert abs(float(obs["time"]) - expected) <= time_tol
        if "load_sample_time_s" in obs:
            assert abs(float(obs["load_sample_time_s"]) - expected) <= time_tol
        mass = float(row["mass"])
        assert mass > 0 and float(row["current_volume"]) > 0
        masses.append(mass)
        row_min_j, row_max_j = float(row["min_J"]), float(row["max_J"])
        assert 0 < row_min_j <= row_max_j
        min_j, max_j = min(min_j, row_min_j), max(max_j, row_max_j)
        damage = float(row["max_damage"])
        plastic = float(row["max_plastic_strain"])
        temperature = float(row["max_temperature"])
        assert 0 <= damage <= 1 and plastic >= 0 and temperature > 0
        max_damage = max(max_damage, damage)
        max_plastic_strain = max(max_plastic_strain, plastic)
        max_temperature = max(max_temperature, temperature)
        assert obs["all_material_history_valid"] is True
        failed_fraction = float(obs["failed_reference_volume_fraction"])
        assert 0 <= failed_fraction <= 1
        max_failed_fraction = max(max_failed_fraction, failed_fraction)
        clamp_motion = float(obs["fully_supported_clamp_max_motion_m"])
        assert clamp_motion <= 1e-12
        max_clamp_motion = max(max_clamp_motion, clamp_motion)

    assert max(masses) - min(masses) <= 1e-12 * masses[0]
    manifest_state = data / cp["states"][0]
    with np.load(manifest_state, allow_pickle=False) as archive:
        keys = list(archive.files)
        state = {}
        for key in keys:
            value = archive[key]
            if value.dtype.kind in "fiu":
                assert np.isfinite(value).all(), key
            if key in {"x", "F", "mass", "volume", "time", "valid"} or key.startswith("history_"):
                state[key] = value
    assert state["x"].shape == (n, 3)
    assert state["F"].shape == (n, 3, 3)
    assert state["mass"].shape == state["volume"].shape == (n,)
    assert np.all(state["mass"] > 0) and np.all(state["volume"] > 0)
    assert np.all(np.linalg.det(state["F"]) > 0)
    assert bool(np.asarray(state["valid"]).all())
    assert abs(float(np.asarray(state["time"])) - float(status["time"])) <= time_tol
    history_valid = state.get("history_valid")
    if history_valid is not None:
        assert bool(np.asarray(history_valid).all())
    for key, value in state.items():
        if key in {"history_damage", "history_failed", "history_valid"}:
            continue
        if key.startswith("history_") and "damage" in key.lower():
            assert np.all((value >= 0) & (value <= 1)), key
        if key.startswith("history_") and "temperature" in key.lower():
            assert np.all(value > 0), key
        if key.startswith("history_") and "plastic" in key.lower() and ("strain" in key.lower() or "equivalent" in key.lower()):
            assert np.all(value >= 0), key
    state_damage = state.get("history_damage")
    state_temperature = state.get("history_temperature")

    geometry = case["geometry"]
    lower, upper = geometry["lower_m"], geometry["upper_m"]
    width = float(geometry["clamp_width_m"])
    files = [data / "frames_data" / f"frame_{i:05d}.npz" for i in range(frames + 1)]
    assert all(p.is_file() for p in files), "missing actual saved frame"
    with np.load(files[0], allow_pickle=False) as archive:
        reference = archive["x"].astype(np.float64, copy=True)
    assert reference.shape == (n, 3) and np.isfinite(reference).all()
    clamp_mask = ((reference[:, 0] <= lower[0] + width) |
                  (reference[:, 0] >= upper[0] - width) |
                  (reference[:, 2] <= lower[2] + width) |
                  (reference[:, 2] >= upper[2] - width))
    assert clamp_mask.any() and (~clamp_mask).any()
    field_hashes = []
    failed_points_over_course = 0
    max_nominal_strip_motion = 0.0
    frame_keys = None
    for i, path in enumerate(files):
        with np.load(path, allow_pickle=False) as archive:
            if frame_keys is None:
                frame_keys = set(archive.files)
                required = {"x", "speed", "damage", "temperature", "time", "contact_impulse"}
                assert required <= frame_keys
            else:
                assert set(archive.files) == frame_keys, (i, archive.files)
            x = archive["x"]
            speed = archive["speed"]
            damage = archive["damage"]
            temperature = archive["temperature"]
            time_s = float(np.asarray(archive["time"]).reshape(-1)[0])
            impulse = archive["contact_impulse"]
            assert x.shape == reference.shape and np.isfinite(x).all()
            assert speed.shape == damage.shape == temperature.shape == (n,)
            assert np.isfinite(speed).all() and np.isfinite(damage).all()
            assert np.isfinite(temperature).all() and np.isfinite(impulse).all()
            assert abs(time_s - i * dt) <= time_tol
            assert np.all((damage >= 0) & (damage <= 1)) and np.all(temperature > 0)
            if "failed" in archive.files:
                failed = archive["failed"]
                assert failed.shape == (n,) and np.isfinite(failed).all()
                assert np.all((failed == 0) | (failed == 1))
                failed_points_over_course = max(failed_points_over_course, int(np.count_nonzero(failed)))
            clamp_move = np.linalg.norm(x[clamp_mask] - reference[clamp_mask], axis=1)
            max_nominal_strip_motion = max(max_nominal_strip_motion, float(clamp_move.max()))
            if i == frames:
                assert np.array_equal(x, state["x"])
                if state_damage is not None:
                    assert np.array_equal(damage, state_damage)
                if state_temperature is not None:
                    assert np.array_equal(temperature, state_temperature)
            field_hashes.append({"index": i, "file": path.name,
                                 "sha256": sha256(path), "bytes": path.stat().st_size})

    checkpoint_sha = sha256(manifest_state)
    receipt = {
        "schema": "hjc-saved-terminal-checkpoint-independent-review-v1",
        "review_status": "passed",
        "completed": True,
        "source_commit": "3328c738d1a25da63b1244f63e9291c110776e00",
        "model": "HJC legacy-v3 demonstration preset",
        "mat111_compatible": False,
        "mat111_compacted_tension": False,
        "frame": frames,
        "steps": steps,
        "time_s": float(state["time"]),
        "particle_count": n,
        "checkpoint_file": cp["states"][0],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_matches_chain": checkpoint_sha == "cdf009d9dd65567bc314e15a7a6a89740b7647fee45487b3b4aae9d11587ace7",
        "checkpoint_signature": cp["signature"],
        "stale_segment_preserved": {"completed": segment["completed"],
                                     "frame": segment["frame"],
                                     "steps_completed": segment["steps_completed"]},
        "source_completed_flags_preserved": True,
        "diagnostic_rows": len(rows),
        "observation_rows": len(observations),
        "frame_count": len(field_hashes),
        "frame_fields": sorted(frame_keys),
        "all_frames_and_times_valid": True,
        "all_saved_particle_arrays_finite": True,
        "positive_deformation_determinants_all": True,
        "positive_mass_and_volume_all": True,
        "material_damage_in_0_1": True,
        "material_history_valid_all": True,
        "max_failed_reference_volume_fraction": max_failed_fraction,
        "max_failed_particle_count_in_frame": failed_points_over_course,
        "max_motion_of_particles_in_nominal_clamp_strips_m": max_nominal_strip_motion,
        "mass_range_kg": [min(masses), max(masses)],
        "min_J_over_course": min_j,
        "max_J_over_course": max_j,
        "max_damage_over_course": max_damage,
        "max_plastic_strain_over_course": max_plastic_strain,
        "max_temperature_over_course_K": max_temperature,
        "max_fixed_clamp_motion_m": max_clamp_motion,
        "frame_sha256": field_hashes,
        "scope": "Read-only audit of saved HJC terminal checkpoint and all saved real frames. No physics was run; original status/summary/segment files were not changed.",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "frame_sha256"}))


if __name__ == "__main__":
    main()
