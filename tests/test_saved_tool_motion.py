import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

# Loading this module directly also exercises ParaView's no-JAX import route.
PATH = Path(__file__).resolve().parents[1]/"mpm_jax/saved_tool_motion.py"
spec = importlib.util.spec_from_file_location("saved_tool_motion_test", PATH)
motion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)


def fixture(folder):
    times = [0., 1e-5, 2e-5]
    centers = [[0., .052, 0.], [0., .05185, 0.], [0., .0518, 0.]]
    velocities = [[0., -20., 0.], [0., -10., 0.], [0., 0., 0.]]
    rows = []
    for time, center, velocity in zip(times, centers, velocities):
        row = {"time": time, "ball_kinetic_energy_J": .005*sum(v*v for v in velocity)}
        for axis, position, speed in zip("xyz", center, velocity):
            row[f"ball_center_{axis}_m"] = position
            row[f"ball_velocity_{axis}_m_s"] = speed
        rows.append(row)
    with (folder/"observables.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    definition = {"motion_kind": "integrated-saved-states", "state_file": "observables.csv",
                  "state_file_sha256": motion.file_sha256(folder/"observables.csv"),
                  "producer_source_fingerprint": "fixed-physics", "radius": .05, "mass": .01,
                  "times": times, "centers": centers, "velocities": velocities}
    summary = {"completed": True, "probe_complete": False, "steps": 2, "full_steps": 2,
               "source": {"fingerprint": "fixed-physics"},
               "definition": {"loading": "sphere-impact", "probe_steps": 0, "rigid_spheres": [definition],
                              "tool": {"mass_kg": .01, "diameter_m": .1, "initial_center_m": centers[0],
                                       "initial_velocity_m_s": velocities[0]}}}
    (folder/"summary.json").write_text(json.dumps(summary))
    plan = {"source": summary["source"], "loading": "sphere-impact", "probe_steps": 0,
            "full_steps": 2, "tool": summary["definition"]["tool"]}
    (folder/"plan.json").write_text(json.dumps(plan))
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (folder/"status.json").write_text(json.dumps({"completed": True, "coupling_valid": True,
                                                "signature": signature, "steps": 2}))
    np.savez(folder/"checkpoint.npz", signature=signature, state_valid=True, aggregate=[2., 1.])
    np.savez(folder/"trajectory.npz", time=times)
    np.savez(folder/"final_state_0.npz", time=times[-1], sphere_center=centers[-1], sphere_velocity=velocities[-1],
             sphere_mass=.01, sphere_radius=.05)
    for name in motion.REVIEW_INPUTS:
        if not (folder/name).exists():
            (folder/name).write_bytes(b"independent-state-review-fixture")
    review = {"passed": True, "full_saved_history_checked": True, "checks": {"state": True},
              "producer_source": summary["source"], "review_source_sha256": motion.file_sha256(PATH.parents[1]/"tools/review_woven_large.py"),
              "input_sha256": {name: motion.file_sha256(folder/name) for name in motion.REVIEW_INPUTS}}
    (folder/"interpretation.json").write_text(json.dumps(review))
    return definition, times


def update_record(folder, name, mutate, *, refresh_review=False):
    path = folder/name
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record))
    if refresh_review:
        update_record(folder, "interpretation.json", lambda report: report["input_sha256"].update({name: motion.file_sha256(path)}))


def test_exact_saved_pose_and_velocity_are_bound_to_all_inputs(tmp_path):
    definition, times = fixture(tmp_path)
    centers, proof = motion.integrated_sphere_states(tmp_path, definition, times)
    np.testing.assert_array_equal(centers, definition["centers"])
    assert proof["type"] == "integrated_finite_mass_rigid_sphere"
    assert proof["input_sha256"]["observables.csv"] == definition["state_file_sha256"]
    assert proof["motion_reader_source_sha256"] == motion.file_sha256(PATH)


@pytest.mark.parametrize("times", [[0., 1e-5], [0., 1.1e-5, 2e-5], [0., 1e-5, 1e-5], [0., float("nan"), 2e-5]])
def test_interpolation_missing_or_ambiguous_output_times_are_rejected(tmp_path, times):
    definition, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="saved state"):
        motion.integrated_sphere_states(tmp_path, definition, times)


@pytest.mark.parametrize("checks", [{}, {"state": False}, {"state": 1}])
def test_empty_or_false_review_does_not_authorize_media(tmp_path, checks):
    definition, times = fixture(tmp_path)
    update_record(tmp_path, "interpretation.json", lambda report: report.update(checks=checks))
    with pytest.raises(ValueError, match="acceptance"):
        motion.integrated_sphere_states(tmp_path, definition, times)


def test_partial_result_and_wrong_source_remain_unrenderable(tmp_path):
    definition, times = fixture(tmp_path)
    update_record(tmp_path, "summary.json", lambda summary: summary.update(completed=False), refresh_review=True)
    with pytest.raises(ValueError, match="acceptance"):
        motion.integrated_sphere_states(tmp_path, definition, times)
    definition, times = fixture(tmp_path)
    update_record(tmp_path, "interpretation.json", lambda report: report.update(producer_source={"fingerprint": "other"}))
    with pytest.raises(ValueError, match="acceptance"):
        motion.integrated_sphere_states(tmp_path, definition, times)


@pytest.mark.parametrize("name", ["summary.json", "observables.csv", "final_state_0.npz", "trajectory.npz", "trajectory_x.npy", "checkpoint.npz"])
def test_changed_input_bytes_are_rejected_before_render(tmp_path, name):
    definition, times = fixture(tmp_path)
    with (tmp_path/name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="acceptance"):
        motion.integrated_sphere_states(tmp_path, definition, times)


def test_changed_sphere_endpoint_cannot_hide_behind_fresh_review_hash(tmp_path):
    definition, times = fixture(tmp_path)
    np.savez(tmp_path/"final_state_0.npz", time=times[-1], sphere_center=[0., .052, 0.],
             sphere_velocity=definition["velocities"][-1], sphere_mass=.01, sphere_radius=.05)
    update_record(tmp_path, "interpretation.json", lambda record: record["input_sha256"].update(
        {"final_state_0.npz": motion.file_sha256(tmp_path/"final_state_0.npz")}))
    with pytest.raises(ValueError, match="endpoint"):
        motion.integrated_sphere_states(tmp_path, definition, times)


def test_arbitrary_ledger_path_or_prescribed_motion_is_rejected(tmp_path):
    definition, times = fixture(tmp_path)
    for key, value in (("state_file", "../observables.csv"), ("motion_kind", "prescribed")):
        changed = {**deepcopy(definition), key: value}
        with pytest.raises(ValueError, match="declared local"):
            motion.integrated_sphere_states(tmp_path, changed, times)


def test_one_ball_ledger_cannot_be_reused_to_display_multiple_balls(tmp_path):
    definition, times = fixture(tmp_path)
    update_record(tmp_path, "summary.json", lambda summary: summary["definition"]["rigid_spheres"].append(deepcopy(definition)), refresh_review=True)
    with pytest.raises(ValueError, match="tool declaration"):
        motion.integrated_sphere_states(tmp_path, definition, times)


def test_checkpoint_from_another_plan_cannot_be_hidden_by_its_fresh_raw_hash(tmp_path):
    definition, times = fixture(tmp_path)
    np.savez(tmp_path/"checkpoint.npz", signature="wrong-plan", state_valid=True, aggregate=[2., 1.])
    update_record(tmp_path, "interpretation.json", lambda review: review["input_sha256"].update(
        {"checkpoint.npz": motion.file_sha256(tmp_path/"checkpoint.npz")}))
    with pytest.raises(ValueError, match="checkpoint"):
        motion.integrated_sphere_states(tmp_path, definition, times)


@pytest.mark.parametrize("change", ["millimeter_radius", "velocity_units", "swapped_axes"])
def test_wrong_units_or_axes_fail_even_when_summary_hash_is_refreshed(tmp_path, change):
    definition, times = fixture(tmp_path)
    if change == "millimeter_radius":
        definition["radius"] *= 1000
    elif change == "velocity_units":
        definition["velocities"] = (np.asarray(definition["velocities"])*3.6).tolist()
    else:
        definition["centers"] = np.asarray(definition["centers"])[:, [1, 0, 2]].tolist()
    update_record(tmp_path, "summary.json", lambda summary: summary["definition"].update(rigid_spheres=[definition]), refresh_review=True)
    with pytest.raises(ValueError, match="saved state|physical plan"):
        motion.integrated_sphere_states(tmp_path, definition, times)


def test_partial_checkpoint_status_is_not_promoted_by_completed_summary(tmp_path):
    definition, times = fixture(tmp_path)
    update_record(tmp_path, "status.json", lambda status: status.update(completed=False), refresh_review=True)
    with pytest.raises(ValueError, match="complete status"):
        motion.integrated_sphere_states(tmp_path, definition, times)
