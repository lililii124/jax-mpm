from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from examples import woven_large_plate as driver
from examples.woven_large_geometry import sample_large_weave


def small_case(monkeypatch, loading="pressure-bending"):
    monkeypatch.setattr(driver, "sample_large_weave", lambda architecture:
                        sample_large_weave(architecture, size=(.02, .02)))
    return driver.make_plan("plain", loading, frames=4)


def write_reference(folder, sample, plan):
    folder.mkdir(parents=True)
    driver.save_json(folder/"plan.json", plan)
    driver.save_npz(folder/"reference.npz", x=sample.positions, volume=sample.volume,
                    mass=driver.validate_reference(sample, plan), phase=sample.phase,
                    yarn_id=sample.yarn_id, family_id=sample.family_id, orientation=sample.orientation)


def test_runtime_pause_keeps_complete_physical_plan_and_signature(monkeypatch):
    _, plan = small_case(monkeypatch)
    before = deepcopy(plan)
    signature = driver.digest(plan)
    for frame in (0, 1, 3, 4, None):
        assert driver.runtime_stop_frame(plan, frame) == (4 if frame is None else frame)
        assert plan == before and driver.digest(plan) == signature
    assert plan["duration_s"] == 200e-6 and plan["stop_steps"] == plan["full_steps"]


@pytest.mark.parametrize("stop", [-1, 5, True, 1.5, "1"])
def test_invalid_runtime_pause_is_rejected(monkeypatch, stop):
    _, plan = small_case(monkeypatch)
    with pytest.raises(ValueError, match="frame index"):
        driver.runtime_stop_frame(plan, stop)


def test_probe_is_not_promoted_to_a_full_run_by_runtime_pause(monkeypatch):
    monkeypatch.setattr(driver, "sample_large_weave", lambda architecture:
                        sample_large_weave(architecture, size=(.02, .02)))
    _, plan = driver.make_plan(probe_steps=128)
    assert driver.runtime_stop_frame(plan, None) == 1
    with pytest.raises(ValueError, match="prefix probes"):
        driver.runtime_stop_frame(plan, 1)


def test_resume_reads_original_reference_bytes_without_reconstruction(tmp_path, monkeypatch):
    sample, plan = small_case(monkeypatch)
    write_reference(tmp_path/"case", sample, plan)
    monkeypatch.setattr(driver, "sample_large_weave", lambda *a, **k: pytest.fail("resume must not resample"))
    loaded, restored = driver.load_saved_case(tmp_path/"case")
    assert restored == plan
    for name in ("positions", "volume", "phase", "yarn_id", "family_id", "orientation"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(sample, name))


@pytest.mark.parametrize("field", ["x", "mass", "orientation", "phase"])
def test_modified_reference_archive_rejected_before_resume(tmp_path, monkeypatch, field):
    sample, plan = small_case(monkeypatch)
    folder = tmp_path/"case"
    write_reference(folder, sample, plan)
    with np.load(folder/"reference.npz", allow_pickle=False) as data:
        values = {key: data[key].copy() for key in data.files}
    values[field].flat[0] += 1
    driver.save_npz(folder/"reference.npz", **values)
    with pytest.raises(ValueError, match="reference bytes"):
        driver.load_saved_case(folder)


def test_different_source_cannot_resume_even_unchanged_reference(tmp_path, monkeypatch):
    sample, plan = small_case(monkeypatch)
    folder = tmp_path/"case"
    write_reference(folder, sample, plan)
    monkeypatch.setattr(driver, "source_record", lambda: {"fingerprint": "different"})
    with pytest.raises(ValueError, match="numerical source"):
        driver.load_saved_case(folder)


def folder_hashes(folder):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in folder.iterdir() if path.is_file()}


@pytest.mark.parametrize("name,value", [
    ("aggregate", np.array([2., 1.])),
    ("aggregate", np.array([2., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.])),
    ("aggregate", np.array([2., 1., 0., 0., 0., 0., 0., 0., np.nan, 0., 1.])),
    ("sphere_velocity", np.zeros(4)), ("sphere_center", np.array([0., np.inf, 0.])),
    ("sphere_mass", np.array(.02)), ("sphere_radius", np.array(.05, dtype=np.float32)),
])
def test_malformed_contact_aggregate_or_ball_state_cannot_resume(name, value):
    data = {"state_valid": np.array(True), "aggregate": np.array([2., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
            "sphere_center": np.array([0., .052, 0.]), "sphere_velocity": np.array([0., -20., 0.]),
            "sphere_mass": np.array(.01), "sphere_radius": np.array(.05)}
    plan = {"tool": {"mass_kg": .01, "diameter_m": .1}}
    driver.validate_checkpoint_control(data, plan)
    data[name] = value
    with pytest.raises(ValueError):
        driver.validate_checkpoint_control(data, plan)


@pytest.mark.parametrize("loading", ["sphere-impact", "pressure-bending"])
def test_actual_jax_full_and_split_saved_histories_match(tmp_path, monkeypatch, loading):
    """A short integration fixture, not the physical large-plate protocol."""
    sample, plan = small_case(monkeypatch, loading)
    # This explicitly small integration protocol is not a production plan.
    # Keep material/dx/dt/operators; exercise the complete pressure ramp in three
    # steps so its work is resolved above coordinate roundoff in this fixture.
    plan.update(duration_s=4*plan["dt_s"], full_steps=4, stop_steps=4,
                frame_steps=[0, 1, 2, 3, 4], output_steps=[0, 1, 2, 3, 4])
    if loading == "pressure-bending":
        plan["loading_duration_s"] = 3*plan["dt_s"]
    if loading == "sphere-impact":
        # Exercise actual contact history and finite-mass impulse during the test.
        plan["tool"]["initial_center_m"][1] = plan["tool"]["initial_occupied_grid_top_m"]+.05+2e-8
        plan["tool"]["initial_nodal_gap_m"] = 2e-8
        plan["tool"]["initial_physical_surface_gap_m"] = plan["tool"]["initial_center_m"][1]-.05-.001
    original_plan = deepcopy(plan)
    full = driver.run_case(sample, plan, tmp_path/"whole")
    first = driver.run_case(sample, plan, tmp_path/"split", stop_after_frame=2)
    assert full["completed"] and not full["segment_complete"]
    assert first["segment_complete"] and not first["completed"] and not first["probe_complete"]
    assert first["phase"] == "checkpoint-ready" and first["steps"] == 2
    split_folder = tmp_path/"split"/"plain"/loading
    whole_folder = tmp_path/"whole"/"plain"/loading
    untouched = folder_hashes(split_folder)
    for stop in (0, 2):
        with pytest.raises(ValueError, match="advance beyond"):
            driver.run_case(sample, plan, tmp_path/"split", resume=True, stop_after_frame=stop)
        assert folder_hashes(split_folder) == untouched
    restored, restored_plan = driver.load_saved_case(split_folder)
    final = driver.run_case(restored, restored_plan, tmp_path/"split", resume=True)
    assert final["completed"] and not final["segment_complete"] and final["steps"] == 4
    assert plan == original_plan
    assert json.loads((split_folder/"plan.json").read_text()) == original_plan
    for filename in ("trajectory_x.npy", "trajectory_speed.npy", "trajectory_damage.npy", "trajectory_temperature.npy"):
        np.testing.assert_array_equal(np.load(split_folder/filename), np.load(whole_folder/filename))
    for filename in ("final_state_0.npz", "checkpoint.npz"):
        with np.load(split_folder/filename, allow_pickle=False) as split, np.load(whole_folder/filename, allow_pickle=False) as whole:
            assert set(split.files) == set(whole.files)
            for key in split.files:
                if key != "solver_seconds":
                    np.testing.assert_array_equal(split[key], whole[key], err_msg=key)
    for filename in ("observables.csv", "diagnostics.csv"):
        assert (split_folder/filename).read_bytes() == (whole_folder/filename).read_bytes()
    assert final["final"]["coupling_valid"]
    if loading == "sphere-impact":
        assert final["observables"]["final"]["max_contact_nodes"] > 0
    else:
        assert final["observables"]["final"]["particle_external_work_J"] > 0
