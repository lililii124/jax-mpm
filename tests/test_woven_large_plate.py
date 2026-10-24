from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from examples import woven_large_plate as example


@pytest.fixture(scope="module")
def planned():
    return example.make_plan("plain", "sphere-impact", probe_steps=128)


def test_plan_is_full_physics_even_when_only_128_steps_are_requested(planned):
    sample, plan = planned
    assert plan["particles"] == 819200
    assert plan["duration_s"] == 200e-6
    assert plan["full_steps"] == 14460
    assert plan["frame_steps"] == list(range(0, 14461, 241))
    assert plan["output_steps"] == [0, 128]
    assert plan["probe_steps"] == plan["stop_steps"] == 128
    assert len(sample.positions) == 819200
    assert plan["tool"]["initial_velocity_m_s"] == [0., -20., 0.]
    assert plan["tool"]["mass_kg"] == .01
    assert plan["tool"]["diameter_m"] == .1
    assert not plan["production_eligible"]


def test_initial_clearance_is_above_actual_positive_grid_support(planned):
    _, plan = planned
    assert plan["tool"]["initial_occupied_grid_top_m"] == pytest.approx(.0015)
    assert plan["tool"]["initial_center_m"][1] == pytest.approx(.052)
    assert plan["tool"]["initial_nodal_gap_m"] == pytest.approx(.0005)
    assert plan["tool"]["initial_physical_surface_gap_m"] == pytest.approx(.001)


def test_pressure_is_a_reference_area_dead_traction_not_a_displacement_constraint(planned):
    sample, source = planned
    plan = deepcopy(source)
    plan["loading"] = "pressure-bending"
    plan["pressure"]["peak_Pa"] = .25e6
    forces = example.pressure_force(sample, plan)
    np.testing.assert_array_equal(forces[:, (0, 2)], 0.)
    assert forces[:, 1].sum() == pytest.approx(-.25e6*.15*.15)
    indices = np.flatnonzero(forces[:, 1])
    assert np.all(sample.positions[indices, 1] == sample.positions[:, 1].max())
    assert np.all(np.abs(sample.positions[indices][:, (0, 2)]) < .075)
    assert plan["clamp"]["components"] == [0, 1, 2]
    assert not plan["clamp"]["backing"]


def test_plan_roundtrip_and_reference_fingerprints_are_deterministic(planned, tmp_path):
    sample, plan = planned
    output = tmp_path/"plan.json"
    example.save_json(output, plan)
    restored = json.loads(output.read_text())
    assert restored == plan
    assert example.digest(restored) == example.digest(plan)
    assert example.array_descriptor(sample.positions) == plan["reference_arrays"]["x"]
    changed = deepcopy(plan)
    changed["tool"]["mass_kg"] *= 2
    assert example.digest(changed) != example.digest(plan)


def test_atomic_complete_archive_has_no_pickle_objects(tmp_path):
    target = tmp_path/"checkpoint.npz"
    original = np.arange(48, dtype=float).reshape(4, 3, 4)
    example.save_npz(target, complete=original, metadata=np.array("source"))
    with np.load(target, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["complete"], original)
        assert str(data["metadata"]) == "source"
    assert not target.with_suffix(".npz.tmp").exists()


@pytest.mark.parametrize("kwargs", [{"frames": 0}, {"probe_steps": -1}, {"dx": 0.}, {"loading": "plane-compression"}])
def test_invalid_plan_is_rejected_before_geometry_allocation(monkeypatch, kwargs):
    monkeypatch.setattr(example, "sample_large_weave", lambda *a, **k: pytest.fail("must validate first"))
    with pytest.raises(ValueError):
        example.make_plan(**kwargs)


def test_static_planning_does_not_import_jax_in_a_clean_process(tmp_path):
    import subprocess
    script = tmp_path/"check_import.py"
    script.write_text("import sys\nfrom examples.woven_large_plate import acoustic_speed, WOVEN_MATERIAL\n"
                      "assert acoustic_speed(WOVEN_MATERIAL)>5000\nassert 'jax' not in sys.modules\n")
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(example.ROOT)
    result = subprocess.run([sys.executable, str(script)], cwd=example.ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr


def test_actual_jax_driver_prefix_checkpoint_and_invalid_resume(tmp_path, monkeypatch):
    """Small integration only; the physical full-size probe is a separate job."""
    from examples.woven_large_geometry import sample_large_weave

    monkeypatch.setattr(example, "sample_large_weave", lambda architecture:
                        sample_large_weave(architecture, size=(.02, .02)))
    sample, plan = example.make_plan("plain", "pressure-bending", frames=2,
                                     probe_steps=3, dx=.001)
    result = example.run_case(sample, plan, tmp_path)
    assert result["completed"] is False
    assert result["probe_complete"] is True
    assert result["steps"] == 3 < result["full_steps"]
    folder = tmp_path/"plain"/"pressure-bending"
    with np.load(folder/"checkpoint.npz", allow_pickle=False) as data:
        values = {name: data[name].copy() for name in data.files}
    assert float(values["aggregate"][1]) == 1.
    assert bool(values["state_valid"])
    assert np.any(values["state_v"] != 0.)
    history = np.load(folder/"trajectory_x.npy", mmap_mode="r")
    try:
        np.testing.assert_array_equal(history[-1], values["state_x"])
    finally:
        history._mmap.close()
    resumed = example.run_case(sample, plan, tmp_path, resume=True)
    assert resumed["probe_complete"] and resumed["steps"] == 3
    values["aggregate"][1] = 0.
    example.save_npz(folder/"checkpoint.npz", **values)
    before = (folder/"checkpoint.npz").read_bytes()
    with pytest.raises(ValueError, match="rejected step"):
        example.run_case(sample, plan, tmp_path, resume=True)
    assert (folder/"checkpoint.npz").read_bytes() == before
