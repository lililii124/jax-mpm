import copy
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools import run_triaxial_study as study


def fake_source():
    producer_files = {"examples/triaxial_failure.py": "a"*64}
    producer = {"files": producer_files,
        "fingerprint": hashlib.sha256(json.dumps(producer_files, sort_keys=True).encode()).hexdigest()}
    return producer, {"files": producer_files, "runner_source_sha256": "b"*64,
        "review_sources": {study.REVIEW_FILES[0]: "c"*64, study.REVIEW_FILES[1]: "d"*64}}


def accepted_fixture(folder):
    folder.mkdir(parents=True)
    case = study.selection()
    producer, source = fake_source()
    plan = {"scope": "formal", "probe_steps": 0, "production_eligible": False,
        "theta_deg": 0, "law": "hertz", "load_transits": 10., "material": study.MATERIAL,
        "geometry": {"dimensions_m": [.024, .016], "grip_length_m": .003,
            "diameter_m": 235e-6, "pitch_m": .001, "crimp_amplitude_m": 140e-6, "clearance_m": 2e-6},
        "grid": {"storage": "tube", "dx_m": 75e-6, "padding_cells": 6, "max_nodes": 35_000_000, "nodes": 25_000_000},
        "particle_spacing_m": 45e-6, "sampling_parameters": {"packing": 1., "max_particles": 1_000_000},
        "particles": 683232, "stop_steps": 19200, "steps": 19200, "saved_states": 61,
        "frame_steps": list(range(0, 19201, 320)), "max_engineering_strain": .02,
        "duration_s": 19200*.15*75e-6/math.sqrt(8e6), "load_time_s": 10*.018/math.sqrt(8e6),
        "CFL": .15, "dt": .15*75e-6/math.sqrt(8e6), "hold_fraction_of_loading": .2,
        "requested_duration_s": 1.2*10*.018/math.sqrt(8e6), "coverage_skin_m": .0005,
        "search_window": 6, "requested_work_of_adhesion_J_per_m2": .1,
        "work_of_adhesion_J_per_m2": 0., "source": producer}
    layout = {"storage": "tube", "active_nodes": 6_500_000, "dense_aabb_nodes": 25_000_000}
    summary = {"completed": True, "steps": 19200, "simulation_time": plan["duration_s"],
        "production_eligible": False, "source": producer, "layout": layout}
    status = {"completed": True, "steps": 19200}
    for name, value in (("plan.json", plan), ("summary.json", summary), ("status.json", status), ("layout.json", layout)):
        study.save(folder/name, value)
    for name in ("observables.csv", "diagnostics.csv"):
        (folder/name).write_text("time,value\n0,1\n", encoding="utf-8")
    for name in ("trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy"):
        np.save(folder/name, np.zeros(3))
    for name in ("final_state_0.npz", "particle_fields.npz", "trajectory.npz"):
        np.savez(folder/name, values=np.arange(4))
    record = {"passed": True, "production_eligible": False, "checks": {"fixture": True},
        "auditor_sha256": source["review_sources"][study.REVIEW_FILES[0]], "source": producer,
        "input_sha256": {name: study.digest(folder/name) for name in study.RECORD_INPUTS}}
    study.save(folder/"record-audit.json", record)
    state = {"passed": True, "production_eligible": False, "checks": {"fixture": True},
        "full_history_checked": True, "review_source_sha256": source["review_sources"][study.REVIEW_FILES[1]],
        "producer_source": producer, "input_sha256": {name: study.digest(folder/name) for name in study.STATE_INPUTS}}
    study.save(folder/"state-review.json", state)
    return case, source, plan


@pytest.mark.parametrize("arguments", [dict(theta=1), dict(law="bonded"), dict(size="small"),
    dict(load_transits=0), dict(load_transits=float("nan")), dict(load_transits=float("inf")), dict(load_transits=101)])
def test_invalid_or_unbounded_study_selection_rejected(arguments):
    with pytest.raises(ValueError):
        study.selection(**arguments)


def test_fixed_case_commands_keep_scale_and_support_noninteger_transits():
    case = study.selection(theta=30, law="jkr", load_transits=12.857142857, size="20x14")
    arguments = study.driver_arguments(case)
    assert arguments[arguments.index("--load-transits")+1] == "12.857142857"
    assert arguments[arguments.index("--grid-storage")+1] == "tube"
    assert arguments[arguments.index("--frames")+1] == "60"
    assert arguments[arguments.index("--max-particles")+1] == "1000000"
    assert "--resume" not in arguments and "--probe-steps" not in arguments and "--smoke" not in arguments
    assert study.selection(size="28x18")["max_nodes"] == 60_000_000
    assert study.selection(size="24x16")["max_nodes"] == 35_000_000
    assert study.bounded(arguments, 265)[:4] == ["timeout", "--signal=INT", "--kill-after=60s", "265m"]


def test_solver_walltime_reserves_review_and_media_without_altering_case():
    case = study.selection(load_transits=40.)
    before = copy.deepcopy(case)
    assert study.solve_time_budget(295*60, 60) == 259*60
    assert study.solve_time_budget(400*60, 0) == 265*60
    with pytest.raises(TimeoutError, match="reserving"):
        study.solve_time_budget(35*60, 0)
    assert case == before


def test_real_numpy_driver_plan_matches_fixed_study_and_fractional_loading():
    from examples.triaxial_failure import plan_case

    case = study.selection(law="jkr", size="20x14", load_transits=12.857142857)
    _, plan = plan_case(law="jkr", dimensions=case["dimensions_m"], grip_length=.003,
                        grid_storage="tube", load_transits=case["load_transits"])
    assert all(study.validate_plan(plan, case, {"files": plan["source"]["files"]}).values())
    protocol = study.physical_loading(plan)
    assert math.isclose(protocol["load_time_s"], 10*.018/plan["wave_speed_m_per_s"], rel_tol=2e-11)


def test_acceptance_requires_both_independent_reviews_and_retains_dynamic_scope(tmp_path):
    case, source, plan = accepted_fixture(tmp_path/"case")
    result = study.acceptance(tmp_path/"case", case, source, plan, True)
    assert result["passed"] and all(result["checks"].values())
    assert not result["production_eligible"] and not result["media_created"]
    assert result["physical_loading"]["average_engineering_strain_rate_per_s"] == .02/plan["load_time_s"]
    assert set(result["input_sha256"]) >= {"record-audit.json", "state-review.json"}
    assert "runner-acceptance.json" not in result["input_sha256"]


@pytest.mark.parametrize("mutation", ["partial-history", "false-check", "wrong-source", "missing-final-state-hash", "source-changed"])
def test_final_only_or_tampered_review_cannot_be_accepted(tmp_path, mutation):
    folder = tmp_path/"case"
    case, source, plan = accepted_fixture(folder)
    state = study.read_json(folder/"state-review.json")
    if mutation == "partial-history":
        state["full_history_checked"] = False
    elif mutation == "false-check":
        state["checks"]["fixture"] = False
    elif mutation == "wrong-source":
        state["review_source_sha256"] = "0"*64
    elif mutation == "missing-final-state-hash":
        state["input_sha256"].pop("final_state_0.npz")
    study.save(folder/"state-review.json", state)
    report = study.acceptance(folder, case, source, plan, mutation != "source-changed")
    assert not report["passed"]


def test_raw_data_change_breaks_both_review_hash_chains(tmp_path):
    folder = tmp_path/"case"
    case, source, plan = accepted_fixture(folder)
    (folder/"observables.csv").write_text("time,value\n0,2\n", encoding="utf-8")
    report = study.acceptance(folder, case, source, plan, True)
    assert not report["checks"]["record_inputs"] and not report["checks"]["state_inputs"]
    assert not study.inputs_match(folder, {"../outside": "0"*64})


def test_shortened_scope_is_rejected_before_solve(tmp_path):
    case, source, plan = accepted_fixture(tmp_path/"case")
    plan["stop_steps"] = 8
    with pytest.raises(ValueError, match="unshortened_protocol"):
        study.validate_plan(plan, case, source)


def test_part_manifest_reconstructs_whole_file_and_detects_corruption(tmp_path):
    path = tmp_path/"original.npz"
    np.savez(path, values=np.arange(3000, dtype=np.float64))
    destination = tmp_path/"archive"
    record = study.split_complete_file(path, destination, "state/original.npz", part_bytes=4096)
    assert record["complete_file"] and len(record["parts"]) > 1
    data = b"".join((destination/part["path"]).read_bytes() for part in record["parts"])
    assert data == path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    for part in record["parts"]:
        assert part["bytes"] <= 4096 and study.digest(destination/part["path"]) == part["sha256"]
    with pytest.raises(ValueError, match="relative"):
        study.split_complete_file(path, destination, "../outside")
    with pytest.raises(ValueError, match="part size"):
        study.split_complete_file(path, destination, "state/file", part_bytes=0)


def test_early_eof_cannot_create_an_unbounded_sequence_of_empty_parts(tmp_path, monkeypatch):
    source = tmp_path/"state.npz"
    source.write_bytes(b"x"*8192)
    original_open = Path.open

    def short_stream(path, *args, **kwargs):
        if path == source and args and args[0] == "rb":
            return io.BytesIO(b"x"*1024)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", short_stream)
    with pytest.raises(ValueError, match="ended early"):
        study.split_complete_file(source, tmp_path/"archive", "state.npz", part_bytes=4096)
    assert len(list((tmp_path/"archive").rglob("*.part-*"))) == 1


def test_bounded_packaging_prioritizes_complete_checkpoint_and_omits_full_trajectory(tmp_path):
    case = study.selection()
    folder = tmp_path/case["folder"]
    folder.mkdir(parents=True)
    logs, delivery = tmp_path/"logs", tmp_path/"delivery"
    study.save(logs/"failure.json", {"completed": False, "error": "timeout"})
    (logs/"solve.log").write_bytes(b"x"*(700*1024))
    study.save(folder/"status.json", {"completed": False, "steps": 320})
    np.savez(folder/"checkpoint.npz", values=np.arange(70_000, dtype=np.float64))
    np.savez(folder/"final_state_0.npz", values=np.arange(80_000, dtype=np.float64))
    np.save(folder/"trajectory_x.npy", np.zeros((2, 4, 3)))
    (folder/"checkpoint.npz.tmp").write_bytes(b"unfinished")
    manifest = study.package(case, root=tmp_path, logs=logs, delivery=delivery,
                             total_bytes=2*1024**2, part_bytes=128*1024)
    checkpoint = (Path(case["folder"])/"checkpoint.npz").as_posix()
    final = (Path(case["folder"])/"final_state_0.npz").as_posix()
    trajectory = (Path(case["folder"])/"trajectory_x.npy").as_posix()
    assert checkpoint in manifest["state_files"] and final in manifest["omitted"]
    assert "logs/solve.log" in manifest["omitted"]
    assert trajectory in manifest["omitted"] and trajectory not in manifest["files"]
    assert not any("final_state_0.npz.part" in path for path in manifest["files"])
    assert not any(path.endswith(".tmp") for path in manifest["files"])
    assert sum(path.stat().st_size for path in delivery.rglob("*") if path.is_file()) <= 2*1024**2
    assert manifest["production_eligible"] is False and manifest["media_created"] is False
    assert study.verify_delivery(delivery) == manifest
    assert study.package(case, root=tmp_path, logs=logs, delivery=delivery) == manifest
    changed = copy.deepcopy(manifest)
    changed["state_files"][checkpoint]["sha256"] = "0"*64
    study.save(delivery/"delivery-manifest.json", changed)
    with pytest.raises(ValueError, match="whole-file"):
        study.verify_delivery(delivery)
    study.save(delivery/"delivery-manifest.json", manifest)
    first = next(iter(manifest["files"]))
    (delivery/first).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        study.verify_delivery(delivery)


def test_truncated_archive_never_becomes_checkpoint(tmp_path):
    case = study.selection()
    folder = tmp_path/case["folder"]
    folder.mkdir(parents=True)
    logs, delivery = tmp_path/"logs", tmp_path/"delivery"
    study.save(logs/"failure.json", {"error": "interrupted"})
    np.savez(folder/"checkpoint.npz", values=np.arange(100))
    data = (folder/"checkpoint.npz").read_bytes()
    (folder/"checkpoint.npz").write_bytes(data[:-30])
    manifest = study.package(case, root=tmp_path, logs=logs, delivery=delivery)
    name = (Path(case["folder"])/"checkpoint.npz").as_posix()
    assert name in manifest["omitted"] and not manifest["state_files"]


def test_success_archive_prioritizes_final_state_and_verified_gif_over_duplicate_checkpoint(tmp_path):
    case = study.selection()
    folder = tmp_path/case["folder"]
    case, source, plan = accepted_fixture(folder)
    np.savez(folder/"final_state_0.npz", values=np.arange(70_000, dtype=np.float64))
    np.savez(folder/"checkpoint.npz", values=np.arange(80_000, dtype=np.float64))
    state = study.read_json(folder/"state-review.json")
    state["input_sha256"]["final_state_0.npz"] = study.digest(folder/"final_state_0.npz")
    study.save(folder/"state-review.json", state)
    review = study.acceptance(folder, case, source, plan, True)
    assert review["passed"]
    study.save(folder/"runner-acceptance.json", review)
    selected = [folder/"frames"/f"frame_{i:05d}.png" for i in (0, 30, 60)] + [folder/"simulation.gif"]
    for path in selected:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"already-verified-media"*500)
    study.save(folder/"media-check.json", {"passed": True, "media_created": True,
        "input_sha256": state["input_sha256"],
        "output_sha256": {path.relative_to(folder).as_posix(): study.digest(path) for path in selected}})
    manifest = study.package(case, root=tmp_path, logs=tmp_path/"logs", delivery=tmp_path/"delivery",
                             total_bytes=2*1024**2, part_bytes=128*1024)
    assert (Path(case["folder"])/"final_state_0.npz").as_posix() in manifest["state_files"]
    assert (Path(case["folder"])/"checkpoint.npz").as_posix() in manifest["omitted"]
    assert manifest["media_created"] and manifest["selected_media_fully_archived"]
    assert not manifest["production_eligible"]


def test_failed_solve_still_records_source_after_without_running_reviews(tmp_path, monkeypatch):
    case = study.selection()
    _, source, plan = accepted_fixture(tmp_path/"fixture")
    logs = tmp_path/"logs"
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "LOGS", logs)
    monkeypatch.setattr(study, "source_record", lambda: copy.deepcopy(source))
    calls = []

    def command(name, arguments, **kwargs):
        calls.append(name)
        if name == "planning":
            study.save(logs/"planning.log", plan)
            return {"completed": True}
        raise RuntimeError("simulated GNU timeout")

    monkeypatch.setattr(study.engine, "command", command)
    with pytest.raises(RuntimeError, match="timeout"):
        study.run(case)
    assert calls == ["planning", "solve"]
    assert study.read_json(logs/"source-after.json")["unchanged"] is True


def test_verified_media_keeps_original_times_damage_range_and_coalesced_gif_duration(tmp_path):
    from PIL import Image

    folder = tmp_path/"case"
    _, _, plan = accepted_fixture(folder)
    summary = study.read_json(folder/"summary.json")
    summary["particles"] = [plan["particles"]]
    study.save(folder/"summary.json", summary)
    times = np.arange(61, dtype=float)*1e-6
    np.savez(folder/"trajectory.npz", time=times)
    source = {"files": {name: study.digest(study.ROOT/name) for name in
        ("mpm_jax/visualization.py", "mpm_jax/paraview_scene.py", "mpm_jax/export.py", "tools/check_vtk.py", "tools/encode_video.py")}}
    original = {"summary.json": study.digest(folder/"summary.json"), "trajectory.npz": study.digest(folder/"trajectory.npz")}
    study.save(folder/"vtk_metadata.json", {"frames": 61, "particles": plan["particles"]})
    scene = {"frames_rendered": 61, "renderer": "ParaView", "field": "damage", "color_range": [0., 1.],
             "geometry_scale": 1., "axes": False, "annotations": False, "animation": "simulation.gif"}
    study.save(folder/"render.json", scene)
    (folder/"simulation.mp4").write_bytes(b"video-fixture")
    study.save(folder/"video.json", {"frames": 61, "render_sha256": study.digest(folder/"render.json"),
                                    "sha256": study.digest(folder/"simulation.mp4")})
    items = "".join(f'<DataSet timestep="{time:.17g}"/>' for time in times)
    (folder/"particles.pvd").write_text(f"<VTKFile><Collection>{items}</Collection></VTKFile>")
    study.save(folder/"vtk_check.json", {"passed": True, "sampled_frames": [0, 60], "particles_per_frame": plan["particles"],
        "checker_sha256": source["files"]["tools/check_vtk.py"], "input_sha256": original,
        "export_sha256": {"particles.pvd": study.digest(folder/"particles.pvd")}})
    (folder/"frames").mkdir()
    with Image.new("RGB", (1200, 1200), color="white") as image:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        for frame in range(61):
            (folder/"frames"/f"frame_{frame:05d}.png").write_bytes(buffer.getvalue())
        image.save(folder/"simulation.gif", loop=0, duration=61*60)
    record = study.check_media(folder, source, original)
    assert record["passed"] and record["media_created"] and not record["production_eligible"]
    assert record["gif"]["encoded_frames"] == 1 and record["gif"]["total_duration_ms"] == 3660
    assert len(record["png_sha256"]) == 61
    scene["geometry_scale"] = 2
    study.save(folder/"render.json", scene)
    with pytest.raises(ValueError, match="physical result"):
        study.check_media(folder, source, original)


def test_workflow_is_manual_bounded_and_renders_without_publication():
    workflow = Path(__file__).resolve().parents[1]/".github/workflows/triaxial-runs.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text and "runs-on: ubuntu-24.04" in text
    assert "timeout-minutes: 360" in text and "include-hidden-files: true" in text
    assert "--package-only" in text and text.count("if: always()") >= 2
    assert "git push" not in text and "--smoke" not in text
    assert "official ParaView" in text and "run_triaxial_study.py install" in text
    assert "install_runner_dependencies.py" in text
    assert '--load-transits "$STUDY_LOAD_TRANSITS"' in text
