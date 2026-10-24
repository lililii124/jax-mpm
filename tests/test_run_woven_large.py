from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import run_woven_large as runner


def records():
    case = runner.case_record("plain", "sphere-impact")
    summary = {"completed": False, "probe_complete": True, "particles": [819200],
               "steps": 128, "full_steps": 14460, "solver_seconds": 64.,
               "observables": {"final": {"max_contact_nodes": 0}}, "source": {"fingerprint": "fixed-source"}}
    review = {"probe_passed": True, "passed": False, "full_saved_history_checked": True,
              "checks": {"physical_check": True}, "producer_source": dict(summary["source"])}
    command = {"completed": True, "memory": {"samples": 64, "stopped_for_memory": False,
               "peak_tree_uss_bytes": 4*1024**3, "minimum_system_available_bytes": 10*1024**3}}
    return case, summary, review, command


def test_valid_early_probe_is_not_a_full_impact_or_media_acceptance():
    result = runner.probe_assessment(*records(), verified_inputs=True)
    assert result["probe_accepted"]
    assert result["early_linear_full_solver_estimate_seconds"] == 7230.
    assert not result["full_protocol_completed"] and not result["production_eligible"]
    assert not result["media_created"] and not result["contact_exercised"]


@pytest.mark.parametrize("part,key,value", [(1, "completed", True), (1, "steps", 127),
                                           (1, "particles", [8192]), (2, "probe_passed", False),
                                           (2, "full_saved_history_checked", False), (3, "completed", False)])
def test_probe_admission_rejects_false_completion_wrong_scale_or_failed_review(part, key, value):
    data = list(deepcopy(records()))
    data[part][key] = value
    assert not runner.probe_assessment(*data, verified_inputs=True)["probe_accepted"]


@pytest.mark.parametrize("key,value", [("samples", 0), ("stopped_for_memory", True),
                                      ("peak_tree_uss_bytes", 13*1024**3),
                                      ("minimum_system_available_bytes", 1024**3)])
def test_memory_evidence_is_required(key, value):
    data = list(deepcopy(records()))
    data[3]["memory"][key] = value
    assert not runner.probe_assessment(*data, verified_inputs=True)["probe_accepted"]


def test_slow_valid_probe_does_not_qualify_for_a_single_job_full_run():
    data = list(deepcopy(records()))
    data[1]["solver_seconds"] = 128*2.
    result = runner.probe_assessment(*data, verified_inputs=True)
    assert result["probe_accepted"]
    assert not result["fits_265_minute_solver_budget_by_early_estimate"]


def test_command_keeps_full_protocol_and_changes_only_runtime_probe_flag():
    command = runner.command_line(runner.case_record("triaxial", "pressure-bending"))
    assert command[command.index("--probe-steps")+1] == "128"
    assert command[command.index("--frames")+1] == "60"
    assert "--resume" not in command and "--duration" not in command


def test_bounded_package_keeps_complete_files_and_records_omitted_whole_files(tmp_path):
    case = runner.case_record("plain", "sphere-impact")
    folder = tmp_path/case["folder"]
    folder.mkdir(parents=True)
    (folder/"status.json").write_text("{}")
    (folder/"checkpoint.npz").write_bytes(b"1234567890123456")
    target = tmp_path/"delivery"
    report = runner.package(case, root=tmp_path, logs=tmp_path/"logs", delivery=target, limit=1024**2+10)
    assert report["copied_bytes"] == 2
    assert str(Path(case["folder"])/"checkpoint.npz").replace("\\", "/") in report["omitted"]
    assert not (target/case["folder"]/"checkpoint.npz").exists()
    assert not report["media_created"]
    assert json.loads((target/"manifest.json").read_text()) == report
    assert sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) <= report["budget_bytes"]
    with pytest.raises(FileExistsError):
        runner.package(case, root=tmp_path, logs=tmp_path/"logs", delivery=target)


def test_mutation_while_copying_cannot_produce_a_verified_manifest(tmp_path, monkeypatch):
    case = runner.case_record("plain", "sphere-impact")
    folder = tmp_path/case["folder"]
    folder.mkdir(parents=True)
    source = folder/"checkpoint.npz"
    source.write_bytes(b"complete")
    copy = runner.shutil.copyfile
    def mutate(first, second):
        result = copy(first, second)
        Path(first).write_bytes(b"modified")
        return result
    monkeypatch.setattr(runner.shutil, "copyfile", mutate)
    target = tmp_path/"delivery"
    with pytest.raises(ValueError, match="changed"):
        runner.package(case, root=tmp_path, logs=tmp_path/"logs", delivery=target)
    assert not (target/"manifest.json").exists()


@pytest.mark.parametrize("checks", [None, {}, {"field": False}, {"field": 1}, {"field": "true"}])
def test_empty_malformed_or_false_check_maps_cannot_be_admitted(checks):
    values = list(deepcopy(records()))
    values[2]["checks"] = checks
    assert not runner.probe_assessment(*values, verified_inputs=True)["probe_accepted"]


def test_missing_input_verification_or_different_producer_is_rejected():
    assert not runner.probe_assessment(*records())["probe_accepted"]
    values = list(deepcopy(records()))
    values[2]["producer_source"] = {"fingerprint": "another-source"}
    assert not runner.probe_assessment(*values, verified_inputs=True)["probe_accepted"]


def test_input_hash_chain_is_checked_against_actual_files(tmp_path):
    _, summary, review, _ = deepcopy(records())
    for name in runner.REVIEW_INPUTS:
        (tmp_path/name).write_bytes(name.encode())
    review["input_sha256"] = {name: runner.engine.digest(tmp_path/name) for name in runner.REVIEW_INPUTS}
    review["review_source_sha256"] = runner.engine.digest(runner.ROOT/"tools/review_woven_large.py")
    assert runner.verify_review_inputs(tmp_path, summary, review)
    (tmp_path/"trajectory_x.npy").write_bytes(b"changed history")
    assert not runner.verify_review_inputs(tmp_path, summary, review)


def test_missing_or_extra_hashed_input_is_rejected(tmp_path):
    _, summary, review, _ = deepcopy(records())
    review["input_sha256"] = {}
    review["review_source_sha256"] = runner.engine.digest(runner.ROOT/"tools/review_woven_large.py")
    assert not runner.verify_review_inputs(tmp_path, summary, review)


def test_manifest_budget_is_reserved_before_any_delivery_is_created(tmp_path):
    target = tmp_path/"delivery"
    with pytest.raises(ValueError, match="manifest"):
        runner.package(runner.case_record("plain", "sphere-impact"), root=tmp_path, logs=tmp_path/"logs",
                       delivery=target, limit=1024**2)
    assert not target.exists()
