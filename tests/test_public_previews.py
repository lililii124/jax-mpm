import hashlib
import json

import pytest

from tools.public_runs import is_ready_for_publication


def write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def preview(folder, loading="transverse-bending", extra_failure=None):
    summary = {"completed": True, "definition": {"architecture": "plain", "loading": loading}}
    write(folder / "summary.json", summary)
    for name in ("diagnostics.csv", "observables.csv"):
        (folder / name).write_text("time,value\n0,0\n1,1\n", encoding="utf-8")
    checks = {
        "at_least_one_hundred_thousand_particles": False, "grid_resolves_yarn_height": False,
        "positive_final_stored_elastic_energy": True, "final_piola_section_forces_match": True,
        "final_reference_section_moment_matches": True,
    }
    allowed = ["at_least_one_hundred_thousand_particles", "grid_resolves_yarn_height"]
    if loading == "transverse-compression":
        checks["required_moving_samples_exist"] = False
        allowed.append("required_moving_samples_exist")
    if extra_failure:
        checks[extra_failure] = False
    core_names = ("summary.json", "diagnostics.csv", "observables.csv")
    hashes = {name: digest(folder / name) for name in core_names}
    review = {"passed": False, "checks": checks, "sources": hashes}
    for name in ("interpretation.json", "preview-review.json"):
        write(folder / name, review)
    hashes = {name: digest(folder / name) for name in (*core_names, "interpretation.json", "preview-review.json")}
    write(folder / "preview-eligibility.json", {
        "kind": "coarse-woven-preview-v1", "eligible": True,
        "review_file": "preview-review.json", "source_review_passed": False,
        "allowed_failed_checks": allowed,
        "observed_failed_checks": [key for key, value in checks.items() if not value],
        "limitations": ["Coarse sampling; no exact inner tracking where the required sample is absent."],
        "sources": hashes,
    })
    return summary, f"woven-preview--plain--{loading}"


@pytest.mark.parametrize("loading", ["transverse-bending", "transverse-compression"])
def test_coarse_preview_keeps_failed_production_review(tmp_path, loading):
    summary, identifier = preview(tmp_path, loading)
    assert is_ready_for_publication(tmp_path, identifier, summary)
    assert not is_ready_for_publication(tmp_path, identifier.replace("woven-preview", "woven"), summary)
    assert json.loads((tmp_path / "interpretation.json").read_text())["passed"] is False


@pytest.mark.parametrize("failure", ["positive_final_J", "final_piola_section_forces_match", "required_moving_samples_exist"])
def test_other_failures_are_not_waived_for_a_preview(tmp_path, failure):
    summary, identifier = preview(tmp_path, extra_failure=failure)
    assert not is_ready_for_publication(tmp_path, identifier, summary)


def test_preview_rejects_changed_sources_or_wrong_case(tmp_path):
    summary, identifier = preview(tmp_path)
    assert not is_ready_for_publication(tmp_path, identifier.replace("plain", "triaxial"), summary)
    (tmp_path / "observables.csv").write_text("changed\n", encoding="utf-8")
    assert not is_ready_for_publication(tmp_path, identifier, summary)


def test_original_report_status_cannot_be_relabelled(tmp_path):
    summary, identifier = preview(tmp_path)
    original = json.loads((tmp_path / "interpretation.json").read_text())
    original["passed"] = True
    write(tmp_path / "interpretation.json", original)
    eligibility = json.loads((tmp_path / "preview-eligibility.json").read_text())
    eligibility["sources"]["interpretation.json"] = digest(tmp_path / "interpretation.json")
    write(tmp_path / "preview-eligibility.json", eligibility)
    assert not is_ready_for_publication(tmp_path, identifier, summary)
