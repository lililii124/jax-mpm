import hashlib
import json

import pytest

from tools.public_runs import homepage_records, is_ready_for_publication


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


@pytest.mark.parametrize("loading", ["weft-tension", "transverse-compression"])
def test_archived_previews_are_hidden_without_a_fine_replacement(tmp_path, loading):
    identifier = f"woven-preview--plain--{loading}"
    folder = tmp_path / "docs/results" / identifier
    folder.mkdir(parents=True)
    summary, _ = preview(folder, loading)
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert is_ready_for_publication(folder, identifier, summary)
    assert homepage_records(tmp_path, [{"id": identifier}]) == []
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


def test_fine_result_remains_displayed_beside_its_archived_preview(tmp_path):
    coarse = "woven-preview--plain--warp-tension"
    fine = coarse.replace("woven-preview", "woven", 1)
    folder = tmp_path / "docs/results" / coarse
    folder.mkdir(parents=True)
    preview(folder, "warp-tension")
    refined = folder.with_name(fine)
    refined.mkdir()
    for name in ("summary.json", "diagnostics.csv", "observables.csv"):
        (refined / name).write_bytes((folder / name).read_bytes())
    write(refined / "interpretation.json", {
        "passed": True, "checks": {"independent_review": True},
        "sources": {name: digest(refined / name) for name in ("summary.json", "diagnostics.csv", "observables.csv")},
    })
    catalogue = [{"id": coarse}, {"id": fine}]
    assert homepage_records(tmp_path, catalogue) == [catalogue[1]]


def test_hidden_concrete_bending_does_not_restore_the_previous_exhibit(tmp_path):
    previous = "material-engineering--concrete-bending--hjc"
    current = "material-engineering--concrete-bending-large--hjc"
    for identifier in (previous, current):
        folder = tmp_path / "docs/results" / identifier
        folder.mkdir(parents=True)
        write(folder / "summary.json", {"completed": True})
        for name in ("diagnostics.csv", "observables.csv"):
            (folder / name).write_text("time,value\n0,0\n1,1\n", encoding="utf-8")
        write(folder / "interpretation.json", {
            "passed": True, "checks": {"independent_review": True},
            "sources": {name: digest(folder / name) for name in ("summary.json", "diagnostics.csv", "observables.csv")},
        })
    catalogue = [{"id": previous}, {"id": current}]
    assert homepage_records(tmp_path, catalogue) == []
    assert all((tmp_path / "docs/results" / record["id"] / "interpretation.json").is_file() for record in catalogue)


@pytest.mark.parametrize("architecture", ["plain", "triaxial", "twill-2x2"])
@pytest.mark.parametrize("loading", ["transverse-compression", "transverse-bending", "hemispherical-compression"])
def test_superseded_woven_loading_stays_archived_but_is_not_displayed(tmp_path, architecture, loading):
    identifier = f"woven--{architecture}--{loading}"
    folder = tmp_path / "docs/results" / identifier
    folder.mkdir(parents=True)
    summary = {"completed": True, "definition": {"architecture": architecture, "loading": loading}}
    write(folder / "summary.json", summary)
    for name in ("diagnostics.csv", "observables.csv"):
        (folder / name).write_text("time,value\n0,0\n1,1\n", encoding="utf-8")
    write(folder / "interpretation.json", {
        "passed": True, "checks": {"independent_review": True},
        "sources": {name: digest(folder / name) for name in ("summary.json", "diagnostics.csv", "observables.csv")},
    })
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert is_ready_for_publication(folder, identifier, summary)
    assert homepage_records(tmp_path, [{"id": identifier}]) == []
    assert before == {path.name: path.read_bytes() for path in folder.iterdir()}


def test_gallery_regeneration_keeps_superseded_woven_loadings_hidden(tmp_path, monkeypatch):
    from tools import collect_gallery

    catalogue = []
    for architecture in ("plain", "triaxial", "twill-2x2"):
        for loading in ("transverse-compression", "transverse-bending", "hemispherical-compression"):
            test_superseded_woven_loading_stays_archived_but_is_not_displayed(tmp_path, architecture, loading)
            catalogue.append({"id": f"woven--{architecture}--{loading}"})
    write(tmp_path / "docs/results/index.json", catalogue)
    monkeypatch.setattr(collect_gallery, "ROOT", tmp_path)
    collect_gallery.main()
    assert json.loads((tmp_path / "docs/images/manifest.json").read_text()) == []
    assert "woven--" not in (tmp_path / "docs/GALLERY.md").read_text()
    assert json.loads((tmp_path / "docs/results/index.json").read_text()) == catalogue
