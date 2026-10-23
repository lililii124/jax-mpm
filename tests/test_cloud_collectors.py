import hashlib
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from tools import collect_gallery, collect_home, collect_results


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def raw(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def descriptors(folder):
    return {path.relative_to(folder).as_posix(): {"bytes": path.stat().st_size, "sha256": raw(path)}
            for path in folder.rglob("*") if path.is_file() and path.name != "import-receipt.json"}


def refresh_receipt(folder):
    path = folder / "import-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["files"] = descriptors(folder)
    write(path, receipt)


def run_fixture(root, family, suffix, *, cloud=True, run_id=10):
    folder = root / "outputs" / family / suffix
    folder.mkdir(parents=True)
    adhesion = family.startswith("adhesion-visible")
    reviewer = "tools/review_adhesion.py" if adhesion else "tools/review_material_engineering.py"
    files = {"tools/review_adhesion.py": "a" * 64, "tools/review_material_engineering.py": "b" * 64,
             "tools/check_vtk.py": "c" * 64}
    source = {"source_kind": "github-actions", "repository": "owner/repo", "commit": "1" * 40,
              "tree": "2" * 40, "run_id": run_id, "run_attempt": 1, "files": files,
              "review_source_sha256": files[reviewer],
              "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    summary = {"completed": True, "case": family + "/" + suffix, "particles": [4], "steps": 1,
               "simulation_time": 1., "min_J": 1., "max_damage": 0., "solver_seconds": .1}
    write(folder / "summary.json", summary)
    write(folder / "status.json", {"completed": True, "frame": 1, "frames": 1})
    for name in ("observables.csv", "diagnostics.csv"):
        (folder / name).write_text("time,value\n0,0\n1,1\n", encoding="utf-8")
    write(folder / "interpretation.json", {"passed": True, "checks": {"physical_check": True},
        "review_source_sha256": files[reviewer],
        "sources": {name: lf(folder / name) for name in ("summary.json", "diagnostics.csv", "observables.csv")}})
    write(folder / "numerical-source.json", source)
    np.savez_compressed(folder / "trajectory_index.npz", time=np.array([0., 1.]), body=np.zeros(4, dtype=np.int32))
    if not cloud:
        shutil.copy2(folder / "trajectory_index.npz", folder / "trajectory.npz")
    np.savez_compressed(folder / "final_state_0.npz", x=np.zeros((4, 3)))
    write(folder / "memory.json", {"scope": "synthetic protocol fixture"})
    (folder / "scene.pvsm").write_text("fixture", encoding="utf-8")
    (folder / "particles.pvd").write_text('<VTKFile><Collection><DataSet timestep="0"/><DataSet timestep="1"/></Collection></VTKFile>', encoding="utf-8")
    bounds = [.22, .78, .13, .82, .22, .78]
    write(folder / "render.json", {"renderer": "ParaView", "style_revision": 8, "version": [6, 2],
        "frames_rendered": 2, "particles": 4, "field": "displacement" if adhesion else "damage",
        "axes": False, "annotations": False, "geometry_scale": 1, "animation": "simulation.gif",
        "camera_position": [1., 1., 1.], "camera_focal_point": [0., 0., 0.], "camera_parallel_scale": 1.,
        "camera_bounds": bounds, "color_range": [0., 1.], "render_overrides": {"focus_bounds": bounds}})
    write(folder / "vtk_metadata.json", {"frames": 2, "particles": 4, "field_ranges": {"displacement": [0., 1.]}})
    write(folder / "vtk_check.json", {"passed": True, "sampled_frames": [0, 1], "particles_per_frame": 4,
        "checker_sha256": files["tools/check_vtk.py"]})
    frame_hashes, concatenated = {}, hashlib.sha256()
    (folder / "frames").mkdir()
    for index in range(2):
        frame = folder / "frames" / f"frame_{index:05d}.png"
        frame.write_bytes(b"synthetic PNG protocol fixture " + str(index).encode())
        frame_hashes[frame.name] = raw(frame)
        concatenated.update(frame.read_bytes())
    for name in ("final.png", "simulation.gif", "simulation.mp4"):
        (folder / name).write_bytes(b"synthetic media protocol fixture")
    video = {"frames": 2, "frames_sha256": concatenated.hexdigest(), "render_sha256": raw(folder / "render.json"),
             "bytes": (folder / "simulation.mp4").stat().st_size, "sha256": raw(folder / "simulation.mp4")}
    write(folder / "video.json", video)
    write(folder / "media-check.json", {"passed": True, "original_png_frames": 2, "png_sha256": frame_hashes,
        "gif": {"encoded_frames": 1, "original_png_frames": 2, "total_duration_ms": 120, "expected_duration_ms": 120},
        "render_sha256": video["render_sha256"], "video_sha256": video["sha256"]})
    if cloud:
        receipt = {key: source[key] for key in ("repository", "commit", "tree", "run_id", "run_attempt")}
        receipt.update(verified=True, source_verification={
            "source_kind": "github-actions", "repository": source["repository"], "commit": source["commit"],
            "tree": source["tree"], "run_id": run_id, "attempt": 1, "checked_source_files": len(files),
            "fingerprint": source["fingerprint"], "review_source_sha256": source["review_source_sha256"]}, files=descriptors(folder))
        write(folder / "import-receipt.json", receipt)
    return folder


def pair_fixture(root, family, *, cloud=True, second_run=10):
    folders = [run_fixture(root, family, name, cloud=cloud, run_id=10 if index == 0 else second_run)
               for index, name in enumerate(collect_results.ADHESION_ROLES.values())]
    parent = folders[0].parent
    names = ("summary.json", "observables.csv", "diagnostics.csv", "interpretation.json", "status.json")
    table = parent / "adhesion_comparison.csv"
    table.write_text("time_s,response\n0,0\n1,1\n", encoding="utf-8")
    write(parent / "adhesion_comparison.json", {"passed": True,
        "checks": {"matched_inputs": True, "lift_hold_has_visible_deformation_contrast": True},
        "sources": {role: {"files": {name: lf(folder / name) for name in names}}
                    for role, folder in zip(collect_results.ADHESION_ROLES, folders)},
        "csv": {"filename": table.name, "lf_sha256": lf(table), "rows": 2}, "sampled_time_s": [0., 1.],
        "deformation_contrast": {"qualifying_saved_intervals": [{"first_saved_time_s": 0., "last_saved_time_s": 1., "saved_span_s": 1.}]}})
    write(parent / "paired-render.json", {"scene_bounds": [.22, .78, .13, .82, .22, .78],
        "color_range": [0., 1.], "renderer_argument": "--focus-bounds", "camera_scope": "synthetic matched view"})
    if cloud:
        for folder in folders:
            receipt = json.loads((folder / "import-receipt.json").read_text())
            receipt["comparison_files"] = {name: {"bytes": (parent / name).stat().st_size, "sha256": raw(parent / name)}
                                            for name in collect_results.COMPARISON_FILES}
            write(folder / "import-receipt.json", receipt)
    return folders


def collect(root, monkeypatch, sources):
    monkeypatch.setattr(collect_results, "SOURCES", tuple(sources))
    target = root / "docs/results"
    collect_results.collect(root, target)
    return json.loads((target / "index.json").read_text())


def test_verified_material_cloud_preferred_without_touching_local_checkpoint(tmp_path, monkeypatch):
    suffix = "ceramic-indentation-wide/jh2"
    local = run_fixture(tmp_path, "material-engineering", suffix, cloud=False)
    cloud = run_fixture(tmp_path, "material-engineering-cloud", suffix)
    (local / "checkpoint_0_0.npz").write_bytes(b"preserve unfinished local checkpoint")
    before = {path.name: raw(path) for path in local.iterdir() if path.is_file()}
    records = collect(tmp_path, monkeypatch, ("material-engineering-cloud", "material-engineering"))
    assert len(records) == 1
    assert records[0]["id"] == "material-engineering--ceramic-indentation-wide--jh2"
    assert records[0]["source_directory"] == cloud.relative_to(tmp_path).as_posix()
    assert before == {path.name: raw(path) for path in local.iterdir() if path.is_file()}
    groups = collect_gallery.gallery_groups(records)
    assert ("ceramic-indentation-wide", cloud.relative_to(tmp_path / "outputs").as_posix()) in groups["Material engineering"]
    assert collect_home.describe(records[0]["id"])[0] == "Material engineering"


@pytest.mark.parametrize("problem", ["missing-receipt", "malformed-receipt", "missing-frame", "changed-frame", "failed-physics", "wrong-source", "wrong-vtk-checker"])
def test_invalid_cloud_falls_back_to_one_complete_local_case(tmp_path, monkeypatch, problem):
    suffix = "concrete-bending-large/hjc"
    local = run_fixture(tmp_path, "material-engineering", suffix, cloud=False)
    cloud = run_fixture(tmp_path, "material-engineering-cloud", suffix)
    if problem == "missing-receipt":
        shutil.copy2(cloud / "trajectory_index.npz", cloud / "trajectory.npz")
        (cloud / "import-receipt.json").unlink()
    elif problem == "malformed-receipt":
        write(cloud / "import-receipt.json", [])
    elif problem == "missing-frame":
        (cloud / "frames/frame_00001.png").unlink()
    elif problem == "changed-frame":
        (cloud / "frames/frame_00001.png").write_bytes(b"replacement does not belong to the audited animation")
        refresh_receipt(cloud)
    else:
        path = cloud / ("vtk_check.json" if problem == "wrong-vtk-checker" else "interpretation.json")
        report = json.loads(path.read_text())
        if problem == "failed-physics":
            report.update(passed=False, checks={"positive_final_J": False})
        elif problem == "wrong-source":
            report["review_source_sha256"] = "d" * 64
        else:
            report["checker_sha256"] = "d" * 64
        write(path, report)
        refresh_receipt(cloud)
    records = collect(tmp_path, monkeypatch, ("material-engineering-cloud", "material-engineering"))
    assert len(records) == 1 and records[0]["source_directory"] == local.relative_to(tmp_path).as_posix()


def test_cloud_pair_and_comparison_are_selected_from_the_same_root(tmp_path, monkeypatch):
    pair_fixture(tmp_path, "adhesion-visible", cloud=False)
    cloud = pair_fixture(tmp_path, "adhesion-visible-cloud")
    records = collect(tmp_path, monkeypatch, ("adhesion-visible-cloud", "adhesion-visible"))
    assert len(records) == 2 and all("outputs/adhesion-visible-cloud/" in item["source_directory"] for item in records)
    comparison = json.loads((tmp_path / "docs/results/adhesion-visible-comparison.json").read_text())
    assert comparison["csv"]["filename"] == "adhesion-visible-comparison.csv"
    assert all("outputs/adhesion-visible-cloud/" in value["source_directory"] for value in comparison["sources"].values())
    entries = collect_gallery.gallery_groups(records)["Adhesion comparison"]
    assert all(relative.startswith("adhesion-visible-cloud/") for _, relative in entries)
    selected, frame, _ = collect_gallery.preview(cloud[0], "adhesion-visible-nonadhesive-pull-off", {"frames_rendered": 2})
    assert selected.parent == cloud[0] / "frames" and frame in (0, 1)


@pytest.mark.parametrize("problem", ["missing-member-receipt", "different-run", "stale-comparison", "missing-comparison-proof"])
def test_pair_never_mixes_cloud_and_local_members(tmp_path, monkeypatch, problem):
    pair_fixture(tmp_path, "adhesion-visible", cloud=False)
    cloud = pair_fixture(tmp_path, "adhesion-visible-cloud", second_run=11 if problem == "different-run" else 10)
    if problem == "missing-member-receipt":
        (cloud[1] / "import-receipt.json").unlink()
    elif problem == "stale-comparison":
        (cloud[0].parent / "adhesion_comparison.csv").write_text("time_s,response\n0,1\n1,2\n", encoding="utf-8")
    elif problem == "missing-comparison-proof":
        receipt = json.loads((cloud[1] / "import-receipt.json").read_text())
        receipt.pop("comparison_files")
        write(cloud[1] / "import-receipt.json", receipt)
    records = collect(tmp_path, monkeypatch, ("adhesion-visible-cloud", "adhesion-visible"))
    assert len(records) == 2 and all(item["source_directory"].startswith("outputs/adhesion-visible/") for item in records)


def test_half_imported_pair_is_not_published(tmp_path, monkeypatch):
    run_fixture(tmp_path, "adhesion-visible-cloud", "nonadhesive-pull-off")
    records = collect(tmp_path, monkeypatch, ("adhesion-visible-cloud", "adhesion-visible"))
    assert records == []


def test_catalogue_cannot_route_a_pair_to_different_roots():
    records = [{"id": "adhesion-visible--nonadhesive-pull-off", "source_directory": "outputs/adhesion-visible-cloud/nonadhesive-pull-off"},
               {"id": "adhesion-visible--adhesive-pull-off", "source_directory": "outputs/adhesion-visible/adhesive-pull-off"}]
    with pytest.raises(ValueError, match="cannot mix"):
        collect_results.catalogue_sources(records)
    with pytest.raises(ValueError, match="cannot mix"):
        collect_gallery.gallery_groups(records)


def test_woven_preview_alias_stays_canonical():
    for source in collect_results.WOVEN_PREVIEW_SOURCES:
        assert collect_results.canonical_identifier(f"{source}/plain/transverse-bending") == "woven-preview--plain--transverse-bending"


@pytest.mark.parametrize("invalid_v3", [False, True])
def test_v3_pair_priority_and_atomic_fallback(tmp_path, monkeypatch, invalid_v3):
    local = pair_fixture(tmp_path, "adhesion-visible", cloud=False)
    v2 = pair_fixture(tmp_path, "adhesion-visible-cloud")
    v3 = pair_fixture(tmp_path, "adhesion-visible-v3-cloud")
    prior = {str(path): raw(path) for folder in (local[0].parent, v2[0].parent)
             for path in folder.rglob("*") if path.is_file()}
    if invalid_v3:
        (v3[1] / "import-receipt.json").unlink()
    records = collect(tmp_path, monkeypatch, collect_results.ADHESION_SOURCES)
    expected = v2 if invalid_v3 else v3
    assert len(records) == 2
    assert {item["source_directory"] for item in records} == {folder.relative_to(tmp_path).as_posix() for folder in expected}
    assert {item["id"] for item in records} == {"adhesion-visible--" + name for name in collect_results.ADHESION_ROLES.values()}
    assert all(raw(Path(path)) == digest for path, digest in prior.items())
    entries = collect_gallery.gallery_groups(records)["Adhesion comparison"]
    assert {relative for _, relative in entries} == {folder.relative_to(tmp_path / "outputs").as_posix() for folder in expected}
    image, _, _ = collect_gallery.preview(expected[0], "adhesion-visible-nonadhesive-pull-off", {"frames_rendered": 2})
    assert image.parent == expected[0] / "frames"


def jc_fixture(root, *, cut_frame=25, failure_frame=20):
    folder = run_fixture(root, "material-engineering-cloud", "jc-necking/johnson-cook")
    times = np.linspace(0., 150e-6, 61)
    summary = json.loads((folder / "summary.json").read_text())
    summary.update(particles=[102320], steps=57900, simulation_time=150e-6,
                   configs=[{"n_grid": 100, "domain_size": .01, "dt": 150e-6 / 57900}])
    write(folder / "summary.json", summary)
    write(folder / "status.json", {"completed": True, "frame": 60, "frames": 60})
    for name in ("observables.csv", "diagnostics.csv"):
        (folder / name).write_text("time,value\n" + "".join(f"{time:.17g},0\n" for time in times))
    np.savez(folder / "trajectory_index.npz", time=times, body=np.zeros(102320, dtype=np.int32))
    source = json.loads((folder / "numerical-source.json").read_text())
    source["files"]["tools/run_engineering_preview.py"] = "d" * 64
    source["fingerprint"] = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    write(folder / "numerical-source.json", source)
    original = {"passed": True, "checks": {"physical_check": True},
                "sources": {name: lf(folder / name) for name in ("summary.json", "diagnostics.csv", "observables.csv")}}
    write(folder / "constitutive-review.json", original)
    history, cuts = [], []
    detail = {"gap_m": .0005, "current_edge_x_m": [.0048, .0053], "initial_edge_x_m": [.0049, .0051],
              "base_nodes_x": [47, 52], "positive_x_node_intervals": [[47, 49], [52, 54]], "edge_damage": [.5, .5]}
    for frame, time in enumerate(times):
        disconnected = frame >= cut_frame
        history.append({"frame": frame, "time_s": float(time), "failed_particles": 16 if frame >= failure_frame else 0,
            "maximum_damage": 1. if frame >= failure_frame else .1, "maximum_axial_gap_m": .0005,
            "axial_support_cuts": int(disconnected), "positive_axial_support_components": 1 + int(disconnected),
            "central_active_particles": 0 if frame >= 30 else 16, "central_geometric_rms_radius_ratio": .8})
        cuts.append({"frame": frame, "time_s": float(time), "cuts": [detail] if disconnected else []})
    with (folder / "jc-support-history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    sources = {name: raw(folder / name) for name in ("summary.json", "diagnostics.csv", "observables.csv", "constitutive-review.json", "final_state_0.npz")}
    sources.update({name: hashlib.sha256(("synthetic omitted array " + name).encode()).hexdigest() for name in collect_results.JC_ARRAY_FILES})
    ordering = ("support_disconnection_observed_before_any_material_failure" if cut_frame < failure_frame else
                "same_saved_interval_unresolved" if cut_frame == failure_frame else "material_failure_observed_at_an_earlier_saved_state")
    audit = {"completed": True, "diagnostic_only": False, "production_eligible": False, "strict_review_passed": True,
        "failed_checks": [], "grid": 100, "particles": 102320, "runner_source_sha256": "d" * 64, "sources": sources,
        "history_csv": {"filename": "jc-support-history.csv", "sha256": raw(folder / "jc-support-history.csv"), "rows": 61},
        "saved_history": history, "saved_positive_axial_support_cuts": cuts,
        "first_axial_stencil_disconnection": {**history[cut_frame], "cuts": [detail]},
        "first_saved_material_failure": history[failure_frame], "first_saved_central_full_failure": history[30],
        "support_disconnection_before_material_failure": cut_frame < failure_frame,
        "support_disconnection_before_central_full_failure": cut_frame < 30,
        "support_disconnection_with_no_failed_material_observed": cut_frame < failure_frame,
        "failure_disconnection_ordering": ordering}
    write(folder / "jc-support-audit.json", audit)
    checks = {"constitutive_review_passed": True, "support_audit_complete": True,
              "no_saved_support_disconnection_before_material_failure": True}
    acceptance = {"passed": True, "checks": checks, "runner_source_sha256": "d" * 64,
        "failure_disconnection_ordering": ordering,
        "sources": {name: raw(folder / name) for name in ("constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv")}}
    write(folder / "runner-acceptance.json", acceptance)
    augmented = {**original, "checks": {**original["checks"], **{"runner_" + key: value for key, value in checks.items()}},
        "review_source_sha256": "b" * 64, "runner_source_sha256": "d" * 64,
        "supplemental_reviews": {name: raw(folder / name) for name in ("constitutive-review.json", "jc-support-audit.json", "runner-acceptance.json")}}
    write(folder / "interpretation.json", augmented)
    render = json.loads((folder / "render.json").read_text())
    render.update(frames_rendered=61, particles=102320)
    write(folder / "render.json", render)
    write(folder / "vtk_metadata.json", {"frames": 61, "particles": 102320})
    write(folder / "vtk_check.json", {"passed": True, "sampled_frames": [0, 60], "particles_per_frame": 102320, "checker_sha256": "c" * 64})
    (folder / "particles.pvd").write_text('<VTKFile><Collection>' + ''.join(f'<DataSet timestep="{time:.17g}"/>' for time in times) + '</Collection></VTKFile>')
    joined, pngs = hashlib.sha256(), {}
    for frame in range(61):
        path = folder / "frames" / f"frame_{frame:05d}.png"
        path.write_bytes(f"synthetic frame {frame}".encode())
        pngs[path.name] = raw(path)
        joined.update(path.read_bytes())
    video = {"frames": 61, "frames_sha256": joined.hexdigest(), "render_sha256": raw(folder / "render.json"),
             "bytes": (folder / "simulation.mp4").stat().st_size, "sha256": raw(folder / "simulation.mp4")}
    write(folder / "video.json", video)
    write(folder / "media-check.json", {"passed": True, "original_png_frames": 61, "png_sha256": pngs,
        "gif": {"encoded_frames": 61, "original_png_frames": 61, "total_duration_ms": 3660, "expected_duration_ms": 3660},
        "render_sha256": video["render_sha256"], "video_sha256": video["sha256"]})
    receipt = json.loads((folder / "import-receipt.json").read_text())
    receipt["preset"] = "jc-necking-formal"
    receipt["source_verification"].update(checked_source_files=4, fingerprint=source["fingerprint"])
    receipt["artifacts"] = [{"id": 123, "name": "synthetic-engineering-main", "part": "main", "zip_sha256": "e" * 64,
                              "github_digest": "sha256:" + "e" * 64, "directory": "private/unpublished"}]
    receipt["source_evidence"] = {"jc_support_audit": {"verified": True, "basis": "verified-actions-artifact",
        "audit_sha256": raw(folder / "jc-support-audit.json"), "runner_source_sha256": "d" * 64,
        "artifact_id": 123, "artifact_zip_sha256": "e" * 64,
        "array_hash_verification": "recorded-by-runner-not-recomputed-after-import",
        "unavailable_array_hashes": {name: sources[name] for name in collect_results.JC_ARRAY_FILES}}}
    write(folder / "import-receipt.json", receipt)
    refresh_receipt(folder)
    return folder


def jc_ready(folder):
    return collect_results.complete_engineering_import(folder, collect_results.JC_FORMAL_ID, json.loads((folder / "summary.json").read_text()))


@pytest.mark.parametrize("cut_frame", [20, 25])
def test_jc_chain_accepts_verified_later_or_unresolved_same_frame_ordering(tmp_path, monkeypatch, cut_frame):
    folder = jc_fixture(tmp_path, cut_frame=cut_frame)
    assert jc_ready(folder)
    records = collect(tmp_path, monkeypatch, ("material-engineering-cloud",))
    assert len(records) == 1 and records[0]["id"] == collect_results.JC_FORMAL_ID
    target = tmp_path / "docs/results" / collect_results.JC_FORMAL_ID
    for name in collect_results.JC_CHAIN_FILES:
        assert (target / name).read_bytes() == (folder / name).read_bytes()
        assert records[0]["sha256"][name] == raw(folder / name)
    evidence = json.loads((target / "source-evidence.json").read_text())
    assert "directory" not in evidence["artifact"]
    assert "not independently" in evidence["scope"]
    assert evidence["source_evidence"]["jc_support_audit"]["array_hash_verification"] == "recorded-by-runner-not-recomputed-after-import"


@pytest.mark.parametrize("name", collect_results.JC_CHAIN_FILES)
def test_jc_chain_missing_file_is_not_production(tmp_path, name):
    folder = jc_fixture(tmp_path)
    (folder / name).unlink()
    refresh_receipt(folder)
    assert not jc_ready(folder)


def test_jc_chain_rejects_unfailed_disconnection_even_with_passed_review_flags(tmp_path):
    folder = jc_fixture(tmp_path, cut_frame=10, failure_frame=20)
    assert not jc_ready(folder)


@pytest.mark.parametrize("problem", ["original-false", "original-source", "original-reviewer", "audit-cycle", "audit-runner",
    "audit-data", "acceptance-false", "acceptance-cycle", "supplemental", "csv-incomplete", "missing-proof",
    "false-recomputation-claim", "wrong-artifact", "diagnostic-preset", "dishonest-ordering"])
def test_jc_chain_rejects_tampering_after_receipt_refresh(tmp_path, problem):
    folder = jc_fixture(tmp_path)
    filename = ("constitutive-review.json" if problem.startswith("original") else
                "runner-acceptance.json" if problem.startswith("acceptance") else
                "interpretation.json" if problem == "supplemental" else
                "import-receipt.json" if problem in ("missing-proof", "false-recomputation-claim", "wrong-artifact", "diagnostic-preset") else "jc-support-audit.json")
    document = json.loads((folder / filename).read_text())
    if problem == "original-false": document["checks"]["physical_check"] = False
    elif problem == "original-source": document["sources"]["summary.json"] = "f" * 64
    elif problem == "original-reviewer": document["review_source_sha256"] = "f" * 64
    elif problem in ("audit-cycle", "acceptance-cycle"): document["sources"]["interpretation.json"] = raw(folder / "interpretation.json")
    elif problem == "audit-runner": document["runner_source_sha256"] = "f" * 64
    elif problem == "audit-data": document["sources"]["final_state_0.npz"] = "f" * 64
    elif problem == "acceptance-false": document["checks"]["support_audit_complete"] = False
    elif problem == "supplemental": document["supplemental_reviews"]["jc-support-audit.json"] = "f" * 64
    elif problem == "csv-incomplete":
        path = folder / "jc-support-history.csv"
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        document["history_csv"]["sha256"] = raw(path)
    elif problem == "missing-proof": document.pop("source_evidence")
    elif problem == "false-recomputation-claim": document["source_evidence"]["jc_support_audit"]["array_hash_verification"] = "independently-recomputed"
    elif problem == "wrong-artifact": document["source_evidence"]["jc_support_audit"]["artifact_zip_sha256"] = "f" * 64
    elif problem == "diagnostic-preset": document["preset"] = "jc-density-diagnostic"
    elif problem == "dishonest-ordering": document["failure_disconnection_ordering"] = "same_saved_interval_unresolved"
    write(folder / filename, document)
    # Refresh transport-level hashes so malformed semantic chains remain the cause of rejection.
    receipt = json.loads((folder / "import-receipt.json").read_text())
    if "source_evidence" in receipt:
        receipt["source_evidence"]["jc_support_audit"]["audit_sha256"] = raw(folder / "jc-support-audit.json")
    write(folder / "import-receipt.json", receipt)
    refresh_receipt(folder)
    assert not jc_ready(folder)


@pytest.mark.parametrize("matches", [False, True])
def test_jc_locally_available_array_is_actually_hashed(tmp_path, matches):
    folder = jc_fixture(tmp_path)
    (folder / "trajectory_failed.npy").write_bytes(b"synthetic omitted array trajectory_failed.npy" if matches else b"changed array")
    receipt = json.loads((folder / "import-receipt.json").read_text())
    receipt["source_evidence"]["jc_support_audit"]["unavailable_array_hashes"].pop("trajectory_failed.npy")
    write(folder / "import-receipt.json", receipt)
    refresh_receipt(folder)
    assert jc_ready(folder) is matches
