import hashlib
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
