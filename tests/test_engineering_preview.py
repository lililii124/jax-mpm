import importlib.metadata
import json

import numpy as np
import pytest

from tools import run_engineering_preview as engineering


def configure(monkeypatch, root):
    monkeypatch.setattr(engineering, "ROOT", root)
    monkeypatch.setattr(engineering, "LOGS", root / ".local/engineering-ci")
    monkeypatch.setattr(engineering, "DELIVERY", root / ".local/engineering-delivery")
    monkeypatch.setattr(engineering, "identity", lambda: {"run_id": "fixture"})


def test_v3_keeps_v2_inputs_and_output_separate():
    v2, v3 = [engineering.plan(name) for name in ("adhesion-v2-pair", "adhesion-v3-pair")]
    for old, new in zip(v2, v3):
        for key in ("name", "module", "grid", "ppc", "frames", "particles", "duration", "field", "review"):
            assert old[key] == new[key]
        assert old["root"] == "outputs/adhesion-visible"
        assert new["root"] == "outputs/adhesion-visible-v3"
        assert new["folder"] == new["root"] + "/" + new["name"]
        assert old["expected_preset"] == "lift-hold"
        assert new["expected_preset"] == "lift-hold-v3"
        assert old["arguments"] == ["--case", old["name"]]
        assert new["arguments"] == ["--case", new["name"], "--profile", "v3"]


@pytest.mark.parametrize("failure", ["individual", "pair", None])
def test_v3_requires_all_reviews_before_media(tmp_path, monkeypatch, failure):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(engineering.shutil, "disk_usage", lambda root: type("Disk", (), {"free": 10 * engineering.GIB})())
    source = {"review_sources": {"tools/review_adhesion.py": "a" * 64}}
    monkeypatch.setattr(engineering, "source_record", lambda: source)
    cases = engineering.plan("adhesion-v3-pair")
    calls = []

    def command(name, arguments, **options):
        calls.append(name)
        if name.startswith("solve-"):
            case = next(case for case in cases if name == "solve-" + case["name"])
            assert arguments[arguments.index("--profile") + 1] == "v3"
            assert arguments[arguments.index("--output") + 1] == case["root"]
            folder = tmp_path / case["folder"]
            engineering.save(folder / "summary.json", {
                "completed": True, "particles": [case["particles"]], "simulation_time": 1.3,
                "steps": 3700, "solver_seconds": .1, "configs": [{"n_grid": 48, "dt": 1.3 / 3700}],
                "definition": {"preset": "lift-hold-v3", "particles_per_cell_per_axis": 3,
                    "young_modulus_Pa": 8000, "adhesion": {"work_of_adhesion": 0. if case["name"].startswith("non") else 72.,
                        "interaction_range": .025, "penalty_stiffness": 4.8e6}}})
            engineering.save(folder / "status.json", {"completed": True, "frame": 50, "frames": 50})
        elif name == "review-pair":
            assert arguments[-1] == "outputs/adhesion-visible-v3"
            engineering.save(tmp_path / arguments[-1] / "adhesion_comparison.json",
                             {"passed": failure != "pair", "checks": {"contrast": failure != "pair"}})
        else:
            case = next(case for case in cases if name == "review-" + case["name"])
            engineering.save(tmp_path / case["folder"] / "interpretation.json",
                             {"passed": failure != "individual", "checks": {"physics": failure != "individual"}})
        return {"memory": {}}

    monkeypatch.setattr(engineering, "command", command)

    def export_pair(folders):
        calls.append("export")
        assert folders == [tmp_path / case["folder"] for case in cases]
        return {"geometry_scale": 1}

    def render(case, options):
        calls.append("render")
        engineering.save(tmp_path / case["folder"] / "render.json", {
            "camera_position": [1, 1, 1], "camera_focal_point": [0, 0, 0],
            "camera_parallel_scale": 1, "color_range": [0, 1]})

    monkeypatch.setattr(engineering, "export_pair", export_pair)
    monkeypatch.setattr(engineering, "render", render)
    monkeypatch.setattr(engineering, "package", lambda *args, **kwargs: calls.append("package"))
    if failure:
        with pytest.raises(ValueError, match="review|comparison"):
            engineering.run("adhesion-v3-pair")
        assert not {"export", "render", "package"}.intersection(calls)
        if failure == "individual":
            assert "review-pair" not in calls
    else:
        engineering.run("adhesion-v3-pair")
        assert calls[-4:] == ["export", "render", "render", "package"]
        assert (tmp_path / "outputs/adhesion-visible-v3/paired-render.json").is_file()
        assert not (tmp_path / "outputs/adhesion-visible").exists()


def test_failure_preserves_pair_evidence_and_previous_failures(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    cases = engineering.plan("adhesion-v3-pair")
    for case in cases:
        engineering.save(tmp_path / case["folder"] / "interpretation.json", {"passed": False})
    root = tmp_path / cases[0]["root"]
    for name in engineering.PAIR_FILES:
        (root / name).write_text("first failed comparison " + name, encoding="utf-8")
    engineering.preserve_failure("adhesion-v3-pair", ValueError("first"))
    original = {path.relative_to(engineering.LOGS): path.read_bytes()
                for path in engineering.LOGS.rglob("*") if path.is_file()}
    saved = engineering.LOGS / "failed-comparisons/adhesion-visible-v3"
    assert all((saved / name).read_bytes() == (root / name).read_bytes() for name in engineering.PAIR_FILES)
    (root / "adhesion_comparison.json").write_text("second failure", encoding="utf-8")
    engineering.preserve_failure("adhesion-v3-pair", ValueError("second"))
    assert all((engineering.LOGS / path).read_bytes() == data for path, data in original.items())
    history = list((engineering.LOGS / "failure-history").iterdir())
    assert len(history) == 1
    assert (history[0] / "failed-comparisons/adhesion-visible-v3/adhesion_comparison.json").read_text() == "second failure"


@pytest.mark.parametrize("preset", ["adhesion-v2-pair", "adhesion-v3-pair"])
def test_package_uses_the_pair_root(tmp_path, monkeypatch, preset):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(engineering.shared, "identity", lambda: {"run_id": "fixture"})
    cases = engineering.plan(preset)
    for case in cases:
        folder = tmp_path / case["folder"]
        folder.mkdir(parents=True)
        for name in ("summary.json", "status.json", "observables.csv", "diagnostics.csv", "interpretation.json", "memory.json", "numerical-source.json",
                     "particles.pvd", "vtk_metadata.json", "render.json", "video.json", "vtk_check.json", "media-check.json", "scene.pvsm",
                     "final_state_0.npz", "final.png", "simulation.gif", "simulation.mp4"):
            (folder / name).write_bytes(b"package protocol fixture")
        np.savez(folder / "trajectory.npz", time=np.linspace(0, 1.3, 51), body=np.array([0]))
        (folder / "frames").mkdir()
        for index in range(51):
            (folder / "frames" / f"frame_{index:05d}.png").write_bytes(b"PNG package fixture")
    for name in engineering.PAIR_FILES:
        (tmp_path / cases[0]["root"] / name).write_text(preset + name, encoding="utf-8")
    engineering.save(engineering.LOGS / "numerical-source.json", {})
    engineering.save(engineering.LOGS / "paraview-install.json", {})
    engineering.package(cases, {"preset": preset, "completed": True, "production_eligible": True, "strict_review_passed": True})
    main = engineering.DELIVERY / "main"
    for name in engineering.PAIR_FILES:
        assert (main / cases[0]["root"] / name).read_text() == preset + name
    manifest = json.loads((main / "delivery-manifest.json").read_text())
    assert all(cases[0]["root"] + "/" + name in manifest["files"] for name in engineering.PAIR_FILES)
