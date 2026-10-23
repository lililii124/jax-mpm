import importlib.metadata
import hashlib
import json

import numpy as np
import pytest

from tools import run_engineering_preview as engineering


def configure(monkeypatch, root):
    monkeypatch.setattr(engineering, "ROOT", root)
    monkeypatch.setattr(engineering, "LOGS", root / ".local/engineering-ci")
    monkeypatch.setattr(engineering, "DELIVERY", root / ".local/engineering-delivery")
    monkeypatch.setattr(engineering, "identity", lambda: {"run_id": "fixture"})


def test_jc_formal_preserves_loading_and_increases_resolution():
    diagnostic = engineering.plan("jc-density-diagnostic")[0]
    formal = engineering.plan("jc-necking-formal")[0]
    assert (diagnostic["grid"], diagnostic["ppc"], diagnostic["particles"]) == (64, 2, 33224)
    assert (formal["grid"], formal["ppc"], formal["particles"]) == (100, 2, 102320)
    for name in ("module", "arguments", "duration", "frames", "review", "field"):
        assert formal[name] == diagnostic[name]
    assert formal["arguments"] == ["--end-time", "150e-6", "--cfl", ".15"]
    assert formal["duration"] == 150e-6 and formal["frames"] == 60
    assert formal["folder"] == "outputs/material-engineering/jc-necking/johnson-cook"
    assert formal["root"] != diagnostic["root"]


def jc_fixture(folder, *, cut_frame=None, failure_frame=None, diagnostic=False, passed=True):
    folder.mkdir(parents=True)
    positions = np.zeros((61, 8, 3))
    positions[:, :, 0] = np.repeat([.0047, .0049, .0051, .0053], 2)
    positions[:, :, 1] = np.tile([.0049, .0051], 4)
    positions[:, :, 2] = .005
    damage = np.zeros((61, 8))
    failed = np.zeros((61, 8), dtype=np.uint8)
    if cut_frame is not None:
        positions[cut_frame:, :4, 0] -= .001
        positions[cut_frame:, 4:, 0] += .001
    if failure_frame is not None:
        failed[failure_frame:, 0] = 1
        damage[failure_frame:, 0] = 1
    np.save(folder / "trajectory_x.npy", positions)
    np.save(folder / "trajectory_damage.npy", damage)
    np.save(folder / "trajectory_failed.npy", failed)
    np.savez(folder / "trajectory.npz", time=np.linspace(0, 150e-6, 61))
    np.savez(folder / "final_state_0.npz", x=positions[-1])
    engineering.save(folder / "summary.json", {"completed": True, "configs": [{"domain_size": .01, "n_grid": 100}]})
    engineering.save(folder / ("interpretation.json" if diagnostic else "constitutive-review.json"),
                     {"passed": passed, "checks": {"physics": passed}, "review_source_sha256": "a" * 64})
    (folder / "observables.csv").write_text("fixture\n")
    (folder / "diagnostics.csv").write_text("fixture\n")
    return folder


@pytest.mark.parametrize("diagnostic", [True, False])
def test_jc_audit_serializes_completion_and_binds_actual_saved_arrays(tmp_path, diagnostic):
    folder = jc_fixture(tmp_path / "jc", diagnostic=diagnostic)
    result = engineering.jc_diagnostic(folder, expected_grid=100, expected_particles=8, diagnostic=diagnostic)
    assert type(result["completed"]) is bool and result["completed"] is True
    assert result["diagnostic_only"] is diagnostic
    assert result["failure_disconnection_ordering"] == "no_saved_support_disconnection"
    for name in ("trajectory_x.npy", "trajectory_damage.npy", "trajectory_failed.npy"):
        assert result["sources"][name] == engineering.digest(folder / name)
    filename = "jc-density-diagnostic.json" if diagnostic else "jc-support-audit.json"
    saved = json.loads((folder / filename).read_text())
    assert saved == result
    assert saved["history_csv"]["rows"] == 61


@pytest.mark.parametrize("cut,failure,ordering,accepted", [
    (10, 20, "support_disconnection_observed_before_any_material_failure", False),
    (10, None, "support_disconnection_observed_before_any_material_failure", False),
    (20, 20, "same_saved_interval_unresolved", True),
    (20, 10, "material_failure_observed_at_an_earlier_saved_state", True),
])
def test_jc_ordering_gate_and_one_way_review_hashes(tmp_path, cut, failure, ordering, accepted):
    folder = jc_fixture(tmp_path / "jc", cut_frame=cut, failure_frame=failure)
    original = (folder / "constitutive-review.json").read_bytes()
    audit = engineering.jc_diagnostic(folder, expected_grid=100, expected_particles=8, diagnostic=False)
    assert audit["failure_disconnection_ordering"] == ordering
    assert audit["first_axial_stencil_disconnection"]["frame"] == cut
    acceptance = engineering.accept_jc_support(folder, audit)
    assert acceptance["passed"] is accepted
    final = json.loads((folder / "interpretation.json").read_text())
    assert final["passed"] is accepted
    assert final["checks"]["physics"] is True
    assert (folder / "constitutive-review.json").read_bytes() == original
    assert "interpretation.json" not in audit["sources"]
    assert "interpretation.json" not in acceptance["sources"]
    for name, digest in acceptance["sources"].items():
        assert digest == engineering.digest(folder / name)
    for name, digest in final["supplemental_reviews"].items():
        assert digest == engineering.digest(folder / name)


def test_jc_uses_strictly_positive_weights_not_three_dx_shortcut():
    _, lower, upper, cuts = engineering.axial_support_intervals(np.array([.5, 2.5]), 1.)
    assert lower.tolist() == [0, 2] and upper.tolist() == [1, 3]
    assert cuts.tolist() == [0]
    _, _, _, cuts = engineering.axial_support_intervals(np.array([.5, 2.5 - 1e-12]), 1.)
    assert cuts.size == 0


def test_jc_incomplete_saved_arrays_are_rejected(tmp_path):
    folder = jc_fixture(tmp_path / "jc")
    np.save(folder / "trajectory_failed.npy", np.zeros((60, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="Incomplete"):
        engineering.jc_diagnostic(folder, expected_grid=100, expected_particles=8, diagnostic=False)


@pytest.mark.parametrize("constitutive_passed,support_passed", [(False, True), (True, False), (True, True)])
def test_formal_jc_requires_material_and_support_reviews_before_render(tmp_path, monkeypatch, constitutive_passed, support_passed):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(engineering.shutil, "disk_usage", lambda root: type("Disk", (), {"free": 10 * engineering.GIB})())
    source = {"review_sources": {"tools/review_material_engineering.py": "b" * 64}}
    monkeypatch.setattr(engineering, "source_record", lambda: source)
    case = engineering.plan("jc-necking-formal")[0]
    folder = tmp_path / case["folder"]
    calls = []
    original_review_bytes = json.dumps({"passed": constitutive_passed, "checks": {"physics": constitutive_passed}}, separators=(",", ":")).encode()

    def command(name, arguments, **options):
        calls.append(name)
        if name.startswith("solve-"):
            assert arguments[arguments.index("--grid") + 1] == "100"
            assert arguments[arguments.index("--ppc") + 1] == "2"
            assert arguments[arguments.index("--end-time") + 1] == "150e-6"
            engineering.save(folder / "summary.json", {
                "completed": True, "particles": [102320], "simulation_time": 150e-6,
                "steps": 57900, "solver_seconds": 1., "configs": [{"n_grid": 100, "dt": 150e-6 / 57900}],
                "definition": {"requested_particles_per_cell_per_axis": 2, "acoustic_CFL_requested": .15}})
            engineering.save(folder / "status.json", {"completed": True, "frame": 60, "frames": 60})
        else:
            (folder / "interpretation.json").write_bytes(original_review_bytes)
        return {"memory": {}}

    def audit(directory, **parameters):
        assert parameters == {"expected_grid": 100, "expected_particles": 102320, "diagnostic": False}
        assert directory == folder and (folder / "constitutive-review.json").is_file()
        record = {"completed": True, "support_disconnection_before_material_failure": not support_passed,
                  "support_disconnection_with_no_failed_material_observed": not support_passed,
                  "failure_disconnection_ordering": "fixture"}
        engineering.save(folder / "jc-support-audit.json", record)
        (folder / "jc-support-history.csv").write_text("fixture\n")
        calls.append("audit")
        return record

    monkeypatch.setattr(engineering, "command", command)
    monkeypatch.setattr(engineering, "jc_diagnostic", audit)
    monkeypatch.setattr(engineering, "render", lambda *args: calls.append("render"))
    monkeypatch.setattr(engineering, "package", lambda *args, **kwargs: calls.append("package"))
    if constitutive_passed and support_passed:
        engineering.run("jc-necking-formal")
        assert calls[-3:] == ["audit", "render", "package"]
    else:
        with pytest.raises(ValueError, match="strict numerical review"):
            engineering.run("jc-necking-formal")
        assert "audit" in calls and not {"render", "package"}.intersection(calls)
    assert json.loads((folder / "constitutive-review.json").read_text())["passed"] is constitutive_passed
    assert (folder / "constitutive-review.json").read_bytes() == original_review_bytes
    assert json.loads((folder / "interpretation.json").read_text())["review_source_sha256"] == "b" * 64


@pytest.mark.parametrize("limited", [False, True])
def test_failure_keeps_raw_state_and_trace_with_a_bounded_manifest(tmp_path, monkeypatch, limited):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(engineering, "LIMIT", 300000 if limited else 2 * 1024**2)
    case = engineering.plan("jc-necking-formal")[0]
    folder = tmp_path / case["folder"]
    folder.mkdir(parents=True)
    # Byte fixtures exercise bounded copying; the helper never claims to validate NPZ contents.
    (folder / "final_state_0.npz").write_bytes(b"state" * 200)
    (folder / "trajectory.npz").write_bytes(b"trajectory" * 20000)
    engineering.save(folder / "interpretation.json", {"passed": False})
    engineering.preserve_failure("jc-necking-formal", TypeError("bool is not JSON serializable"))
    index = json.loads((engineering.LOGS / "failed-data-index.json").read_text())
    state, trace = index["files"]
    assert state["preserved"] is True
    assert trace["preserved"] is (not limited)
    for record in index["files"]:
        data = b""
        for chunk in record["chunks"]:
            payload = (engineering.LOGS / chunk["path"]).read_bytes()
            assert len(payload) == chunk["bytes"]
            assert hashlib.sha256(payload).hexdigest() == chunk["sha256"]
            data += payload
        original = (tmp_path / record["source"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(original).hexdigest()
        assert data == original[:record["copied_bytes"]]
        if record["preserved"]:
            assert data == original
        else:
            assert "not a usable NPZ" in record["reason"]
    assert sum(path.stat().st_size for path in engineering.LOGS.rglob("*") if path.is_file()) <= engineering.LIMIT


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


def test_v4_has_a_separate_root_and_fixed_work_without_changing_v3_plan():
    v3, v4 = [engineering.plan(name) for name in ("adhesion-v3-pair", "adhesion-v4-pair")]
    for old, new in zip(v3, v4):
        for key in ("name", "module", "grid", "ppc", "frames", "particles", "duration", "field", "review"):
            assert old[key] == new[key]
        assert old["root"] == "outputs/adhesion-visible-v3"
        assert new["root"] == "outputs/adhesion-visible-v4"
        assert new["folder"] == new["root"] + "/" + new["name"]
        assert new["expected_preset"] == "lift-hold-v4"
        assert new["expected_work"] == 2 * old["expected_work"]
        assert new["arguments"] == ["--case", new["name"], "--profile", "v4"]
    assert [case["expected_work"] for case in v4] == [0., 144.]


@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("failure", ["individual", "pair", None])
def test_versioned_pair_requires_all_reviews_before_media(tmp_path, monkeypatch, failure, version):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(engineering.shutil, "disk_usage", lambda root: type("Disk", (), {"free": 10 * engineering.GIB})())
    source = {"review_sources": {"tools/review_adhesion.py": "a" * 64}}
    monkeypatch.setattr(engineering, "source_record", lambda: source)
    preset = f"adhesion-v{version}-pair"
    cases = engineering.plan(preset)
    calls = []

    def command(name, arguments, **options):
        calls.append(name)
        if name.startswith("solve-"):
            case = next(case for case in cases if name == "solve-" + case["name"])
            assert arguments[arguments.index("--profile") + 1] == f"v{version}"
            assert arguments[arguments.index("--output") + 1] == case["root"]
            folder = tmp_path / case["folder"]
            engineering.save(folder / "summary.json", {
                "completed": True, "particles": [case["particles"]], "simulation_time": 1.3,
                "steps": 3700, "solver_seconds": .1, "configs": [{"n_grid": 48, "dt": 1.3 / 3700}],
                "definition": {"preset": f"lift-hold-v{version}", "particles_per_cell_per_axis": 3,
                    "young_modulus_Pa": 8000, "adhesion": {"work_of_adhesion": case["expected_work"],
                        "interaction_range": .025, "penalty_stiffness": 4.8e6}}})
            engineering.save(folder / "status.json", {"completed": True, "frame": 50, "frames": 50})
        elif name == "review-pair":
            assert arguments[-1] == f"outputs/adhesion-visible-v{version}"
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
            engineering.run(preset)
        assert not {"export", "render", "package"}.intersection(calls)
        if failure == "individual":
            assert "review-pair" not in calls
    else:
        engineering.run(preset)
        assert calls[-4:] == ["export", "render", "render", "package"]
        assert (tmp_path / f"outputs/adhesion-visible-v{version}/paired-render.json").is_file()
        assert not (tmp_path / "outputs/adhesion-visible").exists()


@pytest.mark.parametrize("version", [3, 4])
def test_failure_preserves_pair_evidence_and_previous_failures(tmp_path, monkeypatch, version):
    configure(monkeypatch, tmp_path)
    preset = f"adhesion-v{version}-pair"
    cases = engineering.plan(preset)
    for case in cases:
        engineering.save(tmp_path / case["folder"] / "interpretation.json", {"passed": False})
    root = tmp_path / cases[0]["root"]
    for name in engineering.PAIR_FILES:
        (root / name).write_text("first failed comparison " + name, encoding="utf-8")
    engineering.preserve_failure(preset, ValueError("first"))
    original = {path.relative_to(engineering.LOGS): path.read_bytes()
                for path in engineering.LOGS.rglob("*") if path.is_file()}
    saved = engineering.LOGS / f"failed-comparisons/adhesion-visible-v{version}"
    assert all((saved / name).read_bytes() == (root / name).read_bytes() for name in engineering.PAIR_FILES)
    (root / "adhesion_comparison.json").write_text("second failure", encoding="utf-8")
    engineering.preserve_failure(preset, ValueError("second"))
    assert all((engineering.LOGS / path).read_bytes() == data for path, data in original.items())
    history = list((engineering.LOGS / "failure-history").iterdir())
    assert len(history) == 1
    assert (history[0] / f"failed-comparisons/adhesion-visible-v{version}/adhesion_comparison.json").read_text() == "second failure"


@pytest.mark.parametrize("preset", ["adhesion-v2-pair", "adhesion-v3-pair", "adhesion-v4-pair"])
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


def test_formal_jc_package_contains_both_reviews_and_support_history(tmp_path, monkeypatch):
    configure(monkeypatch, tmp_path)
    monkeypatch.setattr(engineering.shared, "identity", lambda: {"run_id": "fixture"})
    cases = engineering.plan("jc-necking-formal")
    case = cases[0]
    folder = tmp_path / case["folder"]
    folder.mkdir(parents=True)
    evidence = ("constitutive-review.json", "jc-support-audit.json", "jc-support-history.csv", "runner-acceptance.json")
    for name in ("summary.json", "status.json", "observables.csv", "diagnostics.csv", "interpretation.json", "memory.json", "numerical-source.json",
                 "particles.pvd", "vtk_metadata.json", "render.json", "video.json", "vtk_check.json", "media-check.json", "scene.pvsm",
                 "final_state_0.npz", "final.png", "simulation.gif", "simulation.mp4", *evidence):
        (folder / name).write_bytes(name.encode())
    np.savez(folder / "trajectory.npz", time=np.linspace(0, 150e-6, 61), body=np.array([0]))
    (folder / "frames").mkdir()
    for frame in range(61):
        (folder / "frames" / f"frame_{frame:05d}.png").write_bytes(b"PNG package fixture")
    engineering.save(engineering.LOGS / "numerical-source.json", {})
    engineering.save(engineering.LOGS / "paraview-install.json", {})
    engineering.package(cases, {"preset": "jc-necking-formal", "completed": True,
                               "production_eligible": True, "strict_review_passed": True})
    manifest = json.loads((engineering.DELIVERY / "main/delivery-manifest.json").read_text())
    assert all(case["folder"] + "/" + name in manifest["files"] for name in evidence)
    assert manifest["diagnostic_only"] is False and manifest["production_eligible"] is True
    assert (engineering.DELIVERY / "final-state" / case["folder"] / "final_state_0.npz").is_file()
    assert len(list((engineering.DELIVERY / "media" / case["folder"] / "frames").glob("*.png"))) == 61
