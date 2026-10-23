"""Synthetic saved states exercise the independent symmetry gates; no solver run."""

import ast
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import review_material_engineering as reviewer
from tools import run_engineering_preview as runner


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConfigFixture:
    boundaries: tuple
    n_grid: int = 64
    domain_size: float = .4
    dt: float = .0015 / 10500
    dim: int = 3
    boundary_cells: int = 0
    gravity: tuple = (0., 0., 0.)
    material: str = "hjc"
    failure_mode: str = "retain"

    @property
    def dx(self):
        return self.domain_size / self.n_grid


@pytest.fixture(scope="module")
def constructed():
    spec = importlib.util.spec_from_file_location("symmetry_boundary_test_helpers", ROOT / "tests/test_concrete_bending_symmetric.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    env = helpers.source_functions.__wrapped__()
    original, _ = helpers.boundaries(env, 64)
    config = ConfigFixture(original)
    lower, upper = np.array([.05, .15, .17]), np.array([.35, .21, .23])
    counts = np.ceil((upper - lower) / config.dx * 3).astype(int)
    axes = [lower[a] + (np.arange(counts[a]) + .5) * (upper[a] - lower[a]) / counts[a] for a in range(3)]
    reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    volume = np.full(len(reference), np.prod(upper - lower) / len(reference))
    state = SimpleNamespace(x=reference, v=np.zeros_like(reference), mass=2440 * volume, volume=volume,
                            history=SimpleNamespace(damage=np.zeros(len(reference)), failed=np.zeros(len(reference), dtype=bool), plastic_strain=np.zeros(len(reference))))
    definition = {
        "purpose": "concrete-bending-large", "lower": lower.tolist(), "upper": upper.tolist(),
        "duration_s": .0012 + .0003, "particles_per_cell_per_axis": 3,
        "material_profile": "strain-floor", "material_parameters_SI": reviewer.HJC_PARAMETERS.copy(),
        "loading_revision": "centered-grid-patch-v2", "loading_duration_s": .0012,
        "hold_duration_s": .0003, "startup_ramp_s": .0003, "stopping_ramp_s": .0003,
        "final_prescribed_displacement_m": .012,
        "loading_boundary": {key: value for key, value in asdict(original[-1]).items() if key != "end"},
        "support_centers_x_m": [.075, .325], "support_patch_width_m": .01875,
    }
    base = SimpleNamespace(states=(state,), configs=(config,), materials=(reviewer.HJC_PARAMETERS,),
                           duration=.0015, description=definition)
    env.update(concrete_bending_large=lambda **kwargs: base,
               Case=lambda states, materials, configs, duration, description, **kwargs:
                   SimpleNamespace(states=states, materials=materials, configs=configs, duration=duration, description=description, **kwargs),
               _reactions=lambda current, material, cfg, names: {f"{name}_reaction_{axis}_N": 0. for name in names for axis in "xyz"})
    tree = ast.parse((ROOT / "examples/brittle_engineering.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in ("_damage_observables", "concrete_bending_symmetric")]
    env.update(asdict=asdict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual-symmetric-case>", "exec"), env)
    case = env["concrete_bending_symmetric"]()
    return case, state


def data(constructed, *, damage=None, displacement=None):
    case, state = constructed
    count = len(state.x)
    d = np.zeros(count) if damage is None else damage
    x = state.x if displacement is None else state.x + displacement
    current = SimpleNamespace(**vars(state))
    current.x = x
    current.history = SimpleNamespace(**vars(state.history))
    current.history.damage = d
    observed = case.observer((current,), 0.)
    time = np.linspace(0., .0015, 61)
    observations = []
    for t in time:
        shift, speed = case.configs[0].boundaries[-1].motion(t)
        observations.append({**observed, "time": float(t), "boundary_sample_time_s": float(t + case.configs[0].dt),
                             "prescribed_displacement_y_m": float(shift[1]), "prescribed_speed_y_m_per_s": float(speed[1])})
    # Initial reference remains unchanged. Other saved states are synthetic.
    positions = np.broadcast_to(x, (61, count, 3))
    if displacement is not None:
        positions = positions.copy()
        positions[0] = state.x
        initial = case.observer((state,), 0.)
        observations[0].update(initial)
    final = {"x": x, "history_damage": d, "mass": state.mass, "volume": state.volume, "v": state.v}
    config = asdict(case.configs[0])
    for boundary in config["boundaries"]:
        boundary["end"] = None
    definition, config = json.loads(json.dumps([case.description, config]))
    return definition, config, time, observations, final, positions, np.broadcast_to(d, (61, count)), reviewer.HJC_PARAMETERS.copy()


def checked(arguments):
    result = {"checks": {}}
    reviewer.check_hjc_symmetric(result, *arguments)
    return result


def test_current_case_matches_independent_bvp_and_saved_statistics(constructed):
    result = checked(data(constructed))
    assert all(result["checks"].values()), result["checks"]
    report = result["runtime_symmetry"]
    assert report["passed"] is True and len(report["history"]) == 61
    assert report["boundary_audit"]["node_planes"] == {"x": [32], "z": [32]}
    assert report["thresholds"] == {"maximum_saved_damage_mirror_rms": .01, "maximum_saved_left_right_damaged_fraction_difference": .01}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("alteration,expected_failed", [
    ("left-pin", "symmetric_complete_grid_operator_reflection"),
    ("thick-plane", "symmetric_one_middle_node_plane"),
    ("old-endpoint", "symmetric_exact_boundary_geometry_components_and_motion"),
    ("wrong-loader", "symmetric_five_centered_loading_nodes"),
    ("one-side-anchor", "symmetric_complete_grid_operator_reflection"),
])
def test_boundary_defects_are_rejected_even_when_metadata_is_updated(constructed, alteration, expected_failed):
    definition, config, *_ = data(constructed)
    boundary = config["boundaries"]
    if alteration == "left-pin":
        boundary[0]["components"][0] = True
    elif alteration == "thick-plane":
        boundary[2]["lower"][0] = .2 - 1.01 * .4 / 64
        boundary[2]["upper"][0] = .2 + 1.01 * .4 / 64
    elif alteration == "old-endpoint":
        boundary[0]["upper"][0] = .075 + .009375
    elif alteration == "wrong-loader":
        boundary[-1]["upper"][0] = .2125
    else:
        boundary[3]["upper"][0] = .1
    definition["boundary_definitions"] = [{k: v for k, v in item.items() if k != "end"} for item in boundary]
    checks, report = reviewer.hjc_symmetric_boundaries(definition, config)
    assert not checks[expected_failed]
    assert not all(checks.values())


@pytest.mark.parametrize("axis", [0, 2])
def test_unmodified_observer_does_not_hide_true_damage_asymmetry(constructed, axis):
    reference = constructed[1].x
    damage = np.where(reference[:, axis] < .2 - 1e-12, .8, .2)
    result = checked(data(constructed, damage=damage))
    assert result["checks"]["symmetric_all_saved_statistics_match_observer"]
    assert not result["checks"][f"symmetric_runtime_damage_{'xz'[axis // 2]}_rms_within_one_percent"]
    if axis == 0:
        assert not result["checks"]["symmetric_runtime_half_damage_fraction_within_one_percent"]
    assert not result["runtime_symmetry"]["passed"]


def test_false_observer_symmetry_is_rejected(constructed):
    args = list(data(constructed))
    args[3][-1]["damage_mirror_x_rms"] = .2
    result = checked(args)
    assert not result["checks"]["symmetric_all_saved_statistics_match_observer"]
    assert not result["checks"]["symmetric_final_state_statistics_match_observer"]


@pytest.mark.parametrize("difference,accepted", [(.009, True), (.010001, False)])
def test_damage_rms_gate_keeps_its_predeclared_limit(constructed, difference, accepted):
    reference = constructed[1].x
    damage = np.where(reference[:, 0] < .2, .2 + difference, .2)
    result = checked(data(constructed, damage=damage))
    assert result["checks"]["symmetric_runtime_damage_x_rms_within_one_percent"] is accepted
    assert result["checks"]["symmetric_runtime_half_damage_fraction_within_one_percent"]


def test_damage_half_fraction_has_an_independent_limit(constructed):
    damage = np.where(constructed[1].x[:, 0] < .2, .500001, .499999)
    result = checked(data(constructed, damage=damage))
    assert result["checks"]["symmetric_runtime_damage_x_rms_within_one_percent"]
    assert not result["checks"]["symmetric_runtime_half_damage_fraction_within_one_percent"]


def test_intermediate_asymmetry_cannot_be_hidden_by_a_symmetric_final_state(constructed):
    args = list(data(constructed))
    args[6] = args[6].copy()
    args[6][30, constructed[1].x[:, 0] < .2] = .5
    result = checked(args)
    assert not result["checks"]["symmetric_runtime_damage_x_rms_within_one_percent"]
    assert result["runtime_symmetry"]["final_state_metrics"]["damage_mirror_x_rms"] == 0.


def test_reference_material_and_mass_changes_are_rejected(constructed):
    args = list(data(constructed))
    args[-1]["D1"] *= 2
    args[4] = {**args[4], "mass": args[4]["mass"] * 2}
    args[5] = args[5].copy()
    args[5][0, 0, 0] += 1e-5
    result = checked(args)
    assert not result["checks"]["symmetric_fixed_material"]
    assert not result["checks"]["symmetric_initial_reference_lattice"]
    assert not result["runtime_symmetry"]["passed"]


def test_reflected_displacement_and_com_are_recomputed(constructed):
    reference = constructed[1].x
    displacement = np.zeros_like(reference)
    displacement[:, 1] = -.01 * np.exp(-((reference[:, 0] - .2) / .03)**2)
    result = checked(data(constructed, displacement=displacement))
    assert result["checks"]["symmetric_all_saved_statistics_match_observer"]
    assert result["checks"]["symmetric_final_state_statistics_match_observer"]
    final = result["runtime_symmetry"]["final_state_metrics"]
    assert final["displacement_mirror_x_rms_m"] < 1e-15
    assert final["center_of_mass_displacement_y_m"] < 0


def test_cloud_preset_is_independent_and_preserves_large_plan():
    symmetric, large = runner.plan("hjc-symmetric")[0], runner.plan("hjc-large")[0]
    assert symmetric["folder"] == "outputs/material-engineering/concrete-bending-symmetric/hjc"
    assert symmetric["expected_steps"] == 10500
    assert symmetric["expected_supports_profile"] == "mirror-rollers-with-centre-planes"
    for key in ("grid", "ppc", "frames", "particles", "duration", "field", "review"):
        assert symmetric[key] == large[key]
    workflow = (ROOT / ".github/workflows/engineering-previews.yml").read_text(encoding="utf-8")
    assert "hjc-symmetric" in workflow
    matrix = next(line for line in workflow.splitlines() if "fromJSON" in line)
    assert '"hjc-symmetric"' not in matrix


def test_renderer_uses_three_physical_fixtures_only(constructed):
    tree = ast.parse((ROOT / "mpm_jax/paraview_scene.py").read_text(encoding="utf-8"))
    node = next(item for item in tree.body if getattr(item, "name", None) == "velocity_patch_fixtures")
    env = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual-fixture-renderer>", "exec"), env)
    fixtures = env["velocity_patch_fixtures"](constructed[0].description)
    assert len(fixtures) == 3
    assert [item["role"] for item in fixtures] == ["support", "support", "loading"]


def test_failed_review_still_writes_a_bound_runtime_report(tmp_path, monkeypatch):
    folder = tmp_path / "concrete-bending-symmetric/hjc"
    folder.mkdir(parents=True)
    runtime = {"passed": False, "checks": {"symmetric_runtime_damage_x_rms_within_one_percent": False},
               "sources": {"trajectory_damage.npy": "a" * 64}, "review_source_sha256": "b" * 64}
    record = {"passed": False, "checks": dict(runtime["checks"]), "runtime_symmetry": runtime}
    monkeypatch.setattr(reviewer, "review", lambda *args: record)
    monkeypatch.setattr("sys.argv", ["review", "--root", str(tmp_path), "--case", "concrete-bending-symmetric"])
    with pytest.raises(SystemExit) as error:
        reviewer.main()
    assert error.value.code == 1
    saved = json.loads((folder / "interpretation.json").read_text())
    assert saved["runtime_symmetry_report"] == {"file": "runtime-symmetry.json", "sha256": reviewer.binary_digest(folder / "runtime-symmetry.json")}
    assert "interpretation.json" not in runtime["sources"]
    assert json.loads((folder / "runtime-symmetry.json").read_text()) == runtime


def test_symmetric_success_and_failure_packages_keep_the_runtime_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "LOGS", tmp_path / ".local/engineering-ci")
    monkeypatch.setattr(runner, "DELIVERY", tmp_path / ".local/engineering-delivery")
    monkeypatch.setattr(runner.shared, "identity", lambda: {"run_id": "fixture"})
    cases = runner.plan("hjc-symmetric")
    case = cases[0]
    folder = tmp_path / case["folder"]
    folder.mkdir(parents=True)
    for name in ("summary.json", "status.json", "observables.csv", "diagnostics.csv", "interpretation.json", "memory.json", "numerical-source.json",
                 "particles.pvd", "vtk_metadata.json", "render.json", "video.json", "vtk_check.json", "media-check.json", "scene.pvsm",
                 "final_state_0.npz", "final.png", "simulation.gif", "simulation.mp4", "runtime-symmetry.json"):
        (folder / name).write_bytes(name.encode())
    np.savez(folder / "trajectory.npz", time=np.linspace(0, .0015, 61), body=np.array([0]))
    (folder / "frames").mkdir()
    for index in range(61):
        (folder / "frames" / f"frame_{index:05d}.png").write_bytes(b"package protocol fixture")
    runner.save(runner.LOGS / "numerical-source.json", {})
    runner.save(runner.LOGS / "paraview-install.json", {})
    runner.package(cases, {"preset": "hjc-symmetric", "completed": True, "production_eligible": True, "strict_review_passed": True})
    manifest = json.loads((runner.DELIVERY / "main/delivery-manifest.json").read_text())
    assert case["folder"] + "/runtime-symmetry.json" in manifest["files"]
    runner.preserve_failure("hjc-symmetric", ValueError("symmetry gate fixture"))
    assert (runner.LOGS / "failed-cases/concrete-bending-symmetric/runtime-symmetry.json").read_bytes() == b"runtime-symmetry.json"
