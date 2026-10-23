import copy

import numpy as np
import pytest

from tools import review_adhesion_comparison as comparison
from tools.review_adhesion_comparison import lift_hold_path_matches, lift_hold_profile


def definition(version):
    clearance = .055 if version == 2 else .035
    centers = np.full((8, 3), .5)
    centers[:, 1] = .38 + .16 + np.array([.045, .045, -.035, -.035, clearance, clearance, .295, .295])
    result = {
        "preset": "lift-hold" if version == 2 else f"lift-hold-v{version}",
        "comparison_pair": "single-sphere-lift-hold-v1" if version == 2 else f"single-sphere-lift-hold-v{version}",
        "contact_sampling_plane_y_m": .38,
        "rigid_spheres": [{"radius": .16, "times": [0., .08, .28, .42, .72, .94, 1.12, 1.3], "centers": centers.tolist()}],
    }
    if version != 2:
        result.update(loading_revision=version, comparison_hold_bottom_clearance_m=.035)
    return result


@pytest.mark.parametrize("version", [2, 3, 4])
def test_profiles_require_their_own_unscaled_physical_motion(version):
    case = definition(version)
    profile = lift_hold_profile([case, copy.deepcopy(case)])
    assert lift_hold_path_matches(case, profile)
    changed = copy.deepcopy(case)
    changed["rigid_spheres"][0]["centers"][4][1] += .02
    assert not lift_hold_path_matches(changed, profile)
    changed = copy.deepcopy(case)
    changed["rigid_spheres"][0]["times"][5] = .85
    assert not lift_hold_path_matches(changed, profile)


def test_unknown_or_mixed_profiles_cannot_skip_the_hold_contrast_gate():
    case = definition(3)
    altered = copy.deepcopy(case)
    altered["comparison_pair"] = "single-sphere-lift-hold-v5"
    with pytest.raises(ValueError, match="profile"):
        lift_hold_profile([altered, altered])
    with pytest.raises(ValueError, match="profile"):
        lift_hold_profile([case, definition(2)])
    with pytest.raises(ValueError, match="profile"):
        lift_hold_profile([case, definition(4)])
    assert lift_hold_profile([{"comparison_pair": "single-sphere-normal-pull-off-v1"}]) is None


def controlled_records(root, version=4, work=144.0, uplift=.025, last_visible=.936):
    """Synthetic review inputs exercise gates, not numerical physics."""
    result = {}
    times = np.linspace(0., 1.3, 51)
    for role, name in comparison.CASES.items():
        folder = root / name
        folder.mkdir()
        np.save(folder / "trajectory_x.npy", np.zeros((51, 1, 3)))
        case = definition(version)
        case.update(series="adhesion-suite-v1", suite_case=name, comparison_role=role,
                    geometry_scale=1., adhesion={"work_of_adhesion": 0. if role == "nonadhesive" else work})
        rows = []
        for time in times:
            holding = .72 <= time <= .94
            active = role == "adhesive" and holding and time <= last_visible + 1e-12
            row = {"time": float(time), "surface_displacement_y_min_m": 0.,
                   "surface_displacement_y_max_m": uplift if active else 0.}
            for key in ("reaction_y_N", "velocity_y_m_per_s", "compressed_points", "adhesive_points",
                        "actuator_power_W", "contact_reference_area_m2", "adhesive_reference_area_m2",
                        "min_gap_m", "potential_J", "deformation_feedback_force_difference_N"):
                row["sphere_0_" + key] = 0.
            row.update(sphere_0_reaction_y_N=-1. if active else 0., sphere_0_adhesive_points=int(active))
            for axis in "xyz":
                row["sphere_0_center_" + axis + "_m"] = .5
                row["sphere_0_velocity_" + axis + "_m_per_s"] = 0.
            rows.append(row)
        result[folder] = {"folder": folder, "summary": {"definition": case, "configs": [],
            "materials": [], "steps": 3700, "simulation_time": 1.3, "particles": [1],
            "dtype": "float64", "jax_version": "protocol-fixture", "initial": {}},
            "status": {"frames": 50, "frame": 50, "steps_completed": 3700, "time": 1.3},
            "rows": rows, "hashes": {}, "review": {"passed": True}}
    return result


@pytest.mark.parametrize("version,work", [(2, 72.), (3, 72.), (4, 144.)])
def test_fixed_profile_work_and_original_hold_thresholds(tmp_path, monkeypatch, version, work):
    records = controlled_records(tmp_path, version=version, work=work)
    monkeypatch.setattr(comparison, "read_run", records.__getitem__)
    report, _ = comparison.review(tmp_path)
    assert report["passed"] is True
    assert report["checks"]["lift_hold_fixed_work_of_adhesion"] is True
    assert report["deformation_contrast"]["minimum_uplift_difference_m"] == .025
    assert report["deformation_contrast"]["required_saved_span_s"] == .10


@pytest.mark.parametrize("problem", ["wrong-work", "short-span", "insufficient-uplift", "wrong-revision"])
def test_v4_candidate_must_satisfy_unchanged_gates(tmp_path, monkeypatch, problem):
    records = controlled_records(tmp_path, work=72. if problem == "wrong-work" else 144.,
        uplift=.024999 if problem == "insufficient-uplift" else .025,
        last_visible=.806 if problem == "short-span" else .936)
    if problem == "wrong-revision":
        for run in records.values():
            run["summary"]["definition"]["loading_revision"] = 3
    monkeypatch.setattr(comparison, "read_run", records.__getitem__)
    report, _ = comparison.review(tmp_path)
    assert report["passed"] is False
    key = ("lift_hold_fixed_work_of_adhesion" if problem == "wrong-work" else
           "lift_hold_loading_revision_identified" if problem == "wrong-revision" else
           "lift_hold_has_visible_deformation_contrast")
    assert report["checks"][key] is False
