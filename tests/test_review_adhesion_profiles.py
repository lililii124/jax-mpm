import copy

import numpy as np
import pytest

from tools.review_adhesion_comparison import lift_hold_path_matches, lift_hold_profile


def definition(version):
    clearance = .055 if version == 2 else .035
    centers = np.full((8, 3), .5)
    centers[:, 1] = .38 + .16 + np.array([.045, .045, -.035, -.035, clearance, clearance, .295, .295])
    result = {
        "preset": "lift-hold" if version == 2 else "lift-hold-v3",
        "comparison_pair": "single-sphere-lift-hold-v1" if version == 2 else "single-sphere-lift-hold-v3",
        "contact_sampling_plane_y_m": .38,
        "rigid_spheres": [{"radius": .16, "times": [0., .08, .28, .42, .72, .94, 1.12, 1.3], "centers": centers.tolist()}],
    }
    if version == 3:
        result.update(loading_revision=3, comparison_hold_bottom_clearance_m=.035)
    return result


@pytest.mark.parametrize("version", [2, 3])
def test_both_profiles_require_their_own_unscaled_physical_motion(version):
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
    altered["comparison_pair"] = "single-sphere-lift-hold-v4"
    with pytest.raises(ValueError, match="profile"):
        lift_hold_profile([altered, altered])
    with pytest.raises(ValueError, match="profile"):
        lift_hold_profile([case, definition(2)])
    assert lift_hold_profile([{"comparison_pair": "single-sphere-normal-pull-off-v1"}]) is None
