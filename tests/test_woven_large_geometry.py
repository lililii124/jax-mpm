import numpy as np
import pytest

from examples.woven_geometry import ARCHITECTURES, build_paths, sample_weave
from examples.woven_large_geometry import (
    audit_periodic_geometry, check_translation, sample_large_weave, unit_cell,
)


PARAMETERS = dict(size=(.01, .01), pitch=.0025, yarn_width=.000625,
                  yarn_height=.0005, gap=.0005, matrix_cover=.00025)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_anisotropic_points_fill_the_same_physical_plate(architecture):
    sample = sample_large_weave(architecture, **PARAMETERS)
    assert len(sample.positions) == 20*8*20
    assert sample.positions.dtype == sample.volume.dtype == sample.orientation.dtype == np.float64
    np.testing.assert_allclose(sample.volume, .0005*.00025*.0005, rtol=0, atol=0)
    np.testing.assert_allclose(sample.volume.sum(), .01*.002*.01, rtol=2e-15)
    for axis, spacing in enumerate((.0005, .00025, .0005)):
        np.testing.assert_allclose(np.diff(np.unique(sample.positions[:, axis])), spacing, atol=2e-18)
    np.testing.assert_allclose(np.swapaxes(sample.orientation, 1, 2)@sample.orientation,
                               np.broadcast_to(np.eye(3), sample.orientation.shape), atol=2e-14)
    np.testing.assert_allclose(np.linalg.det(sample.orientation), 1., atol=2e-14)
    assert sample.metadata["periodic_geometry_audit"]["passed"]
    assert sample.metadata["sampled_yarn_intersections"] == 0
    assert set(sample.family_id[sample.phase == 1]) == set(range(3 if architecture == "triaxial" else 2))
    assert np.all(sample.family_id[sample.phase == 0] == -1)
    assert np.all(sample.yarn_id[sample.phase == 0] == -1)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_strip_pruning_retains_old_isotropic_phase_and_orientation(architecture):
    old = sample_weave(architecture, **PARAMETERS, spacing=.00025)
    new = sample_large_weave(architecture, **PARAMETERS, spacing=(.00025,)*3)
    for name in ("positions", "volume", "phase", "family_id", "yarn_id"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    np.testing.assert_allclose(old.orientation, new.orientation, rtol=0, atol=3e-15)


def test_chunking_and_origin_do_not_change_phase_ownership():
    small = sample_large_weave("triaxial", **PARAMETERS, chunk_size=7)
    shift = np.array((.11, .02, -.03))
    translated = sample_large_weave("triaxial", **PARAMETERS, origin=shift)
    np.testing.assert_allclose(translated.positions-small.positions, np.broadcast_to(shift, small.positions.shape))
    for name in ("volume", "phase", "family_id", "yarn_id", "orientation"):
        np.testing.assert_array_equal(getattr(small, name), getattr(translated, name))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_true_periods_repeat_and_pitch_is_not_substituted(architecture):
    cell = unit_cell(architecture, .0025)
    audit = audit_periodic_geometry(architecture)
    assert audit["passed"]
    paths, _, _ = build_paths(architecture, size=(.04, .04), pitch=.0025, amplitude=.0005)
    nonperiod = (.005, 0) if architecture == "twill-2x2" else (.0025, 0)
    assert not check_translation(paths, nonperiod, .0025, .0005)["passed"]
    assert .1/max(cell["orthogonal_repeating_cell_lengths_xz_m"]) >= 10


def test_default_large_budget_is_checked_before_geometry_or_grid_allocation(monkeypatch):
    import examples.woven_large_geometry as module
    monkeypatch.setattr(module, "audit_periodic_geometry", lambda *a, **k: pytest.fail("budget must be checked first"))
    with pytest.raises(ValueError, match="819200 exceeds budget"):
        module.sample_large_weave("plain", max_particles=819199)


@pytest.mark.parametrize("options", [
    {"spacing": (.00051, .00025, .0005)}, {"spacing": (.0005, 0., .0005)},
    {"spacing": .0005}, {"origin": (np.nan, 0., 0.)}, {"chunk_size": False},
    {"max_particles": 0}, {"yarn_width": .0025},
])
def test_invalid_sampling_is_rejected_without_adjustment(options):
    with pytest.raises(ValueError):
        sample_large_weave("plain", **options)


def test_overlap_cannot_be_assigned_by_priority():
    with pytest.raises(ValueError, match="geometry"):
        sample_large_weave("triaxial", size=(.01, .01), yarn_width=.0009375,
                           yarn_height=.000625, gap=.0003125, matrix_cover=.00046875,
                           spacing=(.00025,)*3)
