import numpy as np
import pytest

from examples.metal_plate_geometry import sample_metal_plate, solid_sphere_mass


def test_declared_metal_plate_is_320000_points_and_two_mm_thick():
    sample = sample_metal_plate()
    assert len(sample.positions) == 320000
    np.testing.assert_allclose(sample.volume.sum(), .1*.1*.002, rtol=2e-15)
    assert sample.metadata["lattice_counts"] == [200, 8, 200]
    assert sample.metadata["thickness_m"] == .002
    assert np.all(sample.phase == 0)
    assert np.all(sample.family_id == -1) and np.all(sample.yarn_id == -1)
    np.testing.assert_allclose(sample.orientation[[0, -1]], np.broadcast_to(np.eye(3), (2, 3, 3)))


def test_translation_does_not_change_mass_measure_or_internal_labels():
    first = sample_metal_plate(size=(.005, .005))
    shift = np.array((.1, -.2, .3))
    second = sample_metal_plate(size=(.005, .005), origin=shift)
    np.testing.assert_allclose(second.positions-first.positions, np.broadcast_to(shift, first.positions.shape))
    np.testing.assert_array_equal(first.volume, second.volume)
    np.testing.assert_array_equal(first.orientation, second.orientation)


def test_twenty_mm_steel_ball_geometric_mass():
    mass = solid_sphere_mass(.02)
    assert mass == pytest.approx(.03279822730347744, rel=2e-15)


@pytest.mark.parametrize("options", [{"max_particles": 319999}, {"spacing": (.00051, .00025, .0005)},
                                     {"thickness": 0.}, {"size": (.1, np.nan)}, {"spacing": (.0005,)*2}])
def test_invalid_or_over_budget_lattice_is_not_silently_changed(options):
    with pytest.raises(ValueError):
        sample_metal_plate(**options)
