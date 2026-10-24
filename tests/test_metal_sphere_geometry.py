import numpy as np
import pytest

from examples.metal_sphere_geometry import plan_case, sample_sphere


def test_default_5000_m_per_s_case_has_two_deformable_volume_sampled_bodies():
    arrays, plan = plan_case()
    sphere, plate = arrays['body_id'] == 0, arrays['body_id'] == 1
    assert sphere.sum() > 8000 and plate.sum() == 400000
    np.testing.assert_array_equal(np.unique(arrays['body_id']), [0, 1])
    np.testing.assert_array_equal(arrays['v'][sphere], np.broadcast_to([0., -5000., 0.], (sphere.sum(), 3)))
    np.testing.assert_array_equal(arrays['v'][plate], 0.)
    np.testing.assert_allclose(plan['initial_kinetic_energy_J'], .5*plan['sphere_mass_kg']*5000**2, rtol=2e-15)
    assert plan['boundary']['backing'] is False and plan['boundary']['clamp_width_m'] == .001
    assert plan['sphere']['diameter_m'] == .002 and plan['plate']['thickness_m'] == .0005
    assert plan['speed_m_s'] == 5000 and plan['end_time_s'] == 1e-6
    assert abs(plan['sphere']['volume_relative_error']) < .001
    np.testing.assert_allclose(plan['plate_mass_kg'], 2700*.02*.02*.0005, rtol=3e-15)
    assert arrays['x'][sphere, 1].min()-arrays['x'][plate, 1].max() > plan['initial_physical_gap_m']


def test_sphere_centroid_quadrature_is_symmetric_positive_and_inside_the_actual_sphere():
    sphere = sample_sphere(center=(0., 0., 0.))
    x, volume = sphere['x'], sphere['volume']
    assert np.all(volume > 0)
    assert np.max(np.linalg.norm(x, axis=1)) < .001
    np.testing.assert_allclose((x*volume[:, None]).sum(axis=0)/volume.sum(), 0., atol=1e-18)
    moment = (x*x*volume[:, None]).sum(axis=0)/volume.sum()
    fine = sample_sphere(center=(0., 0., 0.), spacing=(.00005, .000025, .00005))
    fine_moment = (fine['x']**2*fine['volume'][:, None]).sum(axis=0)/fine['volume'].sum()
    exact = .001**2/5
    # Point masses omit within-cell inertia; verify its decay under refinement.
    assert np.max(abs(fine_moment/exact-1)) < .0011
    assert np.max(abs(fine_moment/exact-1)) < .35*np.max(abs(moment/exact-1))
    assert np.unique(volume).size > 1


@pytest.mark.parametrize('options', [dict(speed=0.), dict(speed=np.nan), dict(gap=.0001),
    dict(max_particles=415999), dict(spacing=(.00011,.00005,.0001)), dict(clamp_width=.01),
    dict(frames=1), dict(plate_thickness=0.), dict(spacing=(1e-100,)*3)])
def test_invalid_or_over_budget_impact_geometry_is_refused_without_silent_rescaling(options):
    with pytest.raises(ValueError):
        plan_case(**options)
