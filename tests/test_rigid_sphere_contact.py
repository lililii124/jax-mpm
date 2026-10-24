import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.rigid_sphere_contact import SphereState, project_sphere_contact

jax.config.update('jax_enable_x64', True)


def project(mass, velocity, coordinates, *, tool_velocity=(0., -1., 0.), tool_mass=2., fixed=None):
    sphere = SphereState(jnp.zeros(3), jnp.array(tool_velocity), jnp.asarray(tool_mass), jnp.asarray(1.))
    result = project_sphere_contact(jnp.array(mass), jnp.array(velocity, dtype=float),
        jnp.array(coordinates, dtype=float), sphere, fixed=None if fixed is None else jnp.asarray(fixed))
    return result, sphere


def test_shared_finite_mass_is_used_once_for_many_contacting_nodes():
    result, sphere = project([1., 2., 3.], np.zeros((3, 3)), [[0., -.4, 0.], [0., -.5, 0.], [0., -.6, 0.]])
    assert bool(result.valid)
    np.testing.assert_allclose(result.sphere_velocity, [0., -.25, 0.], atol=1e-14)
    np.testing.assert_allclose(result.velocity[:, 1], -.25, atol=1e-14)
    np.testing.assert_allclose(result.impulse_on_body, [0., -1.5, 0.], atol=1e-14)
    np.testing.assert_allclose(result.impulse_on_body + sphere.mass*(result.sphere_velocity-sphere.velocity), 0., atol=1e-14)
    np.testing.assert_allclose(result.kinetic_loss, .75, atol=1e-14)


def test_three_dimensional_normals_produce_the_coupled_rigid_translation():
    result, _ = project([1., 2., 3.], np.zeros((3, 3)), [[-.5, 0., 0.], [0., -.5, 0.], [0., 0., -.5]],
        tool_velocity=(-1., -2., -3.), tool_mass=4.)
    assert bool(result.valid)
    np.testing.assert_allclose(result.sphere_velocity, 4*np.array([-1., -2., -3.])/np.array([5., 6., 7.]), atol=1e-14)
    assert float(result.kinetic_loss) > 0 and float(result.normal_velocity_violation) < 1e-13


def test_separating_nodes_and_empty_nodes_receive_no_tensile_impulse():
    result, sphere = project([1., 0.], np.zeros((2, 3)), [[0., -.5, 0.], [0., 0., 0.]], tool_velocity=(0., 1., 0.))
    assert bool(result.valid)
    np.testing.assert_array_equal(result.velocity, 0.)
    np.testing.assert_array_equal(result.sphere_velocity, sphere.velocity)
    assert float(result.kinetic_loss) == 0


def test_oblique_impulses_preserve_tangential_velocity_and_balance_force_and_torque():
    point = np.array([-.3, -.4, 0.])
    old = np.array([[2., -1., 3.]])
    result, sphere = project([1.7], old, [point], tool_velocity=(-3., -2., 0.))
    assert bool(result.valid)
    normal = point/np.linalg.norm(point)
    change = np.asarray(result.velocity[0])-old[0]
    np.testing.assert_allclose(np.cross(change, normal), 0., atol=1e-14)
    np.testing.assert_allclose(np.cross(point, result.impulse_on_body), 0., atol=1e-14)
    old_ke = .5*1.7*np.sum(old**2)+.5*float(sphere.mass)*np.sum(np.asarray(sphere.velocity)**2)
    new_ke = .5*1.7*np.sum(np.asarray(result.velocity)**2)+.5*float(sphere.mass)*np.sum(np.asarray(result.sphere_velocity)**2)
    np.testing.assert_allclose(old_ke-new_ke, result.kinetic_loss, rtol=1e-13)


def test_galilean_shift_preserves_contact_impulses_and_projection_loss():
    old = np.array([[.1, .2, .3], [.2, -.3, -.1]])
    points = [[-.3, -.4, 0.], [.3, -.4, 0.]]
    base, _ = project([.5, .7], old, points)
    shift = np.array([3., -2., 1.])
    boosted, _ = project([.5, .7], old+shift, points, tool_velocity=tuple(shift+[0., -1., 0.]))
    assert bool(base.valid) and bool(boosted.valid)
    np.testing.assert_allclose(boosted.velocity, base.velocity+shift, atol=1e-13)
    np.testing.assert_allclose(boosted.sphere_velocity, base.sphere_velocity+shift, atol=1e-13)
    np.testing.assert_allclose(boosted.impulse_on_body, base.impulse_on_body, atol=1e-13)
    np.testing.assert_allclose(boosted.kinetic_loss, base.kinetic_loss, atol=1e-13)


def test_energy_loss_stays_resolved_for_a_very_heavy_tool():
    result, _ = project([1.], [[0., 0., 0.]], [[0., -.5, 0.]], tool_mass=1e15)
    assert bool(result.valid)
    np.testing.assert_allclose(result.kinetic_loss, .5, rtol=1e-12)


@pytest.mark.parametrize('defect', ['fixed_overlap', 'zero_mass', 'negative_mass', 'center_node'])
def test_invalid_contacts_are_explicitly_rejected(defect):
    values = {'tool_mass': 2., 'fixed': None}
    point = [0., -.5, 0.]
    if defect == 'fixed_overlap':
        values['fixed'] = [True]
    elif defect in ('zero_mass', 'negative_mass'):
        values['tool_mass'] = 0. if defect == 'zero_mass' else -1.
    else:
        point = [0., 0., 0.]
    result, _ = project([1.], [[0., 0., 0.]], [point], **values)
    assert not bool(result.valid)
