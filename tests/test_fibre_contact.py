import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.fibre_contact import (
    contact_potential,
    effective_modulus,
    fibre_contact_parameters,
    initial_fibre_contact_history,
    jkr_critical_state,
    normal_pair_forces,
    update_fibre_contact,
)

jax.config.update("jax_enable_x64", True)


def parameters(**kwargs):
    return fibre_contact_parameters(
        effective_modulus=effective_modulus(10e9, .25, 10e9, .25),
        effective_radius=117.5e-6, **kwargs,
    )


def test_two_material_modulus_and_host_validation():
    value = jax.jit(effective_modulus)(10e9, .25, 20e9, .3)
    np.testing.assert_allclose(value, 1 / (.9375 / 10e9 + .91 / 20e9), rtol=1e-14)
    for field, value in (("effective_modulus", 0), ("effective_radius", -1),
                         ("work_of_adhesion", -.1), ("work_of_adhesion", np.nan)):
        args = dict(effective_modulus=1e9, effective_radius=1e-4, work_of_adhesion=.1)
        args[field] = value
        with pytest.raises(ValueError):
            fibre_contact_parameters(**args)
    with pytest.raises(ValueError):
        parameters(model="md")


def test_hertz_scalar_force_area_and_stored_energy():
    p = parameters(model="hertz")
    delta = 1e-7
    _, response = jax.jit(update_fibre_contact)(initial_fibre_contact_history(), delta, p)
    a = np.sqrt(float(p.effective_radius) * delta)
    force = 4 / 3 * float(p.effective_modulus) * np.sqrt(float(p.effective_radius)) * delta**1.5
    np.testing.assert_allclose(response.radius, a, rtol=1e-13)
    np.testing.assert_allclose(response.normal_force, force, rtol=1e-13)
    np.testing.assert_allclose(response.area, np.pi * a*a, rtol=1e-13)
    np.testing.assert_allclose(response.potential, .4 * force * delta, rtol=1e-13)
    assert bool(response.active) and bool(response.valid)


def test_zero_adhesion_is_exactly_hertz_for_all_history_flags():
    delta = jnp.array([-1e-7, 0., 1e-9, 1e-7])
    for attached in (False, True):
        old = initial_fibre_contact_history(delta.shape, active=attached)
        hertz = jax.jit(update_fibre_contact)(old, delta, parameters(model="hertz"))
        zero = jax.jit(update_fibre_contact)(old, delta, parameters(work_of_adhesion=0))
        for a, b in zip(jax.tree.leaves(hertz), jax.tree.leaves(zero)):
            np.testing.assert_array_equal(a, b)


def test_jkr_stable_root_matches_independent_quartic():
    p = parameters()
    ac, dc = map(float, jkr_critical_state(p))
    values = jnp.asarray([.999 * dc, .8 * dc, 0, -dc, -20*dc, 1e-7])
    _, response = jax.jit(update_fibre_contact)(initial_fibre_contact_history(values.shape, active=True), values, p)
    beta = np.sqrt(2 * np.pi * float(p.work_of_adhesion) / float(p.effective_modulus))
    roots = []
    for delta in np.asarray(values):
        polynomial = np.roots([1 / float(p.effective_radius), 0, 0, -beta, -delta])
        real = [z.real**2 for z in polynomial if abs(z.imag) < 1e-11 and z.real > 0]
        roots.append(max(real))
    np.testing.assert_allclose(response.radius, roots, rtol=3e-11, atol=1e-18)
    assert np.all(np.asarray(response.radius) > ac)
    recovered = np.asarray(response.radius)**2 / float(p.effective_radius) - beta * np.sqrt(response.radius)
    np.testing.assert_allclose(recovered, values, atol=1e-20, rtol=3e-12)


def test_jkr_attachment_hysteresis_and_displacement_threshold():
    p = parameters()
    _, dc = jkr_critical_state(p)
    separated = dc * (1 + 1e-10)
    path = jnp.array([2*dc, .5*dc, 0, -dc, .5*dc, separated, .5*dc, 0])

    def advance(old, delta):
        new, response = update_fibre_contact(old, delta, p)
        return new, response

    _, response = jax.jit(lambda x: jax.lax.scan(advance, initial_fibre_contact_history(), x))(path)
    np.testing.assert_array_equal(response.active, [False, False, True, True, True, False, False, True])
    np.testing.assert_array_equal(response.snap_in, [False, False, True, False, False, False, False, True])
    np.testing.assert_array_equal(response.snap_out, [False, False, False, False, False, True, False, False])
    np.testing.assert_array_equal(np.asarray(response.normal_force)[[0, 1, 5, 6]], 0)
    assert float(response.normal_force[4]) < 0


def test_eager_jit_critical_endpoint_uses_the_evaluated_threshold(capsys):
    p = parameters()
    _, eager_dc = jkr_critical_state(p)
    delta = jnp.array([eager_dc * (1 + 1e-10), eager_dc, eager_dc * (1 - 1e-10)])
    old = initial_fibre_contact_history(delta.shape, active=True)
    _, response = jax.jit(lambda values: update_fibre_contact(old, values, p))(delta)
    actual_dc = np.asarray(response.critical_indentation)
    np.testing.assert_allclose(actual_dc, float(eager_dc), rtol=64*np.finfo(float).eps, atol=0)
    np.testing.assert_array_equal(response.active, np.asarray(delta) > actual_dc)
    assert not bool(response.active[0])
    assert bool(response.active[-1])
    offset_ulps = (float(eager_dc) - float(actual_dc[1])) / abs(np.spacing(float(eager_dc)))
    with capsys.disabled():
        print(f"\nJKR critical endpoint: eager={float(eager_dc):.17g}, "
              f"jit={float(actual_dc[1]):.17g}, difference={offset_ulps:g} ULP")


def test_threshold_constants_and_zero_indentation_adhesion():
    p = parameters()
    E, R, W = map(float, (p.effective_modulus, p.effective_radius, p.work_of_adhesion))
    ac, dc = map(float, jkr_critical_state(p))
    np.testing.assert_allclose(ac**3, np.pi*W*R*R/(8*E), rtol=1e-13)
    np.testing.assert_allclose(dc, -3*ac*ac/R, rtol=1e-13)
    _, at_zero = update_fibre_contact(initial_fibre_contact_history(), 0., p)
    np.testing.assert_allclose(at_zero.radius**3, 2*np.pi*W*R*R/E, rtol=1e-13)
    np.testing.assert_allclose(at_zero.normal_force, -4/3*np.pi*W*R, rtol=1e-13)
    _, near_detach = update_fibre_contact(initial_fibre_contact_history(active=True), dc*(1-1e-10), p)
    np.testing.assert_allclose(near_detach.normal_force, -5/6*np.pi*W*R, rtol=3e-5)
    _, detached = update_fibre_contact(initial_fibre_contact_history(active=True), dc, p)
    assert not bool(detached.active)
    assert float(detached.normal_force) == 0


@pytest.mark.parametrize("factor", [-.8, 0., 2.])
def test_energy_force_stationarity_and_stable_radius_curvature(factor):
    p = parameters()
    _, dc = jkr_critical_state(p)
    delta = factor * -dc
    _, response = update_fibre_contact(initial_fibre_contact_history(active=True), delta, p)
    a = response.radius
    force = jax.grad(lambda d: contact_potential(a, d, p))(delta)
    stationary = jax.grad(lambda radius: contact_potential(radius, delta, p))(a)
    curvature = jax.grad(jax.grad(lambda radius: contact_potential(radius, delta, p)))(a)
    np.testing.assert_allclose(force, response.normal_force, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(stationary, 0, atol=1e-15)
    assert float(curvature) > 0


def test_implicit_radius_derivative_inside_attached_branch():
    p = parameters()
    _, dc = jkr_critical_state(p)
    delta = -.4 * -dc
    history = initial_fibre_contact_history(active=True)
    radius = lambda d: update_fibre_contact(history, d, p)[1].radius
    derivative = jax.grad(radius)(delta)
    step = abs(float(dc)) * 1e-5
    numerical = (float(radius(delta+step)) - float(radius(delta-step))) / (2*step)
    np.testing.assert_allclose(derivative, numerical, rtol=2e-7)


def test_normal_action_reaction_has_no_tangent_and_is_rotation_covariant():
    p = parameters()
    _, response = update_fibre_contact(initial_fibre_contact_history(), jnp.array([0., 1e-7]), p)
    normal = jnp.array([[1., 2., 3.], [-2., 1., 4.]])
    pair = jax.jit(normal_pair_forces)(normal, response)
    np.testing.assert_array_equal(pair.first + pair.second, 0)
    np.testing.assert_allclose(jnp.cross(pair.second, normal), 0, atol=1e-18)
    R = jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rotated = normal_pair_forces(normal @ R.T, response)
    np.testing.assert_allclose(rotated.first, pair.first @ R.T, atol=1e-18)
    assert float(jnp.dot(pair.second[0], normal[0])) < 0
    assert float(jnp.dot(pair.second[1], normal[1])) > 0


def test_batched_parameters_and_invalid_input_flags():
    p = fibre_contact_parameters(effective_modulus=[1e9, 2e9], effective_radius=1e-4,
                                work_of_adhesion=[0, .1])
    history, response = jax.jit(update_fibre_contact)(initial_fibre_contact_history((2,)), jnp.zeros(2), p)
    np.testing.assert_array_equal(history.active, [False, True])
    invalid, bad = update_fibre_contact(history, jnp.array([jnp.nan, jnp.inf]), p)
    np.testing.assert_array_equal(invalid.valid, False)
    np.testing.assert_array_equal(invalid.active, history.active)
    np.testing.assert_array_equal(bad.normal_force, 0)
    pair = normal_pair_forces(jnp.array([[0., 0., 0.], [jnp.nan, 1., 0.]]), response)
    np.testing.assert_array_equal(pair.valid, False)
    np.testing.assert_array_equal(pair.second, 0)
