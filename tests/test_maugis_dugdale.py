import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.maugis_dugdale import (
    initial_maugis_history, maugis_branch, maugis_parameters,
    maugis_pressure, update_maugis,
)
from mpm_jax.fibre_contact import normal_pair_forces

jax.config.update("jax_enable_x64", True)


def model(lam=1., work=None):
    E, R, L = 1e6, .01, 1e-4
    W = (4/3)*E*L**3/(np.pi*R**2) if work is None else work
    h = 2*L*L/(np.pi*R*lam)
    return maugis_parameters(effective_modulus=E, effective_radius=R,
                             work_of_adhesion=W, interaction_range=h)


def evaluate(delta, lam=1., active=True):
    p = model(lam)
    return jax.jit(update_maugis)(initial_maugis_history(np.shape(delta), active=active), delta, p)[1]


def test_host_validation_and_integer_indentation():
    for key, value in (("effective_modulus", 0), ("effective_radius", -1),
                       ("interaction_range", 0), ("work_of_adhesion", np.nan)):
        kwargs = dict(effective_modulus=1e6, effective_radius=.01,
                      work_of_adhesion=.1, interaction_range=1e-6)
        kwargs[key] = value
        with pytest.raises(ValueError):
            maugis_parameters(**kwargs)
    assert bool(evaluate(0).valid)


def test_zero_work_has_hertz_force_radius_and_energy_without_history_dependence():
    p = model(work=0)
    d = jnp.asarray([-1e-6, 0, 1e-7, 2e-6])
    a = np.sqrt(float(p.effective_radius)*np.maximum(np.asarray(d), 0))
    f = 4*float(p.effective_modulus)*a**3/(3*float(p.effective_radius))
    results = []
    for active in (False, True):
        _, r = jax.jit(update_maugis)(initial_maugis_history(d.shape, active=active), d, p)
        assert np.all(r.valid)
        np.testing.assert_allclose(r.radius, a, rtol=3e-15)
        np.testing.assert_allclose(r.normal_force, f, rtol=3e-15)
        np.testing.assert_allclose(r.potential, .4*f*np.asarray(d), rtol=3e-15)
        results.append(r)
    for field in ("radius", "normal_force", "area", "potential", "phase"):
        np.testing.assert_array_equal(getattr(results[0], field), getattr(results[1], field))


@pytest.mark.parametrize("lam", [.01, .2, 1., 10., 1000.])
def test_original_maugis_contact_equations(lam):
    p = model(lam)
    delta = np.asarray([0., .5e-6, 2e-6])
    r = evaluate(delta, lam)
    assert np.all(r.valid) and np.all(r.phase == 2)
    K = 4*float(p.effective_modulus)/3
    L = (np.pi*float(p.work_of_adhesion)*float(p.effective_radius)**2/K)**(1/3)
    A, m = np.asarray(r.radius)/L, np.asarray(r.adhesive_radius/r.radius)
    mu = np.sqrt(m*m-1)
    theta = np.arctan(mu)
    condition = .5*lam*A*A*(mu+(m*m-2)*theta)+(4/3)*lam*lam*A*(mu*theta-m+1)
    D = A*A-(4/3)*A*lam*mu
    P = A**3-lam*A*A*(mu+m*m*theta)
    np.testing.assert_allclose(condition, 1, atol=2e-8, rtol=0)
    np.testing.assert_allclose(D*L*L/float(p.effective_radius), delta, atol=3e-14, rtol=1e-8)
    np.testing.assert_allclose(P*np.pi*float(p.work_of_adhesion)*float(p.effective_radius),
                               r.normal_force, atol=2e-12, rtol=1e-8)


def test_noncontact_cohesive_disk_is_retained_until_its_fold():
    lam = .5
    p = model(lam)
    Dscale = 1e-6
    b = 8*lam/(3*np.pi)
    critical = (-2/(np.pi*lam)-b*b/2)*Dscale
    delta = -1.1e-6
    r = evaluate(delta, lam)
    assert r.valid and r.active and int(r.phase) == 1
    assert r.radius == 0 and r.adhesive_radius > 0 and r.normal_force < 0
    sigma = float(p.work_of_adhesion/p.interaction_range)
    c = float(r.adhesive_radius)
    np.testing.assert_allclose(r.normal_force, -np.pi*sigma*c*c, rtol=2e-14)
    np.testing.assert_allclose(r.central_gap, -delta-2*sigma*c/float(p.effective_modulus), rtol=2e-14)
    np.testing.assert_allclose(r.critical_indentation, critical, rtol=2e-14)
    assert evaluate(critical*.999, lam).active
    assert not evaluate(critical*1.001, lam).active
    assert not evaluate(-1.3e-6, lam, active=False).active
    assert evaluate(-1.3e-6, lam, active=True).active


def test_attachment_and_detachment_history_under_jitted_scan():
    p = model(2.)
    ref = evaluate(0., 2.)
    dc, attach = float(ref.critical_indentation), -float(p.interaction_range)
    # Cross thresholds explicitly; no equality assumption across host/JIT rounding.
    path = jnp.asarray([1.01*attach, .99*attach, 0., .999*dc, 1.001*dc, .5*(dc+attach), 0.])
    step = lambda history, d: update_maugis(history, d, p)
    _, response = jax.jit(lambda: jax.lax.scan(step, initial_maugis_history(), path))()
    np.testing.assert_array_equal(response.active, [False, True, True, True, False, False, True])
    assert np.all(response.valid)


@pytest.mark.parametrize("lam,delta", [(.5, -1.1e-6), (1., 1e-6), (20., 1e-6)])
def test_potential_derivative_is_force_on_smooth_stable_branch(lam, delta):
    p = model(lam)
    history = initial_maugis_history(active=True)
    potential = lambda d: update_maugis(history, d, p)[1].potential
    force = update_maugis(history, delta, p)[1].normal_force
    derivative = jax.jit(jax.grad(potential))(delta)
    np.testing.assert_allclose(derivative, force, rtol=3e-7, atol=1e-12)
    df = jax.jit(jax.grad(lambda d: update_maugis(history, d, p)[1].normal_force))(delta)
    h = 1e-10
    finite = (update_maugis(history, delta+h, p)[1].normal_force
              -update_maugis(history, delta-h, p)[1].normal_force)/(2*h)
    np.testing.assert_allclose(df, finite, rtol=2e-6)


def test_pressure_integrates_to_the_solved_total_force():
    p = model(1.)
    response = evaluate(1e-6)
    a, c = float(response.radius), float(response.adhesive_radius)
    z, w = np.polynomial.legendre.leggauss(256)
    # r=a*sqrt(1-t*t) removes the Hertz square-root endpoint.
    t = (z+1)/2
    radius = a*np.sqrt(1-t*t)
    pressure = np.asarray(maugis_pressure(radius, response, p))
    inner = 2*np.pi*a*a*np.sum(w/2*t*pressure)
    outer = -np.pi*float(p.work_of_adhesion/p.interaction_range)*(c*c-a*a)
    np.testing.assert_allclose(inner+outer, response.normal_force, rtol=2e-10, atol=1e-14)


def test_finite_lambda_approaches_jkr_and_dmt_contact_limits():
    # Independent fixed-contact-radius comparison, not a claim about load-control snap.
    for lam, expected in ((1e-8, "dmt"), (1e4, "jkr")):
        lo, hi = 0., max(2., 10*lam)
        for _ in range(80):
            mid = (lo+hi)/2
            if float(maugis_branch(mid, lam)[0]) < 1: lo = mid
            else: hi = mid
        A, _, D, P, *_ = maugis_branch((lo+hi)/2, lam)
        if expected == "dmt":
            np.testing.assert_allclose([D, P], [1., -1.], atol=2e-4, rtol=0)
        else:
            np.testing.assert_allclose([D, P], [1-(2/3)*np.sqrt(6), 1-np.sqrt(6)], atol=2e-7, rtol=0)


def test_batched_eager_and_jit_and_nonfinite_rejection():
    lam = np.asarray([.1, .5, 1., 10.])
    p = model(lam)
    d = jnp.asarray([0., -1.1e-6, 1e-6, np.nan])
    history = initial_maugis_history(d.shape, active=True)
    a = update_maugis(history, d, p)
    b = jax.jit(update_maugis)(history, d, p)
    np.testing.assert_allclose(a[1].normal_force, b[1].normal_force, rtol=2e-12, atol=1e-15)
    np.testing.assert_array_equal(b[1].valid, [True, True, True, False])
    assert b[1].normal_force[-1] == 0


def test_local_scalar_law_gives_objective_frictionless_pair_forces():
    response = evaluate(-.5e-6)
    normal = jnp.asarray([1., 2., -3.])
    force = normal_pair_forces(normal, response)
    Q = jnp.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    transformed = normal_pair_forces(Q@normal, response)
    np.testing.assert_allclose(force.first+force.second, 0, atol=0, rtol=0)
    np.testing.assert_allclose(transformed.second, Q@force.second, rtol=2e-15)
    np.testing.assert_allclose(jnp.cross(force.second, normal), 0, atol=1e-18, rtol=0)
