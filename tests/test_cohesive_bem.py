from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.cohesive_bem import (
    annular_compliance, cohesive_bem, cohesive_bem_potential,
    initial_cohesive_bem_history, update_cohesive_bem,
)
from mpm_jax.maugis_dugdale import initial_maugis_history, maugis_parameters, update_maugis

jax.config.update("jax_enable_x64", True)


@lru_cache
def model(rings=32, adhesive=True, outer=4e-4):
    E, R, L = 1e6, .01, 1e-4
    W = (4/3)*E*L**3/(np.pi*R**2) if adhesive else 0.
    return cohesive_bem(effective_modulus=E, effective_radius=R,
                        work_of_adhesion=W, interaction_range=2e-6/np.pi,
                        outer_radius=outer, rings=rings)


solve = jax.jit(partial(update_cohesive_bem, max_sweeps=1200, tolerance=1e-9))


def response(delta, m):
    return solve(initial_cohesive_bem_history(m, delta), delta, m)[1]


def test_reciprocal_positive_compliance_and_quadrature_refinement():
    edges = np.linspace(0, 3e-4, 17)
    c12 = annular_compliance(edges, 1e6, quadrature_order=12)
    c24 = annular_compliance(edges, 1e6, quadrature_order=24)
    c48 = annular_compliance(edges, 1e6, quadrature_order=48)
    np.testing.assert_array_equal(c24, c24.T)
    assert np.linalg.eigvalsh(c24)[0] > 0
    assert np.linalg.norm(c48-c24) < np.linalg.norm(c24-c12)
    np.testing.assert_allclose(c24, c48, rtol=.003, atol=0)
    for edges_bad in ([0., 0., 1.], [1., 2., 3.], [0., np.nan, 1.]):
        with pytest.raises(ValueError):
            annular_compliance(edges_bad, 1e6)


@pytest.mark.parametrize("rings", [16, 32, 64])
def test_hertz_limit_with_resolved_patch_and_energy(rings):
    m = model(rings, False)
    delta = 1e-6
    r = response(delta, m)
    assert r.valid and r.domain_clear and r.converged
    exact_a = np.sqrt(float(m.effective_radius)*delta)
    exact_force = (4/3)*float(m.effective_modulus)*exact_a**3/float(m.effective_radius)
    np.testing.assert_allclose(r.normal_force, exact_force, rtol=.025)
    np.testing.assert_allclose(r.potential, .4*exact_force*delta, rtol=.04)
    assert abs(float(r.radius)-exact_a) <= float(m.edges[1])
    assert float(r.equilibrium_residual) <= 1e-9
    assert r.adhesive_area == 0


def test_independent_gap_and_traction_equilibrium_conditions():
    m = model()
    r = response(1e-6, m)
    assert r.valid
    g = np.asarray(r.gap)
    p = np.asarray(r.pressure)
    sigma = float(m.work_of_adhesion/m.interaction_range)
    contact = g == 0
    cohesive = (g > 0) & (g < float(m.interaction_range))
    open_region = g >= float(m.interaction_range)
    assert contact.any() and cohesive.any() and open_region.any()
    assert np.all(p[contact] >= -sigma)
    np.testing.assert_allclose(p[cohesive], -sigma, atol=4e-5, rtol=0)
    np.testing.assert_allclose(p[open_region], 0, atol=4e-5, rtol=0)
    np.testing.assert_allclose(g, np.asarray(m.profile)-1e-6+np.asarray(m.compliance)@np.asarray(r.ring_force), atol=1e-18, rtol=0)
    assert float(r.potential) <= float(r.initial_potential)+1e-20
    np.testing.assert_allclose(r.potential, r.elastic_energy+r.cohesive_energy, rtol=2e-13)


def test_energy_gradient_and_generalized_normal_force():
    m = model()
    r = response(1e-6, m)
    gap = r.gap
    grad = jax.grad(cohesive_bem_potential, argnums=0)(gap, 1e-6, m)
    # At an interior gap, ring-force plus cohesive traction is the residual.
    interior = np.asarray((gap > 0) & (gap < m.interaction_range))
    np.testing.assert_allclose(np.asarray(grad)[interior], 0, atol=4e-12, rtol=0)
    derivative = jax.grad(cohesive_bem_potential, argnums=1)(gap, 1e-6, m)
    np.testing.assert_allclose(derivative, r.normal_force, rtol=2e-12, atol=1e-16)


def test_contact_and_cohesive_ring_refinement_against_md():
    values = [0., 1e-6]
    errors = {}
    for rings in (32, 64):
        m = model(rings)
        p = maugis_parameters(effective_modulus=m.effective_modulus,
                              effective_radius=m.effective_radius,
                              work_of_adhesion=m.work_of_adhesion,
                              interaction_range=m.interaction_range)
        errors[rings] = []
        for delta in values:
            r = response(delta, m)
            _, exact = update_maugis(initial_maugis_history(active=True), delta, p)
            assert r.valid and exact.valid
            force_scale = np.pi*float(m.work_of_adhesion*m.effective_radius)
            errors[rings].append((float(r.normal_force-exact.normal_force)/force_scale)**2)
            assert abs(float(r.radius-exact.radius)) <= 2*float(m.edges[1])
            assert abs(float(r.adhesive_radius-exact.adhesive_radius)) <= 2*float(m.edges[1])
            np.testing.assert_allclose(r.potential, exact.potential, rtol=.04)
    # Ring-edge force errors oscillate; this checks an explicitly specified pair
    # of states, not a universal convergence rate or an exact continuum fold.
    assert np.sqrt(np.mean(errors[64])) < np.sqrt(np.mean(errors[32]))
    assert np.sqrt(np.mean(errors[64])) < .12


def test_warm_start_full_separation_and_explicit_nonconvergence_rejection():
    m = model(16)
    history, loaded = solve(initial_cohesive_bem_history(m), 1e-6, m)
    assert loaded.valid
    history, released = solve(history, -4e-6, m)
    assert released.valid and not released.active
    np.testing.assert_allclose(released.normal_force, 0, atol=1e-10, rtol=0)
    _, short = jax.jit(partial(update_cohesive_bem, max_sweeps=1, tolerance=1e-12))(
        initial_cohesive_bem_history(m), 1e-6, m,
    )
    assert not short.converged and not short.valid
    assert int(short.sweeps) == 1


def test_domain_truncation_and_nonfinite_input_are_not_accepted():
    small = model(16, True, outer=1.2e-4)
    r = response(1e-6, small)
    assert not r.domain_clear and not r.valid
    m = model(16)
    _, invalid = solve(initial_cohesive_bem_history(m), jnp.nan, m)
    assert not invalid.valid


def test_independent_batched_contacts_use_vmap():
    m = model(16, False)
    values = jnp.asarray([.5e-6, 1e-6])
    fn = lambda d: update_cohesive_bem(initial_cohesive_bem_history(m, d), d, m,
                                      max_sweeps=1200, tolerance=1e-9)[1]
    r = jax.jit(jax.vmap(fn))(values)
    assert np.all(r.valid)
    np.testing.assert_allclose(r.normal_force[1]/r.normal_force[0], 2**1.5, rtol=.04)
