import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, block, elastic
from mpm_jax.cohesive_crack import (
    CohesiveLaw, Interface, discrete_traction, face_transfer, face_velocity,
    midpoint_bulk, response, scatter_face_force, step,
)

jax.config.update('jax_enable_x64', True)


def test_complete_separation_consumes_prescribed_energy_and_unloading_is_reversible():
    law = CohesiveLaw(strength=12e6, energy=200., stiffness=1e13)
    d0, df = law.peak_opening, law.final_opening
    path = np.concatenate((np.linspace(0, d0, 101), np.linspace(d0, .6*df, 401)[1:],
                           np.linspace(.6*df, 0, 201)[1:], np.linspace(0, .6*df, 201)[1:],
                           np.linspace(.6*df, df, 401)[1:], np.linspace(df, 1.2*df, 101)[1:]))
    gaps = jnp.stack((jnp.zeros(len(path)), jnp.asarray(path)), axis=-1)

    def body(maximum, gap):
        r = response(gap[None], maximum, jnp.asarray([0., 1.]), jnp.asarray([True]), law)
        return r.maximum, (r.traction[0, 1], r.stored[0], r.dissipated[0])

    _, (traction, stored, dissipated) = jax.lax.scan(body, jnp.zeros(1), gaps)
    work = np.sum(.5 * (np.asarray(traction)[1:] + np.asarray(traction)[:-1]) * np.diff(path))
    np.testing.assert_allclose(work, law.energy, rtol=2e-6)
    np.testing.assert_allclose(dissipated[-1], law.energy, rtol=1e-13)
    assert np.all(np.diff(np.asarray(dissipated)) >= -1e-12)
    assert float(stored[-1]) == 0
    np.testing.assert_allclose(dissipated[500:901], dissipated[500], rtol=1e-13)


def test_precrack_is_traction_free_in_opening_but_resists_compression():
    law = CohesiveLaw(strength=12e6, energy=200., stiffness=1e13)
    r = response(jnp.asarray([[1e-6, 2e-6], [1e-6, -2e-6]]), jnp.zeros(2),
                 jnp.asarray([0., 1.]), jnp.asarray([False, False]), law)
    np.testing.assert_allclose(r.traction, [[0., 0.], [0., -2e7]])
    np.testing.assert_array_equal(r.damage, 0.)
    np.testing.assert_array_equal(r.dissipated, 0.)
    np.testing.assert_allclose(r.stored, [0., 20.])


def test_face_scatter_conserves_force_and_discrete_virtual_work():
    config = Config(dim=2, n_grid=16, gravity=(0., 0.))
    mass = jnp.ones(2 * 17**2).at[::3].set(0.)
    segment = jnp.asarray([[.1, .5], [.9, .5]])
    xp = jnp.asarray([[.4, .51], [.6, .52]])
    xm = jnp.asarray([[.4, .49], [.6, .48]])
    p, m = face_transfer(xp, mass, config, segment, True), face_transfer(xm, mass, config, segment, False)
    assert bool(p.valid & m.valid)
    velocity = jnp.asarray(np.random.default_rng(21).normal(size=(len(mass), 2)))
    traction_force = jnp.asarray([[12., 50.], [-3., 70.]])
    force = scatter_face_force(-traction_force, p, velocity.shape) + scatter_face_force(traction_force, m, velocity.shape)
    np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-13)
    face_power = -jnp.sum(traction_force * (face_velocity(velocity, p) - face_velocity(velocity, m)))
    np.testing.assert_allclose(jnp.sum(force * velocity), face_power, atol=1e-13)


def test_coupled_step_preserves_total_momentum_and_damage_history():
    config = Config(dim=2, n_grid=32, dt=1e-6, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.25, .375), (.75, .625), dtype=jnp.float64)
    direction = jnp.where(state.x[:, 1] > .5, 1., -1.)
    state = state._replace(v=jnp.stack((jnp.zeros_like(direction), direction), axis=-1))
    x = jnp.stack((jnp.linspace(.265625, .734375, 16), jnp.full(16, .5)), axis=-1)
    interface = Interface(x, x, jnp.zeros(16), jnp.full(16, .5/16), jnp.ones(16, dtype=bool), jnp.asarray(True))
    law = CohesiveLaw(strength=1000., energy=1., stiffness=1e7)
    segment = jnp.asarray([[.1, .5], [.9, .5]])
    first, interface, _ = step(state, interface, elastic(), config, segment, law)
    second, interface2, work = step(first, interface, elastic(), config, segment, law)
    assert bool(second.valid & interface2.valid)
    np.testing.assert_allclose((second.mass[:, None]*second.v).sum(axis=0),
                               (state.mass[:, None]*state.v).sum(axis=0), atol=1e-11)
    assert bool(jnp.all(interface2.maximum >= interface.maximum))
    assert float(work.cohesive) > 0


def test_bulk_discrete_work_matches_finite_strain_energy_and_rigid_rotation():
    material = elastic(young=1e6, poisson=.2)
    F = jnp.asarray([[[1.1, .05], [.02, .95]], [[.9, .1], [-.1, 1.2]]])
    L = jnp.asarray([[[20., 80.], [-40., -10.]], [[-30., 20.], [50., 40.]]])
    dt = .001
    following, tau, valid = midpoint_bulk(F, L, dt, material)
    assert bool(valid)

    def energy(A):
        q = jnp.log(jnp.linalg.det(A))
        return .5*material.mu*(jnp.sum(A*A, axis=(1, 2))-2)-material.mu*q+.5*material.lam*q*q

    np.testing.assert_allclose(energy(following)-energy(F), dt*jnp.sum(tau*L, axis=(1, 2)), rtol=1e-11, atol=1e-8)
    rotation = jnp.asarray([[[0., -200.], [200., 0.]]])
    F, tau, valid = midpoint_bulk(jnp.eye(2)[None], rotation, dt, material)
    assert bool(valid)
    np.testing.assert_allclose(F @ jnp.swapaxes(F, -1, -2), jnp.eye(2)[None], atol=1e-14)
    np.testing.assert_allclose(tau, 0., atol=1e-9)


def test_discrete_interface_work_across_softening_and_load_direction_changes():
    law = CohesiveLaw(strength=12e6, energy=200., stiffness=1e13)
    old = jnp.asarray([[0., .5e-6], [1e-5, 5e-6], [1e-5, 2e-5]])
    new = jnp.asarray([[0., 4e-5], [-4e-6, 3e-5], [0., -1e-6]])
    maximum = jnp.linalg.norm(old, axis=-1)
    force, after, energy = discrete_traction(old, new, maximum, jnp.asarray([0., 1.]), jnp.ones(3, dtype=bool), law)
    np.testing.assert_allclose(jnp.sum(force*(new-old), axis=-1), energy, rtol=1e-12, atol=1e-12)
    assert bool(jnp.all(after.maximum >= maximum))
