import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from mpm_jax import Config
from mpm_jax.cohesive_crack import CohesiveLaw
from mpm_jax.implicit_contact import line_interface, trace, solve, advect
from mpm_jax.normal_cohesive import (
    Interface,
    discrete_traction,
    energy,
    initialize,
    opening,
    update_history,
)


def law():
    return CohesiveLaw(strength=10.0, energy=0.2, stiffness=1000.0)


def scalar_interface(gap, maximum=0.0, bonded=True):
    return Interface(
        plus=jnp.asarray([[gap, 0.0]]),
        minus=jnp.asarray([[0.0, 0.0]]),
        area=jnp.asarray([1.0]),
        normal=jnp.asarray([1.0, 0.0]),
        valid=jnp.asarray(True),
        maximum=jnp.asarray([maximum]),
        bonded=jnp.asarray([bonded]),
    )


def set_gap(interface, gap):
    center = 0.5 * (interface.plus + interface.minus)
    offset = 0.5 * jnp.asarray(gap)[:, None] * interface.normal
    return interface._replace(plus=center + offset, minus=center - offset)


def total_energy(interface, cohesive_law):
    stored, dissipated = energy(interface, cohesive_law)
    return stored + dissipated


def test_loading_unloading_reloading_and_compression_satisfy_incremental_work():
    cohesive_law = law()
    d0, df = cohesive_law.peak_opening, cohesive_law.final_opening
    interface = scalar_interface(0.0)
    gaps = [0.005, 0.015, 0.020, 0.012, 0.018, -0.003,
            0.010, 0.025, 0.039, df]
    previous_maximum = 0.0

    for new_gap in gaps:
        old_gap = float(opening(interface)[0])
        old_total = total_energy(interface, cohesive_law)
        traction, _ = discrete_traction(
            opening(interface), jnp.asarray([new_gap]), interface.maximum,
            interface.bonded, cohesive_law,
        )
        trial = set_gap(interface, jnp.asarray([new_gap]))
        following = update_history(trial)
        new_total = total_energy(following, cohesive_law)
        assert float(traction[0] * (new_gap - old_gap)) == pytest.approx(
            float(new_total - old_total), rel=2e-11, abs=2e-13
        )
        assert float(following.maximum[0]) >= previous_maximum - 1e-14
        previous_maximum = float(following.maximum[0])
        interface = following

    stored, dissipated = energy(interface, cohesive_law)
    assert float(interface.maximum[0]) == pytest.approx(df, abs=1e-14)
    assert float(dissipated) == pytest.approx(cohesive_law.energy, abs=1e-14)
    assert float(stored) == pytest.approx(0.0, abs=1e-14)
    assert d0 < df
    # Once fully failed, opening has no tensile work or history growth.
    failed = set_gap(interface, jnp.asarray([df + 0.01]))
    traction, _ = discrete_traction(opening(interface), opening(failed),
                                     interface.maximum, interface.bonded, cohesive_law)
    assert float(traction[0]) == pytest.approx(0.0, abs=1e-14)
    beyond_failure = update_history(failed)
    assert float(beyond_failure.maximum[0]) >= float(interface.maximum[0])
    assert float(energy(beyond_failure, cohesive_law)[1]) == pytest.approx(cohesive_law.energy, abs=1e-14)


@pytest.mark.parametrize(
    "old,maximum,new",
    [
        (0.002, 0.002, 0.005),     # elastic loading
        (0.020, 0.030, 0.024),     # unloading at a softened history
        (0.015, 0.025, 0.028),     # reloading below the prior maximum
        (0.020, 0.020, 0.025),     # new record in the softening branch
        (-0.004, 0.020, 0.006),    # compression to tension at fixed history
        (-0.004, 0.020, -0.002),   # compression through prior damage
    ],
)
def test_analytic_discrete_traction_tangent_matches_finite_difference(old, maximum, new):
    cohesive_law = law()
    old_array = jnp.asarray([old])
    maximum_array = jnp.asarray([maximum])
    bonded = jnp.asarray([True])

    def value(z):
        return discrete_traction(old_array, jnp.asarray([z]), maximum_array,
                                 bonded, cohesive_law)[0][0]

    _, tangent = discrete_traction(old_array, jnp.asarray([new]), maximum_array,
                                   bonded, cohesive_law)
    eps = 1.0e-7
    finite_difference = (value(new + eps) - value(new - eps)) / (2.0 * eps)
    assert float(tangent[0]) == pytest.approx(float(finite_difference), rel=2e-7, abs=2e-7)


def test_unbonded_interface_has_no_tension_but_keeps_compressive_recontact():
    cohesive_law = law()
    unbonded = scalar_interface(0.01, maximum=0.0, bonded=False)
    tensile, _ = discrete_traction(opening(unbonded), jnp.asarray([0.02]),
                                   unbonded.maximum, unbonded.bonded, cohesive_law)
    assert float(tensile[0]) == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(energy(unbonded, cohesive_law)[1], 0.0, atol=1e-14)

    compressed = set_gap(unbonded, jnp.asarray([-0.01]))
    pressure, _ = discrete_traction(opening(unbonded), opening(compressed),
                                    unbonded.maximum, unbonded.bonded, cohesive_law)
    expected = 0.5 * cohesive_law.stiffness * 0.01**2 / (-0.01 - 0.01)
    assert float(pressure[0]) == pytest.approx(expected, abs=1e-13)
    assert float(energy(compressed, cohesive_law)[0]) > 0.0
    assert float(update_history(compressed).maximum[0]) == pytest.approx(0.0, abs=1e-14)


def test_two_field_implicit_cohesive_loading_updates_history_and_closes_energy():
    config = Config(n_grid=16, dt=1.0e-3, domain_size=1.0, boundary_cells=0,
                    material="elastic", dim=2, gravity=(0.0, 0.0))
    cohesive_law = law()
    line = line_interface((0.5, 0.3), (0.5, 0.7), (1.0, 0.0), config, gap=0.02)
    interface = initialize(line, jnp.ones_like(line.area, dtype=bool))
    size = (config.n_grid + 1) ** 2
    mass = jnp.linspace(0.4, 1.6, 2 * size)
    free = jnp.ones((2 * size, 2))
    support = trace(interface, mass, free, config)
    assert bool(support.valid)
    initial = jnp.zeros((2 * size, 2))
    initial = initial.at[:size, 0].set(0.15)
    initial = initial.at[size:, 0].set(-0.15)
    dt = config.dt

    result = solve(interface, support, initial, initial, cohesive_law.stiffness,
                   dt, config.dx, cohesive_law)
    assert bool(result.valid)
    moved = advect(interface, support, 0.5 * (initial + result.velocity), dt,
                   transport_tangent=True)
    new_interface = update_history(moved)
    np.testing.assert_allclose(opening(new_interface), result.gap, rtol=2e-11, atol=2e-13)
    assert bool(jnp.all(new_interface.maximum >= interface.maximum))
    assert float(jnp.max(new_interface.maximum)) > float(jnp.max(interface.maximum))

    before = total_energy(interface, cohesive_law)
    after = total_energy(new_interface, cohesive_law)
    midpoint = 0.5 * (initial + result.velocity)
    contact_work = dt * jnp.sum(result.force * midpoint)
    kinetic_change = 0.5 * jnp.sum(mass[:, None] * (result.velocity**2 - initial**2))
    assert float(after - before + contact_work) == pytest.approx(0.0, abs=2e-11)
    assert float(kinetic_change + after - before) == pytest.approx(0.0, abs=2e-10)


def test_softening_loading_jacobian_can_be_indefinite():
    config = Config(n_grid=16, dt=1.0, domain_size=1.0, boundary_cells=0,
                    material="elastic", dim=2, gravity=(0.0, 0.0))
    cohesive_law = law()
    line = line_interface((0.5, 0.3), (0.5, 0.7), (1.0, 0.0), config, gap=0.020)
    interface = initialize(line, jnp.ones_like(line.area, dtype=bool))
    size = (config.n_grid + 1) ** 2
    mass = jnp.full((2 * size,), 1.0e-3)
    free = jnp.ones((2 * size, 2))
    support = trace(interface, mass, free, config)
    assert bool(support.valid)
    trial_gap = jnp.full_like(interface.maximum, 0.025)
    _, derivative = discrete_traction(opening(interface), trial_gap,
                                     interface.maximum, interface.bonded, cohesive_law)
    assert bool(jnp.all(derivative < 0.0))

    alpha = 0.5 * config.dt**2
    weight = interface.area * (-derivative)
    root = jnp.sqrt(weight)
    symmetric_scaled = jnp.eye(len(weight)) - alpha * (
        root[:, None] * support.compliance * root[None, :]
    )
    eigenvalues = jnp.linalg.eigvalsh(symmetric_scaled)
    assert float(jnp.min(eigenvalues)) < 0.0

    initial = jnp.zeros((2 * size, 2))
    maximum_before = np.asarray(interface.maximum).copy()
    result = solve(interface, support, initial, initial, cohesive_law.stiffness,
                   config.dt, config.dx, cohesive_law)
    assert float(result.softening_number) >= 0.95
    assert not bool(result.valid)
    np.testing.assert_array_equal(interface.maximum, maximum_before)
