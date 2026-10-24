from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.fibre_mpm import grid_coordinates, stencil
from mpm_jax.impact_mpm import (
    ImpactConfig, advance_to, build_grid, cayley_kinematics, clamp_grid,
    energy_ledger, make_state, paired_contact, stable_timestep, step,
    stress_kick, thermodynamics, transfer_to_grid, transfer_to_particles,
    validate_state, compatible_contact_work,
)
from mpm_jax.materials.shock_metal import deviatoric_return, eos_state, hugoniot, shock_metal

jax.config.update('jax_enable_x64', True)


def small_state(*, velocity=(0., 0., 0.), affine=None):
    dx = 1e-4
    cube = np.array(np.meshgrid([.25, .75], [.25, .75], [.25, .75], indexing='ij')).reshape(3, -1).T*dx
    x = np.concatenate((cube, cube+[10*dx, 0, 0]))
    ids = np.repeat([0, 1], len(cube))
    grid = build_grid(x, ids, dx=dx)
    material = shock_metal()
    state = make_state(x, np.full(len(x), (dx/2)**3), ids, material, grid,
                       v=np.broadcast_to(velocity, x.shape), C=affine)
    return state, material, grid


def test_pair_map_is_unique_and_matches_actual_global_nodes():
    state, _, grid = small_state()
    pairs = np.asarray(grid.pairs)
    assert len(pairs) > 0
    assert len(np.unique(pairs[:, 0])) == len(pairs) == len(np.unique(pairs[:, 1]))
    a, body_a = grid_coordinates(grid.layout, pairs[:, 0])
    b, body_b = grid_coordinates(grid.layout, pairs[:, 1])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(body_a, 0)
    np.testing.assert_array_equal(body_b, 1)


def test_apic_transfers_convert_only_the_measured_variance_to_heat():
    state, _, grid = small_state()
    rng = np.random.default_rng(18)
    state = state._replace(v=jnp.asarray(rng.normal(size=state.v.shape)),
                           C=jnp.asarray(rng.normal(size=state.C.shape)*1e4))
    support = stencil(state.x, state.body_id, grid.layout)
    mass, velocity, loss = jax.jit(transfer_to_grid)(state, support, grid)
    before = energy_ledger(state, grid)[:2].sum()
    grid_kinetic = .5*jnp.sum(mass*jnp.sum(velocity**2, axis=-1))
    assert jnp.all(loss >= 0)
    np.testing.assert_allclose(before, grid_kinetic+jnp.sum(state.mass*loss), rtol=3e-15)
    v, C, back_loss = jax.jit(transfer_to_particles)(velocity, support, grid)
    after = energy_ledger(state._replace(v=v, C=C), grid)[:2].sum()
    assert jnp.all(back_loss >= 0)
    np.testing.assert_allclose(grid_kinetic, after+jnp.sum(state.mass*back_loss), rtol=3e-15)
    np.testing.assert_allclose(jnp.sum(mass[:, None]*velocity, axis=0),
                               jnp.sum(state.mass[:, None]*v, axis=0), atol=1e-23)


def test_affine_rigid_spin_has_no_transfer_heating():
    spin = np.array([[0., -300., 0.], [300., 0., 0.], [0., 0., 0.]])
    state, _, grid = small_state(affine=np.broadcast_to(spin, (16, 3, 3)))
    state = state._replace(v=state.x@spin.T)
    support = stencil(state.x, state.body_id, grid.layout)
    _, velocity, loss = transfer_to_grid(state, support, grid)
    v, C, back_loss = transfer_to_particles(velocity, support, grid)
    np.testing.assert_allclose(v, state.v, atol=1e-16)
    np.testing.assert_allclose(C, state.C, atol=3e-12)
    assert float(loss.max()) < 1e-31 and float(back_loss.max()) < 1e-31


def test_pair_contact_balances_momentum_and_normal_kinetic_loss():
    velocity = jnp.array([[3., 4., 2.], [-1., -2., 5.], [-2., 1., 0.], [3., 7., 0.]])
    mass = jnp.array([2., 3., 4., 1.])
    pairs = jnp.array([[0, 1], [2, 3]])
    normal = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    result, heat, impulse, valid = jax.jit(paired_contact)(velocity, mass, normal, pairs)
    assert valid
    np.testing.assert_allclose(jnp.sum(mass[:, None]*(result-velocity), axis=0), 0., atol=2e-15)
    change = .5*jnp.sum(mass*jnp.sum(result**2-velocity**2, axis=-1))
    np.testing.assert_allclose(change+heat.sum(), 0., atol=1e-14)
    np.testing.assert_allclose(result[0, 0], result[1, 0], atol=1e-15)
    np.testing.assert_array_equal(result[:, 1:], velocity[:, 1:])
    np.testing.assert_array_equal(result[2:], velocity[2:])
    np.testing.assert_array_equal(impulse[1], 0.)
    assert jnp.all(heat >= 0)


def test_unresolved_occupied_interface_is_rejected():
    _, _, _, valid = paired_contact(jnp.ones((2, 3)), jnp.ones(2),
                                    jnp.zeros((1, 3)), jnp.array([[0, 1]]))
    assert not valid


def test_closing_contact_intersecting_a_fixed_normal_dof_is_rejected():
    velocity = jnp.array([[0., 0., 0.], [-1., 0., 0.]])
    mass, pairs, normal = jnp.ones(2), jnp.array([[0, 1]]), jnp.array([[1., 0., 0.]])
    clamped = jnp.array([[True, False, False], [False, False, False]])
    projected, _, _, valid = jax.jit(paired_contact)(velocity, mass, normal, pairs, clamped)
    assert not valid
    final, _, _ = clamp_grid(projected, mass, clamped)
    assert final[0, 0] > final[1, 0]  # A free projection followed by a clamp still closes.
    tangential = clamped.at[0].set(jnp.array([False, True, True]))
    assert paired_contact(velocity, mass, normal, pairs, tangential)[-1]
    assert paired_contact(-velocity, mass, normal, pairs, clamped)[-1]


@pytest.mark.parametrize('boost', [(0., 0., 0.), (1100., -200., 80.)])
def test_compatible_contact_energy_balances_each_body_and_is_galilean_consistent(boost):
    mass = jnp.array([2., 3.])
    old = jnp.array([[3., 4., 2.], [-1., -2., 5.]])+jnp.asarray(boost)
    force = jnp.array([[8., 1., -2.], [-5., 2., 1.]])
    dt = .2
    kicked = old+dt*force/mass[:, None]
    pairs = jnp.array([[0, 1]])
    final, physical, _, valid = paired_contact(kicked, mass, jnp.array([[1., 0., 0.]]), pairs)
    assert valid
    impulse = mass[:, None]*(final-kicked)
    allocation, projection, correction, interface_work = jax.jit(compatible_contact_work)(
        old, final, impulse, mass, pairs, physical)
    midpoint = .5*(old+final)
    center_velocity = jnp.sum(mass[:, None]*midpoint, axis=0)/mass.sum()
    expected_body_work = jnp.sum(impulse*center_velocity, axis=-1)
    np.testing.assert_allclose(interface_work, expected_body_work, atol=2e-10)
    np.testing.assert_allclose(interface_work.sum(), 0., atol=2e-10)
    change = .5*mass*jnp.sum((final-old)*(final+old), axis=-1)
    new_stress_work = -dt*jnp.sum(force*midpoint, axis=-1)
    np.testing.assert_allclose(change+new_stress_work+allocation, interface_work, atol=2e-10)
    old_work = -dt*jnp.sum(force*.5*(old+kicked), axis=-1)
    np.testing.assert_allclose((new_stress_work+allocation).sum(), (old_work+physical).sum(), atol=2e-10)
    np.testing.assert_allclose(projection+correction, allocation, atol=2e-13)
    if any(boost):
        base = compatible_contact_work(old-jnp.asarray(boost), final-jnp.asarray(boost),
                                       impulse, mass, pairs, physical)
        np.testing.assert_allclose(allocation, base[0], atol=2e-10)
        np.testing.assert_allclose(interface_work-base[3], jnp.sum(impulse*jnp.asarray(boost), axis=-1), atol=2e-10)


def test_opening_then_closing_can_have_signed_coupling_work_without_negative_projection_loss():
    mass = jnp.array([2., 3.])
    old = jnp.array([[-2., 0., 0.], [2., 0., 0.]])
    kicked = -old
    pairs = jnp.array([[0, 1]])
    final, physical, _, _ = paired_contact(kicked, mass, jnp.array([[1., 0., 0.]]), pairs)
    impulse = mass[:, None]*(final-kicked)
    allocation, projection, correction, _ = compatible_contact_work(old, final, impulse, mass, pairs, physical)
    assert jnp.all(physical >= 0) and projection.sum() > 0
    assert allocation.sum() < 0 and correction.sum() < -projection.sum()
    np.testing.assert_allclose(allocation.sum(), -jnp.sum(impulse*.5*(old+final)), rtol=2e-15)


def test_midpoint_stress_work_exactly_opposes_grid_kinetic_change():
    state, _, grid = small_state(velocity=(120., -30., 6.))
    support = stencil(state.x, state.body_id, grid.layout)
    mass, velocity, _ = transfer_to_grid(state, support, grid)
    rng = np.random.default_rng(17)
    tensor = rng.normal(size=state.F.shape)*1e9
    tau = jnp.asarray(.5*(tensor+tensor.swapaxes(-1, -2)))
    result, work = jax.jit(stress_kick)(velocity, mass, state.volume, tau, support, grid, 1e-10)
    change = .5*jnp.sum(mass*jnp.sum((result-velocity)*(result+velocity), axis=-1))
    np.testing.assert_allclose(change+work.sum(), 0., atol=1e-18)
    np.testing.assert_allclose(jnp.sum(mass[:, None]*(result-velocity), axis=0), 0., atol=2e-21)


def test_fixed_boundary_energy_goes_to_reservoir_not_particle_heat():
    state, material, grid = small_state(velocity=(12., 0., 0.))
    indices = jnp.arange(grid.layout.node_count)
    _, body = grid_coordinates(grid.layout, indices)
    mask = jnp.broadcast_to((body == 0)[:, None], (grid.layout.node_count, 3))
    grid = replace(grid, clamped=mask)
    config = ImpactConfig(strength=False)
    dt = stable_timestep(state, material, grid, config)*.5
    new, report = step(state, material, grid, config, dt)
    assert new.valid and report.valid
    expected = .5*jnp.sum(state.mass[state.body_id == 0])*12**2
    np.testing.assert_allclose(new.boundary_reservoir, expected, rtol=2e-15)
    np.testing.assert_allclose(new.internal_energy, 0., atol=1e-27)
    np.testing.assert_allclose(energy_ledger(new, grid).sum(), energy_ledger(state, grid).sum(), rtol=3e-15)


def test_cayley_constitutive_spin_is_objective_without_invented_orbital_motion():
    spin = jnp.array([[0., -700., 0.], [700., 0., 0.], [0., 0., 0.]])
    F, rotation, strain = jax.jit(cayley_kinematics)(jnp.eye(3), spin, 1e-4)
    np.testing.assert_allclose(F.T@F, jnp.eye(3), atol=3e-16)
    np.testing.assert_allclose(jnp.linalg.det(F), 1., atol=3e-16)
    np.testing.assert_array_equal(strain, jnp.zeros((3, 3)))
    material = shock_metal(A=1e12)
    tau = jnp.diag(jnp.array([2e8, -1e8, -1e8]))
    response = deviatoric_return(tau, 0., strain, rotation, 1., 1e-4, 300., material)
    assert response.valid and response.plastic_increment == 0
    np.testing.assert_allclose(response.kirchhoff_stress, rotation@tau@rotation.T, atol=3e-8)


def test_uniform_translation_and_output_time_are_preserved():
    state, material, grid = small_state(velocity=(75., -20., 5.))
    config = ImpactConfig()
    target = stable_timestep(state, material, grid, config)*3.4
    result, count = advance_to(state, material, grid, config, target, max_steps=8)
    assert result.valid and int(count) == 4 and result.time == target
    np.testing.assert_allclose(result.v, state.v, rtol=3e-14, atol=1e-12)
    np.testing.assert_allclose(result.x, state.x+target*state.v, atol=3e-18)
    np.testing.assert_allclose(result.F, state.F, atol=3e-14)
    np.testing.assert_allclose(result.internal_energy, 0., atol=1e-18)
    np.testing.assert_allclose(energy_ledger(result, grid).sum(), energy_ledger(state, grid).sum(), rtol=2e-14)


def test_strength_return_repartitions_existing_internal_energy_only_once():
    state, material, grid = small_state()
    C = jnp.broadcast_to(jnp.diag(jnp.array([5e6, -2.5e6, -2.5e6])), state.C.shape)
    center = jnp.array([[.5e-4, .5e-4, .5e-4], [10.5e-4, .5e-4, .5e-4]])
    v = jnp.einsum('pij,pj->pi', C, state.x-center[state.body_id])
    state = state._replace(v=v, C=C)
    config = ImpactConfig()
    current, report = step(state, material, grid, config,
                           .5*stable_timestep(state, material, grid, config))
    assert current.valid and current.plastic_strain.max() > 0 and current.cold_work.max() > 0
    np.testing.assert_allclose(energy_ledger(current, grid).sum(), energy_ledger(state, grid).sum(), rtol=5e-14)
    expected_gain = report.p2g_heat+report.g2p_heat+report.contact_compatible_energy+report.stress_work
    np.testing.assert_allclose(jnp.sum(state.mass*(current.internal_energy-state.internal_energy)),
                               expected_gain, atol=1e-20)
    assert thermodynamics(current, material).valid.all()


def test_invalid_dt_and_exhausted_budget_fail_closed_without_losing_mass():
    state, material, grid = small_state()
    config = ImpactConfig()
    limit = stable_timestep(state, material, grid, config)
    invalid, _ = step(state, material, grid, config, 2*limit)
    assert not invalid.valid
    stopped, count = advance_to(invalid, material, grid, config, 4*limit)
    assert not stopped.valid and int(count) == 0
    np.testing.assert_array_equal(stopped.mass, state.mass)
    np.testing.assert_array_equal(stopped.x, invalid.x)
    budget, _ = advance_to(state, material, grid, config, 5*limit, max_steps=1)
    assert not budget.valid and budget.time < 5*limit


@pytest.mark.parametrize('cfl', [0., -.1, .21, np.nan])
def test_adaptive_cfl_configuration_rejects_unsafe_input(cfl):
    with pytest.raises(ValueError):
        ImpactConfig(cfl=cfl)


@pytest.mark.parametrize('field', ['plastic_strain', 'cold_work', 'mass', 'boundary_projection_loss'])
def test_resume_validation_checks_history_and_reference_mass(field):
    state, material, grid = small_state()
    if field == 'mass':
        changed = state.mass*1.01
    else:
        changed = -jnp.ones_like(getattr(state, field))
    bad = state._replace(**{field: changed})
    with pytest.raises(FloatingPointError):
        validate_state(bad, material, grid)


@pytest.mark.parametrize('field', ['v', 'C', 'plastic_strain', 'cold_work',
    'initial_entropy', 'minimum_entropy', 'negative_entropy_increments',
    'boundary_impulse', 'contact_projection_loss', 'time', 'ever_phase_extrapolated',
    'ever_tension_extrapolated', 'maximum_temperature', 'minimum_J', 'maximum_J'])
def test_resume_rejects_broadcastable_but_wrong_field_shapes(field):
    state, material, grid = small_state()
    value = getattr(state, field)
    changed = value[:1] if value.ndim else value[None]
    with pytest.raises(FloatingPointError, match='shape or dtype'):
        validate_state(state._replace(**{field: changed}), material, grid)


@pytest.mark.parametrize('field', ['body_id', 'valid', 'v', 'ever_phase_extrapolated'])
def test_resume_rejects_incorrect_field_dtypes(field):
    state, material, grid = small_state()
    changed = getattr(state, field).astype(jnp.float32)
    with pytest.raises(FloatingPointError, match='shape or dtype'):
        validate_state(state._replace(**{field: changed}), material, grid)


def test_domain_scope_is_retained_even_after_temperature_returns_to_reference():
    state, material, grid = small_state()
    state = state._replace(ever_phase_extrapolated=jnp.ones_like(state.ever_phase_extrapolated),
        ever_tension_extrapolated=jnp.ones_like(state.ever_tension_extrapolated),
        maximum_temperature=jnp.full_like(state.maximum_temperature, 2000.),
        minimum_J=jnp.full_like(state.minimum_J, .8), maximum_J=jnp.full_like(state.maximum_J, 1.2))
    target = stable_timestep(state, material, grid, ImpactConfig(strength=False))
    result, _ = advance_to(state, material, grid, ImpactConfig(strength=False), target)
    assert result.valid and result.ever_phase_extrapolated.all() and result.ever_tension_extrapolated.all()
    np.testing.assert_array_equal(result.maximum_temperature, state.maximum_temperature)
    np.testing.assert_array_equal(result.minimum_J, state.minimum_J)
    np.testing.assert_array_equal(result.maximum_J, state.maximum_J)
    assert validate_state(result, material, grid)
    with pytest.raises(FloatingPointError, match='domain history'):
        validate_state(result._replace(maximum_temperature=jnp.zeros_like(result.maximum_temperature)), material, grid)
    with pytest.raises(FloatingPointError, match='domain history'):
        validate_state(result._replace(ever_phase_extrapolated=jnp.zeros_like(result.ever_phase_extrapolated)), material, grid)
    with pytest.raises(FloatingPointError, match='domain history'):
        validate_state(result._replace(minimum_J=jnp.full_like(result.minimum_J, material.minimum_J-.01)), material, grid)


@pytest.mark.parametrize('path', ['elastic', 'isentropic'])
def test_explicit_thermal_error_on_smooth_reversible_paths_decreases_with_dt(path):
    material = shock_metal(A=1e12, B=0.)

    def cycle(n):
        t = jnp.linspace(0., 1., n+1)
        amplitude = jnp.sin(jnp.pi*t)**2
        eye = jnp.eye(3)

        def increment(carry, pair):
            F, tau, internal = carry
            a, b = pair
            old_J = jnp.linalg.det(F)
            stored = jnp.sum(tau*tau)/(4*material.shear_modulus*material.density)
            old = eos_state(old_J, internal-stored, material)
            if path == 'elastic':
                strain = .005*(b-a)*jnp.diag(jnp.array([1., -1., 0.]))
                new_F, rotation = F, eye
                work = jnp.sum(tau*strain)/material.density
            else:
                ratio = ((1-.03*b)/(1-.03*a))**(1/3)
                gradient = 2*n*(ratio-1)/(ratio+1)*eye
                new_F, rotation, strain = cayley_kinematics(F, gradient, 1/n)
                work = -old_J*old.pressure*jnp.trace(gradient)/(n*material.density)
            response = deviatoric_return(tau, 0., strain, rotation, jnp.linalg.det(new_F),
                                         1/n, old.temperature, material)
            internal = internal+work
            current = eos_state(jnp.linalg.det(new_F), internal-response.specific_elastic_energy, material)
            return (new_F, response.kirchhoff_stress, internal), jnp.array([current.entropy, current.valid])

        _, report = jax.lax.scan(increment, (eye, jnp.zeros((3, 3)), jnp.asarray(0.)),
                                jnp.stack((amplitude[:-1], amplitude[1:]), axis=-1))
        assert jnp.all(report[:, 1])
        # Frozen-stress work is first order, so pointwise entropy can decrease.
        # This is measured, not clipped or repaired by duplicate heat addition.
        return float(jnp.max(jnp.abs(report[:, 0]))), float(report[-1, 0])

    coarse, final_coarse = cycle(40)
    fine, final_fine = cycle(80)
    assert final_coarse < 0 and final_fine < 0
    assert 0 < fine < .6*coarse


def planar_collision(refinement):
    material = shock_metal()
    dx, up = 1e-4/refinement, 500.
    planes = 192*refinement
    yz = np.array([[.25, .25], [.75, .25], [.25, .75], [.75, .75]])*dx
    axis = (np.arange(planes)+.5)*dx/2
    slab = np.column_stack((np.repeat(axis, 4), np.tile(yz, (planes, 1))))
    x = np.concatenate((slab*np.array([-1., 1., 1.]), slab))
    ids = np.repeat([0, 1], len(slab))
    v = np.zeros_like(x)
    v[:, 0] = np.where(ids == 0, up, -up)
    grid = build_grid(x, ids, dx=dx, padding_cells=4)
    mask = jnp.zeros((grid.layout.node_count, 3), dtype=bool).at[:, 1:].set(True)
    grid = replace(grid, clamped=mask)
    state = make_state(x, np.full(len(x), (dx/2)**3), ids, material, grid, v=v)
    expected = hugoniot(up, material)
    end_time = 50*1e-4/expected[3]
    result, steps = advance_to(state, material, grid, ImpactConfig(strength=False), end_time, max_steps=12000)
    assert result.valid and result.time == end_time
    thermo = thermodynamics(result, material)
    selected = (np.abs(x[:, 0]) > .0022) & (np.abs(x[:, 0]) < .0032)
    measured = np.array([jnp.linalg.det(result.F)[selected].mean(),
                         thermo.pressure[selected].mean(), result.internal_energy[selected].mean()])
    reference = np.array(expected[:3], dtype=float)
    relative_energy_error = abs(float(energy_ledger(result, grid).sum()/energy_ledger(state, grid).sum()-1))
    assert relative_energy_error < 3e-11
    assert float(jnp.sum(result.mass*thermo.entropy)) > 0
    np.testing.assert_allclose(jnp.sum(result.mass[:, None]*result.v, axis=0), 0., atol=2e-15)
    assert not bool(jnp.any(result.ever_phase_extrapolated))
    assert float(jnp.min(result.minimum_entropy-result.initial_entropy)) >= -1e-8
    np.testing.assert_allclose(measured[:2], reference[:2], rtol=.02)
    return {'refinement': refinement, 'steps': int(steps), 'dx_m': dx,
        'relative_J_error': float(abs(measured[0]/reference[0]-1)),
        'relative_pressure_error': float(abs(measured[1]/reference[1]-1)),
        'relative_specific_energy_error': float(abs(measured[2]/reference[2]-1)),
        'minimum_entropy_from_initial': float(jnp.min(result.minimum_entropy-result.initial_entropy)),
        'worst_cumulative_negative_entropy_increments': float(result.negative_entropy_increments.min()),
        'mass_weighted_cumulative_negative_entropy_increments': float(jnp.sum(result.mass*result.negative_entropy_increments)/result.mass.sum()),
        'contact_projection_loss_J_by_body': np.asarray(result.contact_projection_loss).tolist(),
        'signed_contact_coupling_work_J_by_body': np.asarray(result.contact_coupling_work).tolist(),
        'relative_total_energy_error': relative_energy_error,
        'maximum_temperature_K': float(result.maximum_temperature.max()),
        'minimum_J_entire_history': float(result.minimum_J.min()),
        'maximum_J_entire_history': float(result.maximum_J.max()),
        'ever_phase_extrapolated': bool(jnp.any(result.ever_phase_extrapolated)),
        'ever_tension_extrapolated': bool(jnp.any(result.ever_tension_extrapolated))}


def test_planar_hugoniot_energy_error_reduces_with_spatial_refinement(capsys):
    # The contact-startup hot layer is excluded by a fixed physical sampling band.
    # Total energy is exact up to roundoff; the shock jump itself is discretized.
    reports = [planar_collision(refinement) for refinement in (1, 2, 4)]
    with capsys.disabled():
        print(json.dumps({'planar_strength_off_500_m_s': reports,
                          'scope': 'Explicit approximation; signed coupling work and local entropy increments are not physical heat or a discrete entropy-stability proof.'}))
    errors = [row['relative_specific_energy_error'] for row in reports]
    assert errors[0] < .08 and errors[1] < .05 and errors[2] < .02
    assert errors[1] < .75*errors[0] and errors[2] < .75*errors[1]
    assert reports[-1]['relative_pressure_error'] < .01
    assert reports[-1]['relative_J_error'] < .005
