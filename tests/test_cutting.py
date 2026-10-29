import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingWedgeTool, block, step
from mpm_jax.materials import johnson_cook, johnson_cook_displacement
from mpm_jax.materials.history import equivalent, initial_history
from mpm_jax.materials.johnson_cook import update_johnson_cook
from mpm_jax.materials.johnson_cook_displacement import update_johnson_cook_displacement

jax.config.update('jax_enable_x64', True)


def test_displacement_softening_and_legacy_initiation_are_distinct():
    parameters = dict(young=1e6, poisson=.25, A=100., B=0., n=1., C=0.,
                      heat_fraction=0., failure=(.001, 0., 0., 0., 0.))
    material = johnson_cook_displacement(**parameters, damage_length=.01,
                                        failure_displacement=1e-4)
    old = initial_history(1, jnp.float64, material)
    strain = jnp.diag(jnp.array([.01, -.005, -.005]))[None]
    new = update_johnson_cook_displacement(old, strain, jnp.ones(1), 1e-3, material)
    increment = (3 * material.mu * .01 - material.A) / (3 * material.mu)
    expected_softening = .01 * (increment - .001) / 1e-4
    np.testing.assert_allclose(new.damage_softening, expected_softening, rtol=1e-12)
    np.testing.assert_allclose(equivalent(new.stress), (1-expected_softening)*100., rtol=1e-12)
    assert bool(new.valid[0]) and not bool(new.failed[0]) and float(new.damage[0]) == 1.
    failed = update_johnson_cook_displacement(new, strain, jnp.ones(1), 1e-3, material)
    assert bool(failed.valid[0]) and bool(failed.failed[0])
    np.testing.assert_array_equal(failed.stress, 0.)
    legacy = update_johnson_cook(old, strain, jnp.ones(1), 1e-3,
                               johnson_cook(**parameters))
    assert bool(legacy.valid[0]) and bool(legacy.failed[0])


def test_plane_strain_retains_out_of_plane_stress():
    material = johnson_cook_displacement(young=1e6, poisson=.25, A=1e12)
    config = Config(dim=2, n_grid=16, dt=1e-4, material='johnson-cook-displacement',
                    gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.3,.3), (.7,.7), particles_per_cell=1,
                  material=material, density=material.density, dtype=jnp.float64)
    gradient = jnp.diag(jnp.array([.1, 0.]))
    state = state._replace(v=state.x @ gradient.T,
                           C=jnp.broadcast_to(gradient,state.C.shape))
    result = step(state, material, config)
    assert bool(result.valid) and bool(result.history.valid.all())
    np.testing.assert_allclose(result.F[:,2,2], 1., atol=1e-14)
    np.testing.assert_allclose(result.F[:,:2,2], 0., atol=1e-14)
    np.testing.assert_allclose(result.F[:,2,:2], 0., atol=1e-14)
    expected = (material.bulk - 2*material.mu/3)*np.log1p(config.dt*.1)
    np.testing.assert_allclose(result.history.stress[:,2,2], expected, rtol=3e-5)


def test_positive_clearance_and_event_aligned_motion():
    tool = MovingWedgeTool(tip=(0.,0.), speed=1., clearance_angle_deg=6.,
                           translation_path=((0.,0.,0.),(1.,1.,0.),(2.,1.,1.)))
    polygon = np.asarray(tool._polygon(0.,jnp.float64))
    assert polygon[2,0] < 0 and polygon[2,1] > 0
    assert bool(tool.interval_is_linear(.9,1.))
    assert not bool(tool.interval_is_linear(.9,1.1))
    np.testing.assert_allclose(tool.interval_velocity(1.,1.1,jnp.float64),[0.,1.])
