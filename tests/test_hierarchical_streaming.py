import jax
import numpy as np
import pytest

from mpm_jax.boundaries import VelocityBox, fixed_box
from mpm_jax.hierarchical_dynamics import advance, initial_state, kinetic_energy, strain_energy
from mpm_jax.hierarchical_grid import grid_stencil, hierarchical_grid, moment_matrix, uniform_grid
from mpm_jax.hierarchical_streaming import advance_streamed, initial_state_streamed, streamed_energy
from mpm_jax.materials import neo_hookean

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("mode,batch", [("uniform", 31), ("hierarchical", 64), ("hierarchical", 512)])
def test_particle_batches_retain_full_dynamics_and_boundary_work(mode, batch):
    grid = (uniform_grid(cells=16, domain_size=1., offset=.5) if mode == "uniform" else
            hierarchical_grid(cells=16, domain_size=1., refined_lower=(3.5/16,)*3,
                              refined_upper=(12.5/16,)*3))
    rng = np.random.default_rng(571)
    x = rng.uniform((.12, .3, .3), (.88, .7, .7), (293, 3))
    volume = rng.uniform(1e-5, 2e-5, len(x))
    density = rng.uniform(9., 11., len(x))
    A = np.array([[.05, -.2, .02], [.2, -.01, 0.], [0., .03, .01]])
    deformation = np.broadcast_to(np.eye(3), (len(x), 3, 3)).copy()
    deformation[:, 0, 0] += .02*np.sin(3*x[:, 0])
    deformation[:, 1, 2] += .01*np.cos(x[:, 1])
    kwargs = dict(velocity=(x-.5)@A.T, deformation=deformation, affine=A)
    original = initial_state(x, volume, density, grid, **kwargs)
    streamed = initial_state_streamed(x, volume, density, grid, batch_size=batch, **kwargs)
    for a, b in zip(original, streamed):
        np.testing.assert_allclose(a, b, rtol=2e-14, atol=1e-16)
    material = neo_hookean(20., .25)
    expected_energy = kinetic_energy(original, moment_matrix(grid_stencil(original.x, grid)))+strain_energy(original, material)
    np.testing.assert_allclose(streamed_energy(streamed, material, grid, batch_size=batch),
                               expected_energy, rtol=2e-13, atol=1e-16)
    conditions = (fixed_box((0., 0., 0.), (.2, 1., 1.)),
                  VelocityBox((.8, 0., 0.), (1., 1., 1.), (.02, 0., 0.)))
    options = dict(steps=25, gravity=(.2, -.3, .1), boundaries=conditions)
    expected, a = advance(original, material, grid, .001, **options)
    actual, b = advance_streamed(streamed, material, grid, .001, batch_size=batch, **options)
    assert bool(actual.valid) and np.asarray(b['valid']).all()
    assert float(np.abs(np.asarray(a['boundary_work'])).sum()) > 0
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=2e-11, atol=3e-13)
    assert a.keys() == b.keys()
    for name in a:
        np.testing.assert_allclose(a[name], b[name], rtol=2e-10, atol=3e-13, err_msg=name)
