from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax import Config, block, elastic
from mpm_jax.core import stencil
from mpm_jax.crack_mpm import TensileTipLaw, crack_fields
from mpm_jax.polyline_crack import advance_tip, link_fields, make_path, step

jax.config.update('jax_enable_x64', True)


def fields(starts, ends, vertices):
    starts, ends = jnp.asarray(starts, dtype=float), jnp.asarray(ends, dtype=float)
    support = SimpleNamespace(dpos=(ends-starts)[:, None], indices=jnp.zeros((len(starts), 1), dtype=int))
    path = make_path(vertices, capacity=len(vertices))
    return link_fields(starts, support, path.vertices, path.count)[:, 0]


def test_vertex_crossings_count_once_and_tangencies_do_not_split():
    starts = [[0., -.5], [0., 1.5], [0., .5], [0., -.5]]
    ends = [[0., 1.5], [0., -.5], [0., 1.5], [0., .5]]
    np.testing.assert_array_equal(fields(starts, ends, [[-1., 0.], [0., 1.], [1., 0.]]), [1, 1, 1, 0])
    np.testing.assert_array_equal(fields([[-2., 1.]], [[2., 1.]], [[-1., 0.], [0., 1.], [1., 0.]]), [0])
    np.testing.assert_array_equal(fields([[-2., .5]], [[2., .5]], [[-1., 0.], [0., 1.], [1., 0.]]), [0])
    np.testing.assert_array_equal(fields([[0., -.5], [0., 1.5]], [[0., 1.], [0., 1.]],
                                         [[-1., 0.], [0., 1.], [1., 0.]]), [1, 0])


def test_straight_path_matches_existing_transfer_and_rejects_self_intersection():
    rng = np.random.default_rng(7)
    position = jnp.asarray(rng.uniform(.1, .9, (100, 2)))
    config = Config(dim=2, n_grid=16)
    support = stencil(position, config)
    segment = jnp.asarray([[.2, .5], [.8, .5]])
    np.testing.assert_array_equal(link_fields(position, support, segment, jnp.asarray(2)), crack_fields(position, support, segment))
    with pytest.raises(ValueError, match='Self-intersecting'):
        make_path([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])


def test_bent_crack_keeps_both_rigid_velocities_and_total_momentum():
    config = Config(dim=2, n_grid=32, dt=1e-6, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.25, .375), (.75, .625), particles_per_cell=2, dtype=jnp.float64)
    line = .5+.25*jnp.abs(state.x[:, 0]-.5)
    side = jnp.where(state.x[:, 1] > line, 1., -1.)
    state = state._replace(v=jnp.stack((jnp.zeros_like(side), side), axis=-1))
    path = make_path([[.15, .5875], [.5, .5], [.85, .5875]], capacity=8)
    # The endpoints are outside this patch. Test the physical particle transfer;
    # face support is tested separately with material-attached endpoints below.
    following, _ = step(state, path, elastic(), config)
    np.testing.assert_allclose(following.v, state.v, atol=3e-13)
    np.testing.assert_allclose(jnp.sum(state.mass[:, None]*state.v, axis=0),
                               jnp.sum(following.mass[:, None]*following.v, axis=0), atol=1e-12)


def test_material_crack_markers_translate_with_the_body():
    config = Config(dim=2, n_grid=24, dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    state = block(config, (.25, .25), (.75, .75), velocity=(.3, -.2), dtype=jnp.float64)
    path = make_path([[.25, .5], [.4, .5], [.55, .55]], capacity=8)
    following, moved = step(state, path, elastic(), config)
    assert bool(following.valid & moved.valid)
    np.testing.assert_allclose(moved.vertices[:3]-path.vertices[:3],
                               np.tile(config.dt*np.array([.3, -.2]), (3, 1)), atol=1e-14)
    np.testing.assert_allclose(following.v, np.tile([.3, -.2], (len(state.x), 1)), atol=1e-13)


def test_tension_selects_kink_direction_and_capacity_is_explicit():
    config = Config(dim=2, n_grid=32, dt=1e-5, gravity=(0., 0.), boundary_cells=0)
    material = elastic(young=1e6, poisson=.2)
    state = block(config, (.3, .3), (.7, .7), dtype=jnp.float64)
    path = make_path([[.2, .5], [.5, .5]], capacity=8)
    law = TensileTipLaw(1000., .1, 100., .1)
    same, report = advance_tip(state, path, material, config, law, .01)
    assert int(same.count) == 2 and float(report.increment) == 0
    normal = jnp.array([-.5, jnp.sqrt(.75)])
    state = state._replace(F=state.F+.02*jnp.outer(normal, normal))
    @jax.jit
    def grow(s, p):
        result, report = advance_tip(s, p, material, config, law, .0005)
        result = jax.lax.cond(p.valid, lambda _: result, lambda _: p, operand=None)
        return result, report
    advanced, report = grow(state, path)
    assert bool(advanced.valid) and int(advanced.count) == 3 and float(report.increment) > 0
    np.testing.assert_allclose(report.direction, [np.sqrt(.75), .5], atol=1e-13)
    slow = TensileTipLaw(1000., .1, 1., .1)
    pending, slow_report = advance_tip(state, path, material, config, slow, .0005)
    assert int(pending.count) == 2 and float(pending.pending) > 0 and float(slow_report.increment) == 0
    np.testing.assert_array_equal(pending.vertices, path.vertices)
    full = make_path([[.2, .5], [.5, .5]], capacity=2)
    stopped, report = advance_tip(state, full, material, config, law, .0005)
    assert not bool(stopped.valid) and bool(report.capacity_exceeded)
    np.testing.assert_array_equal(stopped.vertices, full.vertices)
