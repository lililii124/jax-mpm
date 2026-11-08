import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.shared_cell_design import (
    build_shared_cell_map, fixed_count_material_mask, project_fixed_volume,
)


def test_shared_3x3_design_preserves_volume_and_derivatives():
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(73)
    q = jnp.asarray(rng.normal(size=(3, 4, 5)) * .25)
    direction = jnp.asarray(rng.normal(size=q.shape))
    direction /= jnp.linalg.norm(direction)
    weights = jnp.asarray(rng.normal(size=q.shape))
    objective = lambda x: jnp.sum(project_fixed_volume(x, .35) * weights)
    step = 1e-5
    fd = (objective(q + step * direction) - objective(q - step * direction)) / (2 * step)
    np.testing.assert_allclose(jnp.vdot(jax.grad(objective)(q), direction), fd, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(jax.grad(lambda x: jnp.sum(project_fixed_volume(x, .35)))(q), 0, atol=1e-10)
    density = project_fixed_volume(q, .35)
    np.testing.assert_allclose(density.mean(), .35, atol=1e-12)
    mapped = jax.jit(lambda x: build_shared_cell_map(x, (5, 4, 3), (3, 3, 1),
                                                     cell_size=(2., 3., 4.)))(density)
    xyz, ids, ijk = map(np.asarray, (mapped.full_xyz, mapped.cell_id, mapped.local_ijk))
    expected = (ijk + .5) * np.array([2/5, 3/4, 4/3]) + ids * np.array([2, 3, 4])
    np.testing.assert_allclose(xyz, expected, atol=1e-12)
    np.testing.assert_array_equal(mapped.full_density, np.tile(np.asarray(density).ravel(), 9))
    _, mask = fixed_count_material_mask(mapped, 21)
    cells = np.asarray(mask).reshape(9, -1)
    assert int(cells.sum()) == 189
    assert np.all(cells == cells[0])
