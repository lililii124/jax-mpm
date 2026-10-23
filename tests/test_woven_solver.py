import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.common import run
from mpm_jax import Config, block, woven_elastic
from mpm_jax.visualization import trajectory_arrays


def test_woven_run_restores_labels_and_heterogeneous_material(tmp_path):
    jax.config.update("jax_enable_x64", True)
    config = Config(
        dim=3, n_grid=12, dt=1e-4, material="woven-composite",
        gravity=(0.0, 0.0, 0.0), boundary_cells=0,
    )
    state = block(config, (0.35,) * 3, (0.65,) * 3, particles_per_cell=1, dtype=jnp.float64)
    count = len(state.x)
    phase = np.arange(count) % 2
    orientation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    orientation[::3] = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    material = woven_elastic(
        matrix_young=2000.0, matrix_poisson=0.3, matrix_density=1100.0,
        yarn_young=(40000.0, 4000.0, 4000.0), yarn_poisson=(0.25, 0.25, 0.3),
        yarn_shear=(1500.0, 2000.0, 2000.0), yarn_density=1400.0,
        phase=phase, orientation=orientation,
    )
    state = state._replace(
        mass=state.volume * jnp.where(phase == 1, 1400.0, 1100.0),
        v=state.v.at[:, 0].set(0.1 * (state.x[:, 1] - 0.5)),
    )
    fields = {"phase": phase.astype(np.uint8), "family_id": np.where(phase, 0, -1).astype(np.int8)}
    arguments = dict(frames=3, substeps=2, label="woven-checkpoint", render_output=False, point_fields=fields)

    def interrupt(states, time):
        if time >= 4e-4:
            raise RuntimeError("interrupt test")
        return {}

    with pytest.raises(RuntimeError, match="interrupt test"):
        run((state,), (material,), (config,), output=tmp_path / "resumed", observer=interrupt, **arguments)
    resumed, summary = run(
        (state,), (material,), (config,), output=tmp_path / "resumed", resume=True, **arguments,
    )
    continuous, _ = run((state,), (material,), (config,), output=tmp_path / "continuous", **arguments)
    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(continuous)):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert summary["completed"] and summary["min_J"] > 0
    assert summary["final"][0]["strain_energy"] > 0
    np.testing.assert_allclose(resumed[0].mass, state.mass, rtol=0, atol=0)
    with trajectory_arrays(tmp_path / "resumed") as trajectory:
        for name, value in fields.items():
            np.testing.assert_array_equal(trajectory[name], value)
    checkpoint = json.loads((tmp_path / "resumed/checkpoint.json").read_text())
    assert checkpoint["frame"] == 3
    changed = material._replace(orientation=material.orientation.at[0].set(jnp.eye(3)))
    with pytest.raises(ValueError, match="different case"):
        run((state,), (changed,), (config,), output=tmp_path / "resumed", resume=True, **arguments)
