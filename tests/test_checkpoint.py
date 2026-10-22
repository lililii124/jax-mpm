import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.common import run
from mpm_jax import Config, bilinear_plastic, block


def test_resume_preserves_particle_and_material_history(tmp_path):
    jax.config.update("jax_enable_x64", True)
    material = bilinear_plastic(
        young=2e4, poisson=0.2, yield_stress=20.0, tangent=100.0, density=1000.0
    )
    cfg = Config(
        dim=3, n_grid=12, dt=1e-3, material="bilinear", boundary_cells=0, gravity=(0.0, 0.0, 0.0)
    )
    state = block(
        cfg,
        (0.3,) * 3,
        (0.7,) * 3,
        particles_per_cell=1,
        material=material,
        density=1000.0,
        dtype=jnp.float64,
    )
    state = state._replace(v=state.v.at[:, 0].set(2 * (state.x[:, 1] - 0.5)))

    def interrupted(states, time):
        if time >= 0.004:
            raise RuntimeError("simulated interruption")
        return {}

    arguments = dict(frames=4, substeps=2, label="checkpoint", render_output=False)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(
            (state,),
            (material,),
            (cfg,),
            output=tmp_path / "resumed",
            observer=interrupted,
            **arguments,
        )
    checkpoint = json.loads((tmp_path / "resumed/checkpoint.json").read_text())
    assert checkpoint["frame"] == 1
    resumed, summary = run(
        (state,), (material,), (cfg,), output=tmp_path / "resumed", resume=True, **arguments
    )
    expected, _ = run((state,), (material,), (cfg,), output=tmp_path / "continuous", **arguments)
    for actual, reference in zip(jax.tree.leaves(resumed), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    assert float(resumed[0].history.plastic_strain.max()) > 0
    assert summary["completed"]
    with np.load(tmp_path / "resumed/trajectory.npz") as a, np.load(
        tmp_path / "continuous/trajectory.npz"
    ) as b:
        np.testing.assert_array_equal(a["time"], b["time"])
        np.testing.assert_allclose(a["x"], b["x"], atol=1e-12)
