import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mpm_jax.materials import initial_history, jh2, hjc, rht, update_jh2, update_hjc, update_rht

jax.config.update("jax_enable_x64", True)
DATA = Path(__file__).parent / "data/materials"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", MANIFEST["fixtures"], ids=lambda e: e["file"][:-4])
def test_supplied_reference_trace(entry):
    path = DATA / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    data = np.load(path)
    factory, update = {
        "jh2": (jh2, update_jh2),
        "hjc": (hjc, update_hjc),
        "rht": (rht, update_rht),
    }[entry["model"]]
    p = factory()
    de = data["increments"]
    strain = np.zeros((len(de), 3, 3))
    strain[:, 0, 0], strain[:, 1, 1], strain[:, 2, 2] = de[:, 0], de[:, 1], de[:, 2]
    strain[:, 0, 1] = strain[:, 1, 0] = de[:, 3]
    strain[:, 1, 2] = strain[:, 2, 1] = de[:, 4]
    strain[:, 0, 2] = strain[:, 2, 0] = de[:, 5]
    volumes = jnp.asarray(np.exp(np.cumsum(de[:, :3].sum(1))))
    dt = float(data["dt"])

    def advance(h, values):
        increment, J = values
        h = update(h, increment[None], J[None], dt, p)
        return h, h

    h = initial_history(1, jnp.float64, p, p.density)
    _, history = jax.jit(lambda h: jax.lax.scan(advance, h, (jnp.asarray(strain), volumes)))(h)
    assert np.all(history.valid)
    np.testing.assert_allclose(history.stress[:, 0], data["stress"], rtol=2e-7, atol=1.0)
    for i, field in enumerate(entry["fields"]):
        tolerance = 1.0 if field in ("pressure", "bulking") else 1e-5 if field == "energy" else 1e-8
        np.testing.assert_allclose(
            np.asarray(getattr(history, field))[:, 0],
            data["values"][:, i],
            rtol=2e-6,
            atol=tolerance,
        )
