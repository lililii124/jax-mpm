"""New scalar-oracle points; these are not the historical impact finalcheck."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpm_jax.materials.hjc import hjc, update_hjc, MAT111_SOURCE_SHA256
from mpm_jax.materials.history import initial_history

jax.config.update('jax_enable_x64', True)
TABLE = json.loads((Path(__file__).parent/'data/materials/hjc_current_mat111_points.json').read_text())


@pytest.mark.parametrize('case', TABLE['cases'], ids=lambda c:c['case'])
def test_current_mat111_scalar_oracle(case):
    assert TABLE['source_sha256']==MAT111_SOURCE_SHA256
    assert TABLE['jax_impact_validation_passed'] is False
    x = case['inputs']
    p = hjc(profile='mat111-compatible', D1=x['D1'])
    old = initial_history(1,jnp.float64,p,p.density)._replace(damage=jnp.array([x['initial_damage']]))
    if x['compacted']:
        old = old._replace(compaction=jnp.ones(1),plastic_volume=jnp.full(1,.1),max_compression=jnp.full(1,.2))
    new = jax.jit(update_hjc)(old,jnp.array([x['strain']]),jnp.array([1/(1+x['mu'])]),x['dt'],p)
    for name, expected in case['oracle'].items():
        np.testing.assert_allclose(np.asarray(getattr(new,name)[0]),expected,
            rtol=2e-11,atol=1e-6 if name in ('stress','pressure') else 1e-12)


def test_profile_is_explicit_and_existing_profiles_stay_separate():
    assert hjc().mat111_compatible is False
    assert hjc(profile='strain-floor').mat111_compatible is False
    assert hjc(profile='mat111-compatible').mat111_compatible is True
    with pytest.raises(ValueError):
        hjc(profile='mat111-compatible',cold_start_damage=True)
