"""Strain and validity guards used by the toe excavation case."""
import jax
import jax.numpy as jnp


def principal_hencky_max(F):
    # Works for both soil 3x3 and foundation 2x2. No singular-value decomposition.
    values = jnp.linalg.eigvalsh(jnp.swapaxes(F, -1, -2) @ F)
    logs = 0.5 * jnp.log(jnp.maximum(values, jnp.finfo(F.dtype).tiny))
    return jnp.max(jnp.abs(logs))


def soil_elastic_compliance_max(stress):
    G = 5.86e6
    nu = 0.28
    bulk = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))
    trace = jnp.trace(stress, axis1=-2, axis2=-1)
    dev = stress - trace[:, None, None] * jnp.eye(3) / 3
    estimate = dev / (2 * G) + trace[:, None, None] * jnp.eye(3) / (9 * bulk)
    return jnp.max(jnp.abs(jnp.linalg.eigvalsh(estimate)))


def step_guards(before, after):
    total = jnp.asarray(0.0)
    increment = jnp.asarray(0.0)
    minimum_J = jnp.asarray(jnp.inf)
    good = jnp.asarray(True)
    for old, new in zip(before, after):
        total = jnp.maximum(total, principal_hencky_max(new.F))
        increment = jnp.maximum(increment, principal_hencky_max(new.F @ jnp.linalg.inv(old.F)))
        minimum_J = jnp.minimum(minimum_J, jnp.min(jnp.linalg.det(new.F)))
        good = good & jnp.all(new.valid) & jnp.all(new.history.valid)
        for leaf in jax.tree_util.tree_leaves(new):
            good = good & jnp.all(jnp.isfinite(leaf))
    elastic = soil_elastic_compliance_max(after[0].history.stress)
    metrics = jnp.stack((total, increment, elastic, -minimum_J))
    good = good & jnp.all(jnp.isfinite(metrics)) & (minimum_J > 0)
    return metrics, good
