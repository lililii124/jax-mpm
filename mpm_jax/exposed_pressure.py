"""Smooth front and back material visibility for a column of cell densities."""
import jax.numpy as jnp


def exposed_weights(density, *, from_top, epsilon=1e-8):
    density = jnp.asarray(density)
    if density.ndim < 1 or not 0 < epsilon < .5:
        raise ValueError('Expected a density array and a positive visibility regularization')
    rho = jnp.clip(density, epsilon, 1-epsilon)
    ordered = rho[..., ::-1] if from_top else rho
    product = jnp.cumprod(1-ordered, axis=-1)
    previous = jnp.concatenate((jnp.ones_like(product[..., :1]), product[..., :-1]), axis=-1)
    visible = ordered*previous
    visible = visible/jnp.maximum(jnp.sum(visible, axis=-1, keepdims=True), epsilon)
    return visible[..., ::-1] if from_top else visible
