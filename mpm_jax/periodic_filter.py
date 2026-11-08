"""Differentiable cell-centered filter for shared periodic material designs."""
import jax.numpy as jnp


def filter_cell_logits(logits, passes=0):
    """Smooth logits with wrap in x/y and zero normal gradient at z faces.

    Each pass applies a [1, 2, 1]/4 kernel along all three grid axes. This
    limits grid-scale variation but does not guarantee connected solid regions.
    """
    field = jnp.asarray(logits)
    if field.ndim != 3 or passes < 0:
        raise ValueError('Expected a three-dimensional field and nonnegative passes')
    for _ in range(passes):
        for axis in (1, 2):
            field = (jnp.roll(field, 1, axis=axis) + 2 * field +
                     jnp.roll(field, -1, axis=axis)) * .25
        padded = jnp.pad(field, ((1, 1), (0, 0), (0, 0)), mode='edge')
        field = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) * .25
    return field
