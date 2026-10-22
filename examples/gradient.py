"""Differentiate a deformed block's response with respect to Young's modulus.

Run from the project root: python -m examples.gradient
"""

import json

import jax
import jax.numpy as jnp

from mpm_jax import Config, advance, block, elastic, validate_state


def main():
    jax.config.update("jax_enable_x64", True)
    config = Config(n_grid=24, dt=2e-4, gravity=(0.0, 0.0), boundary_cells=0)
    state = block(config, lower=(0.4, 0.4), upper=(0.6, 0.6), dtype=jnp.float64)
    state = state._replace(F=state.F.at[:, 0, 0].set(1.04).at[:, 1, 1].set(0.98))

    def response(young):
        final = advance(state, elastic(young), config, steps=60)
        return jnp.mean((final.x[:, 0] - 0.5) ** 2)

    evaluate = jax.jit(jax.value_and_grad(response))
    young = 20000.0
    value, derivative = evaluate(young)
    h = 2.0
    finite_difference = (response(young + h) - response(young - h)) / (2 * h)
    final = advance(state, elastic(young), config, steps=60)
    validate_state(final, config)
    print(
        json.dumps(
            {
                "young": young,
                "response": float(value),
                "autodiff": float(derivative),
                "finite_difference": float(finite_difference),
                "relative_error": float(
                    abs(derivative - finite_difference) / abs(finite_difference)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
