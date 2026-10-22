"""Release a prestrained 3D block for each elastic constitutive law."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import (
    Config,
    block,
    linear_elastic,
    neo_hookean,
    orthotropic,
    transverse_isotropic,
    blatz_ko,
)
from .common import run


def setup(name, grid=48):
    angle = np.pi / 6
    Q = jnp.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
    )
    Q_ti = jnp.array(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    materials = {
        "linear": linear_elastic(2e4, 0.2),
        "orthotropic": orthotropic((8e4, 2e4, 3e4), (0.2, 0.15, 0.25), (1e4, 1.5e4, 8e3), Q),
        "transverse-isotropic": transverse_isotropic(8e4, 2e4, 0.25, 0.3, 1.2e4, Q_ti),
        "neo-hookean": neo_hookean(2e4, 0.2),
        "blatz-ko": blatz_ko(8e3),
    }
    cfg = Config(dim=3, n_grid=grid, dt=2e-4, gravity=(0, 0, 0), boundary_cells=0, material=name)
    state = block(cfg, (0.35,) * 3, (0.65,) * 3, density=1000.0, dtype=jnp.float64)
    stretches = (
        jnp.array([1.12, 0.96, 0.98])
        if name in ("neo-hookean", "blatz-ko")
        else jnp.array([1.01, 0.997, 0.998])
    )
    F = jnp.diag(stretches)
    state = state._replace(x=0.5 + (state.x - 0.5) @ F.T, F=jnp.broadcast_to(F, state.F.shape))
    return state, materials[name], cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", "linear", "orthotropic", "transverse-isotropic", "neo-hookean", "blatz-ko"),
        default="all",
    )
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--substeps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/elastic_models"))
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    names = (
        ("linear", "orthotropic", "transverse-isotropic", "neo-hookean", "blatz-ko")
        if args.model == "all"
        else (args.model,)
    )
    for name in names:
        state, mat, cfg = setup(name, args.grid)
        run(
            (state,),
            (mat,),
            (cfg,),
            frames=args.frames,
            substeps=args.substeps,
            output=args.output / name,
            label=name,
            gif=args.gif,
        )


if __name__ == "__main__":
    main()
