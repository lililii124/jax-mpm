"""3D dam break, sloshing and a fluid-driven deformable block."""

import argparse
from pathlib import Path

import jax
from .scenarios import fluid_case
from .suite import execute


def setup(scene, grid=48, ppc=2):
    case = fluid_case(scene, grid, ppc)
    return case.states, case.materials, case.configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("dam-break", "sloshing", "fsi", "water-entry"), default="fsi"
    )
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/fluids"))
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    execute(
        args.scene,
        grid=args.grid,
        ppc=args.ppc,
        frames=args.frames,
        output=args.output,
        gif=args.gif,
    )


if __name__ == "__main__":
    main()
