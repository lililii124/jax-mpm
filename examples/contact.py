"""Oblique collision of two elastic blocks with optional Coulomb friction."""

import argparse
from pathlib import Path

import jax
from .suite import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", choices=("head-on", "oblique", "sliding", "rebound"), default="oblique"
    )
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/contact"))
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
