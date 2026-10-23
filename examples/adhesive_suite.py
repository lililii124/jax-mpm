"""Compare normal pull-off, lateral release and multiple adhesive indenters."""

import argparse
from pathlib import Path

import jax

from .adhesive_array import SUITE_CASES, make_case as array_case
from .common import run


def make_case(scene, *, grid=48, ppc=3, frames=50):
    return array_case(grid=grid, ppc=ppc, frames=frames, suite_case=scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SUITE_CASES, default="adhesive-pull-off")
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/adhesion"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.grid < 8 or args.ppc < 1 or args.frames < 1:
        parser.error("grid >= 8, ppc >= 1 and frames >= 1 are required")
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(
        args.case, grid=args.grid, ppc=args.ppc, frames=args.frames
    )
    print(
        f"{args.case}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=args.output / args.case,
        label=args.case,
        forces=case.forces,
        observer=case.observer,
        metadata=case.description,
        gif=args.gif,
        render_output=not args.no_render,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
