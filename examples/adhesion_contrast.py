"""A matched normal pull-off pair with a stationary withdrawal stage."""

import argparse
from pathlib import Path

import jax

from .adhesive_array import make_case as array_case
from .common import run


CASES = ("nonadhesive-pull-off", "adhesive-pull-off")
PROFILES = {"v2": "lift-hold", "v3": "lift-hold-v3", "v4": "lift-hold-v4"}


def make_case(scene, *, grid=48, ppc=3, frames=50, profile="v2"):
    if scene not in CASES:
        raise ValueError("contrast requires a nonadhesive or adhesive pull-off case")
    if profile not in PROFILES:
        raise ValueError("unknown adhesion contrast profile")
    return array_case(
        grid=grid, ppc=ppc, frames=frames, suite_case=scene, preset=PROFILES[profile]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="adhesive-pull-off")
    parser.add_argument("--profile", choices=PROFILES, default="v2",
                        help="v2: 55 mm/72 J/m2; v3: 35 mm/72 J/m2; v4: 35 mm/144 J/m2")
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        help="default: outputs/adhesion-visible for v2; outputs/adhesion-visible-v3 or -v4 otherwise")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.grid < 8 or args.ppc < 1 or args.frames < 1:
        parser.error("grid >= 8, ppc >= 1 and frames >= 1 are required")
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(args.case, grid=args.grid, ppc=args.ppc, frames=args.frames, profile=args.profile)
    output = args.output or Path("outputs/adhesion-visible" if args.profile == "v2"
                                else f"outputs/adhesion-visible-{args.profile}")
    print(
        f"{args.case}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps",
        flush=True,
    )
    run(
        case.states, case.materials, case.configs,
        frames=args.frames, substeps=substeps, output=output / args.case,
        label=args.case, forces=case.forces, observer=case.observer,
        metadata=case.description, gif=args.gif,
        render_output=not args.no_render, resume=args.resume,
    )


if __name__ == "__main__":
    main()
