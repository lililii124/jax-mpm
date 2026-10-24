"""Adhesive pull-off with an explicit concentric sinusoidal rigid surface."""
import argparse
from pathlib import Path

import jax
from mpm_jax.adhesion import RadialSinusoid
from .adhesive_array import make_case as array_case
from .common import run


def make_case(*, grid=48, ppc=3, frames=50, amplitude=0.008, wavelength=0.08):
    return array_case(grid=grid, ppc=ppc, frames=frames,
                      suite_case="adhesive-pull-off", preset="lift-hold-v4",
                      radial_profile=RadialSinusoid(amplitude, wavelength))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--amplitude", type=float, default=0.008)
    parser.add_argument("--wavelength", type=float, default=0.08)
    parser.add_argument("--output", type=Path, default=Path("outputs/adhesion-radial-sinusoidal-v1"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(grid=args.grid, ppc=args.ppc, frames=args.frames,
                              amplitude=args.amplitude, wavelength=args.wavelength)
    label = "radial-sinusoidal-pull-off"
    print(f"{label}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps", flush=True)
    run(case.states, case.materials, case.configs, frames=args.frames,
        substeps=substeps, output=args.output / label, label=label,
        forces=case.forces, observer=case.observer, metadata=case.description,
        gif=args.gif, render_output=not args.no_render, resume=args.resume)


if __name__ == "__main__":
    main()
