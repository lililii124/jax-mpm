"""Run the three-dimensional mechanics examples."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import jax
import numpy as np

from .common import run
from .scenarios import ELASTIC, PLASTIC, SCENES, setup


def execute(
    scene,
    *,
    model="neo-hookean",
    grid=48,
    ppc=2,
    frames=20,
    dt_factor=1.0,
    output=Path("outputs/suite"),
    gif=False,
    render_output=True,
    resume=False,
):
    case = setup(scene, model=model, grid=grid, ppc=ppc)
    if scene == "wave":
        model = "linear"
    substeps = int(np.ceil(case.duration / (frames * case.configs[0].dt * dt_factor)))
    configs = tuple(replace(c, dt=case.duration / (frames * substeps)) for c in case.configs)
    print(
        f"{scene}/{model}: grid={grid}^3, particles={sum(s.x.shape[0] for s in case.states)}, steps={frames * substeps}",
        flush=True,
    )
    return run(
        case.states,
        case.materials,
        configs,
        frames=frames,
        substeps=substeps,
        output=Path(output) / scene / model,
        label=f"{scene} / {model}",
        forces=case.forces,
        friction=case.friction,
        observer=case.observer,
        metadata=case.description,
        gif=gif,
        render_output=render_output,
        resume=resume,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all",) + SCENES, default="wave")
    parser.add_argument(
        "--model", choices=("all-elastic", "all-plastic") + ELASTIC + PLASTIC, default="neo-hookean"
    )
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/suite"))
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="continue from a matching saved checkpoint"
    )
    args = parser.parse_args()
    if args.list:
        print("\n".join(SCENES))
        return
    if args.frames < 1 or args.ppc < 1 or not 0 < args.dt_factor <= 1:
        parser.error("frames and ppc must be positive; dt-factor must be in (0, 1]")
    if args.model == "all-plastic" and args.case != "compression":
        parser.error("all-plastic is available with --case compression")
    if args.model == "all-elastic" and args.case not in (
        "cantilever",
        "torsion",
        "shear",
        "compression",
    ):
        parser.error("all-elastic requires a solid mechanics case")
    jax.config.update("jax_enable_x64", True)
    scenes = SCENES if args.case == "all" else (args.case,)
    summaries = []
    for scene in scenes:
        models = (
            ELASTIC
            if args.model == "all-elastic"
            else PLASTIC
            if args.model == "all-plastic"
            else (args.model,)
        )
        for model in models:
            _, summary = execute(
                scene,
                model=model,
                grid=args.grid,
                ppc=args.ppc,
                frames=args.frames,
                dt_factor=args.dt_factor,
                output=args.output,
                gif=args.gif,
                render_output=not args.no_render,
                resume=args.resume,
            )
            summaries.append(summary)
            jax.clear_caches()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
