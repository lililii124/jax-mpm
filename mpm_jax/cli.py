"""Run elastic and fluid examples and export trajectories."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from .core import Config, advance, elastic, fluid
from .initialization import block, diagnostics


def render(positions, speeds, times, config, output, scene, make_gif, field_label="Speed"):
    from .visualization import render_trajectory

    render_trajectory(output, gif=make_gif, field="damage" if field_label == "Damage" else "body")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("elastic", "dam-break"), default="elastic")
    parser.add_argument("--grid", type=int, default=48, help="cells per side")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--substeps", type=int, default=50)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--ppc", type=int, default=2, help="particles per cell per axis")
    parser.add_argument("--density", type=float, default=1000.0)
    parser.add_argument("--young", type=float, default=2e4)
    parser.add_argument("--poisson", type=float, default=0.2)
    parser.add_argument("--bulk", type=float, default=2e5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--x64", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 1 or args.substeps < 1 or args.ppc < 1:
        parser.error("frames, substeps and ppc must be positive")
    if (
        not all(np.isfinite(v) and v > 0 for v in (args.density, args.young, args.bulk))
        or not np.isfinite(args.poisson)
        or not -1 < args.poisson < 0.5
    ):
        parser.error("density, young and bulk must be finite and positive; -1 < poisson < 0.5")
    if args.gif and args.no_render:
        parser.error("--gif cannot be combined with --no-render")
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    kind = "elastic" if args.scene == "elastic" else "fluid"
    config = Config(n_grid=args.grid, dt=args.dt, material=kind, dim=args.dim)
    material = elastic(args.young, args.poisson) if kind == "elastic" else fluid(args.bulk)
    if kind == "elastic":
        lower, upper = (0.35, 0.55), (0.65, 0.85)
    else:
        lower, upper = (3 * config.dx, 3 * config.dx), (0.4, 0.72)
    if args.dim == 3:
        lower, upper = lower + (0.35,), upper + (0.65,)
    state = block(
        config,
        lower,
        upper,
        particles_per_cell=args.ppc,
        density=args.density,
        dtype=np.float64 if args.x64 else np.float32,
    )
    output = args.output or Path("outputs") / args.scene
    output.mkdir(parents=True, exist_ok=True)
    wave_speed = np.sqrt(
        (material.lam + 2 * material.mu if kind == "elastic" else material.bulk) / args.density
    )
    initial_cfl = args.dt * wave_speed / config.dx
    if initial_cfl > 0.4:
        parser.error(f"initial acoustic CFL={initial_cfl:.3f} exceeds 0.4; reduce --dt")
    print(
        f"JAX {jax.__version__} | {jax.devices()} | {state.x.shape[0]} particles | CFL={initial_cfl:.4f}",
        flush=True,
    )

    # Exclude compilation from the solver timing.
    start = perf_counter()
    warm = advance(state, material, config, args.substeps)
    warm.x.block_until_ready()
    compile_seconds = perf_counter() - start
    print(f"Compilation + warm-up: {compile_seconds:.3f} s", flush=True)
    positions, speeds, times = [np.asarray(state.x)], [np.linalg.norm(state.v, axis=1)], [0.0]
    rows = [{"time": 0.0, **diagnostics(state, config)}]
    solver_seconds = 0.0
    for frame in range(1, args.frames + 1):
        start = perf_counter()
        state = advance(state, material, config, args.substeps)
        state.x.block_until_ready()
        solver_seconds += perf_counter() - start
        t = frame * args.substeps * config.dt
        row = {"time": t, **diagnostics(state, config)}
        rows.append(row)
        positions.append(np.asarray(state.x))
        speeds.append(np.linalg.norm(state.v, axis=1))
        times.append(t)
        if frame == 1 or frame % 20 == 0 or frame == args.frames:
            print(
                f"Frame {frame}/{args.frames} t={t:.3f} J=[{row['min_J']:.3f}, {row['max_J']:.3f}]",
                flush=True,
            )
    positions, speeds, times = np.asarray(positions), np.asarray(speeds), np.asarray(times)
    np.savez_compressed(output / "trajectory.npz", x=positions, speed=speeds, time=times)
    arrays = {k: v for k, v in state._asdict().items() if k != "history"}
    arrays.update({f"history_{k}": v for k, v in state.history._asdict().items()})
    np.savez_compressed(output / "final_state.npz", **arrays)
    with (output / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scene": args.scene,
        "config": asdict(config),
        "material": dict(material._asdict()),
        "density": args.density,
        "particles_per_cell_per_axis": args.ppc,
        "particles": int(state.x.shape[0]),
        "frames": args.frames,
        "substeps": args.substeps,
        "steps": args.frames * args.substeps,
        "simulation_time": float(times[-1]),
        "dtype": str(state.x.dtype),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "compile_and_warmup_seconds": compile_seconds,
        "solver_seconds": solver_seconds,
        "steps_per_second": args.frames * args.substeps / solver_seconds,
        "min_J_over_frames": min(r["min_J"] for r in rows),
        "max_J_over_frames": max(r["max_J"] for r in rows),
        "initial": rows[0],
        "final": rows[-1],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_render:
        render(positions, speeds, times, config, output, args.scene, args.gif)
    print(
        f"Saved to {output.resolve()} | solver {solver_seconds:.3f} s ({summary['steps_per_second']:.1f} steps/s)"
    )


if __name__ == "__main__":
    main()
