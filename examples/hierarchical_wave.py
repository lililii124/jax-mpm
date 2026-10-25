"""Plane-strain pulse crossing a fixed coarse/fine/coarse grid transition."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpm_jax.boundaries import fixed_box
from mpm_jax.hierarchical_dynamics import advance, initial_state, kinetic_energy, strain_energy
from mpm_jax.hierarchical_grid import grid_stencil, hierarchical_grid, moment_matrix, uniform_grid
from mpm_jax.hierarchical_streaming import advance_streamed, initial_state_streamed, streamed_energy
from mpm_jax.materials import neo_hookean

jax.config.update("jax_enable_x64", True)


def sample_bar(resolution, h=.0025, transverse_ppc=2):
    lower = np.array([.020, .095, .095])
    dimensions = np.array([.160, .010, .010]) / h
    if not np.allclose(dimensions, np.rint(dimensions), rtol=0, atol=1e-10):
        raise ValueError("Cell spacing must divide the bar dimensions")
    axes = [lower[d] + (np.arange(n) + .5) * h for d, n in enumerate(np.rint(dimensions).astype(int))]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    refined = (centres[:, 0] >= .070) & (centres[:, 0] < .130)
    if resolution != "hierarchical":
        refined[:] = resolution == "fine"
    x, volume = [], []
    base_ppc = np.array([2, transverse_ppc, transverse_ppc])
    for selected, ppc in ((~refined, base_ppc), (refined, 2 * base_ppc)):
        axes = [(np.arange(n) + .5) / n - .5 for n in ppc]
        offsets = h * np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        points = (centres[selected, None, :] + offsets).reshape(-1, 3)
        x.append(points)
        volume.append(np.full(len(points), h**3 / np.prod(ppc)))
    return np.concatenate(x), np.concatenate(volume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=("coarse", "fine", "hierarchical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-cells", type=int, default=80)
    parser.add_argument("--transverse-ppc", type=int, choices=(1, 2), default=2)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--transverse-padding-cells", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    density, amplitude, width, centre = 1200., .02, .006, .045
    material = neo_hookean(20e6, .3)
    wave_speed = np.sqrt((material.lam + 2 * material.mu) / density)
    h = .2 / args.coarse_cells
    duration, chunk = .00065, 20
    planned_steps = int(np.ceil(600 * args.coarse_cells / 80 / chunk)) * chunk
    dt = duration / planned_steps
    steps = planned_steps if args.stop_after_steps is None else args.stop_after_steps
    if steps < chunk or steps > planned_steps or steps % chunk:
        parser.error("Stop step must be a multiple of 20 within the planned run")
    if args.batch_size < 0:
        parser.error("Batch size must be nonnegative; zero selects the full stencil")
    if args.transverse_padding_cells is not None and args.transverse_padding_cells < 4:
        parser.error("Transverse storage padding requires at least four coarse cells")
    reference, volume = sample_bar(args.resolution, h, args.transverse_ppc)
    cells = args.coarse_cells * (2 if args.resolution == "fine" else 1)
    grid_h = .2 / cells
    bounds = ((0, round(.08 / grid_h), round(.08 / grid_h)),
              (cells, round(.12 / grid_h), round(.12 / grid_h)))
    if args.transverse_padding_cells is not None:
        lower = int(np.floor((.095 - args.transverse_padding_cells*h)/grid_h))
        upper = int(np.ceil((.105 + args.transverse_padding_cells*h)/grid_h))
        bounds = ((0, lower, lower), (cells, upper, upper))
    if args.resolution == "hierarchical":
        grid = hierarchical_grid(cells=cells, domain_size=.2, node_bounds=bounds,
                                 refined_lower=(.070 - .5*h, .095 - 3.5*h, .095 - 3.5*h),
                                 refined_upper=(.130 + .5*h, .105 + 3.5*h, .105 + 3.5*h))
    else:
        grid = uniform_grid(cells=cells, domain_size=.2, node_bounds=bounds,
                            offset=0. if args.resolution == "coarse" else .5)
    phase = (reference[:, 0] - centre) / width
    pulse = amplitude * np.exp(-phase**2)
    displacement = -amplitude / wave_speed * width * np.sqrt(np.pi) / 2 * (erf(phase) + 1.)
    x, v = reference.copy(), np.zeros_like(reference)
    x[:, 0] += displacement
    v[:, 0] = pulse
    F = np.broadcast_to(np.eye(3), (len(x), 3, 3)).copy()
    F[:, 0, 0] -= pulse / wave_speed
    C = np.zeros_like(F)
    C[:, 0, 0] = -2 * phase / width * pulse / F[:, 0, 0]
    initializer = initial_state_streamed if args.batch_size else initial_state
    storage_options = {"batch_size": args.batch_size} if args.batch_size else {}
    state = initializer(x, volume, density, grid, velocity=v, deformation=F, affine=C, **storage_options)
    boundaries = (fixed_box((0.,) * 3, (.2,) * 3, components=(False, True, True)),)
    initial_energy = float(streamed_energy(state, material, grid, **storage_options) if args.batch_size else
                           kinetic_energy(state, moment_matrix(grid_stencil(state.x, grid)))
                           + strain_energy(state, material))
    np.savez_compressed(args.output / "initial.npz", reference=reference, **state._asdict())
    start = perf_counter()
    integrator = advance_streamed if args.batch_size else advance
    compiled = integrator.lower(state, material, grid, dt, steps=chunk, boundaries=boundaries,
                                **storage_options).compile()
    compile_seconds = perf_counter() - start
    records, frames = [], []
    start = perf_counter()
    for block in range(steps // chunk):
        state, record = compiled(state, material, grid, dt)
        jax.block_until_ready(state)
        host = jax.tree.map(np.asarray, record)
        records.append(host)
        if not np.all(host["valid"]) or np.max(host["cfl_bound"]) > .25:
            raise RuntimeError("Invalid state or acoustic CFL bound exceeded")
        frames.append(dict(step=(block + 1) * chunk, time=float(state.time),
                           energy=float(host["kinetic_energy"][-1] + host["strain_energy"][-1])))
        np.savez_compressed(args.output / f"records_{block:05d}.npz", **host)
        np.savez_compressed(args.output / "checkpoint.npz", **state._asdict())
        if (block + 1) % 5 == 0:
            print(f"{args.resolution}: {(block + 1) * chunk}/{steps}", flush=True)
    solver_seconds = perf_counter() - start
    values = {key: np.concatenate([r[key] for r in records]) for key in records[0]}
    np.savez_compressed(args.output / "step_records.npz", **values)
    completed_duration = steps*dt
    exact = amplitude * np.exp(-((reference[:, 0] - wave_speed * completed_duration - centre) / width)**2)
    actual = np.asarray(state.v)[:, 0]
    mass = np.asarray(state.mass)
    l2 = np.sqrt(np.sum(mass * (actual - exact)**2) / np.sum(mass * exact**2))
    reflected = reference[:, 0] < .065
    reflection = np.sum(mass[reflected] * actual[reflected]**2) / np.sum(mass * pulse**2)
    energy = float(values["kinetic_energy"][-1] + values["strain_energy"][-1])
    components = ["p2g_change", "g2p_change", "moment_advection", "body_work", "boundary_work",
                  "boundary_loss", "material_remainder", "force_time_remainder", "force_gradient_remainder"]
    files = [Path(__file__).resolve(), ROOT / "mpm_jax/hierarchical_grid.py",
             ROOT / "mpm_jax/hierarchical_dynamics.py", ROOT / "mpm_jax/materials/elastic.py",
             ROOT / "mpm_jax/boundaries.py"]
    if args.batch_size:
        files.append(ROOT / "mpm_jax/hierarchical_streaming.py")
    report = {
        "resolution": args.resolution, "particles": len(x), "grid_nodes": len(grid.positions),
        "coarse_cells": args.coarse_cells, "coarse_spacing_m": h, "transverse_ppc": args.transverse_ppc,
        "node_bounds": bounds,
        "steps": steps, "planned_steps": planned_steps, "duration_s": completed_duration,
        "target_duration_s": duration, "complete": steps == planned_steps,
        "batch_size": args.batch_size, "dt_s": dt, "wave_speed_m_s": wave_speed,
        "physical_case": "0.16 x 0.01 x 0.01 m bar; transverse velocities fixed, axial end faces free",
        "reference": "Linear plane-strain right-going Gaussian; neo-Hookean amplitude/c = 1.34e-4",
        "initial_energy_J": initial_energy, "final_energy_J": energy,
        "relative_energy_change": (energy - initial_energy) / initial_energy,
        "linear_reference_velocity_relative_L2": float(l2), "left_residual_velocity_energy_fraction": float(reflection),
        "max_ledger_error_relative_initial_energy": float(np.max(np.abs(values["ledger_error"])) / initial_energy),
        "summed_ledger_error_relative_initial_energy": float(np.sum(values["ledger_error"]) / initial_energy),
        "max_force_gradient_remainder_J": float(np.max(np.abs(values["force_gradient_remainder"]))),
        "min_J": float(values["min_J"].min()), "max_cfl_bound": float(values["cfl_bound"].max()),
        "energy_components_J": {name: float(values[name].sum()) for name in components},
        "compile_seconds": compile_seconds, "solve_and_checkpoint_seconds": solver_seconds,
        "frames": frames, "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "source": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("source", "frames")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
