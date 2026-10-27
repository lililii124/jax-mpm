"""Re-run the saved mixed holdout with true/fitted MAT002 parameters and save particle frames."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples import inverse_orthotropic_svk as model
from mpm_jax.total_lagrangian import NodalBoundary, initial_state, sample_particles, step


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_frames(engineering, cfg, ref, boundaries, steps: int, stride: int):
    theta = model.encode_engineering(engineering)
    fixed, pattern, phase, phase_v, phase_a, _, _, _receipt = boundaries
    zero = jnp.zeros(3, dtype=jnp.float64)
    indices = jnp.arange(stride, dtype=jnp.int32)

    def run(th):
        material = model.make_material(th)
        state0 = initial_state(ref, material, cfg, zero)

        def group(state, group_index):
            start = group_index * stride

            def substep(s, local_index):
                i = start + local_index
                b0 = NodalBoundary(fixed, phase[i] * pattern,
                                   phase_v[i] * pattern, phase_a[i] * pattern)
                b1 = NodalBoundary(fixed, phase[i + 1] * pattern,
                                   phase_v[i + 1] * pattern,
                                   phase_a[i + 1] * pattern)
                following, _ = step(s, ref, material, cfg, zero, zero, b0, b1)
                return following, None

            following, _ = jax.lax.scan(substep, state, indices)
            positions = sample_particles(following, ref).x
            return following, positions

        final, frames = jax.lax.scan(group, state0,
                                      jnp.arange(steps // stride, dtype=jnp.int32))
        return final, frames

    final, frames = jax.jit(run)(theta)
    jax.block_until_ready(frames)
    if not bool(np.asarray(final.valid)):
        raise RuntimeError("holdout particle trajectory became invalid")
    # Include the actual undeformed state as frame zero; subsequent frames are
    # sampled from the exact MPM state every `stride` integration steps.
    frames = np.concatenate((np.asarray(ref.positions)[None, ...],
                             np.asarray(frames)), axis=0)
    if not np.isfinite(frames).all():
        raise RuntimeError("particle output contains nonfinite coordinates")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--holdout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    result = json.loads(args.result.read_text(encoding="utf-8"))
    with np.load(args.holdout, allow_pickle=False) as data:
        expected_truth = data["truth_u"].copy()
        expected_fit = data["inferred_u"].copy()
        sensor_ids = data["sensor_particle_stencil"].copy()
        sensor_weights = data["sensor_weights"].copy()
        times = data["time_s"].copy()

    h = result["holdout"]
    steps = int(h["steps"])
    dt = float(h["dt_s"])
    if len(times) != steps or not np.array_equal(times, np.arange(1, steps + 1) * dt):
        raise ValueError("saved holdout time axis does not match result receipt")
    if steps % args.stride:
        raise ValueError("stride must divide the saved number of holdout steps")
    grid, ppc = int(h["grid"]), float(h["ppc"])
    lengths = np.asarray(h["geometry_lengths_m"], dtype=np.float64)
    cfg, ref, xyz, _vol = model.make_body(grid, ppc, dt, lengths=lengths)
    boundaries = model.boundary_case(ref, cfg, h["kind"], steps,
        max(12, steps), .0035, lengths=lengths)
    if len(xyz) != int(h["particles"]):
        raise ValueError("rebuilt holdout particle count differs from saved receipt")

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "status": "running",
        "source_result_sha256": file_sha(args.result),
        "source_holdout_sha256": file_sha(args.holdout),
        "grid": grid, "ppc": ppc, "particles": len(xyz), "dt_s": dt,
        "steps": steps, "frame_stride_steps": args.stride,
        "frame_step_indices": list(range(0, steps + 1, args.stride)),
        "density_kg_m3": result["density_kg_m3"],
        "material": result["model"], "load_case": h["kind"],
        "lengths_m": lengths.tolist(),
        "boundary_recipe": "same make_body/make_boundary_case used by saved holdout; 3.5 mm mixed axial/torsion amplitude; period=max(12,steps)",
        "sensor_validation_max_abs_error_m": {},
        "frame_coordinate_source": "sample_particles from each completed velocity-Verlet MPM state; no temporal interpolation",
    }
    responses = ((expected_truth, result["true_engineering_constants"], "truth"),
                 (expected_fit, result["estimated_engineering_constants"], "inferred"))
    sensor_ids_j = jnp.asarray(sensor_ids)
    sensor_w_j = jnp.asarray(sensor_weights)
    for expected, engineering, label in responses:
        frames = run_frames(engineering, cfg, ref, boundaries, steps, args.stride)
        # Validate the repeated real MPM states against the frozen holdout sensor
        # histories at every exported time. This catches load/parameter drift.
        sampled_u = frames[1:] - np.asarray(ref.positions)[None, ...]
        selected = sampled_u[:, sensor_ids, :]
        sensor_u = np.sum(sensor_weights[None, :, :, None] * selected, axis=2)
        expected_at_frames = expected[args.stride - 1::args.stride]
        error = float(np.max(np.abs(sensor_u - expected_at_frames)))
        tolerance = max(1.0e-10, 1.0e-7 * float(np.max(np.abs(expected_at_frames))))
        if error > tolerance:
            raise RuntimeError(f"{label} frame rerun differs from frozen MPM response: {error:g} > {tolerance:g}")
        np.savez_compressed(args.output / f"{label}_particle_positions.npz",
            positions_m=frames, reference_positions_m=np.asarray(ref.positions),
            frame_step_indices=np.asarray(meta["frame_step_indices"], dtype=np.int32),
            frame_time_s=np.asarray(meta["frame_step_indices"], dtype=np.float64) * dt)
        meta["sensor_validation_max_abs_error_m"][label] = error
        del frames
        jax.clear_caches()
    meta["status"] = "complete"
    meta["source_sha256"] = {
        "examples/inverse_orthotropic_svk.py": file_sha(Path(__file__).resolve().parents[1] / "examples" / "inverse_orthotropic_svk.py"),
        "mpm_jax/total_lagrangian.py": file_sha(Path(__file__).resolve().parents[1] / "mpm_jax" / "total_lagrangian.py"),
        "mpm_jax/core.py": file_sha(Path(__file__).resolve().parents[1] / "mpm_jax" / "core.py"),
        "mpm_jax/materials/elastic.py": file_sha(Path(__file__).resolve().parents[1] / "mpm_jax" / "materials" / "elastic.py"),
    }
    (args.output / "particle_frames.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
