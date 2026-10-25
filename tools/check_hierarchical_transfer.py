"""Measure one frozen-position THB/APIC transfer on the proposed plate layout."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpm_jax.hierarchical_grid import (
    apic_from_grid, apic_to_grid, hierarchical_grid, hierarchical_stencil,
    internal_force, moment_matrix,
)

jax.config.update("jax_enable_x64", True)


def plate_points():
    h = .0025
    lower = np.array([.020, .095, .020])
    shape = (64, 4, 64)
    axes = [lower[d] + (np.arange(shape[d]) + .5) * h for d in range(3)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    fine = np.all((centres[:, [0, 2]] >= .070) & (centres[:, [0, 2]] < .130), axis=1)
    positions, volumes = [], []
    for selected, per_axis in ((~fine, 2), (fine, 4)):
        axis = (np.arange(per_axis) + .5) / per_axis - .5
        offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3) * h
        points = (centres[selected, None, :] + offsets).reshape(-1, 3)
        positions.append(points)
        volumes.append(np.full(len(points), (h / per_axis)**3))
    return np.concatenate(positions), np.concatenate(volumes)


def spin(B):
    return jnp.stack((B[:, 2, 1] - B[:, 1, 2], B[:, 0, 2] - B[:, 2, 0],
                      B[:, 1, 0] - B[:, 0, 1]), -1)


@jax.jit
def transfer_check(x, volume, mass, velocity, B, tau, A, support, grid):
    gm, gp = apic_to_grid(mass, velocity, B, support, grid)
    gv = gp / jnp.where(gm > 0, gm, 1)[:, None]
    v1, B1, L = apic_from_grid(gv, support)
    force = internal_force(volume, tau, support, grid)
    angular0 = jnp.sum(mass[:, None] * (jnp.cross(x, velocity) + spin(B)), axis=0)
    angular1 = jnp.sum(mass[:, None] * (jnp.cross(x, v1) + spin(B1)), axis=0)
    angular_grid = jnp.sum(jnp.cross(grid.positions, gp), axis=0)
    momentum0 = jnp.sum(mass[:, None] * velocity, axis=0)
    momentum1 = jnp.sum(mass[:, None] * v1, axis=0)
    power_grid = jnp.sum(force * gv)
    power_material = -jnp.sum(volume[:, None, None] * tau * L)
    return {
        "finite": jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in (gm, gp, v1, B1, L, force)])),
        "grid_mass": gm.sum(), "particle_mass": mass.sum(),
        "occupied_nodes": jnp.count_nonzero(gm > 0),
        "mass_error": jnp.abs(gm.sum() - mass.sum()),
        "momentum_grid_error": jnp.max(jnp.abs(gp.sum(0) - momentum0)),
        "momentum_return_error": jnp.max(jnp.abs(momentum1 - momentum0)),
        "angular_grid_error": jnp.max(jnp.abs(angular_grid - angular0)),
        "angular_return_error": jnp.max(jnp.abs(angular1 - angular0)),
        "velocity_max_error": jnp.max(jnp.abs(v1 - velocity)),
        "affine_moment_max_error": jnp.max(jnp.abs(B1 - B)),
        "physical_gradient_max_error": jnp.max(jnp.abs(L - A)),
        "internal_force_resultant": jnp.max(jnp.abs(force.sum(0))),
        "internal_force_torque": jnp.max(jnp.abs(jnp.cross(grid.positions, force).sum(0))),
        "grid_power": power_grid, "material_power": power_material,
        "virtual_work_error": jnp.abs(power_grid - power_material),
    }


def timed_call(function, *args):
    start = perf_counter()
    compiled = function.lower(*args).compile()
    compile_seconds = perf_counter() - start
    start = perf_counter()
    result = compiled(*args)
    jax.block_until_ready(result)
    return result, {"compile_seconds": compile_seconds, "execution_seconds": perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x_np, volume_np = plate_points()
    x, volume = jnp.asarray(x_np), jnp.asarray(volume_np)
    grid = hierarchical_grid(cells=80, domain_size=.2,
                             refined_lower=(.06875, .09125, .06875),
                             refined_upper=(.13125, .10875, .13125))
    print(f"points={len(x)} nodes={len(grid.positions)}", flush=True)
    support, stencil_time = timed_call(hierarchical_stencil, x, grid)
    print("stencil complete", flush=True)
    D = moment_matrix(support)
    A = jnp.array([[0., 15., 0.], [-3., 0., 2.], [1., 0., 0.]])
    velocity = x @ A.T + jnp.array([.03, -.04, .02])
    B = A @ D
    stress = jnp.array([[2000., 300., -500.], [300., -1000., 200.], [-500., 200., 500.]])
    tau = jnp.broadcast_to(stress, (len(x), 3, 3))
    jax.block_until_ready((D, B, velocity, tau))
    result, transfer_time = timed_call(transfer_check, x, volume, 1200. * volume,
                                     velocity, B, tau, A, support, grid)
    metrics = {key: np.asarray(value).item() for key, value in result.items()}
    metrics.update({
        "valid_support": bool(support.valid),
        "partition_max_error": float(jnp.max(jnp.abs(support.weights.sum(1) - 1.))),
        "minimum_weight": float(support.weights.min()),
        "moment_minimum_eigenvalue": float(jnp.linalg.eigvalsh(D).min()),
        "volume_error": float(abs(volume_np.sum() - .160 * .010 * .160)),
    })
    limits = {
        "mass_error": 1e-12, "momentum_grid_error": 1e-12, "momentum_return_error": 1e-12,
        "angular_grid_error": 1e-12, "angular_return_error": 1e-12,
        "velocity_max_error": 1e-12, "affine_moment_max_error": 1e-15,
        "physical_gradient_max_error": 1e-9, "internal_force_resultant": 1e-10,
        "internal_force_torque": 1e-10, "virtual_work_error": 1e-10,
        "partition_max_error": 2e-14, "volume_error": 1e-16,
    }
    passed = (metrics["finite"] and metrics["valid_support"] and metrics["minimum_weight"] >= 0
              and metrics["moment_minimum_eigenvalue"] > 0
              and all(metrics[key] <= limit for key, limit in limits.items()))
    source_files = [Path(__file__).resolve(), ROOT / "mpm_jax/hierarchical_grid.py"]
    report = {
        "scope": "one frozen-position affine transfer and virtual-work check; no time integration, contact or speedup claim",
        "passed": bool(passed), "particles": len(x), "grid_nodes": len(grid.positions),
        "coarse_nodes": int(grid.fine_start), "fine_nodes": int(len(grid.positions) - grid.fine_start),
        "stencil": stencil_time, "transfer_and_force": transfer_time,
        "metrics": metrics, "absolute_limits_SI": limits,
        "environment": {"python": sys.version, "jax": jax.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "devices": [str(d) for d in jax.devices()]},
        "source": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (args.output / "transfer-review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
