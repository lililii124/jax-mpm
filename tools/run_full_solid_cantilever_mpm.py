"""Run the published cantilever MPM with rho=1 at every material point.

Use this on Linux with JAX_ENABLE_X64=true. It imports the actual example's
geometry, B-spline stencil, load transfer, fixed nodes, and matrix-free MPM
operator; it does not run the SIMP optimizer or use a beam/spring surrogate.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.classic_cantilever_topopt import (
    E0, FORCE_N, HEIGHT, LENGTH, NU, RHO_BASE, THICKNESS,
    build_solver, force_balance, geometry,
)


def beam_reference() -> dict[str, float]:
    area = HEIGHT * THICKNESS
    iz = THICKNESS * HEIGHT**3 / 12.0
    shear_modulus = E0 / (2.0 * (1.0 + NU))
    eb = FORCE_N * LENGTH**3 / (3.0 * E0 * iz)
    shear = FORCE_N * LENGTH / ((5.0 / 6.0) * shear_modulus * area)
    return {
        "area_m2": area,
        "Izz_m4": iz,
        "G_Pa": shear_modulus,
        "euler_bernoulli_deflection_m": eb,
        "timoshenko_shear_deflection_m": shear,
        "timoshenko_total_deflection_m": eb + shear,
        "euler_bernoulli_compliance_Nm": FORCE_N * eb,
        "timoshenko_compliance_Nm": FORCE_N * (eb + shear),
    }


def solve_grid(grid: int, maxiter: int, tol: float) -> dict:
    started = time.perf_counter()
    mesh = geometry(grid)
    solver = build_solver(mesh, beta=1.5, vf=1.0, maxiter=maxiter, tol=tol)
    # equilibrium() accepts per-material-point density and bypasses SIMP's
    # volume projection. Every quadrature point is exactly full solid here.
    rho = jnp.ones((len(mesh["points"]),), dtype=jnp.float64)
    u = solver["equilibrium"](rho)
    u.block_until_ready()
    Ku = solver["action"](rho, u)
    Ku.block_until_ready()
    rhs = jnp.asarray(solver["rhs"])
    free = jnp.asarray(solver["free_flat"])
    residual = float(jnp.linalg.norm(jnp.where(free, Ku - rhs, 0.0)) / jnp.maximum(jnp.linalg.norm(rhs), 1.0))
    compliance = float(jnp.vdot(rhs, u).real)
    balance = force_balance(mesh, np.asarray(Ku), enforce=False)
    if not np.isfinite(compliance) or residual > 2e-7:
        raise RuntimeError(f"grid {grid}: nonfinite compliance or residual too high ({residual:g})")
    if balance["relative_force_balance_error"] > 2e-6 or balance["relative_moment_balance_error"] > 2e-6:
        raise RuntimeError(f"grid {grid}: force/reaction balance failed ({balance})")
    node_u = np.asarray(u).reshape((-1, 3))
    particle_u = np.einsum("pk,pki->pi", mesh["weights"], node_u[mesh["indices"]])
    result = {
        "grid_background": grid,
        "beam_cells": list(mesh["shape"]),
        "material_quadrature_points": len(mesh["points"]),
        "full_solid_mass_kg": RHO_BASE * LENGTH * HEIGHT * THICKNESS,
        "density_at_every_material_point": 1.0,
        "fixed_end": "published full x=X0 beam face fixed; same support nodes as SIMP case",
        "load_resultant_N": balance["applied_resultant_N"],
        "load_moment_about_origin_Nm": balance["applied_moment_about_origin_Nm"],
        "loaded_grid_nodes": balance["loaded_grid_nodes"],
        "compliance_Nm": compliance,
        "load_weighted_end_pad_mean_deflection_m": compliance / FORCE_N,
        "relative_linear_residual": residual,
        "force_balance_relative_error": balance["relative_force_balance_error"],
        "moment_balance_relative_error": balance["relative_moment_balance_error"],
        "max_material_point_displacement_m": float(np.linalg.norm(particle_u, axis=1).max()),
        "material_point_displacement_rms_m": float(np.sqrt(np.mean(np.sum(particle_u**2, axis=1)))),
        "wall_seconds": time.perf_counter() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--cg-maxiter", type=int, default=6000)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("output/full-solid-beam-comparison.json"))
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("full-solid reference requires float64 JAX")
    ref = beam_reference()
    grids = []
    for grid in args.grids:
        print(json.dumps({"event": "grid_start", "grid": grid}), flush=True)
        record = solve_grid(grid, args.cg_maxiter, args.cg_tol)
        record["ratio_to_Euler_Bernoulli_compliance"] = record["compliance_Nm"] / ref["euler_bernoulli_compliance_Nm"]
        record["ratio_to_Timoshenko_compliance"] = record["compliance_Nm"] / ref["timoshenko_compliance_Nm"]
        record["relative_deflection_difference_from_Timoshenko"] = (
            record["load_weighted_end_pad_mean_deflection_m"] / ref["timoshenko_total_deflection_m"] - 1.0
        )
        grids.append(record)
        print(json.dumps({"event": "grid_complete", **record}), flush=True)
        jax.clear_caches()
        gc.collect()
    for left, right in zip(grids, grids[1:]):
        right["compliance_change_from_previous_grid"] = right["compliance_Nm"] / left["compliance_Nm"] - 1.0
        right["deflection_change_from_previous_grid"] = (
            right["load_weighted_end_pad_mean_deflection_m"] / left["load_weighted_end_pad_mean_deflection_m"] - 1.0
        )
    result = {
        "case": "full-solid JAX MPM cantilever versus Euler-Bernoulli and Timoshenko beam theory",
        "model": "published classic_cantilever_topopt geometry/build_solver/action/equilibrium; rho=1 at every material quadrature point",
        "material": {"E_Pa": E0, "nu": NU, "density_kg_m3": RHO_BASE},
        "boundary_and_load": "same complete x=X0 clamp, 100N load transferred by the published 2x2 Gauss end-pad quadrature, same centered 0.125 x 0.083333m non-design pad geometry",
        "beam_theory_full_solid_reference": ref,
        "grid_results": grids,
        "interpretation": "This is the same full-solid material and load case. Differences combine three-dimensional elasticity, finite pad and support discretization, B-spline material-point quadrature, and mesh error. MPM load-weighted pad response is compliance/F and is not asserted to equal a pointwise tip node displacement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "all_complete", "output": str(args.output), "grid_count": len(grids)}), flush=True)


if __name__ == "__main__":
    main()
