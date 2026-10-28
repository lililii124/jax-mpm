"""JAX MPM minimum-material cantilever search with compliance/stress limits.

The outer search brackets the smallest feasible projected volume fraction. For
each trial volume, an Adam inner solve minimizes compliance with augmented
penalties for compliance and a KS bound on solid-constituent stress at points
with rho >= the stated cutoff. This is a bounded, discretized search; it does
not establish a global optimum or a manufacturable part.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.classic_cantilever_topopt import (
    E0, FORCE_N, HEIGHT, LENGTH, NU, RHO_BASE, RHO_FLOOR, THICKNESS,
    build_solver, design_checks, force_balance, geometry,
)
from mpm_jax.materials import linear_elastic, linear_stress


LOAD_N = 50_000.0
DEFLECTION_LIMIT_M = 0.002
COMPLIANCE_LIMIT_NM = LOAD_N * DEFLECTION_LIMIT_M
ALLOWABLE_VM_PA = 100.0e6
ASSUMED_YIELD_PA = 276.0e6
STRESS_DENSITY_CUTOFF = 0.10
KS_RHO_CUTOFF = 0.10
KS_P = 64.0
PROJECTION_BETA = 12.0
FILTER_RADIUS_CELLS = 2.0


def resized_trilinear(source: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(source, dtype=np.float64)
    for axis, target_n in enumerate(target_shape):
        old_n = array.shape[axis]
        coordinate = np.linspace(0.0, old_n - 1.0, target_n)
        lo = np.floor(coordinate).astype(int)
        hi = np.minimum(lo + 1, old_n - 1)
        fraction = coordinate - lo
        shape = [1, 1, 1]
        shape[axis] = target_n
        weight = fraction.reshape(shape)
        array = np.take(array, lo, axis=axis) * (1.0 - weight) + np.take(array, hi, axis=axis) * weight
    return array


def make_problem(grid: int, vf: float, maxiter: int, cg_tol: float):
    mesh = geometry(grid)
    mesh["load"] *= LOAD_N / FORCE_N
    solver = build_solver(
        mesh, beta=PROJECTION_BETA, vf=vf, maxiter=maxiter, tol=cg_tol,
        filter_radius_cells=FILTER_RADIUS_CELLS,
    )
    conn = jnp.asarray(mesh["indices"], dtype=jnp.int32)
    grad = jnp.asarray(mesh["gradients"], dtype=jnp.float64)
    point_cell = jnp.asarray(mesh["cell_id_per_point"], dtype=jnp.int32)
    rhs = jnp.asarray(solver["rhs"], dtype=jnp.float64)
    free = jnp.asarray(solver["free_flat"], dtype=bool)
    bnorm = jnp.maximum(jnp.linalg.norm(rhs), 1.0)
    material = linear_elastic(E0, NU)
    eye = jnp.eye(3, dtype=jnp.float64)
    cscale = LOAD_N * (LOAD_N * LENGTH**3 / (3.0 * E0 * (THICKNESS * HEIGHT**3 / 12.0))
                       + LOAD_N * LENGTH / ((5.0 / 6.0) * (E0 / (2.0 * (1.0 + NU))) * HEIGHT * THICKNESS))

    def full_metrics(q):
        rho = solver["rho"](q)
        u = jnp.where(free, solver["equilibrium"](rho), 0.0)
        Ku = solver["action"](rho, u)
        compliance = jnp.vdot(rhs, u).real
        residual = jnp.linalg.norm(jnp.where(free, Ku - rhs, 0.0)) / bnorm
        nodal_u = u.reshape((mesh["node_xyz"].shape[0], 3))
        grad_u = jnp.einsum("pki,pkj->pij", nodal_u[conn], grad)
        strain = 0.5 * (grad_u + jnp.swapaxes(grad_u, 1, 2))
        sigma = linear_stress(strain, material)
        deviator = sigma - jnp.trace(sigma, axis1=1, axis2=2)[:, None, None] * eye / 3.0
        vm = jnp.sqrt(jnp.maximum(0.0, 1.5 * jnp.sum(deviator**2, axis=(1, 2))))
        point_rho = rho[point_cell]
        mask = lax.stop_gradient(point_rho >= KS_RHO_CUTOFF)
        normalized = vm / ALLOWABLE_VM_PA
        selected = jnp.where(mask, normalized, -1.0e8)
        stress_ks = logsumexp(KS_P * selected) / KS_P
        cuts = jnp.asarray([0.01, 0.05, 0.10, 0.20, 0.50, 0.80], dtype=jnp.float64)
        peak_by_cut = jnp.stack([
            jnp.max(jnp.where(lax.stop_gradient(point_rho >= cut), vm, 0.0)) for cut in cuts
        ])
        return compliance, jnp.mean(rho), residual, stress_ks, peak_by_cut, rho, u

    def loss_with_aux(q):
        compliance, volume, residual, stress_ks, peak_by_cut, rho, u = full_metrics(q)
        g_compliance = jnp.maximum(compliance / COMPLIANCE_LIMIT_NM - 1.0, 0.0)
        g_stress = jnp.maximum(stress_ks - 1.0, 0.0)
        loss = compliance / cscale + 25.0 * g_compliance**2 + 50.0 * g_stress**2
        aux = (compliance, volume, residual, stress_ks, peak_by_cut, rho, u)
        return loss, aux

    loss_jit = jax.jit(loss_with_aux)
    value_grad = jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))
    return mesh, solver, full_metrics, loss_jit, value_grad, cscale


def metrics_record(mesh, aux, target_vf, iteration, objective):
    compliance, vf, residual, stress_ks, peak_by_cut, rho, u = aux
    cuts = [0.01, 0.05, 0.10, 0.20, 0.50, 0.80]
    peak = np.asarray(peak_by_cut, dtype=np.float64)
    stress_max = float(peak[cuts.index(STRESS_DENSITY_CUTOFF)])
    return {
        "target_volume_fraction": float(target_vf),
        "actual_volume_fraction": float(vf),
        "material_mass_kg": float(vf) * RHO_BASE * LENGTH * HEIGHT * THICKNESS,
        "compliance_Nm": float(compliance),
        "pad_mean_deflection_m": float(compliance) / LOAD_N,
        "compliance_limit_Nm": COMPLIANCE_LIMIT_NM,
        "compliance_feasible": bool(float(compliance) <= COMPLIANCE_LIMIT_NM * 1.001),
        "stress_KS_over_allowable_for_rho_ge_0_10": float(stress_ks),
        "stress_KS_feasible": bool(float(stress_ks) <= 1.0),
        "maximum_constituent_von_mises_by_density_cutoff_Pa": {
            str(cut): float(value) for cut, value in zip(cuts, peak)
        },
        "stress_constraint_density_cutoff": STRESS_DENSITY_CUTOFF,
        "maximum_constituent_von_mises_for_rho_ge_0_10_Pa": stress_max,
        "stress_feasible_actual_max": bool(stress_max <= ALLOWABLE_VM_PA),
        "relative_linear_residual": float(residual),
        "objective": float(objective),
        "iteration": int(iteration),
        "rho": np.asarray(rho),
        "u": np.asarray(u),
    }


def run_trial(grid, target_vf, trial_id, args, output, q_seed=None):
    trial_started = time.perf_counter()
    mesh, solver, _, loss_jit, value_grad, cscale = make_problem(grid, target_vf, args.cg_maxiter, args.cg_tol)
    shape = mesh["shape"]
    ndesign = int(np.sum(mesh["design_cell"]))
    ncell = int(np.prod(shape))
    # The published density map accepts a full-cell q array and overwrites
    # non-design cells to solid. Those fixed-cell q slots have zero influence.
    q = jnp.asarray(q_seed if q_seed is not None else np.full(ncell, 0.38), dtype=jnp.float64)
    if q.shape != (ncell,):
        raise ValueError(f"full-cell q shape {q.shape} != {(ncell,)}")
    m = jnp.zeros_like(q); v = jnp.zeros_like(q); adam_step = 0
    trial_dir = output / f"trial-{trial_id:02d}-vf-{target_vf:.5f}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    history_path = output / "iterations.csv"
    trial_history = []
    frames = []
    base_loss = None
    fd_record = None
    for iteration in range(args.iterations):
        (loss_value, aux), gradient = value_grad(q)
        loss_value.block_until_ready()
        grad_norm = float(jnp.linalg.norm(gradient))
        if not np.isfinite(float(loss_value)) or not np.isfinite(grad_norm):
            raise RuntimeError(f"trial {trial_id} iteration {iteration}: nonfinite objective/gradient")
        if base_loss is None:
            base_loss = float(loss_value)
            direction = jax.random.normal(jax.random.PRNGKey(4100 + trial_id), q.shape, dtype=q.dtype)
            direction = direction / jnp.maximum(jnp.linalg.norm(direction), 1e-30)
            step_fd = 2e-4
            fd = (loss_jit(q + step_fd * direction)[0] - loss_jit(q - step_fd * direction)[0]) / (2.0 * step_fd)
            ad = jnp.vdot(gradient, direction).real
            fd_error = abs(float(fd - ad)) / max(abs(float(fd)), abs(float(ad)), 1e-12)
            fd_record = {"ad_directional_derivative": float(ad), "fd_directional_derivative": float(fd),
                         "relative_error": fd_error, "step": step_fd}
            if not np.isfinite(fd_error) or fd_error > 2e-3:
                raise RuntimeError(f"AD/FD objective-direction check failed: {fd_record}")
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient * gradient
        adam_step += 1
        mhat = m / (1.0 - 0.9**adam_step)
        vhat = v / (1.0 - 0.999**adam_step)
        direction = -mhat / (jnp.sqrt(vhat) + 1e-8)
        lr = args.learning_rate
        accepted = False
        candidate_aux = aux
        candidate_loss = float(loss_value)
        for _ in range(8):
            candidate = jnp.clip(q + lr * direction, 1e-4, 1.0 - 1e-4)
            trial_loss, trial_aux = loss_jit(candidate)
            trial_loss.block_until_ready()
            candidate_loss = float(trial_loss)
            if np.isfinite(candidate_loss) and candidate_loss <= float(loss_value) * (1.0 + 1e-10):
                q = candidate
                candidate_aux = trial_aux
                accepted = True
                break
            lr *= 0.5
        if not accepted:
            candidate_loss = float(loss_value)
        rec = metrics_record(mesh, candidate_aux, target_vf, iteration + 1, candidate_loss)
        rec.pop("rho"); rec.pop("u")
        rec.update({"trial_id": trial_id, "accepted": accepted, "learning_rate": lr,
                    "gradient_l2": grad_norm, "ad_fd_relative_error": fd_record["relative_error"] if fd_record else None})
        trial_history.append(rec)
        frames.append(np.asarray(candidate_aux[5], dtype=np.float32).reshape(shape))
        checkpoint = {
            "q": np.asarray(q), "adam_m": np.asarray(m), "adam_v": np.asarray(v),
            "adam_step": np.asarray(adam_step), "iteration": np.asarray(iteration + 1),
            "target_vf": np.asarray(target_vf), "trial_id": np.asarray(trial_id),
        }
        np.savez_compressed(trial_dir / "optimizer-checkpoint.npz", **checkpoint)
        (trial_dir / "latest.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        if (iteration + 1) % 4 == 0 or iteration + 1 == args.iterations:
            print(json.dumps({"event": "trial_iteration", **rec}, separators=(",", ":")), flush=True)

    final_loss, final_aux = loss_jit(q)
    final_loss.block_until_ready()
    result = metrics_record(mesh, final_aux, target_vf, args.iterations, float(final_loss))
    feasible = result["compliance_feasible"] and result["stress_KS_feasible"] and result["stress_feasible_actual_max"]
    result["feasible"] = bool(feasible)
    result["ad_fd_directional_check"] = fd_record
    result["design_grid"] = grid
    result["beam_cells"] = list(shape)
    result["design_controls"] = ndesign
    np.savez_compressed(trial_dir / "design.npz", q=np.asarray(q), rho=np.asarray(final_aux[5]).reshape(shape),
                        displacement=np.asarray(final_aux[6]).reshape((-1, 3)))
    np.savez_compressed(trial_dir / "density-history.npz", rho_history=np.stack(frames),
                        beta=np.full(len(frames), PROJECTION_BETA), iteration=np.arange(1, len(frames) + 1))
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_history[0]))
        if f.tell() == 0:
            writer.writeheader()
        writer.writerows(trial_history)
    result["trial_directory"] = str(trial_dir)
    result["wall_seconds"] = time.perf_counter() - trial_started
    (trial_dir / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("rho", "u")}, indent=2) + "\n", encoding="utf-8")
    return result, np.asarray(q), frames, mesh


def direct_solve_metrics(mesh, solver, rho):
    rho_j = jnp.asarray(rho, dtype=jnp.float64)
    u = solver["equilibrium"](rho_j)
    u.block_until_ready()
    Ku = solver["action"](rho_j, u)
    Ku.block_until_ready()
    rhs = jnp.asarray(solver["rhs"])
    free = jnp.asarray(solver["free_flat"])
    residual = float(jnp.linalg.norm(jnp.where(free, Ku - rhs, 0.0)) / jnp.maximum(jnp.linalg.norm(rhs), 1.0))
    compliance = float(jnp.vdot(rhs, u).real)
    balance = force_balance(mesh, np.asarray(Ku), enforce=False)
    node_u = np.asarray(u).reshape((-1, 3))
    particle_u = np.einsum("pk,pki->pi", mesh["weights"], node_u[mesh["indices"]])
    conn = jnp.asarray(mesh["indices"], dtype=jnp.int32)
    grad = jnp.asarray(mesh["gradients"], dtype=jnp.float64)
    point_cell = np.asarray(mesh["cell_id_per_point"])
    material = linear_elastic(E0, NU)
    eye = jnp.eye(3, dtype=jnp.float64)
    gu = jnp.einsum("pki,pkj->pij", u.reshape((mesh["node_xyz"].shape[0], 3))[conn], grad)
    eps = 0.5 * (gu + jnp.swapaxes(gu, 1, 2))
    sigma = linear_stress(eps, material)
    dev = sigma - jnp.trace(sigma, axis1=1, axis2=2)[:, None, None] * eye / 3.0
    vm = np.asarray(jnp.sqrt(jnp.maximum(0.0, 1.5 * jnp.sum(dev**2, axis=(1, 2)))))
    rho_p = np.asarray(rho_j)[point_cell]
    peaks = {str(cut): float(vm[rho_p >= cut].max()) if np.any(rho_p >= cut) else None
             for cut in (0.01, 0.05, 0.10, 0.20, 0.50, 0.80)}
    return {
        "compliance_Nm": compliance,
        "pad_mean_deflection_m": compliance / LOAD_N,
        "relative_linear_residual": residual,
        "force_balance": balance,
        "maximum_constituent_von_mises_by_density_cutoff_Pa": peaks,
        "peak_displacement_material_point_m": float(np.linalg.norm(particle_u, axis=1).max()),
        "rms_displacement_material_point_m": float(np.sqrt(np.mean(np.sum(particle_u**2, axis=1)))),
        "rho": np.asarray(rho_j),
        "u": np.asarray(u),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--validation-grid", type=int, default=64)
    parser.add_argument("--outer-bisections", type=int, default=6)
    parser.add_argument("--inner-iterations", dest="iterations", type=int, default=32)
    parser.add_argument("--cg-maxiter", type=int, default=2400)
    parser.add_argument("--cg-tol", type=float, default=1e-9)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--volume-lower", type=float, default=0.15)
    parser.add_argument("--initial-q", type=Path, default=None,
                        help="optional warm-start checkpoint NPZ containing q; optimizer moments are reset")
    parser.add_argument("--output", type=Path, default=Path("output/min-volume-cantilever"))
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("this example requires JAX float64")
    if args.grid % 16 or args.validation_grid <= args.grid or args.validation_grid % 16:
        raise ValueError("grids must be divisible by 16 and validation grid must be finer")
    if not 0.05 < args.volume_lower < 0.9:
        raise ValueError("volume lower bound must be in (0.05,0.9)")
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    initial_q = None
    if args.initial_q is not None:
        with np.load(args.initial_q) as checkpoint:
            initial_q = np.asarray(checkpoint["q"], dtype=np.float64).copy()
        expected_q = int(np.prod(geometry(args.grid)["shape"]))
        if initial_q.shape != (expected_q,) or not np.all(np.isfinite(initial_q)):
            raise ValueError(f"warm-start q must be finite with shape {(expected_q,)}, got {initial_q.shape}")
    (out / "problem.json").write_text(json.dumps({
        "objective": "minimize mean projected material volume fraction by outer bisection over fixed-volume SIMP subproblems",
        "compliance_cap_Nm": COMPLIANCE_LIMIT_NM,
        "loaded_pad_mean_deflection_cap_m": DEFLECTION_LIMIT_M,
        "load_N": LOAD_N,
        "stress_allowable_Pa": ALLOWABLE_VM_PA,
        "assumed_yield_strength_Pa": ASSUMED_YIELD_PA,
        "assumed_yield_factor": ASSUMED_YIELD_PA / ALLOWABLE_VM_PA,
        "stress_definition": "unscaled isotropic matrix von Mises stress at MPM Gauss points, constrained by unnormalized KS upper bound over points with projected rho>=0.10; report additional density cutoffs to show threshold sensitivity",
        "material_assumptions": {"E_Pa": E0, "nu": NU, "density_kg_m3": RHO_BASE,
                                 "yield_note": "276 MPa is an illustrative 6061-T6-like minimum-yield assumption; E=70 GPa is the published benchmark value, not a certified alloy card"},
        "continuous_model": {"SIMP_penalty": 3, "residual_modulus_fraction": RHO_FLOOR,
                             "filter_radius_cells": FILTER_RADIUS_CELLS, "projection_beta": PROJECTION_BETA,
                             "stress_density_cutoff": KS_RHO_CUTOFF},
        "boundary_load": "same published full x=X0 face clamp and centered 0.125 x 0.083333 m end-face pad, but force scaled from 100 N to 50 kN; resultant/moment audited per solve",
        "search": {"volume_lower": args.volume_lower, "volume_upper_full_solid": 1.0,
                   "outer_bisections": args.outer_bisections, "inner_iterations_each_target": args.iterations,
                   "trial_grid": args.grid, "independent_validation_grid": args.validation_grid,
                   "initial_q": str(args.initial_q) if args.initial_q else None,
                   "warm_start_note": "q may be initialized from a previous design; Adam moments are reset for each target fraction"},
        "claim_limit": "bounded volume-fraction search with local Adam compliance/stress optimization; no global-optimum or manufacturable-binary certification",
    }, indent=2) + "\n", encoding="utf-8")

    # Verify the full-solid bracket on the coarse grid before volume search.
    mesh_full = geometry(args.grid); mesh_full["load"] *= LOAD_N / FORCE_N
    full_solver = build_solver(mesh_full, PROJECTION_BETA, 0.95, args.cg_maxiter, args.cg_tol)
    full_rho = np.ones(len(mesh_full["points"]), dtype=np.float64)
    full_eval = direct_solve_metrics(mesh_full, full_solver, full_rho)
    full_peaks = full_eval["maximum_constituent_von_mises_by_density_cutoff_Pa"]
    if (full_eval["pad_mean_deflection_m"] > DEFLECTION_LIMIT_M
            or full_peaks["0.1"] is None or full_peaks["0.1"] > ALLOWABLE_VM_PA):
        raise RuntimeError("chosen engineering limits are not feasible even for the full-solid bracket")
    np.savez_compressed(out / "full-solid-coarse.npz", rho=np.ones(mesh_full["shape"], dtype=np.float32),
                        displacement=full_eval["u"].reshape((-1, 3)))
    del full_solver, full_eval
    jax.clear_caches(); gc.collect()

    mesh_for_bound = geometry(args.grid)
    minimum_fixed_fraction = float(np.mean(mesh_for_bound["fixed_cell"]))
    low, high = args.volume_lower, 1.0
    feasible = None
    feasible_q = None
    feasible_frames = None
    feasible_mesh = None
    trials = []
    trial_id = 0
    # Establish that the requested lower test fraction is actually infeasible;
    # if it happens to pass, continue down to the non-design-pad-only bound.
    print(json.dumps({"event": "lower_volume_bound_trial_start", "target_vf": args.volume_lower}), flush=True)
    lower_result, lower_q, lower_frames, lower_mesh = run_trial(
        args.grid, args.volume_lower, trial_id, args, out, q_seed=initial_q
    )
    trials.append({k: v for k, v in lower_result.items() if k not in ("rho", "u")})
    if lower_result["feasible"]:
        high = args.volume_lower
        low = minimum_fixed_fraction
        feasible, feasible_q, feasible_frames, feasible_mesh = lower_result, lower_q.copy(), lower_frames.copy(), lower_mesh
        np.savez_compressed(out / "best-feasible-checkpoint.npz", q=feasible_q,
                            rho=np.asarray(lower_result["rho"]).reshape(lower_mesh["shape"]),
                            iteration=np.asarray(args.iterations), vf=np.asarray(args.volume_lower))
    else:
        low = args.volume_lower
    trial_id += 1
    for _ in range(args.outer_bisections):
        target = 0.5 * (low + high)
        print(json.dumps({"event": "volume_trial_start", "trial": trial_id, "lower": low,
                          "upper": high, "target_vf": target}), flush=True)
        result, q, frames, mesh = run_trial(args.grid, target, trial_id, args, out)
        trials.append({k: v for k, v in result.items() if k not in ("rho", "u")})
        print(json.dumps({"event": "volume_trial_complete", "trial": trial_id,
                          **{k: result[k] for k in ("target_volume_fraction", "actual_volume_fraction", "material_mass_kg",
                                                     "compliance_Nm", "pad_mean_deflection_m", "maximum_constituent_von_mises_for_rho_ge_0_10_Pa",
                                                     "stress_KS_over_allowable_for_rho_ge_0_10", "feasible")}}, separators=(",", ":")), flush=True)
        if result["feasible"]:
            high = target
            feasible, feasible_q, feasible_frames, feasible_mesh = result, q.copy(), frames.copy(), mesh
            np.savez_compressed(out / "best-feasible-checkpoint.npz", q=feasible_q,
                                rho=np.asarray(result["rho"]).reshape(mesh["shape"]),
                                iteration=np.asarray(args.iterations), vf=np.asarray(target))
        else:
            low = target
        (out / "search-state.json").write_text(json.dumps({"low_infeasible_bound": low,
            "high_feasible_bound": high, "minimum_fixed_fraction": minimum_fixed_fraction,
            "lower_bound_note": "At the non-design-pad-only bound there is no design-region load path between fixed and loaded pads.",
            "trial_results": trials}, indent=2) + "\n", encoding="utf-8")
        trial_id += 1
        jax.clear_caches(); gc.collect()

    if feasible is None:
        raise RuntimeError("no feasible material fraction found; see trial records")

    # Independent finer-grid continuous solve; preserve the same projected VF.
    fine_mesh, fine_solver, fine_eval_jit, _, _, _ = make_problem(
        args.validation_grid, float(feasible["target_volume_fraction"]), args.cg_maxiter, args.cg_tol
    )
    fine_q = resized_trilinear(feasible_q.reshape(feasible_mesh["shape"]), fine_mesh["shape"]).ravel()
    fine_eval = fine_eval_jit(jnp.asarray(fine_q))
    fine_eval[0].block_until_ready()
    fine_aux = tuple(np.asarray(x) if hasattr(x, "shape") else x for x in fine_eval)
    fine_record = metrics_record(fine_mesh, fine_aux, feasible["target_volume_fraction"], args.iterations,
                                 float(fine_eval[0]))
    fine_record.pop("rho"); fine_record.pop("u")
    fine_rho = np.asarray(fine_eval[5]).reshape(fine_mesh["shape"])
    fine_u = np.asarray(fine_eval[6]).reshape((-1, 3))
    fine_balance = force_balance(fine_mesh, np.asarray(fine_solver["action"](jnp.asarray(fine_eval[5]),
                                                                               jnp.asarray(fine_eval[6]))), enforce=False)
    fine_record["force_balance"] = fine_balance
    fine_record["compliance_feasible"] = fine_record["compliance_feasible"]
    fine_record["stress_conservative_KS_feasible"] = fine_record["stress_KS_feasible"]
    fine_record["feasible"] = bool(fine_record["compliance_feasible"]
                                   and fine_record["stress_KS_feasible"]
                                   and fine_record["stress_feasible_actual_max"])
    np.savez_compressed(out / "refined-continuous.npz", q=fine_q, rho=fine_rho, displacement=fine_u)
    np.savez_compressed(out / "refined-optimized.npz", q=fine_q, rho=fine_rho, displacement=fine_u)

    # Full-solid 64-grid comparison under the same 50 kN load.
    full_fine = direct_solve_metrics(fine_mesh, fine_solver, np.ones(len(fine_mesh["points"])))
    full_fine_record = {k: v for k, v in full_fine.items() if k not in ("rho", "u")}
    full_fine_record["mass_kg"] = RHO_BASE * LENGTH * HEIGHT * THICKNESS

    # A 0.50 voxel threshold is re-equilibrated with the stated 0.001E ersatz
    # floor. It is a diagnostic of binarization, not a zero-void certification.
    solid = fine_rho >= 0.50
    solid |= fine_mesh["fixed_cell"].reshape(fine_mesh["shape"])
    rho_binary = np.where(solid, 1.0, RHO_FLOOR).astype(np.float64).ravel()
    binary_eval = direct_solve_metrics(fine_mesh, fine_solver, rho_binary)
    binary_record = {k: v for k, v in binary_eval.items() if k not in ("rho", "u")}
    binary_record["volume_fraction"] = float(np.mean(solid))
    binary_record["mass_kg"] = binary_record["volume_fraction"] * RHO_BASE * LENGTH * HEIGHT * THICKNESS
    binary_record["topology"] = design_checks(solid.astype(np.float64), fine_mesh["shape"],
                                                fine_mesh["fixed_cell"], args.validation_grid)
    binary_record["interpretation"] = "independent equilibrium of phi=.50 voxel cells with 0.001E ersatz void; not true-zero-void or manufacturing certification"
    binary_record["compliance_feasible"] = binary_record["pad_mean_deflection_m"] <= DEFLECTION_LIMIT_M
    binary_record["stress_feasible_actual_max"] = binary_record["maximum_constituent_von_mises_by_density_cutoff_Pa"]["0.5"] <= ALLOWABLE_VM_PA
    np.savez_compressed(out / "refined-binary-phi050.npz", rho_binary=rho_binary.reshape(fine_mesh["shape"]),
                        displacement=binary_eval["u"].reshape((-1, 3)))

    # ParaView-compatible history is the actual continuous MPM rho field at
    # each accepted update of the selected volume trial, on the 48-grid cells.
    coarse_history = np.stack(feasible_frames).astype(np.float32)
    np.savez_compressed(out / "design-evolution.npz", rho_history=coarse_history,
                        beta=np.full(len(coarse_history), PROJECTION_BETA),
                        iteration=np.arange(1, len(coarse_history) + 1))
    initial_fine = np.asarray(fine_solver["rho"](jnp.full(fine_q.shape, 0.38))).reshape(fine_mesh["shape"])
    np.savez_compressed(out / "refined-initial.npz", rho=initial_fine)

    tested_infeasible = [float(t["target_volume_fraction"]) for t in trials if not t["feasible"]]
    tested_feasible = [float(t["target_volume_fraction"]) for t in trials if t["feasible"]]
    algorithm_description = (
        f"fixed-volume candidate at vf={feasible['target_volume_fraction']:.6f}; "
        f"{args.iterations} local Adam updates, no outer material search performed"
        if args.outer_bisections == 0 else
        f"minimum projected volume search using {args.outer_bisections} outer bisection trials; "
        "each candidate receives local Adam minimization"
    )
    if args.outer_bisections == 0:
        search_bounds = {
            "lowest_tested_infeasible_volume_fraction": min(tested_infeasible) if tested_infeasible else None,
            "lowest_tested_feasible_volume_fraction": min(tested_feasible) if tested_feasible else None,
            "untested_lower_physical_pad_fraction": minimum_fixed_fraction,
            "interval_width": None,
        }
        volume_claim = "A feasible fixed-volume candidate was evaluated; minimum material was not identified and no global-optimum claim is made."
    else:
        search_bounds = {
            "lowest_tested_infeasible_volume_fraction": min(tested_infeasible) if tested_infeasible else None,
            "lowest_tested_feasible_volume_fraction": min(tested_feasible) if tested_feasible else None,
            "untested_lower_physical_pad_fraction": minimum_fixed_fraction,
            "interval_width": (
                min(tested_feasible) - max(tested_infeasible)
                if tested_infeasible and tested_feasible else None
            ),
        }
        volume_claim = "The selected volume is the lowest feasible point found by finite outer bisection and local optimization; it is not a global minimum."

    result = {
        "case": "bounded minimum-volume JAX MPM cantilever topology search",
        "base_material": {"E_Pa": E0, "nu": NU, "density_kg_m3": RHO_BASE,
                          "assumed_yield_Pa": ASSUMED_YIELD_PA,
                          "allowable_Pa": ALLOWABLE_VM_PA,
                          "assumed_yield_factor": ASSUMED_YIELD_PA / ALLOWABLE_VM_PA},
        "load_N": LOAD_N,
        "load_geometry": "same centered finite right-end pad and x=X0 face clamp as published cantilever MPM case; load nodal pattern scaled from 100 N",
        "hard_constraints": {"pad_average_deflection_max_m": DEFLECTION_LIMIT_M,
                             "compliance_max_Nm": COMPLIANCE_LIMIT_NM,
                             "constituent_von_mises_KS_max_Pa": ALLOWABLE_VM_PA,
                             "stress_evaluated_for_projected_density_at_least": KS_RHO_CUTOFF,
                             "stress_aggregate": f"unnormalized KS upper bound, p={KS_P}",
                             "residual_modulus_fraction": RHO_FLOOR},
        "objective_and_algorithm": algorithm_description + "; local Adam minimizes compliance and augmented penalties for compliance/KS violations; JAX implicit linear solve gradients",
        "coarse_search_grid": args.grid,
        "independent_validation_grid": args.validation_grid,
        "search_bounds_after_last_trial": search_bounds,
        "minimum_fixed_fraction_from_non_design_pads": minimum_fixed_fraction,
        "minimum_fraction_interpretation": "The non-design clamp and load pads alone are not a connected load path; this is the physical lower bound before adding any design material.",
        "selected_continuous_design": {k: v for k, v in feasible.items() if k not in ("rho", "u")},
        "volume_trials": trials,
        "refined_continuous_validation": fine_record,
        "same_load_full_solid_refined_reference": full_fine_record,
        "threshold_phi_050_ersatz_reanalysis": binary_record,
        "binary_design_constraints_passed": bool(binary_record["compliance_feasible"]
                                                  and binary_record["stress_feasible_actual_max"]
                                                  and binary_record["topology"]["six_connected_component_count"] == 1
                                                  and binary_record["topology"]["clamp_to_load_pad_connected"]),
        "limitations": [
            volume_claim,
            "The optimized design is continuous SIMP with residual 0.001E ersatz stiffness. The phi=.50 voxel solve is a separate ersatz reanalysis, not a true-zero-void or manufacturing validation.",
            "The stress cap is an assumed 100 MPa limit against an illustrative 276 MPa yield benchmark and is assessed on solid-constituent von Mises stress at rho>=0.10. Reported lower-density cutoff sensitivity does not certify microstructure or gray-region strength.",
            "The voxel topology audit and minimum cell sections do not establish a manufacturing process or minimum feature certification.",
        ],
        "files": {"coarse_history": "design-evolution.npz", "selected_checkpoint": "best-feasible-checkpoint.npz",
                  "continuous_forward": "refined-continuous.npz", "binary_ersatz_forward": "refined-binary-phi050.npz"},
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(out),
                      "volume_fraction": feasible["actual_volume_fraction"],
                      "deflection_m": feasible["pad_mean_deflection_m"],
                      "stress_MPa": feasible["maximum_constituent_von_mises_for_rho_ge_0_10_Pa"] / 1e6,
                      "fine_binary_pass": result["binary_design_constraints_passed"]}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
