"""Dynamic, mass-constrained solid-volume design using the 3D TL-MPM solver.

The design field redistributes a linear-Voigt solid volume fraction along a
fixed-envelope 3D cantilever. The reference quadrature volume and particle mass
both scale with this fraction, so total mass is conserved and fixed external
forces do not depend on the design. This is a material-layout/continuum-mixture
optimisation, not a geometric thickness/topology change. Particle positions and
grid stencil topology never change during optimisation.
Run with ``python -m examples.structural_design`` on Linux/JAX; see
``docs/STRUCTURAL_DESIGN.md`` for the production/optimisation workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import base64
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.core import Config
from mpm_jax.materials import elastic
from mpm_jax.total_lagrangian import (
    NodalBoundary,
    initial_state,
    momentum,
    reference_grid,
    sample_nodal,
    step as mpm_step,
)


RHO = 1000.0
YOUNG = 2.0e5
POISSON = 0.25
VOLUME_FRACTION = 0.75
FORCE_N = 8.0
N_CONTROL = 10
TIME_HORIZON = 0.24
RAMP_TIME = 0.02
DESIGN_DT = 5.0e-4
PRODUCTION_DT = 2.0e-4
INNER_CHUNK_STEPS = 12
CAPTURE_EVERY_CHUNKS = 5


def particle_cloud(grid_n: int, dtype=jnp.float64):
    """Regular material points in a metre-scale, solid 3D cantilever."""
    dx = 1.0 / grid_n
    # Two particles per background cell in the production check.  The same
    # construction at grid 24 is used for the inexpensive design iterations.
    pitch = dx / 2.0
    bounds = ((0.22, 0.88), (0.38, 0.62), (0.40, 0.60))
    axes = [np.arange(lo + 0.5 * pitch, hi, pitch) for lo, hi in bounds]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    points = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
    # Midpoint quadrature assigns equal material volume to each material point;
    # its total equals the unscaled continuum box volume exactly.
    base_volume = np.full(len(points), np.prod([hi - lo for lo, hi in bounds]) / len(points))
    return jnp.asarray(points, dtype=dtype), jnp.asarray(base_volume, dtype=dtype), bounds


def design_basis(x, n_control=N_CONTROL):
    """Smooth Gaussian partition-of-unity basis along the beam axis."""
    s = (x - 0.22) / (0.88 - 0.22)
    centers = jnp.linspace(0.0, 1.0, n_control, dtype=x.dtype)
    width = 1.65 / (n_control - 1)
    weights = jnp.exp(-0.5 * ((s[:, None] - centers[None, :]) / width) ** 2)
    return weights / jnp.sum(weights, axis=1, keepdims=True)


def design_basis_numpy(x, n_control=N_CONTROL):
    """NumPy copy used for cheap coarse and production-mesh feasibility checks."""
    x = np.asarray(x, dtype=np.float64)
    s = (x - 0.22) / 0.66
    centers = np.linspace(0.0, 1.0, n_control)
    width = 1.65 / (n_control - 1)
    weights = np.exp(-0.5 * ((s[:, None] - centers[None, :]) / width) ** 2)
    return weights / weights.sum(axis=1, keepdims=True)


def volume_scale(controls, basis):
    """Conservative solid-fraction redistribution with an exact mean.

    q lies in [-0.28, 0.28], hence scale is bounded by
    0.75*exp(+-0.56) before physical feasibility checks. The field mean is
    exactly 0.75, so total solid volume and mass are invariant to roundoff.
    Each particle's mass and internal-force quadrature use this same linear
    (p=1 Voigt) solid fraction. The optimizer separately enforces scale <= 1.
    """
    controls = jnp.asarray(controls, dtype=basis.dtype)
    q = 0.28 * jnp.tanh(basis @ controls)
    raw = jnp.exp(q)
    return VOLUME_FRACTION * raw / jnp.mean(raw)


def solid_fraction_feasible(controls, bases):
    """Require a positive, finite local solid fraction no greater than one."""
    controls = np.asarray(controls, dtype=np.float64)
    for basis in bases:
        basis = np.asarray(basis, dtype=np.float64)
        q = 0.28 * np.tanh(basis @ controls)
        raw = np.exp(q)
        scale = VOLUME_FRACTION * raw / raw.mean()
        if not (np.isfinite(scale).all() and np.all(scale > 0.0) and np.max(scale) <= 1.0):
            return False
    return True


def make_model(grid_n: int, n_steps: int, dt: float, dtype=jnp.float64):
    config = Config(
        n_grid=grid_n,
        dt=dt,
        domain_size=1.0,
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        dim=3,
        material="elastic",
        failure_mode="retain",
    )
    points, base_volume, bounds = particle_cloud(grid_n, dtype=dtype)
    basis = design_basis(points[:, 0])
    n_nodes = (grid_n + 1) ** 3
    # A finite-width clamp in reference coordinates.  It is independent of the
    # design and prescribes zero displacement, velocity and acceleration.
    fixed_nodes = jnp.broadcast_to(
        (jnp.arange(n_nodes) // ((grid_n + 1) ** 2) * config.dx <= 0.205)[:, None],
        (n_nodes, 3),
    )
    boundary = NodalBoundary(
        fixed=fixed_nodes,
        displacement=jnp.zeros((n_nodes, 3), dtype=dtype),
        velocity=jnp.zeros((n_nodes, 3), dtype=dtype),
        acceleration=jnp.zeros((n_nodes, 3), dtype=dtype),
    )
    material = elastic(jnp.asarray(YOUNG, dtype=dtype), jnp.asarray(POISSON, dtype=dtype))
    wave_speed = math.sqrt(YOUNG * (1 - POISSON) / (RHO * (1 + POISSON) * (1 - 2 * POISSON)))
    cfl = wave_speed * dt / config.dx
    if cfl > 0.30:
        raise ValueError(f"unstable MPM step: CFL={cfl:.4f} exceeds the 0.30 admission limit")
    tip = points[:, 0] >= 0.79
    y_patch = (points[:, 0] > 0.78) & (points[:, 1] > 0.585)
    z_patch = (points[:, 0] > 0.78) & (points[:, 2] > 0.565)
    # Two independent dead-load patches, on orthogonal exterior faces.  Force
    # per point is fixed, not acceleration: dividing by the design-dependent
    # particle mass cancels the mass factor in the solver's body-force term.
    load_mask = jnp.stack((y_patch, z_patch))
    patch_counts = np.asarray(jnp.sum(load_mask, axis=1))
    if np.any(patch_counts == 0):
        raise ValueError(
            f"grid {grid_n} leaves an orthogonal surface-load patch empty; "
            f"particle counts are {patch_counts.tolist()}"
        )
    weights = load_mask.astype(dtype)
    weights = weights / jnp.sum(weights, axis=1, keepdims=True)
    directions = jnp.asarray(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), dtype=dtype)
    load_shape = weights[:, :, None] * directions[:, None, :] * FORCE_N
    tip_mask = jnp.broadcast_to(tip[None, :], (2, points.shape[0]))
    amps = jnp.arange(n_steps + 1, dtype=dtype)
    times = amps * dt
    horizon = n_steps * dt
    ramp = min(RAMP_TIME, 0.1 * horizon)
    rise_phase = jnp.clip(times / ramp, 0.0, 1.0)
    rise = jnp.where(times < ramp, 0.5 * (1.0 - jnp.cos(jnp.pi * rise_phase)), 1.0)
    unload_start = horizon - ramp
    fall_phase = jnp.clip((times - unload_start) / ramp, 0.0, 1.0)
    fall = jnp.where(times > unload_start, 0.5 * (1.0 + jnp.cos(jnp.pi * fall_phase)), 1.0)
    pulse = rise * fall
    n_blocks = math.ceil(n_steps / INNER_CHUNK_STEPS)
    # Pad only the final block with zero-force steps; its scores are masked.
    padded_steps = n_blocks * INNER_CHUNK_STEPS
    pulse_padded = jnp.pad(pulse, (0, padded_steps - n_steps))
    pulse_pairs = jnp.stack((pulse_padded[:-1], pulse_padded[1:]), axis=1)
    pulse_pairs = pulse_pairs.reshape((n_blocks, INNER_CHUNK_STEPS, 2))
    valid_steps = jnp.arange(padded_steps).reshape((n_blocks, INNER_CHUNK_STEPS)) < n_steps

    def simulate_case(reference, case_index, capture=False):
        initial = initial_state(reference, material, config, jnp.zeros_like(points))
        force_distribution = load_shape[case_index]
        acceleration = force_distribution / reference.particle_mass[:, None]

        def chunk(state, pulse_block):
            def body(carry, pulse_pair):
                start_amp, end_amp = pulse_pair
                following, _ = mpm_step(
                    carry,
                    reference,
                    material,
                    config,
                    start_amp * acceleration,
                    end_amp * acceleration,
                    boundary,
                    boundary,
                )
                up = sample_nodal(following.displacement, reference)
                component = directions[case_index]
                tip_projection = jnp.sum(up * component[None, :], axis=1)
                response = jnp.sum(jnp.where(tip, tip_projection**2, 0.0)) / jnp.sum(tip)
                return following, response

            return jax.lax.scan(body, state, pulse_block)

        if capture:
            def capture_chunk(state, pulse_block):
                def body(carry, pulse_pair):
                    start_amp, end_amp = pulse_pair
                    following, _ = mpm_step(
                        carry, reference, material, config,
                        start_amp * acceleration, end_amp * acceleration,
                        boundary, boundary,
                    )
                    up = sample_nodal(following.displacement, reference)
                    projection = jnp.sum(up * directions[case_index][None, :], axis=1)
                    score = jnp.sum(jnp.where(tip, projection**2, 0.0)) / jnp.sum(tip)
                    return following, score
                final, scores = jax.lax.scan(body, state, pulse_block)
                # Keep just one particle-sampled frame per 12-step block. Saving
                # every nodal state would retain ~1.6 GiB for the 64^3 mesh.
                sampled_frame = sample_nodal(final.displacement, reference)
                return final, (jnp.mean(scores), sampled_frame)

            final, (scores, sampled_frames) = jax.lax.scan(capture_chunk, initial, pulse_pairs)
            return final, jnp.mean(scores), sampled_frames[
                CAPTURE_EVERY_CHUNKS - 1 :: CAPTURE_EVERY_CHUNKS
            ]

        checkpointed_chunk = jax.checkpoint(chunk, prevent_cse=False)

        def outer(state, args):
            pulse_block, mask = args
            following, block_scores = checkpointed_chunk(state, pulse_block)
            score = jnp.sum(block_scores * mask) / jnp.maximum(jnp.sum(mask), 1)
            return following, score

        _, block_scores = jax.lax.scan(outer, initial, (pulse_pairs, valid_steps))
        return jnp.mean(block_scores)

    def response(controls):
        scale = volume_scale(controls, basis)
        volume = base_volume * scale
        reference = reference_grid(points, volume, RHO, config)
        # Sequential case map plus rematerialization avoids retaining both
        # full time histories at once during the reverse pass.
        per_case = jax.lax.map(
            lambda k: jax.checkpoint(
                lambda case_id: simulate_case(reference, case_id),
                prevent_cse=False,
            )(k),
            jnp.arange(2),
            batch_size=1,
        )
        # Average normalized time-integrated tip motion over both orthogonal
        # load cases, rather than fitting a single node at a selected instant.
        dynamic = jnp.mean(per_case) / (0.04**2)
        smooth = jnp.mean(jnp.diff(controls) ** 2)
        return dynamic + 0.002 * smooth

    return config, points, base_volume, basis, material, boundary, load_shape, tip, pulse_pairs, valid_steps, response, simulate_case


def write_history_csv(path: Path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("iteration", "objective", "gradient_l2", "solid_fraction_mean",
                        "solid_fraction_min", "solid_fraction_max", "mass_kg", "controls"),
        )
        writer.writeheader()
        writer.writerows(history)


def save_optimizer_state(path: Path, controls, first, second, iteration, history, signature):
    """Atomically persist design variables and Adam moments after every update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(
        temporary,
        controls=np.asarray(controls),
        first=np.asarray(first),
        second=np.asarray(second),
        iteration=np.asarray(iteration, dtype=np.int64),
        history_json=np.asarray(json.dumps(history)),
        signature_json=np.asarray(json.dumps(signature, sort_keys=True)),
    )
    os.replace(temporary, path)


def optimize(response, evaluate, iterations: int, learning_rate: float, basis, feasibility_bases, base_volume,
             checkpoint_path: Path, history_path: Path, signature, initial_state=None):
    """Adam with line-search acceptance, restartable moments, and CSV history."""
    if initial_state is None:
        controls = jnp.zeros((N_CONTROL,), dtype=jnp.float64)
        first = jnp.zeros_like(controls)
        second = jnp.zeros_like(controls)
        start_iteration = 0
        history = []
    else:
        controls = jnp.asarray(initial_state["controls"], dtype=jnp.float64)
        first = jnp.asarray(initial_state["first"], dtype=jnp.float64)
        second = jnp.asarray(initial_state["second"], dtype=jnp.float64)
        start_iteration = int(initial_state["iteration"])
        history = list(initial_state["history"])
    if not solid_fraction_feasible(controls, feasibility_bases):
        raise ValueError("starting design violates the local solid-fraction bound on a design or production mesh")
    objective = lambda value: evaluate(value)[0]
    stop_iteration = start_iteration + iterations
    for iteration in range(start_iteration, stop_iteration + 1):
        value, gradient = evaluate(controls)
        if not np.isfinite(float(value)) or not np.all(np.isfinite(np.asarray(gradient))):
            raise RuntimeError(f"nonfinite objective/gradient at iteration {iteration}")
        current_scale = volume_scale(controls, basis)
        if float(jnp.max(current_scale)) > 1.0:
            raise RuntimeError(f"infeasible solid fraction at optimizer iteration {iteration}")
        row = {
            "iteration": iteration,
            "objective": float(value),
            "gradient_l2": float(jnp.linalg.norm(gradient)),
            "solid_fraction_mean": float(jnp.mean(current_scale)),
            "solid_fraction_min": float(jnp.min(current_scale)),
            "solid_fraction_max": float(jnp.max(current_scale)),
            "mass_kg": float(RHO * jnp.sum(base_volume * current_scale)),
            "controls": np.asarray(controls).tolist(),
        }
        if history and int(history[-1]["iteration"]) == iteration:
            history[-1] = row
        else:
            history.append(row)
        print(f"OPTIMIZER iteration={iteration} objective={float(value):.9g} gradient_l2={float(jnp.linalg.norm(gradient)):.6g}", flush=True)
        if iteration == stop_iteration:
            save_optimizer_state(checkpoint_path, controls, first, second, iteration, history, signature)
            write_history_csv(history_path, history)
            break
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        mhat = first / (1.0 - 0.9 ** (iteration + 1))
        vhat = second / (1.0 - 0.999 ** (iteration + 1))
        proposal = controls - learning_rate * mhat / (jnp.sqrt(vhat) + 1e-8)
        # Backtrack if Adam's step raises the actual MPM response.
        proposal_feasible = solid_fraction_feasible(proposal, feasibility_bases)
        proposed_value = objective(proposal) if proposal_feasible else float("inf")
        scale = 1.0
        while (not proposal_feasible or float(proposed_value) > float(value)) and scale > 1 / 32:
            scale *= 0.5
            proposal = controls + scale * (proposal - controls)
            proposal_feasible = solid_fraction_feasible(proposal, feasibility_bases)
            proposed_value = objective(proposal) if proposal_feasible else float("inf")
        if proposal_feasible and float(proposed_value) < float(value):
            controls = proposal
        else:
            # A scale-aware gradient fallback avoids treating a failed Adam
            # step as a successful optimisation iteration.
            gscale = learning_rate / (jnp.linalg.norm(gradient) + 1e-8)
            proposal = controls - gscale * gradient
            fallback_feasible = solid_fraction_feasible(proposal, feasibility_bases)
            if fallback_feasible and float(objective(proposal)) < float(value):
                controls = proposal
        save_optimizer_state(checkpoint_path, controls, first, second, iteration + 1, history, signature)
        write_history_csv(history_path, history)
    return controls, history


def load_patch_audit(model, controls):
    """Check nonempty load patches, resultant forces and loaded-region mass."""
    config, points, base_volume, basis = model[:4]
    load_shape = model[6]
    scale = volume_scale(controls, basis)
    particle_mass = RHO * base_volume * scale
    patch = jnp.linalg.norm(load_shape, axis=-1) > 0
    counts = np.asarray(jnp.sum(patch, axis=1))
    resultants = np.asarray(jnp.sum(load_shape, axis=1))
    patch_mass = np.asarray(jnp.sum(patch * particle_mass[None, :], axis=1))
    if np.any(counts <= 0) or np.any(patch_mass <= 0):
        raise RuntimeError(
            f"empty/zero-mass load patch at grid {config.n_grid}: "
            f"counts={counts.tolist()}, masses={patch_mass.tolist()}"
        )
    return {
        "grid": config.n_grid,
        "particles": int(len(points)),
        "patch_particle_counts": counts.astype(int).tolist(),
        "prescribed_resultant_force_N": resultants.tolist(),
        "solid_mass_on_patch_kg": patch_mass.tolist(),
    }


def vtk_binary_array(name, values, vtk_type, ncomp=1):
    arr = np.asarray(values)
    if vtk_type in ("Float32", "Float64"):
        arr = arr.astype("<f4" if vtk_type == "Float32" else "<f8", copy=False)
    else:
        dtype = {"Int64": "<i8", "UInt8": "u1", "Int32": "<i4"}.get(vtk_type)
        if dtype is None:
            raise ValueError(f"unsupported VTK type: {vtk_type}")
        arr = arr.astype(dtype, copy=False)
    payload = struct.pack("<I", arr.nbytes) + arr.tobytes(order="C")
    encoded = base64.b64encode(payload).decode("ascii")
    return f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="binary">{encoded}</DataArray>'


def write_vtu(path: Path, points, volume_ratio, displacement, time_s):
    xyz = np.asarray(points + displacement, dtype=np.float32)
    count = len(xyz)
    connectivity = np.arange(count, dtype=np.int32)
    offsets = np.arange(1, count + 1, dtype=np.int32)
    types = np.ones(count, dtype=np.uint8)
    scalar = np.asarray(volume_ratio, dtype=np.float32)
    disp = np.asarray(displacement, dtype=np.float32)
    text = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">
<UnstructuredGrid><Piece NumberOfPoints="{count}" NumberOfCells="{count}">
<PointData Scalars="volume_ratio">{vtk_binary_array("volume_ratio", scalar, "Float32")}{vtk_binary_array("displacement", disp, "Float32", 3)}{vtk_binary_array("displacement_magnitude", np.linalg.norm(disp, axis=1), "Float32")}</PointData>
<CellData/>
<Points>{vtk_binary_array("Points", xyz, "Float32", 3)}</Points>
<Cells>{vtk_binary_array("connectivity", connectivity, "Int32")}{vtk_binary_array("offsets", offsets, "Int32")}{vtk_binary_array("types", types, "UInt8")}</Cells>
<FieldData>{vtk_binary_array("time_s", [time_s], "Float64")}</FieldData>
</Piece></UnstructuredGrid></VTKFile>'''
    path.write_text(text, encoding="utf-8")


def write_pvd(directory: Path, frames):
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">', '<Collection>']
    for time_s, filename in frames:
        lines.append(f'<DataSet timestep="{time_s:.9g}" group="" part="0" file="{filename}"/>')
    lines.extend(['</Collection>', '</VTKFile>'])
    (directory / "series.pvd").write_text("\n".join(lines), encoding="utf-8")


def write_trajectory(directory, label, controls, model):
    (config, points, base_volume, basis, material, boundary, load_shape, tip,
     pulse_pairs, valid_steps, response, simulate_case) = model
    scale = volume_scale(controls, basis)
    if float(jnp.min(scale)) <= 0.0 or float(jnp.max(scale)) > 1.0:
        raise RuntimeError(
            f"{label} forward design violates the physical solid-fraction bound "
            f"0 < fraction <= 1 (range={float(jnp.min(scale)):.9g}..{float(jnp.max(scale)):.9g})"
        )
    volume = base_volume * scale
    reference = reference_grid(points, volume, RHO, config)
    p = np.asarray(points)
    vol_ratio = np.asarray(scale)
    all_cases = {}
    for case_index, axis in enumerate(("y", "z")):
        print(f"FORWARD_START label={label} load={axis} particles={len(p)} grid={config.n_grid}", flush=True)
        final, scores, sampled = simulate_case(reference, jnp.asarray(case_index), capture=True)
        sampled = np.asarray(sampled, dtype=np.float32)
        scores_value = float(scores)
        print(f"FORWARD_DONE label={label} load={axis} score={scores_value:.9g} valid={bool(final.valid)}", flush=True)
        zero_frame = np.zeros((len(p), 3), dtype=np.float32)
        sampled_frames = np.concatenate((zero_frame[None, ...], sampled), axis=0)
        times = np.arange(len(sampled_frames), dtype=np.float64) * (
            INNER_CHUNK_STEPS * CAPTURE_EVERY_CHUNKS * config.dt
        )
        case_dir = directory / label / f"load-{axis}"
        case_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for frame, (particle_u, time_s) in enumerate(zip(sampled_frames, times)):
            filename = f"frame-{frame:03d}.vtu"
            write_vtu(case_dir / filename, p, vol_ratio, particle_u, time_s)
            frames.append((time_s, filename))
        write_pvd(case_dir, frames)
        particle_xyz = np.asarray(p[None, :, :] + sampled_frames, dtype=np.float32)
        current_momentum, current_angular = momentum(final, reference)
        momentum_error = jnp.linalg.norm(current_momentum - final.impulse)
        angular_momentum_error = jnp.linalg.norm(current_angular - final.angular_impulse)
        all_cases[axis] = {
            "tip_response_m2": scores_value,
            "positions_m": particle_xyz,
            "valid": bool(final.valid),
            "min_J": float(jnp.min(jnp.linalg.det(final.F))),
            "momentum_error_kg_m_s": float(momentum_error),
            "angular_momentum_error_kg_m2_s": float(angular_momentum_error),
        }
    return all_cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-grid", type=int, default=24,
                        help="background grid for differentiable design iterations (default: 24)")
    parser.add_argument("--production-grid", type=int, default=64,
                        help="independent final 3D confirmation grid; default is 64^3")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--time-horizon", type=float, default=TIME_HORIZON,
                        help="same physical analysis time on design and production meshes (s)")
    parser.add_argument("--design-dt", type=float, default=DESIGN_DT)
    parser.add_argument("--production-dt", type=float, default=PRODUCTION_DT)
    parser.add_argument("--output", type=Path, default=Path("outputs/structural-design"))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-state", type=Path,
                        help="resume controls and Adam moments from optimizer-state.npz")
    resume.add_argument("--resume-history", type=Path,
                        help="resume a CSV and reconstruct Adam moments from its saved designs")
    parser.add_argument("--project-resume-amplitude", type=float, default=1.0,
                        help="scale resumed controls by this factor before evaluation (feasibility projection)")
    parser.add_argument("--skip-design", action="store_true",
                        help="verify the uniform baseline only; mainly for resource qualification")
    parser.add_argument("--optimization-only", action="store_true",
                        help="save restartable optimizer history/checkpoint without production forwards")
    args = parser.parse_args(argv)
    if args.time_horizon <= 2 * RAMP_TIME:
        parser.error("--time-horizon must include loading, hold, and unloading")
    design_steps = int(round(args.time_horizon / args.design_dt))
    production_steps = int(round(args.time_horizon / args.production_dt))
    if (design_steps < INNER_CHUNK_STEPS or production_steps < INNER_CHUNK_STEPS or
            design_steps % INNER_CHUNK_STEPS or production_steps % INNER_CHUNK_STEPS):
        parser.error("time horizon / each dt must give a positive step count divisible by 12")
    if (abs(design_steps * args.design_dt - args.time_horizon) > 1e-9 or
            abs(production_steps * args.production_dt - args.time_horizon) > 1e-9):
        parser.error("time horizon must be an exact integer number of steps on both meshes")
    if args.skip_design and (args.resume_state or args.resume_history):
        parser.error("--skip-design cannot be combined with a resume file")
    if not (0.0 < args.project_resume_amplitude <= 1.0):
        parser.error("--project-resume-amplitude must be in (0, 1]")
    if args.project_resume_amplitude != 1.0 and not (args.resume_state or args.resume_history):
        parser.error("--project-resume-amplitude requires a resume file")
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)

    design_model = make_model(args.design_grid, design_steps, args.design_dt)
    response = design_model[-2]
    controls0 = jnp.zeros((N_CONTROL,), dtype=jnp.float64)
    production_points_for_constraint, _, _ = particle_cloud(args.production_grid, dtype=jnp.float64)
    feasibility_bases = (
        np.asarray(design_model[3], dtype=np.float64),
        design_basis_numpy(np.asarray(production_points_for_constraint)[:, 0]),
    )
    evaluate = jax.jit(jax.value_and_grad(response))
    signature = {
        "design_grid": args.design_grid, "design_dt": args.design_dt,
        "time_horizon": args.time_horizon, "design_steps": design_steps,
        "control_count": N_CONTROL, "material": "neo-hookean",
        "young_pa": YOUNG, "poisson": POISSON, "density_kg_m3": RHO,
        "volume_fraction": VOLUME_FRACTION,
    }
    resume_state = None
    if args.resume_state:
        with np.load(args.resume_state, allow_pickle=False) as saved:
            saved_signature = json.loads(str(saved["signature_json"].item()))
            if saved_signature != signature:
                raise ValueError(f"optimizer checkpoint model signature mismatch: {saved_signature}")
            resume_state = {
                "controls": saved["controls"], "first": saved["first"],
                "second": saved["second"], "iteration": int(saved["iteration"]),
                "history": json.loads(str(saved["history_json"].item())),
            }
    elif args.resume_history:
        with args.resume_history.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"resume history is empty: {args.resume_history}")
        history = []
        for row in rows:
            history.append({
                "iteration": int(row["iteration"]), "objective": float(row["objective"]),
                "gradient_l2": float(row["gradient_l2"]),
                "solid_fraction_mean": float(row["solid_fraction_mean"]),
                "solid_fraction_min": float(row["solid_fraction_min"]),
                "solid_fraction_max": float(row["solid_fraction_max"]),
                "mass_kg": float(row["mass_kg"]), "controls": json.loads(row["controls"]),
            })
        first = jnp.zeros((N_CONTROL,), dtype=jnp.float64)
        second = jnp.zeros((N_CONTROL,), dtype=jnp.float64)
        # Recreate the exact Adam moments used to reach the last saved design.
        for row in history[:-1]:
            _, grad = evaluate(jnp.asarray(row["controls"], dtype=jnp.float64))
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
        resume_state = {
            "controls": np.asarray(history[-1]["controls"]), "first": np.asarray(first),
            "second": np.asarray(second), "iteration": int(history[-1]["iteration"]),
            "history": history,
        }
    if resume_state is not None and args.project_resume_amplitude != 1.0:
        resume_state["controls"] = (
            np.asarray(resume_state["controls"], dtype=np.float64)
            * args.project_resume_amplitude
        )
        if args.iterations > 0:
            resume_state["first"] = np.zeros((N_CONTROL,), dtype=np.float64)
            resume_state["second"] = np.zeros((N_CONTROL,), dtype=np.float64)
    controls_start = (controls0 if resume_state is None else
                      jnp.asarray(resume_state["controls"], dtype=jnp.float64))
    start_iteration = 0 if resume_state is None else resume_state["iteration"]
    print(f"DESIGN_START grid={args.design_grid} particles={len(design_model[1])} dt={args.design_dt:g} steps={design_steps} horizon={args.time_horizon:g} start_iteration={start_iteration}", flush=True)
    initial_value, initial_grad = evaluate(controls_start)
    if not np.isfinite(float(initial_value)) or not np.all(np.isfinite(np.asarray(initial_grad))):
        raise RuntimeError("initial design objective/gradient is not finite")
    # One central finite difference along the normalised autodiff direction.
    direction = initial_grad / (jnp.linalg.norm(initial_grad) + 1e-16)
    eps = 2.0e-3
    fd = (evaluate(controls_start + eps * direction)[0] - evaluate(controls_start - eps * direction)[0]) / (2 * eps)
    ad_directional = jnp.vdot(initial_grad, direction)
    fd_rel_error = abs(ad_directional - fd) / jnp.maximum(abs(fd), 1e-12)
    if not np.isfinite(float(fd_rel_error)) or float(fd_rel_error) > 1.0e-3:
        raise RuntimeError(f"directional AD/finite-difference check failed: {float(fd_rel_error)}")
    print(f"INITIAL_AD_DONE objective={float(initial_value):.9g} directional_fd_error={float(fd_rel_error):.6g}", flush=True)

    if args.skip_design:
        controls = controls0
        history = [{"iteration": 0, "objective": float(initial_value),
                    "gradient_l2": float(jnp.linalg.norm(initial_grad)),
                    "solid_fraction_mean": float(jnp.mean(volume_scale(controls0, design_model[3]))),
                    "solid_fraction_min": float(jnp.min(volume_scale(controls0, design_model[3]))),
                    "solid_fraction_max": float(jnp.max(volume_scale(controls0, design_model[3]))),
                    "mass_kg": float(RHO * jnp.sum(design_model[2] * volume_scale(controls0, design_model[3]))),
                    "controls": np.asarray(controls0).tolist()}]
    else:
        controls, history = optimize(response, evaluate, args.iterations, args.learning_rate,
                                     design_model[3], feasibility_bases, design_model[2],
                                     args.output / "optimizer-state.npz",
                                     args.output / "design-history.csv", signature,
                                     initial_state=resume_state)
    if args.optimization_only:
        controls_scale = volume_scale(controls, design_model[3])
        result = {
            "phase": "optimization-only",
            "model_signature": signature,
            "last_iteration": int(history[-1]["iteration"]),
            "objective_initial_design_mesh": float(history[0]["objective"]),
            "objective_final_design_mesh": float(history[-1]["objective"]),
            "mass_kg": float(RHO * jnp.sum(design_model[2] * controls_scale)),
            "solid_fraction_mean": float(jnp.mean(controls_scale)),
            "solid_fraction_min": float(jnp.min(controls_scale)),
            "solid_fraction_max": float(jnp.max(controls_scale)),
            "controls": np.asarray(controls).tolist(),
            "history_csv": "design-history.csv",
            "optimizer_checkpoint": "optimizer-state.npz",
        }
        (args.output / "optimization-status.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        return
    design_value = history[-1]["objective"]
    design_scale = volume_scale(controls, design_model[3])
    if float(jnp.min(design_scale)) <= 0.0 or float(jnp.max(design_scale)) > 1.0:
        raise RuntimeError("final design violates the physical solid-fraction bound 0 < fraction <= 1")
    mass_before = float(RHO * jnp.sum(design_model[2] * volume_scale(controls0, design_model[3])))
    mass_after = float(RHO * jnp.sum(design_model[2] * design_scale))

    # The production confirmation refines to a 64^3 background grid (~70k
    # particles), then runs independent forward analyses under both loads.
    production_model = make_model(args.production_grid, production_steps, args.production_dt, dtype=jnp.float64)
    production_scale = volume_scale(controls, production_model[3])
    if float(jnp.min(production_scale)) <= 0.0 or float(jnp.max(production_scale)) > 1.0:
        raise RuntimeError(
            "final design violates the physical solid-fraction bound on the production mesh: "
            f"range={float(jnp.min(production_scale)):.9g}..{float(jnp.max(production_scale)):.9g}"
        )
    print(f"PRODUCTION_START grid={args.production_grid} particles={len(production_model[1])} dt={args.production_dt:g} steps={production_steps} horizon={args.time_horizon:g}", flush=True)
    load_audit = {
        "design_initial": load_patch_audit(design_model, controls0),
        "design_optimized": load_patch_audit(design_model, controls),
        "production_initial": load_patch_audit(production_model, controls0),
        "production_optimized": load_patch_audit(production_model, controls),
    }
    print("LOAD_PATCH_AUDIT " + json.dumps(load_audit), flush=True)
    before_cases = write_trajectory(args.output, "initial", controls0, production_model)
    after_cases = write_trajectory(args.output, "optimized", controls, production_model)
    baseline_production = (before_cases["y"]["tip_response_m2"] + before_cases["z"]["tip_response_m2"]) / (2 * 0.04**2)
    design_smoothness = 0.002 * float(jnp.mean(jnp.diff(controls) ** 2))
    optimized_production = (after_cases["y"]["tip_response_m2"] + after_cases["z"]["tip_response_m2"]) / (2 * 0.04**2) + design_smoothness
    summary = {
        "model": "3D total-Lagrangian MPM dynamic solid-fraction material-layout redistribution",
        "constitutive_model": "compressible Neo-Hookean, E=200 kPa, nu=0.25, rho=1000 kg/m^3",
        "simulation_precision": "float64 for design, gradient checks, and refined forward confirmation; VTK/NPZ visual state exports are float32",
        "design_variables": "10 smooth axial solid-volume fraction controls, fixed envelope and particle positions/stencil topology; linear p=1 Voigt mixture",
        "constraints": {
            "reference_volume_fraction": VOLUME_FRACTION,
            "mass_before_kg": mass_before,
            "mass_after_kg": mass_after,
            "mass_relative_error": abs(mass_after - mass_before) / mass_before,
            "volume_scale_min": float(jnp.min(design_scale)),
            "volume_scale_max": float(jnp.max(design_scale)),
            "local_solid_fraction_bound": [0.0, 1.0],
            "design_grid_fraction": {
                "min": float(jnp.min(design_scale)),
                "max": float(jnp.max(design_scale)),
                "mean": float(jnp.mean(design_scale)),
            },
            "production_grid_fraction": {
                "min": float(jnp.min(production_scale)),
                "max": float(jnp.max(production_scale)),
                "mean": float(jnp.mean(production_scale)),
            },
            "unconstrained_analytical_scale_bounds": [
                VOLUME_FRACTION * math.exp(-0.56), VOLUME_FRACTION * math.exp(0.56)
            ],
        },
        "differentiation": {
            "jax_value_and_grad": True,
            "jit": True,
            "lax_scan": True,
            "checkpointed_12_step_blocks": True,
            "sequential_checkpointed_load_cases": 2,
            "initial_directional_ad": float(ad_directional),
            "initial_directional_finite_difference": float(fd),
            "directional_relative_error": float(fd_rel_error),
        },
        "load_patch_audit": load_audit,
        "optimization": {
            "design_grid": args.design_grid,
            "production_grid": args.production_grid,
            "particle_count_production": int(len(production_model[1])),
            "time_horizon_s": args.time_horizon,
            "design_time_steps": design_steps,
            "design_dt_s": args.design_dt,
            "production_time_steps": production_steps,
            "production_dt_s": args.production_dt,
            "capture_stride_steps": INNER_CHUNK_STEPS * CAPTURE_EVERY_CHUNKS,
            "design_cfl": (math.sqrt(YOUNG * (1 - POISSON) / (RHO * (1 + POISSON) * (1 - 2 * POISSON))) * args.design_dt / (1.0 / args.design_grid)),
            "production_cfl": (math.sqrt(YOUNG * (1 - POISSON) / (RHO * (1 + POISSON) * (1 - 2 * POISSON))) * args.production_dt / (1.0 / args.production_grid)),
            "iterations": len(history) - 1,
            "optimizer_last_iteration": int(history[-1]["iteration"]),
            "additional_iterations_this_run": 0 if args.skip_design else args.iterations,
            "post_optimizer_feasibility_projection_amplitude": args.project_resume_amplitude,
            "adam_moments_reset_after_projection": bool(
                args.project_resume_amplitude != 1.0 and args.iterations > 0
            ),
            "objective_initial_design_mesh": float(history[0]["objective"]),
            "objective_resume_start_design_mesh": float(initial_value),
            "objective_optimized_design_mesh": float(design_value),
            "objective_initial_production_mesh": float(baseline_production),
            "objective_optimized_production_mesh": float(optimized_production),
            "production_relative_improvement": float((baseline_production - optimized_production) / abs(baseline_production)),
            "two_load_cases": {
                "initial": {axis: {"tip_response_m2": case["tip_response_m2"], "valid": case["valid"], "min_J": case["min_J"], "momentum_error_kg_m_s": case["momentum_error_kg_m_s"], "angular_momentum_error_kg_m2_s": case["angular_momentum_error_kg_m2_s"]} for axis, case in before_cases.items()},
                "optimized": {axis: {"tip_response_m2": case["tip_response_m2"], "valid": case["valid"], "min_J": case["min_J"], "momentum_error_kg_m_s": case["momentum_error_kg_m_s"], "angular_momentum_error_kg_m2_s": case["angular_momentum_error_kg_m2_s"]} for axis, case in after_cases.items()},
            },
            "final_design_controls": np.asarray(controls).tolist(),
        },
        "artifacts": {
            "design_history": "design-history.csv",
            "optimizer_checkpoint": "optimizer-state.npz",
            "initial_states": "initial/load-y/series.pvd and initial/load-z/series.pvd",
            "optimized_states": "optimized/load-y/series.pvd and optimized/load-z/series.pvd",
            "render_command": "pvpython --force-offscreen-rendering tools/render_structural_design_pv.py --input outputs/structural-design --output outputs/structural-design/paraview-gif --case y",
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_history_csv(args.output / "design-history.csv", history)
    production_scale = volume_scale(controls, production_model[3])
    np.savez_compressed(
        args.output / "trajectory.npz",
        points=np.asarray(production_model[1], dtype=np.float32),
        volume_ratio_initial=np.asarray(volume_scale(controls0, production_model[3]), dtype=np.float32),
        volume_ratio_optimized=np.asarray(production_scale, dtype=np.float32),
        **{f"initial_{axis}_positions": case["positions_m"] for axis, case in before_cases.items()},
        **{f"optimized_{axis}_positions": case["positions_m"] for axis, case in after_cases.items()},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
