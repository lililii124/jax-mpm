"""3D slope-slide demonstration with MAT173 soil and a weak elastic-base contact.

The slope is not prescribed to move: self-weight ramps on both bodies and a
recorded Coulomb contact acts at their basal interface. This is a numerical
demonstration, not native MAT173 validation or a site-specific prediction.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from functools import partial
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax.boundaries import PlaneWall, fixed_box
from mpm_jax.contact import coupled_step
from mpm_jax.core import Config, State
from mpm_jax.materials import initial_history, mat173, neo_hookean, update_mat173


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "examples/mat173_landslide.py",
    "mpm_jax/__init__.py",
    "mpm_jax/core.py",
    "mpm_jax/materials/__init__.py",
    "mpm_jax/materials/history.py",
    "mpm_jax/materials/mat173.py",
    "mpm_jax/boundaries.py",
    "mpm_jax/contact.py",
    "mpm_jax/materials/elastic.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return json_safe(np.asarray(value).tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_particles(spacing=0.05):
    """Slope body with a flat back bench and a single descending 33-degree face."""
    xs = np.arange(0.2 + spacing / 2, 3.0, spacing)
    ys = np.arange(0.5 + spacing / 2, 1.8, spacing)
    zs = np.arange(0.5 + spacing / 2, 2.5, spacing)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    top = np.where(X <= 1.0, 1.8, 1.8 - 1.3 * (X - 1.0) / 2.0)
    keep = (Y <= top) & (Y >= 0.5) & (X <= 3.0)
    return np.column_stack((X[keep], Y[keep], Z[keep])).astype(np.float64)


def make_foundation_particles(spacing=0.1):
    """Finite elastic foundation block below the basal contact interface."""
    xs = np.arange(0.2 + spacing / 2, 3.3, spacing)
    ys = np.arange(0.2 + spacing / 2, 0.5, spacing)
    zs = np.arange(0.3 + spacing / 2, 2.7, spacing)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel())).astype(np.float64)


@partial(jax.jit, static_argnames=("configs",))
def _advance_one(states, materials, configs, interface_friction, ramp_time):
    next_time = states[0].time + configs[0].dt
    phase = jnp.clip(next_time / ramp_time, 0.0, 1.0)
    scale = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
    gravity = jnp.asarray((0.0, -9.81, 0.0), dtype=states[0].x.dtype)
    forces = tuple(state.mass[:, None] * scale * gravity[None, :] for state in states)
    return coupled_step(states, materials, configs, interface_friction, forces)


@partial(jax.jit, static_argnames=("configs", "steps"))
def _advance_chunk(states, materials, configs, interface_friction, ramp_time, steps):
    def body(carry, _):
        state_pair, impulse_sum = carry
        valid = state_pair[0].valid & state_pair[1].valid
        def advance(current):
            updated, impulse = _advance_one(current, materials, configs, interface_friction, ramp_time)
            return updated, impulse_sum + impulse
        next_pair, next_impulse = jax.lax.cond(valid, advance, lambda current: carry, state_pair)
        return (next_pair, next_impulse), None

    impulse0 = jnp.zeros((3,), dtype=states[0].x.dtype)
    return jax.lax.scan(body, (states, impulse0), None, length=steps)[0]


def checkpoint_arrays(states):
    arrays = {}
    for prefix, state in zip(("soil_", "foundation_"), states):
        arrays.update({prefix + key: np.asarray(getattr(state, key)) for key in
            ("x", "v", "F", "C", "mass", "volume", "time", "valid")})
        arrays.update({prefix + "history_" + key: np.asarray(getattr(state.history, key))
            for key in state.history._fields})
    return arrays


def constitutive_preflight(material):
    """One batched JAX check of the elastic, plastic-edge and hydrostatic branches."""
    dtype = jnp.float64
    old = initial_history(3, dtype, material, material.density)
    uniaxial = -0.01
    lateral = -material.poisson * uniaxial
    strains = jnp.asarray((
        ((1e-5, 2e-6, -1e-6), (2e-6, -3e-6, 4e-6), (-1e-6, 4e-6, 2e-6)),
        ((uniaxial, 0., 0.), (0., lateral, 0.), (0., 0., lateral)),
        -1e-4 * np.eye(3),
    ), dtype=dtype)
    updated = jax.jit(update_mat173)(old, strains, jnp.ones((3,), dtype=dtype), 1e-4, material)
    stress = np.asarray(updated.stress)
    pstrain = np.asarray(updated.plastic_strain)
    principal = np.linalg.eigvalsh(stress)
    smax, smin = principal[:, 2], principal[:, 0]
    phi = material.friction
    max_yield = max(
        (1 + math.sin(phi)) * float(a) - (1 - math.sin(phi)) * float(b)
        - 2 * material.cohesion * math.cos(phi)
        for a, b in zip(smax, smin)
    )
    G, nu = material.shear, material.poisson
    lame = 2 * G * nu / (1 - 2 * nu)
    expected = 2 * G * np.asarray(strains[0]) + lame * np.trace(np.asarray(strains[0])) * np.eye(3)
    elastic_error = float(np.max(np.abs(stress[0] - expected)))
    result = {"valid": bool(np.asarray(updated.valid).all()),
        "elastic_max_abs_error_Pa": elastic_error,
        "plastic_case_accumulated_strain": float(pstrain[1]),
        "hydrostatic_case_accumulated_strain": float(pstrain[2]),
        "plastic_case_max_yield_residual_Pa": max_yield,
        "branch_check": "plastic path entered, returned MC principal surfaces; hydrostatic compression elastic"}
    if (not result["valid"] or elastic_error > 1e-6 or pstrain[1] <= 0 or
            abs(float(pstrain[2])) > 1e-14 or max_yield > 1e-4):
        raise RuntimeError(f"MAT173 constitutive preflight failed: {result}")
    return result


def save_frame(out, frame, step_number, references, states):
    prefix = Path(out) / "frames"
    prefix.mkdir(parents=True, exist_ok=True)
    soil, foundation = states
    arrays = {
        "step": np.asarray(step_number, dtype=np.int64),
        "time_s": np.asarray(soil.time),
        "soil_reference_x": references[0],
        "soil_x": np.asarray(soil.x),
        "soil_displacement": np.asarray(soil.x) - references[0],
        "soil_velocity": np.asarray(soil.v),
        "soil_plastic_strain": np.asarray(soil.history.plastic_strain),
        "soil_pressure_Pa": np.asarray(soil.history.pressure),
        "soil_stress_Pa": np.asarray(soil.history.stress),
        "foundation_reference_x": references[1],
        "foundation_x": np.asarray(foundation.x),
        "foundation_displacement": np.asarray(foundation.x) - references[1],
        "foundation_velocity": np.asarray(foundation.v),
        "foundation_F": np.asarray(foundation.F),
    }
    atomic_npz(
        prefix / f"frame_{frame:03d}.npz",
        **arrays,
    )


def deformation_metrics(state):
    F = np.asarray(state.F)
    stretches = np.linalg.svd(F, compute_uv=False)
    max_log_strain = float(np.max(np.abs(np.log(stretches))))
    minimum_J = float(np.linalg.det(F).min())
    return max_log_strain, minimum_J


def make_state(x0, spacing, density, material, dtype):
    x = jnp.asarray(x0, dtype=dtype)
    n = x.shape[0]
    return State(
        x, jnp.zeros_like(x), jnp.broadcast_to(jnp.eye(3, dtype=dtype), (n, 3, 3)),
        jnp.zeros((n, 3, 3), dtype=dtype),
        jnp.full((n,), density * spacing**3, dtype=dtype),
        jnp.full((n,), spacing**3, dtype=dtype),
        initial_history(n, dtype, material, density), jnp.asarray(0.0, dtype=dtype), jnp.asarray(True),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--frames", type=int, default=61)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=7.5e-5)
    parser.add_argument("--interface-friction", type=float, default=0.1)
    args = parser.parse_args()
    if args.steps < 1 or args.frames < 2 or args.frames > args.steps + 1:
        parser.error("require steps>=1 and 2<=frames<=steps+1")
    if not np.isfinite(args.interface_friction) or args.interface_friction < 0:
        parser.error("interface-friction must be finite and nonnegative")

    x0 = make_particles(args.spacing)
    xf = make_foundation_particles()
    if len(x0) < 30000:
        raise ValueError("slope particle cloud unexpectedly small")
    density, shear, poisson = 1800.0, 5.86e6, 0.28
    friction, dilation, cohesion = math.radians(28.0), math.radians(2.0), 1000.0
    material = mat173(
        shear_modulus=shear, poisson=poisson, friction_angle=friction,
        cohesion=cohesion, dilation_angle=dilation, density=density, novoid=1,
    )
    foundation_density, foundation_young = 2500.0, 50e6
    foundation_material = neo_hookean(young=foundation_young, poisson=0.28)
    constitutive_check = constitutive_preflight(material)
    dx = 4.0 / args.grid
    soil_wave_speed = math.sqrt((2 * shear * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))) / density)
    foundation_wave_speed = math.sqrt((foundation_material.lam + 2 * foundation_material.mu) / foundation_density)
    cfl = args.dt * max(soil_wave_speed, foundation_wave_speed) / dx
    if cfl > 0.18:
        raise ValueError(f"requested timestep has CFL={cfl:.4g}, above the 0.18 limit")
    soil_config = Config(
        n_grid=args.grid, dt=args.dt, domain_size=4.0, gravity=(0.0, 0.0, 0.0),
        boundary_cells=0, material="mat173", dim=3,
        boundaries=(
            PlaneWall(axis=0, position=0.2, side="lower", friction=0.0),
            PlaneWall(axis=2, position=0.3, side="lower", friction=0.0),
            PlaneWall(axis=2, position=2.7, side="upper", friction=0.0),
        ),
    )
    foundation_config = Config(
        n_grid=args.grid, dt=args.dt, domain_size=4.0, gravity=(0.0, 0.0, 0.0),
        boundary_cells=0, material="neo-hookean", dim=3,
        boundaries=(
            fixed_box((0.0, 0.0, 0.0), (4.0, 0.3, 4.0)),
            PlaneWall(axis=0, position=0.2, side="lower", friction=0.0),
            PlaneWall(axis=2, position=0.3, side="lower", friction=0.0),
            PlaneWall(axis=2, position=2.7, side="upper", friction=0.0),
        ),
    )
    configs = (soil_config, foundation_config)
    materials = (material, foundation_material)
    dtype = jnp.float64
    states = (make_state(x0, args.spacing, density, material, dtype),
              make_state(xf, 0.1, foundation_density, foundation_material, dtype))
    references = (x0, xf)

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be new or empty")
    (out / "frames").mkdir(parents=True, exist_ok=True)
    slope_angle = math.atan2(1.3, 2.0)
    normal_depth = 1.3 / math.sin(slope_angle)
    gamma = density * 9.81
    driving = gamma * normal_depth * math.sin(slope_angle) * math.cos(slope_angle)
    resistance = cohesion + gamma * normal_depth * math.cos(slope_angle) ** 2 * math.tan(friction)
    factor_of_safety = resistance / driving
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    metadata = {
        "schema": "mat173-3d-slope-weak-base-demo-v2",
        "scope": "numerical demonstration; not native MAT173 final-check evidence or site-specific prediction",
        "material_parameters_SI": {"density_kg_m3": density, "shear_modulus_Pa": shear,
            "poisson": poisson, "friction_angle_deg": 28.0, "dilation_angle_deg": 2.0,
            "cohesion_Pa": cohesion, "novoid": 1},
        "geometry": {"type": "3D slope body with a flat back bench and descending face",
            "slope_height_m": 1.3, "slope_run_m": 2.0, "slope_width_m": 2.0,
            "bench_x_m": [0.2, 1.0], "bench_elevation_m": 1.8,
            "toe_x_m": 3.0,
            "face_angle_deg": math.degrees(slope_angle), "particle_spacing_m": args.spacing,
            "slope_particles": len(x0), "domain_m": [4.0, 4.0, 4.0], "grid_cells_per_axis": args.grid,
            "foundation": {"geometry_m": {"x": [0.2, 3.3], "y": [0.2, 0.5], "z": [0.3, 2.7]},
                "particles": len(xf), "particle_spacing_m": 0.1,
                "material": "compressible neo-Hookean elastic block",
                "young_modulus_Pa": foundation_young, "poisson": 0.28,
                "density_kg_m3": foundation_density,
                "fixed_boundary": "all grid velocity components fixed at y<=0.3 m"},
            "basal_interface": {"type": "two-body MPM grid contact", "coulomb_friction": args.interface_friction,
                "initial_material_point_center_gap_m": 0.075,
                "contact_scope": "normal impulse plus Coulomb tangential impulse; no adhesion"},
            "back_bench_boundary": "roller wall at x=0.2 m blocks motion out of the modeled uphill boundary",
            "side_walls_z_m": [0.3, 2.7],
            "initial_stress": "zero in slope and foundation; gravity is ramped over 0.02 s, so the initial stress-building transient is included"},
        "loading": {"trigger": "self-weight only; no prescribed displacement or velocity trigger",
            "body_force": "rho*g multiplied by a half-cosine ramp over 0.02 s on both bodies",
            "gravity_m_s2": [0.0, -9.81, 0.0], "steps": args.steps,
            "dt_s": args.dt, "duration_s": args.steps * args.dt, "cfl": cfl,
            "interface_friction": args.interface_friction,
            "frames": args.frames, "frame_step_indices": np.rint(np.linspace(0,args.steps,args.frames)).astype(int).tolist()},
        "infinite_slope_screen": {"factor_of_safety": factor_of_safety,
            "driving_shear_Pa": driving, "resistance_Pa": resistance,
            "scope": "internal-strength planar infinite-slope screening estimate; excludes finite bench, weak basal contact, foundation and transient stress build-up, and is not validation"},
        "constitutive_scope": "MAT173 six principal-stress surfaces, constant cohesion, nonassociated flow, face/edge/apex active-set return and NOVOID=1 hydrostatic tensile exception. Incremental small strain with corotated chart; finite-deformation host update and native history mapping are not validated.",
        "constitutive_preflight": constitutive_check,
        "initial_mass_kg": {"slope": float(density * args.spacing**3 * len(x0)),
            "foundation": float(foundation_density * 0.1**3 * len(xf))},
        "sources": sources,
    }
    atomic_json(out / "model.json", metadata)
    atomic_npz(out / "reference.npz", soil_x=x0, foundation_x=xf)
    writer = (out / "metrics.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(writer, fieldnames=("frame", "step", "time_s", "valid", "max_displacement_m",
        "max_speed_m_s", "max_downslope_displacement_m", "median_downslope_displacement_m",
        "p90_downslope_displacement_m", "max_downslope_speed_m_s", "max_abs_log_strain",
        "minimum_J", "max_plastic_strain", "plastic_particle_fraction", "max_compression_Pa",
        "foundation_max_displacement_m", "foundation_max_abs_log_strain",
        "contact_impulse_x_Ns", "contact_impulse_y_Ns", "contact_impulse_z_Ns"))
    csv_writer.writeheader()
    ramp_time = 0.02
    frame_steps = np.rint(np.linspace(0, args.steps, args.frames)).astype(int)
    current_step = 0
    start = time.monotonic()
    small_strain_limit_reached = False
    cumulative_contact_impulse = np.zeros(3)
    for frame, target_step in enumerate(frame_steps):
        remaining = int(target_step - current_step)
        while remaining:
            chunk = min(remaining, 10)
            states, contact_impulse = _advance_chunk(
                states, materials, configs, args.interface_friction, ramp_time, chunk)
            cumulative_contact_impulse += np.asarray(contact_impulse)
            current_step += chunk
            remaining -= chunk
            if not bool(states[0].valid & states[1].valid):
                break
        save_frame(out, frame, current_step, references, states)
        arrays = checkpoint_arrays(states)
        atomic_npz(out / "checkpoint.npz", **arrays)
        soil, foundation = states
        soil_disp = np.asarray(soil.x) - x0
        soil_velocity = np.asarray(soil.v)
        displacement = np.linalg.norm(soil_disp, axis=-1)
        speed = np.linalg.norm(soil_velocity, axis=-1)
        plastic = np.asarray(soil.history.plastic_strain)
        compression = np.maximum(-np.trace(np.asarray(soil.history.stress), axis1=1, axis2=2), 0) / 3
        max_log_strain, minimum_J = deformation_metrics(soil)
        foundation_log_strain, foundation_minimum_J = deformation_metrics(foundation)
        beta = slope_angle
        downhill = soil_disp @ np.asarray((math.cos(beta), -math.sin(beta), 0.0))
        downhill_speed = soil_velocity @ np.asarray((math.cos(beta), -math.sin(beta), 0.0))
        face_mask = x0[:, 0] >= 1.0
        csv_writer.writerow({"frame": frame, "step": current_step, "time_s": float(soil.time),
            "valid": bool(soil.valid & foundation.valid), "max_displacement_m": float(displacement.max()),
            "max_speed_m_s": float(speed.max()), "max_downslope_displacement_m": float(downhill[face_mask].max()),
            "median_downslope_displacement_m": float(np.median(downhill[face_mask])),
            "p90_downslope_displacement_m": float(np.percentile(downhill[face_mask], 90)),
            "max_downslope_speed_m_s": float(downhill_speed[face_mask].max()),
            "max_abs_log_strain": max_log_strain,
            "minimum_J": minimum_J, "max_plastic_strain": float(plastic.max()),
            "plastic_particle_fraction": float(np.mean(plastic > 1e-10)),
            "max_compression_Pa": float(compression.max()),
            "foundation_max_displacement_m": float(np.linalg.norm(np.asarray(foundation.x)-xf,axis=-1).max()),
            "foundation_max_abs_log_strain": foundation_log_strain,
            "contact_impulse_x_Ns": float(cumulative_contact_impulse[0]),
            "contact_impulse_y_Ns": float(cumulative_contact_impulse[1]),
            "contact_impulse_z_Ns": float(cumulative_contact_impulse[2])})
        writer.flush()
        if not bool(soil.valid & foundation.valid):
            break
        if max_log_strain >= 0.05 or float(plastic.max()) >= 0.05:
            small_strain_limit_reached = True
            break
    writer.close()
    valid = bool(states[0].valid & states[1].valid)
    status = ("INVALID_OR_INCOMPLETE" if not valid else
              "SMALL_STRAIN_LIMIT" if small_strain_limit_reached else
              "COMPLETE" if current_step == args.steps else "INCOMPLETE")
    max_log_strain, minimum_J = deformation_metrics(states[0])
    foundation_log_strain, foundation_minimum_J = deformation_metrics(states[1])
    summary = {"status": status,
        "actual_steps": current_step, "target_steps": args.steps, "time_s": float(states[0].time),
        "valid": valid, "elapsed_s": time.monotonic() - start,
        "slope_particles": len(x0), "foundation_particles": len(xf),
        "max_displacement_m": float(np.linalg.norm(np.asarray(states[0].x)-x0, axis=-1).max()),
        "max_abs_log_strain": max_log_strain, "minimum_J": minimum_J,
        "max_plastic_strain": float(np.asarray(states[0].history.plastic_strain).max()),
        "foundation_max_displacement_m": float(np.linalg.norm(np.asarray(states[1].x)-xf, axis=-1).max()),
        "foundation_max_abs_log_strain": foundation_log_strain,
        "foundation_minimum_J": foundation_minimum_J,
        "small_strain_limit_reached": small_strain_limit_reached,
        "cumulative_contact_impulse_Ns": cumulative_contact_impulse.tolist(),
        "all_sources_sha256": sources, "constitutive_preflight": constitutive_check}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if summary["status"] in ("INVALID_OR_INCOMPLETE", "INCOMPLETE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
