"""Finite-deformation loading cases rendered at their actual geometry scale."""

import argparse
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import (
    Config,
    CyclicVelocityBox,
    MovingVelocityBox,
    RotatingVelocityBox,
    block,
    fixed_box,
    neo_hookean,
)
from .common import run
from .scenarios import Case


SCENES = ("wave", "cantilever", "surface-load", "shear", "torsion")


def pressure_cycle(time):
    rise = 0.5 * (1 - jnp.cos(jnp.pi * jnp.clip(time / 0.3, 0.0, 1.0)))
    fall = 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip((time - 0.6) / 0.3, 0.0, 1.0)))
    return 1000.0 * rise * fall


def make_case(scene, *, grid=48, ppc=3, frames=50, dt_factor=1.0):
    if scene not in SCENES:
        raise ValueError("unknown visible-motion case")
    if grid < 24 or ppc < 1 or frames < 1 or not 0 < dt_factor <= 1:
        raise ValueError(
            "grid >= 24, ppc >= 1, frames >= 1 and 0 < dt_factor <= 1 are required"
        )
    young, poisson, density = 2e4, 0.2, 1000.0
    material = neo_hookean(young, poisson)
    acoustic_speed = float(np.sqrt((material.bulk + 4 * material.mu / 3) / density))
    shear_speed = float(np.sqrt(material.mu / density))
    lower, upper = (0.15, 0.4, 0.35), (0.8, 0.6, 0.65)
    grip = None
    clamp_axis, clamp_edge = 0, 0.2
    if scene == "wave":
        lower, upper = (0.125, 0.35, 0.375), (0.875, 0.55, 0.625)
        clamp_edge = 0.175
        duration = 4 * (upper[0] - clamp_edge) / shear_speed
        boundaries = (
            fixed_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), components=(True, False, True)),
            fixed_box((0.0, 0.0, 0.0), (clamp_edge, 1.0, 1.0)),
        )
    elif scene == "shear":
        lower, upper = (0.3, 0.25, 0.3), (0.7, 0.75, 0.7)
        clamp_axis, clamp_edge = 1, 0.28
        duration = 0.4
        grip = MovingVelocityBox(
            (0.0, 0.72, 0.0), (1.0, 1.0, 1.0), (0.5, 0.0, 0.0), ramp_time=0.08
        )
        boundaries = (fixed_box((0.0, 0.0, 0.0), (1.0, clamp_edge, 1.0)), grip)
    elif scene == "cantilever":
        duration = 1.4
        grip = CyclicVelocityBox(
            lower=(0.72, 0.3, 0.3),
            upper=(1.0, 0.7, 0.7),
            displacement=(0.0, 0.12, 0.0),
            period=1.2,
            components=(False, True, False),
        )
        boundaries = (fixed_box((0.0, 0.0, 0.0), (clamp_edge, 1.0, 1.0)), grip)
    elif scene == "torsion":
        duration = 1.2
        grip = RotatingVelocityBox(
            lower=(0.73, 0.3, 0.3),
            upper=(1.0, 0.7, 0.7),
            center=(0.0, 0.5, 0.5),
            axis=0,
            angle=float(np.pi / 2),
            period=1.0,
        )
        boundaries = (fixed_box((0.0, 0.0, 0.0), (clamp_edge, 1.0, 1.0)), grip)
    else:
        lower, upper = (0.2, 0.45, 0.2), (0.8, 0.55, 0.8)
        duration = 1.2
        boundaries = (
            fixed_box((0.0, 0.0, 0.0), (0.25, 1.0, 1.0)),
            fixed_box((0.75, 0.0, 0.0), (1.0, 1.0, 1.0)),
            fixed_box((0.0, 0.0, 0.0), (1.0, 1.0, 0.25)),
            fixed_box((0.0, 0.0, 0.75), (1.0, 1.0, 1.0)),
        )
    maximum_dt = 0.12 / grid / acoustic_speed * dt_factor
    substeps = int(np.ceil(duration / (frames * maximum_dt)))
    config = Config(
        dim=3,
        n_grid=grid,
        dt=duration / (frames * substeps),
        material="neo-hookean",
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=boundaries,
    )
    state = block(
        config, lower, upper, particles_per_cell=ppc, density=density, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    nodes_lower = np.floor(reference / config.dx - 0.5).astype(int)
    nodes_upper = nodes_lower + 2
    anchor = nodes_upper[:, clamp_axis] <= int(np.floor(clamp_edge / config.dx))
    if scene == "surface-load":
        anchor = np.any(
            (nodes_upper[:, (0, 2)] <= int(np.floor(0.25 / config.dx)))
            | (nodes_lower[:, (0, 2)] >= int(np.ceil(0.75 / config.dx))),
            axis=1,
        )
    tip = reference[:, 0] > (0.80 if scene == "wave" else 0.76)
    top = (
        reference[:, 1] > 0.72
        if scene == "shear"
        else reference[:, 1] == reference[:, 1].max()
    )
    center = (reference[:, 0] - 0.5) ** 2 + (reference[:, 2] - 0.5) ** 2 < 0.06**2
    forces = None
    if scene == "wave":
        shape = jnp.sin(
            0.5
            * jnp.pi
            * jnp.clip((state.x[:, 0] - clamp_edge) / (upper[0] - clamp_edge), 0.0, 1.0)
        )
        state = state._replace(v=state.v.at[:, 1].set(0.6 * shape))
    elif scene == "surface-load":
        face = jnp.asarray(top)
        area_per_point = (upper[0] - lower[0]) * (upper[2] - lower[2]) / int(top.sum())

        def forces(states, time):
            return (
                (pressure_cycle(time) * area_per_point * face)[:, None]
                * jnp.array([0.0, -1.0, 0.0]),
            )

    def observe(states, time):
        current = states[0]
        positions, velocities = np.asarray(current.x), np.asarray(current.v)
        change = positions - reference
        magnitude = np.linalg.norm(change, axis=1)
        determinants = np.linalg.det(np.asarray(current.F))
        values = {
            "maximum_displacement_m": float(magnitude.max()),
            "rms_displacement_m": float(np.sqrt(np.mean(magnitude**2))),
            "anchor_max_displacement_m": float(magnitude[anchor].max())
            if anchor.any()
            else None,
            "anchor_max_speed_m_per_s": float(
                np.linalg.norm(velocities[anchor], axis=1).max()
            )
            if anchor.any()
            else None,
            "sample_min_J": float(determinants.min()),
            "sample_max_J": float(determinants.max()),
        }
        if scene not in ("surface-load", "shear"):
            for axis, name in enumerate("xyz"):
                values[f"tip_mean_displacement_{name}_m"] = float(
                    change[tip, axis].mean()
                )
            values["tip_max_displacement_m"] = float(magnitude[tip].max())
        if scene == "wave":
            values["constrained_xz_max_displacement_m"] = float(
                np.abs(change[:, (0, 2)]).max()
            )
            values["constrained_xz_max_speed_m_per_s"] = float(
                np.abs(velocities[:, (0, 2)]).max()
            )
            values["maximum_shear_F_yx"] = float(
                np.abs(np.asarray(current.F)[:, 1, 0]).max()
            )
        elif scene == "shear":
            prescribed, _ = grip.motion(time)
            values["target_top_displacement_x_m"] = float(prescribed[0])
            values["top_mean_displacement_x_m"] = float(change[top, 0].mean())
            values["nominal_engineering_shear"] = float(
                prescribed[0] / (upper[1] - lower[1])
            )
            values["maximum_shear_F_xy"] = float(
                np.abs(np.asarray(current.F)[:, 0, 1]).max()
            )
        elif scene == "cantilever":
            prescribed, _ = grip.motion(time)
            values["target_grip_displacement_y_m"] = float(prescribed[1])
            values["tip_displacement_tracking_error_y_m"] = float(
                change[tip, 1].mean() - prescribed[1]
            )
        elif scene == "torsion":
            prescribed, _ = grip.rotation(time)
            initial_section = reference[tip, 1:3] - np.array([0.5, 0.5])
            current_section = positions[tip, 1:3] - np.array([0.5, 0.5])
            dot = float(np.sum(initial_section * current_section))
            cross = float(
                np.sum(
                    initial_section[:, 0] * current_section[:, 1]
                    - initial_section[:, 1] * current_section[:, 0]
                )
            )
            angle = float(np.arctan2(cross, dot))
            values["target_tip_rotation_x_rad"] = float(prescribed)
            values["tip_rotation_x_rad"] = angle
            values["tip_rotation_tracking_error_rad"] = angle - float(prescribed)
        else:
            pressure = float(pressure_cycle(time))
            values["surface_pressure_Pa"] = pressure
            values["prescribed_surface_force_y_N"] = (
                -pressure * (upper[0] - lower[0]) * (upper[2] - lower[2])
            )
            values["surface_force_assignment_error_N"] = float(
                np.asarray(forces(states, time)[0])[:, 1].sum()
                - values["prescribed_surface_force_y_N"]
            )
            values["center_mean_displacement_y_m"] = float(change[center, 1].mean())
            values["center_min_displacement_y_m"] = float(change[center, 1].min())
        return values

    definition = {
        "purpose": f"visible-{scene}",
        "series": "visible-motion-v1",
        "geometry": {"lower": lower, "upper": upper},
        "particles_per_cell_per_axis": ppc,
        "model": "neo-hookean",
        "young_modulus_Pa": young,
        "poisson_ratio": poisson,
        "density_kg_per_m3": density,
        "initial_acoustic_speed_m_per_s": acoustic_speed,
        "initial_shear_speed_m_per_s": shear_speed,
        "time_step_factor": dt_factor,
        "bulk_CFL_limit": 0.12 * dt_factor,
        "gravity": "zero",
        "units": "m, s, kg, Pa",
        "geometry_scale": 1.0,
        "boundary_method": "prescribed grid velocity constraints",
        "anchor_particles_with_fully_constrained_support": int(anchor.sum()),
        "anchor_sampling": "initial particle selected only if every quadratic B-spline support node is in a fully fixed region; maxnode=floor(X/dx-0.5)+2",
        "tip_reference_x_min_m": None
        if scene in ("shear", "surface-load")
        else 0.80
        if scene == "wave"
        else 0.76,
        "interpretation": "new finite-deformation loading case; original small-motion benchmark records remain separate",
    }
    if grip is not None:
        definition["driven_boundary_type"] = type(grip).__name__
        definition["driven_boundary"] = asdict(grip)
    if scene == "wave":
        definition.update(
            comparison_family="finite-transverse-shear-wave-v1",
            constrained_components="all grid nodes: vx=vz=0; nodes x<=0.175 m: all velocity components fixed",
            free_span_m=upper[0] - clamp_edge,
            initial_velocity_amplitude_m_per_s=0.6,
            initial_velocity_profile="vy=0.6*sin(pi*clip((X-0.175)/0.7,0,1)/2)",
            duration_basis="4*free_span/initial_shear_speed",
            comparison_scope="same geometry, material and initial/boundary data across grid and time-step runs; no inherited axial-wave accuracy numbers",
        )
    elif scene == "surface-load":
        definition["surface_pressure"] = {
            "peak_Pa": 1000.0,
            "rise_s": 0.3,
            "hold_s": 0.3,
            "fall_s": 0.3,
            "rest_s": 0.3,
            "area_measure": "reference top-face area",
            "area_m2": 0.36,
        }
    return (
        Case((state,), (material,), (config,), duration, definition, forces, observe),
        substeps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SCENES, default="wave")
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--dt-factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/visible-motion"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(
        args.case,
        grid=args.grid,
        ppc=args.ppc,
        frames=args.frames,
        dt_factor=args.dt_factor,
    )
    name = f"wave-grid{args.grid}" if args.case == "wave" else args.case
    if args.case == "wave" and args.dt_factor != 1:
        if args.grid != 48 or args.dt_factor != 0.5:
            parser.error(
                "the named wave-half-dt comparison uses grid=48 and dt-factor=0.5"
            )
        name = "wave-half-dt"
    print(
        f"{name}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps, t={case.duration:.9g}",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=args.output / name / "neo-hookean",
        label=f"{name} / neo-hookean",
        forces=case.forces,
        observer=case.observer,
        metadata=case.description,
        gif=args.gif,
        render_output=not args.no_render,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
