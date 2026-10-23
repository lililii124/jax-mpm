"""Sequential indentation, lateral translation and pull-off of an adhesive soft layer."""

import argparse
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, block, fixed_box, neo_hookean
from mpm_jax.adhesion import (
    DugdaleLaw,
    SpherePath,
    penalty_step_number,
    sphere_interaction,
)
from mpm_jax.contact_laws import boundary_reaction
from .common import run
from .scenarios import Case


SUITE_CASES = (
    "nonadhesive-pull-off",
    "adhesive-pull-off",
    "lateral-release",
    "three-sphere-array",
)


def make_case(*, grid=48, ppc=3, frames=50, suite_case=None, preset="standard"):
    if suite_case is not None and suite_case not in SUITE_CASES:
        raise ValueError("unknown adhesive suite case")
    if preset not in ("standard", "lift-hold"):
        raise ValueError("unknown adhesive preset")
    paired = suite_case in ("nonadhesive-pull-off", "adhesive-pull-off")
    if preset == "lift-hold" and not paired:
        raise ValueError("lift-hold requires a normal pull-off pair case")
    lift_hold = preset == "lift-hold"
    if grid < 8 or ppc < 1 or frames < 1:
        raise ValueError("grid >= 8, ppc >= 1 and frames >= 1 are required")
    young, poisson, density = 8000.0 if lift_hold else 12000.0, 0.25, 1000.0
    lower, upper = np.array((0.14, 0.15, 0.14)), np.array((0.86, 0.38, 0.86))
    counts = np.ceil((upper - lower) * grid * ppc).astype(int)
    spacings = (upper - lower) / counts
    surface_area = float(spacings[0] * spacings[2])
    surface_y = float(upper[1] - 0.5 * spacings[1])
    law = DugdaleLaw(
        work_of_adhesion=0.0 if suite_case == "nonadhesive-pull-off" else 72.0 if lift_hold else 12.0,
        interaction_range=0.025,
        penalty_stiffness=4.8e6 if lift_hold else 1.2e6,
    )
    material = neo_hookean(young, poisson)
    duration = 1.30 if lift_hold else 0.85
    wave_speed = float(np.sqrt((material.bulk + 4 * material.mu / 3) / density))
    bulk_dt = 0.15 / grid / wave_speed
    local_mass = density * np.prod(spacings)
    penalty_dt = 0.3 * np.sqrt(local_mass / (law.penalty_stiffness * surface_area))
    substeps = int(np.ceil(duration / (frames * min(bulk_dt, penalty_dt))))
    dt = duration / (frames * substeps)
    config = Config(
        dim=3,
        n_grid=grid,
        dt=dt,
        material="neo-hookean",
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=(fixed_box((0.0, 0.0, 0.0), (1.0, lower[1], 1.0)),),
    )
    state = block(
        config, lower, upper, particles_per_cell=ppc, density=density, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    surface_indices = jnp.asarray(
        np.flatnonzero(reference[:, 1] == reference[:, 1].max())
    )
    radius, clearance, indentation, lift = 0.16, 0.045, 0.035, 0.25 if lift_hold else 0.10
    high = surface_y + radius + clearance
    low = surface_y + radius - indentation
    paths = []
    if lift_hold:
        comparison_height = surface_y + radius + 0.055
        paths.append(
            SpherePath(
                radius=radius,
                times=(0.0, 0.08, 0.28, 0.42, 0.72, 0.94, 1.12, duration),
                centers=(
                    (0.5, high, 0.5),
                    (0.5, high, 0.5),
                    (0.5, low, 0.5),
                    (0.5, low, 0.5),
                    (0.5, comparison_height, 0.5),
                    (0.5, comparison_height, 0.5),
                    (0.5, high + lift, 0.5),
                    (0.5, high + lift, 0.5),
                ),
            )
        )
    elif suite_case in (None, "three-sphere-array"):
        for x, z, delay, shift in (
            (0.32, 0.28, 0.0, -0.025),
            (0.68, 0.28, 0.10, 0.025),
            (0.50, 0.70, 0.20, 0.065),
        ):
            paths.append(
                SpherePath(
                    radius=radius,
                    times=(
                        0.0,
                        0.06 + delay,
                        0.25 + delay,
                        0.42 + delay,
                        0.64 + delay,
                        duration,
                    ),
                    centers=(
                        (x, high, z),
                        (x, high, z),
                        (x, low, z),
                        (x + shift, low, z),
                        (x + shift, high + lift, z),
                        (x + shift, high + lift, z),
                    ),
                )
            )
    else:
        initial_x, swept_x, lifted_x = (
            (0.38, 0.55, 0.65) if suite_case == "lateral-release" else (0.5, 0.5, 0.5)
        )
        paths.append(
            SpherePath(
                radius=radius,
                times=(0.0, 0.06, 0.25, 0.42, 0.64, duration),
                centers=(
                    (initial_x, high, 0.5),
                    (initial_x, high, 0.5),
                    (initial_x, low, 0.5),
                    (swept_x, low, 0.5),
                    (lifted_x, high + lift, 0.5),
                    (lifted_x, high + lift, 0.5),
                ),
            )
        )
    reference_surface = jnp.asarray(reference)[surface_indices]

    def interactions(current, time):
        points = current.x[surface_indices]
        return tuple(
            sphere_interaction(
                points, surface_area, path.motion(time)[0], path.radius, law
            )
            for path in paths
        )

    def forces(states, time):
        loads = interactions(states[0], time)
        surface_forces = sum(load.forces for load in loads)
        return (jnp.zeros_like(states[0].x).at[surface_indices].add(surface_forces),)

    def observe(states, time):
        current = states[0]
        points, velocity = current.x[surface_indices], current.v[surface_indices]
        loads = interactions(current, time)
        force = forces(states, time)[0]
        base = -boundary_reaction(
            current, material, config, boundary_index=0, force=force
        )
        reaction = sum(load.reaction for load in loads)
        total_moment = jnp.sum(jnp.cross(current.x, force), axis=0)
        row = {
            "force_sample_time_s": float(time),
            "base_boundary_sample_time_s": float(time + config.dt),
            "surface_reference_area_m2": float(len(surface_indices) * surface_area),
            "contact_potential_J": float(sum(load.potential for load in loads)),
            "penalty_potential_J": float(sum(load.penalty_potential for load in loads)),
            "adhesive_potential_J": float(
                sum(load.adhesive_potential for load in loads)
            ),
            "surface_displacement_y_min_m": float(jnp.min(points[:, 1] - surface_y)),
            "surface_displacement_y_max_m": float(jnp.max(points[:, 1] - surface_y)),
            "force_balance_error_N": float(
                jnp.linalg.norm(jnp.sum(force, axis=0) + reaction)
            ),
            "particle_contact_power_W": float(
                jnp.sum(sum(load.forces for load in loads) * velocity)
            ),
        }
        actuator_power = 0.0
        all_x = np.asarray(current.x)
        for index, (path, load) in enumerate(zip(paths, loads)):
            center, center_velocity = path.motion(time)
            undeformed = sphere_interaction(
                reference_surface, surface_area, center, path.radius, law
            )
            prefix = f"sphere_{index}_"
            for axis, name in enumerate("xyz"):
                row[prefix + f"center_{name}_m"] = float(center[axis])
                row[prefix + f"velocity_{name}_m_per_s"] = float(center_velocity[axis])
                row[prefix + f"reaction_{name}_N"] = float(load.reaction[axis])
                row[prefix + f"reaction_moment_{name}_Nm"] = float(
                    load.reaction_moment[axis]
                )
                row[prefix + f"undeformed_surface_reaction_{name}_N"] = float(
                    undeformed.reaction[axis]
                )
            row.update(
                {
                    prefix
                    + "contact_reference_area_m2": float(load.contact_reference_area),
                    prefix
                    + "adhesive_reference_area_m2": float(load.adhesive_reference_area),
                    prefix + "compressed_points": int(load.compressed_points),
                    prefix + "adhesive_points": int(load.adhesive_points),
                    prefix + "min_gap_m": float(jnp.min(load.gap)),
                    prefix
                    + "max_penetration_m": float(jnp.maximum(-jnp.min(load.gap), 0.0)),
                    prefix
                    + "max_bulk_penetration_m": float(
                        max(
                            0.0,
                            path.radius
                            - np.linalg.norm(all_x - np.asarray(center), axis=1).min(),
                        )
                    ),
                    prefix + "potential_J": float(load.potential),
                    prefix
                    + "deformation_feedback_force_difference_N": float(
                        jnp.linalg.norm(load.reaction - undeformed.reaction)
                    ),
                    prefix
                    + "actuator_power_W": float(
                        -jnp.dot(load.reaction, center_velocity)
                    ),
                }
            )
            actuator_power += row[prefix + "actuator_power_W"]
            total_moment += jnp.cross(center, load.reaction) + load.reaction_moment
        for axis, name in enumerate("xyz"):
            row[f"base_force_on_body_{name}_N"] = float(base[axis])
            row[f"sphere_total_reaction_{name}_N"] = float(reaction[axis])
        row["actuator_power_W"] = actuator_power
        row["moment_balance_error_Nm"] = float(jnp.linalg.norm(total_moment))
        return row

    definition = {
        "purpose": suite_case or "dugdale-array",
        "contact": "Dugdale finite-range nominal adhesion with penalty repulsion; explicit MPM coupling",
        "theory_scope": "MD-type interface law; no analytical Maugis sphere solution or experimental calibration",
        "sources": [
            "https://doi.org/10.1016/0021-9797(92)90285-T",
            "https://bdml.stanford.edu/twiki/pub/Rise/AdhesionModels/JohnsonAdhesionModel.pdf",
            "https://doi.org/10.1016/j.triboint.2017.04.001",
        ],
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "young_modulus_Pa": young,
        "poisson_ratio": poisson,
        "density_kg_per_m3": density,
        "particles_per_cell_per_axis": ppc,
        "particle_counts_per_axis": counts.tolist(),
        "particle_spacings_m": spacings.tolist(),
        "surface_particles": int(len(surface_indices)),
        "contact_sampling_plane_y_m": surface_y,
        "sampling_plane_offset_below_nominal_top_m": float(0.5 * spacings[1]),
        "surface_quadrature": "top-layer particle centers, advected by MPM; fixed reference area weights; no F-dependent surface offset",
        "reference_area_per_surface_point_m2": surface_area,
        "adhesion": asdict(law),
        "cohesive_nominal_stress_Pa": law.cohesive_stress,
        "adhesive_range_per_surface_spacing": float(
            law.interaction_range / max(spacings[0], spacings[2])
        ),
        "penalty_step_number_max": float(
            jnp.max(
                penalty_step_number(state.mass[surface_indices], surface_area, dt, law)
            )
        ),
        "penalty_stability_indicator": "k*A0*dt^2/m <= 0.09; local contact indicator, not a nonlinear global stability proof",
        "base": "all velocity components fixed at grid nodes y <= 0.15 m",
        "other_faces": "traction-free",
        "gravity": "zero",
        "tangential_contact": "frictionless",
        "rigid_spheres": [asdict(path) for path in paths],
        "rigid_sphere_separation_lower_bound_m": 0.36 if len(paths) > 1 else None,
        "rigid_sphere_surface_clearance_lower_bound_m": 0.04
        if len(paths) > 1
        else None,
        "rigid_sphere_motion": "sum of cosine-interpolated waypoint increments; exact prescribed positions at output times",
        "reaction_sampling": "sphere forces at output time; base next-step grid impulse includes the same external contact force",
        "deformation_feedback_reference": "same law and sphere position evaluated on the initial undeformed surface; diagnostic only, never applied to particles",
        "energy_interpretation": "reversible surface potential; no irreversible bond damage or claimed fracture dissipation",
        "resolution_interpretation": "finite interaction range deliberately resolved; coarse runs with range/spacing < 3 are smoke tests only",
        "geometry_scale": 1.0,
    }
    if suite_case is not None:
        definition.update(
            series="adhesion-suite-v1",
            suite_case=suite_case,
            comparison_pair="single-sphere-normal-pull-off-v1" if paired else None,
            comparison_role=(
                "nonadhesive" if suite_case == "nonadhesive-pull-off" else "adhesive"
            )
            if paired
            else None,
            comparison_variable="work_of_adhesion" if paired else None,
            comparison_scope="paired runs share initial state, grid, time step, stiffness, sphere geometry and path; only work of adhesion changes"
            if paired
            else None,
            loading_stages_s={
                "initial_hold_end": 0.06,
                "indentation_end": 0.25,
                "hold_or_lateral_sweep_end": 0.42,
                "pull_off_end": 0.64,
                "final_hold_end": duration,
            }
            if len(paths) == 1
            else None,
            lateral_loading="lateral sweep and oblique pull-off; radial normal forces only, no tangential friction or irreversible bond damage"
            if suite_case == "lateral-release"
            else None,
            explicit_coupling="current MPM surface positions determine gaps and forces each step; their forces update the deformable body on the next explicit step",
        )
    if lift_hold:
        definition.update(
            preset="lift-hold",
            comparison_pair="single-sphere-lift-hold-v1",
            loading_stages_s={
                "initial_hold_end": 0.08,
                "indentation_end": 0.28,
                "hold_or_lateral_sweep_end": 0.42,
                "comparison_lift_end": 0.72,
                "comparison_hold_end": 0.94,
                "pull_off_end": 1.12,
                "final_hold_end": duration,
            },
            comparison_hold_bottom_clearance_m=0.055,
            comparison_targets={
                "minimum_contiguous_time_s": 0.10,
                "minimum_surface_uplift_difference_m": 0.025,
                "baseline": "zero interacting points and zero sphere reaction",
                "adhesive": "tensile sphere reaction and higher maximum surface y displacement at the same saved times",
                "status": "prospective checks; not measured results",
            },
        )
    return (
        Case((state,), (material,), (config,), duration, definition, forces, observe),
        substeps,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/adhesion"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.grid < 8 or args.ppc < 1 or args.frames < 1:
        parser.error("grid >= 8, ppc >= 1 and frames >= 1 are required")
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(grid=args.grid, ppc=args.ppc, frames=args.frames)
    print(
        f"dugdale-array: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=args.output / "dugdale-array",
        label="dugdale-array",
        forces=case.forces,
        observer=case.observer,
        metadata=case.description,
        gif=args.gif,
        render_output=not args.no_render,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
