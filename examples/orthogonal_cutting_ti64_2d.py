"""Independent 2D plane-strain Ti-6Al-4V orthogonal-cutting prescreen.

The grid and particle kinematics are x-y only. Johnson-Cook history and
deformation remain full 3x3 tensors with Fzz=1, so the constitutive update is
the 3D plane-strain path. Forces/energy are reported per unit out-of-plane
thickness. This remains an exploratory model; the sharp analytical wedge,
adiabatic heat update and no chip deletion differ from the cited FE study.
"""

import argparse
import json
import hashlib
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingWedgeTool, fixed_box
from mpm_jax.core import particle_to_grid
from mpm_jax.initialization import block
from mpm_jax.particle_contact import contact_failed_particles_with_intact, project_particles_to_wedge
from mpm_jax.particle_wedge_grid_contact import particle_wedge_step_adapter
from mpm_jax.failed_wedge_swept_contact import collide_failed_particles_with_translating_wedge
from .cutting_runtime import run
from .cutting_presets import cutting_material, FLOW_PARAMETERS


DOMAIN = 0.32e-3
WORK_LOWER = (0.032e-3, 0.064e-3)
WORK_UPPER = (0.192e-3, 0.192e-3)
CUT_DEPTH = 0.020e-3
TOOL_START_X = 0.031e-3
TOOL_SPEED = 3.0
SOURCE_FILES = (
    "examples/orthogonal_cutting_ti64_2d.py", "examples/cutting_runtime.py",
    "examples/cutting_presets.py",
    "mpm_jax/__init__.py", "mpm_jax/materials/__init__.py",
    "mpm_jax/convex_penetration.py", "mpm_jax/particle_fields.py",
    "mpm_jax/core.py", "mpm_jax/boundaries.py", "mpm_jax/materials/johnson_cook_displacement.py",
    "mpm_jax/materials/history.py", "mpm_jax/materials/jc_clipped_rate_increment.py",
    "mpm_jax/particle_wedge_grid_contact.py", "mpm_jax/failed_wedge_swept_contact.py",
    "mpm_jax/initialization.py", "mpm_jax/materials/plane_strain_kinematics.py",
)


def ti64_reference(*, damage=False, dx=None, failure_displacement_cells=2.0,
                   fast_return=False):
    """Classical Johnson flow-stress set; not the article's Seo Table 1 set."""
    return cutting_material('johnson',damage=damage,dx=dx,
                            failure_displacement_cells=failure_displacement_cells,
                            fast_return=fast_return)


def setup(*, grid=128, ppc=1, cfl=0.08, friction=0.3, cut_depth=CUT_DEPTH,
          tip_radius=0.0, tip_arc_segments=8, damage=False,
          failure_displacement_cells=2.0, fast_return=False, translation_path=None,
          material_preset='johnson'):
    if type(grid) is not int or grid < 32:
        raise ValueError("grid must be an integer >= 32")
    if type(ppc) is not int or ppc < 1:
        raise ValueError("ppc must be a positive integer")
    if not 0 < cfl <= 0.1 or friction < 0:
        raise ValueError("CFL must be in (0,0.1] and friction nonnegative")
    if not np.isfinite(cut_depth) or not 0 < cut_depth < WORK_UPPER[1] - WORK_LOWER[1]:
        raise ValueError("cut_depth must be positive and smaller than the workpiece height")
    if not np.isfinite(tip_radius) or tip_radius < 0:
        raise ValueError("tip_radius must be finite and nonnegative")
    if type(tip_arc_segments) is not int or tip_arc_segments < 2:
        raise ValueError("tip_arc_segments must be an integer >= 2")
    dx = DOMAIN / grid
    material = cutting_material(material_preset, damage=damage, dx=dx,
                              failure_displacement_cells=failure_displacement_cells,
                              fast_return=fast_return)
    wave_speed = float(np.sqrt((material.bulk + 4 * material.mu / 3) / material.density))
    dt = cfl * dx / wave_speed
    top = WORK_UPPER[1]
    tool = MovingWedgeTool(
        tip=(TOOL_START_X, top - cut_depth), speed=TOOL_SPEED,
        rake_angle_deg=10.0, clearance_angle_deg=6.0,
        length=0.256e-3, friction=friction, tip_radius=tip_radius,
        tip_arc_segments=tip_arc_segments, translation_path=translation_path,
    )
    config = Config(
        dim=2, n_grid=grid, domain_size=DOMAIN, dt=dt,
        material="johnson-cook-displacement", gravity=(0.0, 0.0), boundary_cells=0,
        boundaries=(fixed_box((0.0, 0.0), (DOMAIN, WORK_LOWER[1] + dx)), tool),
        failure_mode="ballistic" if damage else "retain",
    )
    state = block(config, WORK_LOWER, WORK_UPPER, particles_per_cell=ppc,
                  density=material.density, material=material, dtype=jnp.float64,
                  temperature=material.reference_temperature)
    point_count = int(len(state.x))
    definition = {
        "case": "orthogonal-cutting-ti64-jc-plane-strain-2d",
        "status": "exploratory; no physical validation or grid convergence yet",
        "units": "SI; force and energy are per unit out-of-plane thickness",
        "reference": "Numerical Analysis of Serrated Chip Formation Mechanism with Johnson-Cook Parameters in Micro-Cutting of Ti6Al4V, Metals 11 (2021) 102, doi:10.3390/met11010102",
            "model_scope": "2D x-y grid; full 3x3 JC material history; plane strain enforced by Fzz=1 and zero out-of-plane shear. Prescribed analytical wedge with optional circular fillet. Adiabatic plastic-work heating; no conduction.",
        "material_parameters_SI": {
            "preset": material_preset,
            "E_Pa_at_50C": 109e9, "nu": 0.34, "rho_kg_m3": 4430.0,
            "A_Pa": material.A, "B_Pa": material.B, "n": material.n, "C": material.C,
            "m": material.m, "reference_rate_s-1": 1.0,
            "reference_temperature_K": 298.15, "melt_temperature_K": 1878.15,
            "heat_capacity_J_kgK_at_20C": 611.0, "Taylor_Quinney": 0.9,
            "thermal_conductivity_W_mK_at_20C": 6.8,
            "property_temperature_dependence_in_reference": True,
            "property_temperature_dependence_implemented": False,
        },
        "failure_model": {
            "enabled": damage,
            "JC_D1_D2_D3_D4_D5": [-0.09, 0.25, -0.5, 0.014, 3.87] if damage else None,
            "triaxiality_convention": "tension-positive eta=-pressure/q; pressure=K*(1/J-1) is compression-positive",
            "initiation": "cumulative JC damage index reaches 1",
            "post_initiation_softening": "linear in equivalent plastic displacement L*delta_PEEQ; exploratory uf, not calibrated Gf",
            "negative_fracture_strain_validity": "invalid before initiation when dep>0; after damage initiation reaches 1, post-initiation displacement evolution proceeds from dep and does not require epsilon_f>0",
            "characteristic_length_m": dx if damage else None,
            "failure_displacement_m": material.failure_displacement if damage else None,
            "failed_particle_handling": "ballistic: retained in state/archive, excluded from active P2G; no fragment-fragment or fragment-workpiece collision model",
            "limitations": "not energy-closed fracture; uf is an explicit exploratory input; no remapping of crack-side velocity fields",
        },
        "constitutive_solver": {
            "johnson_cook_return": "safeguarded Newton with reference bisection fallback" if fast_return else "64-iteration bisection reference",
            "fast_return_enabled": fast_return,
            "active_point_compaction": {
                "enabled": fast_return,
                "capacity": 2048,
                "overflow_behavior": "fallback to dense return mapping",
                "physics_equivalence": "bitwise frame-300 100-step comparison against dense return mapping",
            },
        },
        "geometry": {
            "domain_m": DOMAIN, "workpiece_lower_m": WORK_LOWER,
            "workpiece_upper_m": WORK_UPPER,
            "uncut_chip_thickness_m": cut_depth,
            "tool_tip_initial_m": [TOOL_START_X, top - cut_depth],
            "initial_tool_tip_gap_to_workpiece_side_m": TOOL_START_X - WORK_LOWER[0],
            "rake_angle_deg": 10.0, "clearance_angle_deg": 6.0,
            "clearance_convention": "positive angle: clearance face rises behind virtual tip; direction=(-cos(alpha),+sin(alpha))",
            "virtual_tip_initial_m": [TOOL_START_X, top - cut_depth],
            "nominal_cut_depth_definition": "stock top minus virtual sharp-face intersection y; legacy tool_tip field is polygon vertex zero (upper rake tangent when rounded)",
            "actual_polygon_nose_min_y_m": float(np.min(np.asarray(tool._polygon(0., jnp.float64))[:, 1])),
            "effective_polygon_nose_cut_depth_m": float(top - np.min(np.asarray(tool._polygon(0., jnp.float64))[:, 1])),
            "edge_radius_in_reference_m": 1e-6,
            "edge_radius_implemented": tip_radius > 0,
            "tool_tip_radius_m": tip_radius,
            "tool_tip_arc_segments": tip_arc_segments if tip_radius > 0 else 0,
            "tip_definition": "virtual intersection of rake and clearance faces; rounded fillet tangent to both faces" if tip_radius > 0 else "sharp face intersection",
            "out_of_plane_thickness": "unit thickness convention",
        },
        "tool": {
            "type": "prescribed rigid analytical wedge with optional circular tip fillet",
            "speed_m_s": TOOL_SPEED, "friction_coefficient": friction,
            "contact": "occupied grid-node unilateral velocity projection with Coulomb tangential slip",
            "reference_microcut_friction_coefficient": 0.3,
        },
        "numerics": {
            "grid_cells_per_axis": grid, "dx_m": dx,
            "particles_per_cell_per_axis": ppc, "particle_count": point_count,
            "dt_s": dt, "initial_wave_speed_m_s": wave_speed,
            "acoustic_CFL": cfl, "grid_shift_used": False,
            "constitutive_state_shape": [point_count, 3, 3],
            "grid_velocity_dimension": 2,
        },
        "qualification": {
            "uniform_plane_strain_path_checked_against_3d": False,
            "physical_validation_completed": False,
            "resolution_convergence_completed": False,
            "experimental_force_validation_completed": False,
        },
    }

    @jax.jit
    def predicted_tool_impulse(current, time):
        gm, gp = particle_to_grid(current, material, config)
        occupied = gm > 0
        velocity = gp / jnp.where(occupied, gm, 1.0)[:, None]
        velocity = jnp.where(occupied[:, None], velocity, 0.0)
        nodes = jnp.arange((grid + 1) ** 2)
        stride = (grid + 1) ** jnp.arange(1, -1, -1)
        coords = ((nodes[:, None] // stride) % (grid + 1)) * dx
        before_tool = velocity
        for boundary in config.boundaries[:-1]:
            before_tool = boundary.apply(before_tool, coords, time)
        after_tool, impulse, inside = tool._project(before_tool, coords, time, gm)
        contact_nodes = jnp.sum((gm > 0) & inside)
        return impulse, contact_nodes

    return state, material, config, tool, definition, predicted_tool_impulse


def main(*, runner=run, translation_path=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--material-preset", choices=tuple(FLOW_PARAMETERS), default='johnson',
                        help='flow-stress parameter group; shared thermal/failure assumptions')
    parser.add_argument("--ppc", type=int, default=1)
    parser.add_argument("--cfl", type=float, default=0.08)
    parser.add_argument("--friction", type=float, default=0.3)
    parser.add_argument("--cut-depth", type=float, default=CUT_DEPTH,
                        help="uncut chip thickness in metres")
    parser.add_argument("--tool-tip-radius", type=float, default=0.0,
                        help="circular fillet radius at the virtual sharp tip, in metres")
    parser.add_argument("--tool-tip-arc-segments", type=int, default=8,
                        help="polygon facets used to approximate the circular fillet")
    parser.add_argument("--jc-damage", action="store_true",
                        help="enable Table 4 JC initiation and exploratory displacement softening")
    parser.add_argument("--failure-displacement-cells", type=float, default=2.0,
                        help="post-initiation equivalent plastic displacement as a multiple of dx")
    parser.add_argument("--travel", type=float, default=60e-6,
                        help="planned tool travel in metres; default is 60 micrometres")
    parser.add_argument("--steps", type=int, default=0,
                        help="override physical-step count for a benchmark")
    parser.add_argument("--substeps", type=int, default=500)
    parser.add_argument("--stop-after-frames", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/orthogonal-cutting-ti64-2d"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--particle-contact", action="store_true",
                        help="use an isolated after-G2P particle-to-wedge projection experiment")
    parser.add_argument("--hybrid-contact", action="store_true",
                        help="retain nodal wedge contact and project only residual particle penetration")
    parser.add_argument("--failed-particle-contact", action="store_true",
                        help="contact failed material points against intact points and the tool with a bounded spatial bucket")
    parser.add_argument("--particle-contact-warm-start", action="store_true")
    parser.add_argument("--particle-grid-contact", action="store_true",
                        help="replace nodal wedge projection with particle-interpolated PGS grid impulse constraints")
    parser.add_argument("--failed-wedge-swept-contact", action="store_true",
                        help="continuously collide step-start ballistic failed points with the translating wedge")
    parser.add_argument("--particle-contact-capacity", type=int, default=1024)
    parser.add_argument("--particle-contact-sweeps", type=int, default=12)
    parser.add_argument("--particle-contact-max-sweeps", type=int, default=12,
                        help="adaptive PGS hard cap; may exceed initial sweeps")
    parser.add_argument("--particle-contact-activation-cells", type=float, default=0.5)
    parser.add_argument("--particle-contact-gap-tolerance-cells", type=float, default=1e-3)
    parser.add_argument("--fast-jc-return", action="store_true",
                        help="use safeguarded Newton JC return with 64-step bisection fallback")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    state, material, config, tool, definition, tool_impulse = setup(
        grid=args.grid, ppc=args.ppc, cfl=args.cfl, friction=args.friction,
        cut_depth=args.cut_depth, tip_radius=args.tool_tip_radius,
        tip_arc_segments=args.tool_tip_arc_segments, damage=args.jc_damage,
        failure_displacement_cells=args.failure_displacement_cells,
        fast_return=args.fast_jc_return, material_preset=args.material_preset,
        translation_path=translation_path)
    post_step = None
    step_adapter = None
    if args.particle_contact_warm_start and not args.particle_grid_contact:
        parser.error("--particle-contact-warm-start requires --particle-grid-contact")
    if args.particle_grid_contact:
        if args.particle_contact or args.hybrid_contact or args.failed_particle_contact:
            parser.error("--particle-grid-contact cannot be combined with other particle contact modes")
        if args.failed_wedge_swept_contact and not args.jc_damage:
            parser.error("--failed-wedge-swept-contact requires --jc-damage")
        if (args.particle_contact_capacity < 1 or args.particle_contact_sweeps < 1
                or args.particle_contact_max_sweeps < args.particle_contact_sweeps
                or args.particle_contact_activation_cells < 0
                or args.particle_contact_gap_tolerance_cells <= 0):
            parser.error("capacity/sweeps/gap tolerance must be positive and activation cells nonnegative")
        if not args.fast_jc_return:
            parser.error("--particle-grid-contact window requires --fast-jc-return for the validated fast JC branch")
        def configured_particle_wedge_step(state, material, config, force=None, cache=None):
            cache_arguments = {} if cache is None else {"cache": cache}
            result = particle_wedge_step_adapter(state, material, config, force,
                capacity=args.particle_contact_capacity,
                sweeps=args.particle_contact_sweeps,
                max_sweeps=args.particle_contact_max_sweeps,
                activation_cells=args.particle_contact_activation_cells,
                gap_tolerance_cells=args.particle_contact_gap_tolerance_cells,
                **cache_arguments)
            updated, impulse, work, values = result[:4]
            next_cache = None if cache is None else result[4]
            if args.failed_wedge_swept_contact:
                updated, swept = collide_failed_particles_with_translating_wedge(
                    state, updated, config, tool)
                impulse = impulse + swept.particle_impulse
                work = work + swept.tool_work
                extra = jnp.stack((swept.collisions.astype(state.x.dtype),
                    swept.particle_impulse[0], swept.particle_impulse[1],
                    swept.tool_work, swept.delta_kinetic, swept.collision_loss,
                    swept.max_penetration,
                    swept.max_all_particle_penetration,
                    (swept.invalid_start | swept.invalid_energy
                     | swept.all_particle_guard_failed
                     | (swept.max_penetration > 1e-9 * config.dx)
                     | (swept.post_reconstruction_max_penetration > 1e-9 * config.dx)
                     | (swept.post_reconstruction_max_all_particle_penetration > 0.5 * config.dx)).astype(state.x.dtype),
                    swept.post_reconstruction_max_penetration,
                    swept.post_reconstruction_max_all_particle_penetration,
                    swept.persistent_touch_points.astype(state.x.dtype),
                    swept.persistent_touch_max_correction,
                    swept.persistent_touch_max_rejected_correction))
            else:
                extra = jnp.zeros((14,), dtype=state.x.dtype)
            result = (updated, impulse, work, jnp.concatenate((values, extra)))
            if cache is not None:
                next_cache = jnp.where(updated.history.failed[:, None], 0.0, next_cache)
                accepted = updated.valid & jnp.all(updated.history.valid)
                next_cache = jnp.where(accepted, next_cache, cache)
                return (*result, next_cache)
            return result
        step_adapter = configured_particle_wedge_step
        definition["case"] += "-particle-grid-pgs"
        definition["tool"]["contact"] = (
            "particle-interpolated unilateral velocity constraints solved by accumulated normal/tangent PGS "
            "in the nodal mass metric; replaces nodal wedge projection and uses ordinary G2P for x/F/history"
        )
        definition["tool"]["particle_grid_contact"] = {
            "active_particle_capacity": args.particle_contact_capacity,
            "pgs_sweeps": args.particle_contact_sweeps,
            "pgs_adaptive": args.particle_contact_max_sweeps > args.particle_contact_sweeps,
            "pgs_max_sweeps": args.particle_contact_max_sweeps,
            "activation_band_cells": args.particle_contact_activation_cells,
            "post_G2P_gap_tolerance_cells": args.particle_contact_gap_tolerance_cells,
            "fixed_dofs": "excluded from inverse mass; support reaction reported separately",
            "guard": "capacity, normal complementarity, friction projection residual and post-G2P end-gap",
            "units": "2D plane strain; impulse N s/m and work J/m",
        }
        if args.particle_contact_warm_start:
            definition["tool"]["particle_grid_contact"]["warm_start"] = {
                "cache": "previous accepted contact multipliers keyed by particle ID",
                "reset": "zero at the beginning of every output batch",
                "acceptance": "unchanged PGS, material and swept-contact guards",
            }
        if args.failed_wedge_swept_contact:
            definition["tool"]["failed_ballistic_wedge_contact"] = {
                "method": "continuous relative-motion sweep through every convex polygon face over one physical substep; first entry face gets inelastic normal impulse and Coulomb-limited tangential impulse",
                "scope": "only particles whose history.failed was true at the step start; no failed-failed or failed-intact contact",
                "state_update": "position and velocity only; F and stress/plastic/thermal history remain frozen",
                "accounting": "particle impulse, tool work, particle kinetic-energy change, and residual collision loss recorded separately; tool impulse/work included in total contact ledger",
                "guard": "failed point starting inside wedge or negative collision loss beyond roundoff invalidates the state; failed-point residual penetration is guarded at 1e-9 dx and all-particle rounded-wedge penetration retains the 0.5 dx hard guard",
                "start_surface_classification_tolerance": "same 1e-9*dx admissibility as failed-point end-gap residual guard; no position repair on migration",
                "units": "2D plane strain; impulse N s/m and energy J/m",
            }
            definition["tool"]["failed_ballistic_wedge_contact"]["persistent_touch_roundoff"] = {
                "enabled": True,
                "scope": "step-start failed points only; per-particle CCD hit mask must be false",
                "eligibility": "within existing 1e-9*dx signed-gap tolerance; step-start failed, per-particle CCD hit mask false, and start/end coordinates remain on the same finite face without a corner crossing",
                "operation": "reconstruct from actual MovingWedgeTool._polygon(t+dt) in face-local coordinates: g1=g0+vn*dt and s1=s0+vt*dt; preserve the signed normal gap including opening and closing motion",
                "maximum_position_correction_m": "must not exceed the existing 1e-9*dx; larger correction is rejected and original CCD result is retained",
                "unchanged": ["v", "mass", "F", "constitutive history", "impulse", "tool work", "kinetic-energy ledger"],
                "guard_order": "original CCD collision/impulse and pre-reconstruction gaps are preserved; original failed-gap and all-particle guards are also checked on the reconstructed endpoint",
                "metrics": "separate original CCD max gaps from post-reconstruction actual endpoint gaps",
                "qualification": "paired identical-state 2,000-substep comparison; see persistent-touch NPZ audit manifest",
            }
    if args.particle_contact or args.hybrid_contact or args.failed_particle_contact:
        if args.failed_particle_contact and not args.jc_damage:
            parser.error("--failed-particle-contact requires --jc-damage")
        if args.particle_contact:
            # Isolated particle-only experiment: remove nodal contact.
            config = replace(config, boundaries=tuple(b for b in config.boundaries if b is not tool))
            definition["case"] += "-particle-projection"
            definition["tool"]["contact"] = (
                "after-G2P per-particle unilateral projection with Coulomb slip; "
                "nodal tool contact disabled"
            )
        elif args.hybrid_contact:
            definition["case"] += "-hybrid-residual-projection"
            definition["tool"]["contact"] = (
                "nodal unilateral velocity projection first; after-G2P particle projection "
                "acts only on remaining geometric penetration and approaching relative velocity"
            )
        definition["tool"]["particle_projection"] = {
            "separation_m": 1e-9,
            "projection_updates": ["particle position", "particle velocity"],
            "projection_does_not_update": ["deformation gradient F", "constitutive history"],
            "frictional_heat_added": False,
            "impulse_units": "N s per unit out-of-plane thickness",
            "work_units": "J per unit out-of-plane thickness",
            "qualification": "exploratory contact-method comparison; not energy-closed",
            "mode": "particle-only" if args.particle_contact else "nodal-plus-residual-particle-correction",
            "nodal_impulse": "integrated from mass*(v_after-v_before) at each actual grid-boundary projection step",
        }
        if args.failed_particle_contact:
            definition["case"] += "-failed-particle-contact"
            definition["tool"]["failed_particle_contact"] = {
                "method": "nearest failed/intact particle sphere contact; mass-weighted pair position correction and equal/opposite Coulomb impulse; failed particles also receive finite-wedge projection",
                "bucket_capacity": 32, "max_search_cells": 2,
                "overflow_behavior": "State.valid=false; stop and preserve last valid checkpoint",
                "F_and_history_updated_after_position_projection": False,
                "scope": "exploratory local contact, not resolved crack-side MPM or fully energy-closed fracture",
            }
        definition["geometry"]["initial_tool_tip_gap_note"] = (
            "The tip is 1 micrometre left of the workpiece side; geometric separation is +1 micrometre. "
            "The signed x offset is tool_tip_x - workpiece_lower_x = -1 micrometre."
        )

        def post_step(states, end_time):
            current = states[0]
            fragment_metrics = jnp.zeros((6,), dtype=current.x.dtype)
            particle_mask = None
            if args.failed_particle_contact:
                current, fragment_metrics = contact_failed_particles_with_intact(
                    current, args.friction, config.dx, bucket_capacity=32,
                    max_search_cells=2, grid_cells=args.grid, separation=0.0)
                current = current._replace(valid=current.valid & (fragment_metrics[4] == 0))
                particle_mask = current.history.failed
            x, v, metrics = project_particles_to_wedge(
                current.x, current.v, current.mass,
                tool._polygon(end_time, jnp.float64),
                tool.velocity(end_time, current.v.dtype),
                args.friction,
                1e-9,
                particle_mask,
            )
            if args.failed_particle_contact:
                combined = jnp.zeros((13,), dtype=current.x.dtype)
                combined = combined.at[:7].set(metrics)
                combined = combined.at[7].set(fragment_metrics[0])
                combined = combined.at[8].set(fragment_metrics[1])
                combined = combined.at[9].set(fragment_metrics[2])
                combined = combined.at[10].set(fragment_metrics[3])
                combined = combined.at[11].set(fragment_metrics[4])
                combined = combined.at[12].set(fragment_metrics[5])
                metrics = combined
            else:
                metrics = jnp.concatenate((metrics, jnp.zeros((6,), dtype=current.x.dtype)))
            return (current._replace(x=x, v=v),), metrics
    if args.substeps < 1:
        parser.error("substeps must be positive")
    steps = args.steps or int(np.ceil((args.travel / TOOL_SPEED / config.dt) / args.substeps) * args.substeps)
    if steps < 1 or steps % args.substeps:
        parser.error("steps must be a positive multiple of substeps")
    frames = steps // args.substeps
    if args.stop_after_frames is not None and not 0 < args.stop_after_frames <= frames:
        parser.error("stop-after-frames must lie within the full protocol")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    definition['source_sha256']={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                                 for name in SOURCE_FILES}
    if args.particle_grid_contact and args.failed_wedge_swept_contact:
        definition["tool"]["failed_ballistic_wedge_contact"]["polygon_gap_min_sqrt_reduce_sha256"] = (
            definition['source_sha256']['mpm_jax/failed_wedge_swept_contact.py'])
        definition["jc_return_source_sha256"] = (
            definition['source_sha256']['mpm_jax/materials/jc_clipped_rate_increment.py'])
    definition["duration_s"] = steps * config.dt
    definition["planned_steps"] = steps
    definition["substeps_per_saved_frame"] = args.substeps
    definition["saved_frames"] = frames
    definition["planned_tool_travel_m"] = TOOL_SPEED * steps * config.dt
    definition["qualification"]["uniform_plane_strain_path_checked_against_3d"] = True
    if runner is run:
        (output / "case.json").write_text(json.dumps(definition, indent=2) + "\n", encoding="utf-8")

    def observe(states, time):
        current = states[0]
        x = np.asarray(current.x)
        history = current.history
        polygon = np.asarray(tool._polygon(time, jnp.float64))
        rel = x[:, None, :] - polygon[None, :, :]
        edge = np.roll(polygon, -1, axis=0) - polygon
        cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
        signed_distance = cross / np.linalg.norm(edge, axis=1)[None, :]
        inside = np.all(signed_distance >= -1e-14, axis=1)
        depth = np.where(inside, np.min(np.maximum(signed_distance, 0.0), axis=1), 0.0)
        frame = int(round(time / (args.substeps * config.dt)))
        np.savez_compressed(
            output / f"fields_{frame:05d}.npz", particle_id=np.arange(len(x), dtype=np.int32), x=x,
            temperature_K=np.asarray(history.temperature),
            plastic_strain=np.asarray(history.plastic_strain),
            plastic_shear_strain_xy=np.asarray(history.plastic_tensor)[:, 0, 1],
            plastic_rate=np.asarray(history.plastic_rate), damage=np.asarray(history.damage),
            damage_softening=np.asarray(history.damage_softening),
            damage_failure_displacement=np.asarray(history.damage_failure_displacement),
            failed=np.asarray(history.failed), pressure_Pa=np.asarray(history.pressure),
            J=np.linalg.det(np.asarray(current.F)),
            tool_tip=np.asarray(tool._polygon(time, jnp.float64)[0]),
        )
        impulse, contact_nodes = tool_impulse(current, time)
        impulse = np.asarray(impulse)
        if args.particle_grid_contact:
            return {
                "time_s": float(time), "tool_tip_x_m": float(tool.tip_position(time, jnp.float64)[0]),
                "tool_tip_y_m": float(tool.tip_position(time, jnp.float64)[1]),
                "penetrating_particles": int(inside.sum()),
                "penetrating_particle_fraction": float(inside.mean()),
                "maximum_particle_tool_penetration_m": float(depth.max(initial=0.0)),
                "maximum_temperature_K": float(np.max(np.asarray(history.temperature))),
                "maximum_plastic_strain": float(np.max(np.asarray(history.plastic_strain))),
                "failed_points": int(np.asarray(history.failed).sum()),
                "minimum_J": float(np.linalg.det(np.asarray(current.F)).min()),
            }
        if args.particle_contact or args.hybrid_contact or args.failed_particle_contact:
            return {
                "time_s": float(time), "tool_tip_x_m": float(tool.tip_position(time, jnp.float64)[0]),
                "tool_tip_y_m": float(tool.tip_position(time, jnp.float64)[1]),
                "counterfactual_nodal_tool_reaction_x_N_per_m": float(-impulse[0] / config.dt),
                "counterfactual_nodal_tool_reaction_y_N_per_m": float(-impulse[1] / config.dt),
                "counterfactual_nodal_contact_nodes": int(contact_nodes),
                "penetrating_particles": int(inside.sum()),
                "penetrating_particle_fraction": float(inside.mean()),
                "maximum_particle_tool_penetration_m": float(depth.max(initial=0.0)),
                "maximum_temperature_K": float(np.max(np.asarray(history.temperature))),
                "maximum_plastic_strain": float(np.max(np.asarray(history.plastic_strain))),
                "plastic_work_J_per_m": float(np.dot(np.asarray(current.mass), np.asarray(history.energy))),
                "failed_points": int(np.asarray(history.failed).sum()),
                "minimum_J": float(np.linalg.det(np.asarray(current.F)).min()),
                "minimum_Fzz": float(np.asarray(current.F)[:, 2, 2].min()),
                "maximum_Fzz_error": float(np.max(np.abs(np.asarray(current.F)[:, 2, 2] - 1.0))),
            }
        return {
            "time_s": float(time), "tool_tip_x_m": float(tool.tip_position(time, jnp.float64)[0]),
            "tool_tip_y_m": float(tool.tip_position(time, jnp.float64)[1]),
            "predicted_tool_reaction_x_N_per_m": float(-impulse[0] / config.dt),
            "predicted_tool_reaction_y_N_per_m": float(-impulse[1] / config.dt),
            "predicted_occupied_tool_contact_nodes": int(contact_nodes),
            "penetrating_particles": int(inside.sum()),
            "penetrating_particle_fraction": float(inside.mean()),
            "maximum_particle_tool_penetration_m": float(depth.max(initial=0.0)),
            "material_points_above_initial_top": int((x[:, 1] > WORK_UPPER[1] + config.dx).sum()),
            "maximum_temperature_K": float(np.max(np.asarray(history.temperature))),
            "maximum_plastic_strain": float(np.max(np.asarray(history.plastic_strain))),
            "plastic_work_J_per_m": float(np.dot(np.asarray(current.mass), np.asarray(history.energy))),
            "failed_points": int(np.asarray(history.failed).sum()),
            "minimum_J": float(np.linalg.det(np.asarray(current.F)).min()),
            "minimum_Fzz": float(np.asarray(current.F)[:, 2, 2].min()),
            "maximum_Fzz_error": float(np.max(np.abs(np.asarray(current.F)[:, 2, 2] - 1.0))),
        }

    nodes = (args.grid + 1) ** 2
    print(json.dumps({
        "particles": int(len(state.x)), "grid": args.grid, "nodes": nodes,
        "dt_s": config.dt, "planned_steps": steps,
        "planned_duration_s": steps * config.dt,
        "planned_cut_distance_m": TOOL_SPEED * steps * config.dt,
        "output": str(output),
    }), flush=True)
    runner(
        (state,), (material,), (config,), frames=frames, substeps=args.substeps,
        output=output, label="2D plane-strain orthogonal cutting / Ti-6Al-4V / JAX MPM",
        observer=observe, metadata=definition, render_output=False, resume=args.resume,
        stop_after_frame=args.stop_after_frames,
        post_step=post_step,
        capture_boundary_impulse=args.hybrid_contact,
        step_adapter=step_adapter,
        step_adapter_cache_factory=(
            (lambda s: jnp.zeros((s.x.shape[0], 5), dtype=s.x.dtype))
            if args.particle_contact_warm_start else None),
        point_fields={"reference_x": np.asarray(state.x)[:, 0],
                      "reference_y": np.asarray(state.x)[:, 1]},
    )


if __name__ == "__main__":
    main()
