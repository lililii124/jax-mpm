"""Bonded woven composites under prescribed dynamic loading and contact."""

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, State, VelocityBox, fixed_box
from mpm_jax.boundaries import PlaneWall
from mpm_jax.contact_laws import boundary_reaction
from mpm_jax.materials import initial_history
from mpm_jax.materials.woven import (
    woven_elastic, woven_energy, woven_kirchhoff_stress, woven_wave_speed_bound,
)
from .common import run
from .scenarios import Case
from .woven_geometry import ARCHITECTURES, sample_weave
from .woven_hemisphere import RigidHemisphereBoundary


LOADS = (
    "warp-tension", "weft-tension", "in-plane-shear",
    "transverse-compression", "transverse-bending", "hemispherical-compression",
)
DOMAIN = 0.016
ORIGIN = (0.008, 0.008, 0.008)
SIZE = (0.010, 0.010)
PITCH = 0.0025
YARN_WIDTH = 0.000625
YARN_HEIGHT = 0.0005
GAP = 0.0005
MATRIX_COVER = 0.00025
THICKNESS = 2 * YARN_HEIGHT + GAP + 2 * MATRIX_COVER
LOADING_TIME = 0.45e-3 / np.sqrt(1000.0)
DURATION = 0.60e-3 / np.sqrt(1000.0)
MATERIAL = {
    "matrix_young": 2e9, "matrix_poisson": 0.30, "matrix_density": 1100.0,
    "yarn_young": (40e9, 4e9, 4e9), "yarn_poisson": (0.25, 0.25, 0.30),
    "yarn_shear": (1.5e9, 2e9, 2e9), "yarn_density": 1400.0,
}


@dataclass(frozen=True)
class SmoothDisplacementBox:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    displacement: tuple[float, float, float]
    loading_duration: float
    components: tuple[bool, bool, bool] | None = None
    start: float = 0.0

    def validate(self, dim, domain_size):
        VelocityBox(self.lower, self.upper, self.displacement, self.components).validate(dim, domain_size)
        if not np.isfinite(self.loading_duration) or self.loading_duration <= 0:
            raise ValueError("loading_duration must be finite and positive")
        if not np.isfinite(self.start) or self.start < 0:
            raise ValueError("start must be finite and nonnegative")

    def motion(self, time):
        elapsed = jnp.asarray(time, dtype=jnp.result_type(time, float)) - self.start
        fraction = jnp.clip(elapsed / self.loading_duration, 0.0, 1.0)
        distance = 0.5 * (1 - jnp.cos(jnp.pi * fraction))
        rate = 0.5 * jnp.pi / self.loading_duration * jnp.sin(jnp.pi * fraction)
        rate = jnp.where((elapsed > 0) & (elapsed < self.loading_duration), rate, 0.0)
        target = jnp.asarray(self.displacement, dtype=elapsed.dtype)
        return distance * target, rate * target

    def apply(self, velocity, coordinates, time):
        displacement, speed = self.motion(time)
        reference = coordinates - displacement.astype(coordinates.dtype)
        inside = jnp.all(
            (reference >= jnp.asarray(self.lower)) & (reference <= jnp.asarray(self.upper)),
            axis=-1,
        ) & (time >= self.start)
        components = jnp.ones(3, dtype=bool) if self.components is None else jnp.asarray(self.components)
        return jnp.where(inside[:, None] & components, speed.astype(velocity.dtype), velocity)


@dataclass
class WovenCase(Case):
    point_fields: dict = field(default_factory=dict)


def specimen_bounds():
    half = np.array((SIZE[0], THICKNESS, SIZE[1])) / 2
    return np.asarray(ORIGIN) - half, np.asarray(ORIGIN) + half


def loading_boundaries(loading, dx):
    """Prescribed grid fixtures or frictionless unilateral contact geometry."""
    if loading not in LOADS:
        raise ValueError(f"loading must be one of {LOADS}")
    lower, upper = specimen_bounds()
    grip_width = 0.001
    if loading in ("warp-tension", "weft-tension", "in-plane-shear"):
        face_axis = 0 if loading == "warp-tension" else 2
        motion_axis = 0 if loading != "weft-tension" else 2
        stroke = 0.00015 if loading == "in-plane-shear" else 0.00008
        low_upper, high_lower = np.full(3, DOMAIN), np.zeros(3)
        low_upper[face_axis] = lower[face_axis] + grip_width
        high_lower[face_axis] = upper[face_axis] - grip_width
        direction = np.eye(3)[motion_axis] * stroke
        boundaries = (
            SmoothDisplacementBox((0.0,) * 3, tuple(low_upper), tuple(-direction), LOADING_TIME),
            SmoothDisplacementBox(tuple(high_lower), (DOMAIN,) * 3, tuple(direction), LOADING_TIME),
        )
        names = ("negative_grip", "positive_grip")
        description = {
            "gauge_length_m": float(high_lower[face_axis] - low_upper[face_axis]),
            "grip_width_m": grip_width, "face_normal_axis": "xyz"[face_axis],
            "motion_axis": "xyz"[motion_axis], "stroke_per_grip_m": stroke,
            "nominal_final_strain_or_shear": 2 * stroke / (high_lower[face_axis] - low_upper[face_axis]),
            "constraints": "both edge grips prescribe all velocity components; other faces are traction-free",
        }
    elif loading == "transverse-compression":
        depth = 0.000375
        boundaries = (
            SmoothDisplacementBox(
                (0.0,) * 3, (DOMAIN, lower[1] + depth, DOMAIN),
                (0.0, 0.000025, 0.0), LOADING_TIME, (False, True, False),
            ),
            SmoothDisplacementBox(
                (0.0, upper[1] - depth, 0.0), (DOMAIN,) * 3,
                (0.0, -0.000025, 0.0), LOADING_TIME, (False, True, False),
            ),
        )
        names = ("lower_platen", "upper_platen")
        description = {
            "face_patch_depth_m": depth, "stroke_per_platen_m": 0.000025,
            "nominal_final_thickness_strain": -0.00005 / THICKNESS,
            "free_height_m": THICKNESS - 2 * depth,
            "nominal_final_free_height_strain": -0.00005 / (THICKNESS - 2 * depth),
            "constraints": "opposing y-velocity face patches; x/z slide freely; zero initial velocity and no horizontal forcing",
        }
    elif loading == "hemispherical-compression":
        radius, stroke = SIZE[0] / 8, 0.0001
        boundaries = (
            PlaneWall(axis=1, position=float(lower[1]), side="lower", friction=0.0),
            RigidHemisphereBoundary(
                (ORIGIN[0], float(upper[1] + radius), ORIGIN[2]), radius,
                (0.0, -stroke, 0.0), LOADING_TIME,
            ),
        )
        names = ("lower_plate", "hemisphere")
        description = {
            "indenter_shape": "closed lower hemisphere: curved cap and flat disk at center y",
            "indenter_radius_m": radius, "diameter_to_specimen_width_ratio": 0.25,
            "final_indentation_m": stroke, "support_plane_y_m": float(lower[1]),
            "nominal_contact_radius_m": float(np.sqrt(radius * stroke)),
            "nominal_contact_radius_cells": float(np.sqrt(radius * stroke) / dx),
            "contact_radius_scope": "sqrt(R*indentation) is a geometric resolution estimate, not the measured contact radius or a Hertz solution",
            "center_top_roi_radius_m": 0.00025,
            "center_top_roi_definition": "fixed reference top particle layer within radius 0.25 mm of the specimen center in x/z",
            "constraints": "lower frictionless unilateral plane; upper frictionless unilateral closed hemisphere; no imposed top-face velocity or lateral clamp",
        }
    else:
        left, right = lower[0] + grip_width, upper[0] - grip_width
        support_half_width = 0.0002
        support_depth = 0.25 * dx
        boundaries = (
            fixed_box((left - support_half_width, 0.0, 0.0), (left + support_half_width, lower[1] + support_depth, DOMAIN), components=(True, True, False)),
            fixed_box((right - support_half_width, 0.0, 0.0), (right + support_half_width, lower[1] + support_depth, DOMAIN), components=(False, True, False)),
            fixed_box((left - support_half_width, 0.0, ORIGIN[2] - 0.55 * dx), (left + support_half_width, lower[1] + support_depth, ORIGIN[2] + 0.55 * dx), components=(False, False, True)),
            SmoothDisplacementBox(
                (ORIGIN[0] - 0.0005, upper[1] - 0.5 * dx, 0.0),
                (ORIGIN[0] + 0.0005, DOMAIN, DOMAIN),
                (0.0, -0.00025, 0.0), LOADING_TIME, (False, True, False),
            ),
        )
        names = ("left_support", "right_support", "lateral_anchor", "loading_patch")
        description = {
            "support_centers_x_m": [left, right], "support_span_m": right - left,
            "support_width_m": 2 * support_half_width, "loading_patch_width_m": 0.001,
            "final_loading_displacement_y_m": -0.00025,
            "constraints": "left support fixes x/y, right support fixes y, small left mid-width patch fixes z; central upper patch prescribes y only",
        }
    return boundaries, names, description


def _support_inside(reference, dx, boundary):
    if isinstance(boundary, (PlaneWall, RigidHemisphereBoundary)):
        return np.zeros(len(reference), dtype=bool)
    if isinstance(boundary, SmoothDisplacementBox):
        return np.all(
            (reference >= np.asarray(boundary.lower) + 1.5 * dx)
            & (reference <= np.asarray(boundary.upper) - 1.5 * dx), axis=1,
        )
    lowest = np.floor(reference / dx - 0.5) * dx
    highest = lowest + 2 * dx
    return np.all((lowest >= np.asarray(boundary.lower) - 1e-14) & (highest <= np.asarray(boundary.upper) + 1e-14), axis=1)


def rigid_grips(loading, boundaries, names, dx):
    lower, upper = specimen_bounds()
    if loading == "hemispherical-compression":
        return [{
            "name": "lower_plate", "lower": [lower[0], lower[1] - .00025, lower[2]],
            "upper": [upper[0], lower[1], upper[2]], "displacement": [0., 0., 0.],
            "loading_duration": LOADING_TIME, "start": 0.,
            "motion_kind": "half-cosine-displacement-hold",
            "constraint_kind": "frictionless unilateral plane at the top of the displayed support",
        }]
    fixtures = []
    for boundary, name in zip(boundaries, names):
        if name == "lateral_anchor":
            continue
        lo = np.maximum(boundary.lower, lower).astype(float)
        hi = np.minimum(boundary.upper, upper).astype(float)
        if loading == "transverse-compression":
            edge = lower[1] if name == "lower_platen" else upper[1]
            lo[1], hi[1] = (edge - 0.00025, edge) if name == "lower_platen" else (edge, edge + 0.00025)
        elif loading == "transverse-bending":
            if name == "loading_patch":
                lo[1], hi[1] = upper[1], upper[1] + 0.00025
            else:
                lo[1], hi[1] = lower[1] - 0.00025, lower[1]
        else:
            lo[1] -= 0.5 * dx
            hi[1] += 0.5 * dx
        fixtures.append({
            "name": name, "lower": lo.tolist(), "upper": hi.tolist(),
            "displacement": list(getattr(boundary, "displacement", (0.0,) * 3)),
            "loading_duration": float(getattr(boundary, "loading_duration", LOADING_TIME)),
            "start": float(boundary.start),
            "motion_kind": "half-cosine-displacement-hold",
        })
    return fixtures


def make_case(architecture="plain", loading="warp-tension", *, grid=96, ppc=2, frames=60):
    if architecture not in ARCHITECTURES or loading not in LOADS:
        raise ValueError("unknown weave architecture or loading")
    if not isinstance(grid, int) or isinstance(grid, bool) or grid < 32:
        raise ValueError("grid must be an integer >= 32")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (ppc, frames)):
        raise ValueError("ppc and frames must be positive integers")
    dx = DOMAIN / grid
    sample = sample_weave(
        architecture, size=SIZE, pitch=PITCH, yarn_width=YARN_WIDTH,
        yarn_height=YARN_HEIGHT, gap=GAP, matrix_cover=MATRIX_COVER,
        spacing=dx / ppc, origin=ORIGIN,
    )
    material = woven_elastic(**MATERIAL, orientation=sample.orientation, phase=sample.phase)
    speed = woven_wave_speed_bound(material)
    substeps = int(np.ceil(DURATION / (frames * 0.15 * dx / speed)))
    boundaries, names, load_description = loading_boundaries(loading, dx)
    config = Config(
        dim=3, n_grid=grid, domain_size=DOMAIN,
        dt=DURATION / (frames * substeps), material="woven-composite",
        gravity=(0.0,) * 3, boundary_cells=0, boundaries=boundaries,
    )
    reference = sample.positions
    count = len(reference)
    density = np.where(sample.phase == 1, MATERIAL["yarn_density"], MATERIAL["matrix_density"])
    volume = jnp.asarray(sample.volume, dtype=jnp.float64)
    state = State(
        jnp.asarray(reference, dtype=jnp.float64), jnp.zeros((count, 3), dtype=jnp.float64),
        jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (count, 3, 3)),
        jnp.zeros((count, 3, 3), dtype=jnp.float64),
        jnp.asarray(density) * volume, volume,
        initial_history(count, jnp.float64, density=jnp.asarray(density)),
        jnp.asarray(0.0, dtype=jnp.float64), jnp.asarray(True),
    )
    interior = [_support_inside(reference, dx, boundary) for boundary in boundaries]
    spacing = dx / ppc
    sections = [np.abs(reference[:, axis] - ORIGIN[axis]) <= 0.51 * spacing for axis in range(3)]
    section_widths = [len(np.unique(reference[mask, axis])) * spacing for axis, mask in enumerate(sections)]
    center = (np.abs(reference[:, 0] - ORIGIN[0]) <= 0.0005) & (np.abs(reference[:, 2] - ORIGIN[2]) <= 0.0005)
    top_center = (reference[:, 1] == reference[:, 1].max()) & (np.linalg.norm(reference[:, (0, 2)] - np.asarray(ORIGIN)[[0, 2]], axis=1) <= .00025)

    def observe(states, time):
        current = states[0]
        x, velocity, F = (np.asarray(value) for value in (current.x, current.v, current.F))
        displacement = x - reference
        magnitude = np.linalg.norm(displacement, axis=1)
        strain = 0.5 * (F.swapaxes(-1, -2) @ F - np.eye(3))
        strain_norm = np.sqrt(np.sum(strain**2, axis=(-1, -2)))
        tau = np.asarray(woven_kirchhoff_stress(current.F, material))
        energy = np.asarray(woven_energy(current.F, material))
        piola = tau @ np.linalg.inv(F).swapaxes(-1, -2)
        values = {
            "maximum_displacement_m": float(magnitude.max()),
            "rms_displacement_m": float(np.sqrt(np.mean(magnitude**2))),
            "center_mean_displacement_y_m": float(displacement[center, 1].mean()),
            "maximum_green_strain_norm": float(strain_norm.max()),
            "minimum_J": float(np.linalg.det(F).min()),
            "maximum_J": float(np.linalg.det(F).max()),
            "reference_yarn_volume_fraction": sample.metadata["yarn_volume_fraction"],
            "max_damage": float(np.asarray(current.history.damage).max()),
            "failed_particles": int(np.count_nonzero(np.asarray(current.history.failed))),
            "boundary_sample_time_s": float(time + config.dt),
        }
        for phase, name in ((0, "matrix"), (1, "yarn")):
            selected = sample.phase == phase
            values[f"{name}_strain_energy_J"] = float(np.sum(sample.volume[selected] * energy[selected]))
            values[f"{name}_maximum_green_strain_norm"] = float(strain_norm[selected].max())
        for axis, mask in enumerate(sections):
            weighted = piola[mask, :, axis] * (sample.volume[mask] / section_widths[axis])[:, None]
            for component, name in enumerate("xyz"):
                values[f"central_{'xyz'[axis]}_section_force_{name}_N"] = float(weighted[:, component].sum())
            if axis == 0:
                values["central_x_section_moment_z_Nm"] = float(-np.sum(
                    (reference[mask, 1] - ORIGIN[1]) * weighted[:, 0]
                ))
        for index, (boundary, name, selected) in enumerate(zip(boundaries, names, interior)):
            force = np.asarray(boundary_reaction(current, material, config, boundary_index=index))
            if isinstance(boundary, (PlaneWall, RigidHemisphereBoundary)):
                for axis, label in enumerate("xyz"):
                    values[f"{name}_reaction_{label}_N"] = float(force[axis])
                if isinstance(boundary, PlaneWall):
                    values["lower_plate_maximum_particle_overlap_m"] = float(max(boundary.position - x[:, 1].min(), 0.0))
                else:
                    tool_center, tool_speed = (np.asarray(value) for value in boundary.motion(time))
                    gap = np.asarray(boundary.signed_distance(current.x, time))
                    values["hemisphere_minimum_particle_gap_m"] = float(gap.min())
                    values["hemisphere_maximum_particle_overlap_m"] = float(max(-gap.min(), 0.0))
                    values["hemisphere_particle_centers_inside"] = int(np.count_nonzero(gap < 0))
                    values["hemisphere_flat_face_clearance_m"] = float(tool_center[1] - x[:, 1].max())
                    values["hemisphere_center_top_roi_particles"] = int(top_center.sum())
                    values["hemisphere_center_top_mean_displacement_y_m"] = float(displacement[top_center, 1].mean()) if top_center.any() else None
                    values["hemisphere_center_top_maximum_downward_displacement_m"] = float(max(-displacement[top_center, 1].min(), 0.0)) if top_center.any() else None
                    for axis, label in enumerate("xyz"):
                        values[f"hemisphere_center_{label}_m"] = float(tool_center[axis])
                        values[f"hemisphere_prescribed_velocity_{label}_m_per_s"] = float(tool_speed[axis])
                continue
            target, speed_at_time = boundary.motion(time) if hasattr(boundary, "motion") else (np.zeros(3), np.zeros(3))
            components = np.ones(3, dtype=bool) if boundary.components is None else np.asarray(boundary.components)
            target, speed_at_time = np.asarray(target), np.asarray(speed_at_time)
            node_lower = np.floor(x[selected] / config.dx - 0.5) * config.dx
            node_upper = node_lower + 2 * config.dx
            contained = np.all(
                (node_lower >= np.asarray(boundary.lower) + target - 1e-14)
                & (node_upper <= np.asarray(boundary.upper) + target + 1e-14), axis=1,
            )
            values[f"{name}_interior_particles"] = int(selected.sum())
            values[f"{name}_interior_current_full_support_particles"] = int(contained.sum())
            values[f"{name}_interior_position_error_m"] = float(np.abs((displacement[selected] - target)[:, components]).max()) if selected.any() else None
            values[f"{name}_interior_velocity_error_m_per_s"] = float(np.abs((velocity[selected] - speed_at_time)[:, components]).max()) if selected.any() else None
            for axis, label in enumerate("xyz"):
                values[f"{name}_reaction_{label}_N"] = float(force[axis])
                values[f"{name}_prescribed_displacement_{label}_m"] = float(target[axis])
                values[f"{name}_prescribed_velocity_{label}_m_per_s"] = float(speed_at_time[axis])
        if loading == "hemispherical-compression":
            weights = np.asarray(current.mass)
            com_change = np.sum(weights[:, None] * displacement, axis=0) / weights.sum()
            for axis, label in enumerate("xyz"):
                values[f"center_of_mass_displacement_{label}_m"] = float(com_change[axis])
                values[f"net_boundary_reaction_{label}_N"] = sum(values[f"{name}_reaction_{label}_N"] for name in names)
            values["horizontal_center_of_mass_displacement_m"] = float(np.linalg.norm(com_change[[0, 2]]))
        return values

    lower, upper = specimen_bounds()
    description = {
        "purpose": f"{architecture}-{loading}", "series": "woven-elastic-v1",
        "loading_revision": 2 if loading == "in-plane-shear" else 1,
        "architecture": architecture, "loading": loading, "model": "woven-composite",
        "lower": lower.tolist(), "upper": upper.tolist(), "geometry": sample.metadata,
        "particle_spacing_m": spacing, "particles_per_cell_per_axis": ppc,
        "grid_cells_across_yarn_height": YARN_HEIGHT / dx,
        "grid_cells_across_yarn_width": YARN_WIDTH / dx,
        "nominal_grid_resolution_target_met": YARN_HEIGHT / dx >= 3 - 1e-12,
        "resolution_scope": "three grid cells across yarn height at grid96; an initial resolved discretization, not a convergence result; extra particles do not replace grid refinement",
        "material_parameters": MATERIAL, "units": "m, s, kg, Pa", "geometry_scale": 1.0,
        "material_scope": "generic illustrative engineering constants; no calibrated fibre/resin grade",
        "constitutive_scope": "fully bonded single-grid matrix and continuous interlaced yarns; neo-Hookean matrix and orthotropic Saint Venant-Kirchhoff yarns; no debonding, damage, fracture or Z binders",
        "comparison_scope": "same outer dimensions and engineering constants; architectures have different yarn volume fractions, so differences are not attributed to topology alone",
        "loading_regime": "dynamic displacement control; not a quasi-static material test",
        "weft_loading_definition": "global z loading; triaxial families are 0,+60,-60 degrees and have no 90-degree yarn family",
        "duration_s": DURATION, "loading_duration_s": LOADING_TIME,
        "hold_duration_s": DURATION - LOADING_TIME,
        "motion_law": "u(t)=target*(1-cos(pi*clip(t/loading_duration,0,1)))/2; stationary displacement hold afterwards",
        "initial_acoustic_speed_bound_m_per_s": speed, "initial_acoustic_CFL_limit": 0.15,
        "initial_acoustic_CFL": config.dt * speed / dx,
        "boundary_names": list(names), "boundary_types": [type(boundary).__name__ for boundary in boundaries],
        "boundaries": [asdict(boundary) for boundary in boundaries],
        "rigid_grips": rigid_grips(loading, boundaries, names, dx),
        "loading_definition": load_description,
        "reaction_sampling": "force on each grid fixture in a replay of the next step; sample time=t+dt, includes inertia and earlier constraints",
        "section_force_measure": "reference central slab quadrature of P=tau*F^-T; sum(P_iA*V0/slab_width), two reference layers on an even lattice",
        "section_moment_measure": "central x-plane reference moment about z: -sum((Y0-Ycenter)*P_xx*V0/slab_width)",
        "boundary_tracking_scope": "moving-box samples remain at least 1.5dx from all reference faces, covering every grid phase under exact prescribed translation; static boxes use initial discrete support; current complete support is counted separately; no interior samples means no direct tracking claim",
    }
    fields = {name: getattr(sample, name) for name in ("phase", "yarn_id", "family_id")}
    if loading == "hemispherical-compression":
        description["rigid_hemisphere"] = {
            **asdict(boundaries[-1]), "motion_kind": "half-cosine-displacement-hold",
            "display_opacity": 0.35,
        }
        description["boundary_tracking_scope"] = "unilateral grid contact, no exact particle-clamping claim; saved particle-center overlap is a resolution alarm, not a contact accuracy or area measure"
        description["horizontal_motion_scope"] = "free sides and frictionless support; no artificial horizontal anchor; report mass-weighted center-of-mass drift and next-step net tool reaction without assuming zero coupling for every yarn architecture"
        description["resolution_scope"] = "formal grid192/ppc1 retains the grid96/ppc2 particle spacing while refining the background grid; nominal contact radius must span at least four grid cells; not a convergence result"
    return WovenCase((state,), (material,), (config,), DURATION, description, observer=observe, point_fields=fields), substeps


def loading_resolution(loading, grid=None, ppc=None):
    """Choose CLI defaults without changing explicit resolution requests."""
    if loading not in LOADS:
        raise ValueError("unknown woven loading")
    default_grid, default_ppc = (192, 1) if loading == "hemispherical-compression" else (96, 2)
    return default_grid if grid is None else grid, default_ppc if ppc is None else ppc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=(*ARCHITECTURES, "all"), default="plain")
    parser.add_argument("--loading", choices=(*LOADS, "all"), default="warp-tension")
    parser.add_argument("--grid", type=int, help="grid cells per side (default: hemisphere 192, other loads 96)")
    parser.add_argument("--ppc", type=int, help="points per cell direction (default: hemisphere 1, other loads 2)")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/woven"))
    rendering = parser.add_mutually_exclusive_group()
    rendering.add_argument("--render", action="store_true")
    rendering.add_argument("--no-render", action="store_false", dest="render")
    parser.set_defaults(render=False)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.gif and not args.render:
        parser.error("--gif requires --render; the default only writes numerical output")
    jax.config.update("jax_enable_x64", True)
    architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
    loadings = LOADS if args.loading == "all" else (args.loading,)
    for architecture in architectures:
        for loading in loadings:
            grid, ppc = loading_resolution(loading, args.grid, args.ppc)
            case, substeps = make_case(architecture, loading, grid=grid, ppc=ppc, frames=args.frames)
            print(f"{architecture}/{loading}: {len(case.states[0].x)} points, {args.frames * substeps} steps, t={case.duration:.9g}s", flush=True)
            run(
                case.states, case.materials, case.configs, frames=args.frames, substeps=substeps,
                output=args.output / architecture / loading, label=f"{architecture}/{loading} / woven-composite",
                observer=case.observer, metadata=case.description, point_fields=case.point_fields,
                render_output=False, resume=args.resume,
            )
            if args.render:
                from mpm_jax.visualization import render_trajectory

                render_trajectory(args.output / architecture / loading, gif=args.gif, field="family_id")
            del case
            jax.clear_caches()


if __name__ == "__main__":
    main()
