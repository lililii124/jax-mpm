"""Steel round-bar necking using the geometry and loading of SPHinXsys PR 1127."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, johnson_cook
from mpm_jax.contact_laws import boundary_reaction
from mpm_jax.core import State
from mpm_jax.materials import initial_history
from .common import run
from .scenarios import Case


SOURCE_COMMIT = "c3c1556ec86f9ebc6f0697149dff052702d20a75"
DOMAIN = 0.01
CENTER = DOMAIN / 2
RADIUS = 8e-4
WAIST_RADIUS = 7.5e-4
HALF_LENGTH = 3.2e-3
TRANSITION_LENGTH = 7.5e-4
GRIP_POSITION = 2.8e-3
ARC_RADIUS = (TRANSITION_LENGTH**2 + (RADIUS - WAIST_RADIUS) ** 2) / (4 * (RADIUS - WAIST_RADIUS))
SPEED = 10.0
RAMP_TIME = 20e-6
DURATION = 150e-6


def verification_steel():
    return johnson_cook(
        young=200e9,
        poisson=0.29,
        density=7830.0,
        A=792e6,
        B=510e6,
        n=0.26,
        C=0.014,
        m=1.03,
        reference_rate=1.0,
        reference_temperature=298.0,
        melt_temperature=1793.0,
        heat_capacity=477.0,
        heat_fraction=1.0,
        failure=(0.05, 3.44, -2.12, 0.002, 0.61),
    )


def rod_radius(axial_coordinate):
    coordinate = np.minimum(np.abs(np.asarray(axial_coordinate)), TRANSITION_LENGTH)
    first = WAIST_RADIUS + ARC_RADIUS - np.sqrt(ARC_RADIUS**2 - coordinate**2)
    second = RADIUS - ARC_RADIUS + np.sqrt(ARC_RADIUS**2 - (TRANSITION_LENGTH - coordinate) ** 2)
    return np.where(coordinate <= TRANSITION_LENGTH / 2, first, second)


def reference_lattice(grid, ppc):
    """A centred Cartesian lattice with identical spacing and volume on all axes."""
    if not isinstance(grid, int) or grid < 50:
        raise ValueError("jc-necking requires grid >= 50")
    if not isinstance(ppc, int) or ppc < 1:
        raise ValueError("ppc must be a positive integer")
    target_count = 2 * RADIUS / (DOMAIN / grid) * ppc
    transverse_count = 2 * int(np.ceil(target_count / 2 - 1e-12))
    spacing = 2 * RADIUS / transverse_count
    axial_count = 4 * transverse_count
    transverse = -RADIUS + (np.arange(transverse_count) + 0.5) * spacing
    yz = np.stack(np.meshgrid(transverse, transverse, indexing="ij"), axis=-1).reshape(-1, 2)
    radial_squared = np.sum(yz**2, axis=1)
    layers = []
    for axial in -HALF_LENGTH + (np.arange(axial_count) + 0.5) * spacing:
        cross_section = yz[radial_squared < rod_radius(axial) ** 2]
        layers.append(np.column_stack((np.full(len(cross_section), axial), cross_section)))
    return np.concatenate(layers) + CENTER, spacing, transverse_count


def make_case(*, grid=100, ppc=3, frames=60, end_time=DURATION, cfl=0.15):
    if not isinstance(frames, int) or frames < 1:
        raise ValueError("frames must be a positive integer")
    if not np.isfinite(end_time) or not 0 < end_time <= DURATION:
        raise ValueError("end_time must be positive and at most 150 microseconds")
    if not np.isfinite(cfl) or not 0 < cfl <= 0.3:
        raise ValueError("cfl must be positive and at most 0.3")
    reference, spacing, transverse_count = reference_lattice(grid, ppc)
    material = verification_steel()
    left_face, right_face = CENTER - GRIP_POSITION, CENTER + GRIP_POSITION
    left = MovingVelocityBox(
        (0.0, 0.0, 0.0),
        (left_face, DOMAIN, DOMAIN),
        (-SPEED, 0.0, 0.0),
        ramp_time=RAMP_TIME,
    )
    right = MovingVelocityBox(
        (right_face, 0.0, 0.0),
        (DOMAIN, DOMAIN, DOMAIN),
        (SPEED, 0.0, 0.0),
        ramp_time=RAMP_TIME,
    )
    wave_speed = float(np.sqrt((material.bulk + 4 * material.mu / 3) / material.density))
    substeps = int(np.ceil(end_time * wave_speed / (frames * cfl * DOMAIN / grid)))
    config = Config(
        dim=3,
        n_grid=grid,
        domain_size=DOMAIN,
        dt=end_time / (frames * substeps),
        material="johnson-cook",
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=(left, right),
        failure_mode="ballistic",
    )
    count = len(reference)
    dtype = jnp.float64
    volume = jnp.full(count, spacing**3, dtype=dtype)
    state = State(
        jnp.asarray(reference, dtype=dtype),
        jnp.zeros((count, 3), dtype=dtype),
        jnp.broadcast_to(jnp.eye(3, dtype=dtype), (count, 3, 3)),
        jnp.zeros((count, 3, 3), dtype=dtype),
        material.density * volume,
        volume,
        initial_history(count, dtype, material, material.density),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(True),
    )
    levels, sections = np.unique(reference[:, 0], return_inverse=True)
    initial_count = np.bincount(sections)
    gauge = (levels > left_face + config.dx) & (levels < right_face - config.dx)
    central = np.abs(levels - CENTER) <= 0.51 * spacing
    central_particles = central[sections]
    initial_radius_squared = (
        np.bincount(sections, weights=np.sum((reference[:, 1:] - CENTER) ** 2, axis=1))
        / initial_count
    )
    left_inner = reference[:, 0] <= left_face - 1.5 * config.dx + 1e-14
    right_inner = reference[:, 0] >= right_face + 1.5 * config.dx - 1e-14
    grip = (reference[:, 0] <= left_face) | (reference[:, 0] >= right_face)
    initial_mass = float(np.asarray(state.mass).sum())
    initial_area = np.pi * RADIUS**2
    gauge_length = 2 * GRIP_POSITION

    def observe(states, time):
        current = states[0]
        position, velocity, deformation, mass, reference_volume = (
            np.asarray(value)
            for value in (current.x, current.v, current.F, current.mass, current.volume)
        )
        failed = np.asarray(current.history.failed)
        active = ~failed
        J = np.linalg.det(deformation)
        axial_stretch = np.linalg.norm(deformation[:, :, 0], axis=1)
        area_ratio = J / axial_stretch
        active_count = np.bincount(sections, weights=active, minlength=len(levels))
        section_area_ratio = (
            np.bincount(sections, weights=active * area_ratio, minlength=len(levels))
            / initial_count
        )
        cofactor_normal = J[:, None] * np.swapaxes(np.linalg.inv(deformation), 1, 2)[:, :, 0]
        axial_traction = np.einsum(
            "pi,pi->p", np.asarray(current.history.stress)[:, 0, :], cofactor_normal
        )
        section_force = np.bincount(
            sections,
            weights=active * axial_traction * reference_volume / spacing,
            minlength=len(levels),
        )
        transverse_mean = np.stack(
            [np.bincount(sections, weights=position[:, axis]) / initial_count for axis in (1, 2)],
            axis=-1,
        )
        radial_squared = np.sum((position[:, 1:] - transverse_mean[sections]) ** 2, axis=1)
        geometric_radius_squared = np.bincount(sections, weights=radial_squared) / initial_count
        rms_ratio = np.sqrt(geometric_radius_squared / initial_radius_squared)
        intact_gauge = gauge & (active_count == initial_count)
        gaps = np.diff(np.sort(position[active, 0]))
        components = int(active.any()) + int(np.count_nonzero(gaps > 3 * config.dx))
        reaction_left = np.asarray(boundary_reaction(current, material, config, boundary_index=0))
        reaction_right = np.asarray(boundary_reaction(current, material, config, boundary_index=1))
        displacement_left, velocity_left = (np.asarray(v) for v in left.motion(time))
        displacement_right, velocity_right = (np.asarray(v) for v in right.motion(time))
        velocity_error, position_error, transverse_error = [], [], []
        for mask, displacement, prescribed_velocity in (
            (left_inner, displacement_left, velocity_left),
            (right_inner, displacement_right, velocity_right),
        ):
            if mask.any():
                velocity_error.append(
                    np.linalg.norm(velocity[mask] - prescribed_velocity, axis=1).max()
                )
                position_error.append(
                    np.abs(position[mask, 0] - reference[mask, 0] - displacement[0]).max()
                )
                transverse_error.append(
                    np.linalg.norm(position[mask, 1:] - reference[mask, 1:], axis=1).max()
                )
        extension = float(displacement_right[0] - displacement_left[0])
        load = 0.5 * (reaction_left[0] - reaction_right[0])
        next_velocity_left = np.asarray(left.motion(time + config.dt)[1])
        next_velocity_right = np.asarray(right.motion(time + config.dt)[1])
        return {
            "prescribed_extension_m": extension,
            "engineering_strain": extension / gauge_length,
            "mean_grip_tensile_force_N": float(load),
            "nominal_grip_stress_Pa": float(load / initial_area),
            "left_grip_reaction_x_N": float(reaction_left[0]),
            "right_grip_reaction_x_N": float(reaction_right[0]),
            "boundary_sample_time_s": float(time + config.dt),
            "actuator_power_W": float(
                -reaction_left @ next_velocity_left - reaction_right @ next_velocity_right
            ),
            "central_material_section_force_N": float(section_force[central].mean()),
            "minimum_absolute_gauge_section_force_N": float(np.abs(section_force[gauge]).min()),
            "minimum_gauge_load_bearing_area_ratio": float(section_area_ratio[gauge].min()),
            "failed_gauge_sections": int(np.count_nonzero((active_count == 0) & gauge)),
            "central_active_particles": int(active[central_particles].sum()),
            "central_area_ratio_mean": float(area_ratio[central_particles].mean()),
            "central_geometric_rms_radius_m": float(
                np.sqrt(geometric_radius_squared[central]).mean()
            ),
            "central_geometric_rms_radius_ratio": float(rms_ratio[central].mean()),
            "minimum_intact_gauge_rms_radius_ratio": float(rms_ratio[intact_gauge].min())
            if intact_gauge.any()
            else 1.0,
            "intact_gauge_sections": int(intact_gauge.sum()),
            "load_bearing_axial_gap_m": float(gaps.max(initial=0.0)),
            "gap_exceeds_grid_support": bool(components > 1),
            "support_separated_components": components,
            "maximum_equivalent_plastic_strain": float(current.history.plastic_strain.max()),
            "maximum_temperature_K": float(current.history.temperature.max()),
            "max_damage": float(current.history.damage.max()),
            "mean_damage_volume_weighted": float(
                np.dot(reference_volume, np.asarray(current.history.damage))
                / reference_volume.sum()
            ),
            "failed_particles": int(failed.sum()),
            "failed_mass_kg": float(mass[failed].sum()),
            "failed_reference_volume_fraction": float(
                reference_volume[failed].sum() / reference_volume.sum()
            ),
            "total_mass_error_kg": float(mass.sum() - initial_mass),
            "momentum_x_kg_m_per_s": float(np.dot(mass, velocity[:, 0])),
            "kinetic_energy_J": float(0.5 * np.sum(mass[:, None] * velocity**2)),
            "plastic_work_J": float(np.dot(mass, np.asarray(current.history.energy))),
            "constitutive_work_J": float(np.dot(mass, np.asarray(current.history.work))),
            "grip_inner_particles": int(left_inner.sum() + right_inner.sum()),
            "grip_inner_velocity_error_m_per_s": float(max(velocity_error, default=0.0)),
            "grip_inner_axial_position_error_m": float(max(position_error, default=0.0)),
            "grip_inner_transverse_displacement_m": float(max(transverse_error, default=0.0)),
            "failed_grip_particles": int(failed[grip].sum()),
        }

    grips = []
    for lo, hi, side in (
        (CENTER - HALF_LENGTH - 1e-4, left_face, -1),
        (right_face, CENTER + HALF_LENGTH + 1e-4, 1),
    ):
        grips.append(
            {
                "lower": [lo, CENTER - 1e-3, CENTER - 1e-3],
                "upper": [hi, CENTER + 1e-3, CENTER + 1e-3],
                "velocity": [side * SPEED, 0.0, 0.0],
                "ramp_time": RAMP_TIME,
                "start": 0.0,
                "end": None,
            }
        )
    description = {
        "purpose": "steel circular-arc-waisted rod under symmetric dynamic tension",
        "series": "jc-necking-sphinxsys-pr1127-v1",
        "units": "m, s, kg, N, Pa",
        "source": "https://github.com/Xiangyu-Hu/SPHinXsys/pull/1127",
        "source_commit": SOURCE_COMMIT,
        "source_scope": "unmerged numerical implementation reference; not experimental validation",
        "material": "SPHinXsys PR 1127 verification steel",
        "bar_ends_x_m": [CENTER - HALF_LENGTH, CENTER + HALF_LENGTH],
        "bar_diameter_m": 2 * RADIUS,
        "waist_radius_m": WAIST_RADIUS,
        "transition_half_length_m": TRANSITION_LENGTH,
        "arc_radius_m": ARC_RADIUS,
        "radial_profile": "two tangent circular arcs per waist half, as necking_geometry.h",
        "gauge_length_m": gauge_length,
        "grip_inner_faces_x_m": [left_face, right_face],
        "requested_particles_per_cell_per_axis": ppc,
        "effective_particles_per_cell_per_axis": config.dx / spacing,
        "particle_spacing_m": spacing,
        "transverse_lattice_count": transverse_count,
        "reference_volume": "uniform spacing cubed; circular-arc outline clips particle centres",
        "particle_density_quantization": "round target diameter count upward to an even integer; axial count is four times diameter count",
        "grid_cells_across_nominal_diameter": 2 * RADIUS / config.dx,
        "grips": "opposed translating grid velocity boxes; all three velocity components prescribed",
        "grip_discretization": "grid constraints approximate the PR's reference-particle position/velocity constraints; quadratic support smooths the interface",
        "grip_inner_sampling": "initial centres at least 1.5 dx inside each translating grip face; full support remains constrained at any grid phase",
        "grip_axial_integration": "positions advance with prescribed end-step velocity; expected ramp quadrature error is O(speed*dt)",
        "grip_speed_m_per_s": SPEED,
        "ramp_time_s": RAMP_TIME,
        "nominal_strain_rate_plateau_per_s": 2 * SPEED / gauge_length,
        "duration_s": end_time,
        "reference_duration_s": DURATION,
        "acoustic_CFL_requested": cfl,
        "acoustic_CFL": config.dt * wave_speed / config.dx,
        "flow_integration": "backward-Euler radial return with current plastic rate and old temperature",
        "reference_flow_integration": "PR explicit hardening tangent uses the previous plastic rate",
        "failure_rate": "current equivalent plastic strain increment / dt",
        "reference_failure_rate": "PR current equivalent deviatoric total strain increment / dt",
        "heating": "plastic work / current density / cp; heat fraction one",
        "reference_heating": "PR uses reference density in the plastic-work heating update",
        "triaxiality": "mean tensile stress / von Mises stress; D3=-2.12",
        "failure_mode": "ballistic",
        "failure_interpretation": "D=1 removes stress and grid participation; mass retained as noncontacting ballistic debris; no fracture-energy regularization",
        "separation_measure": "empty reference material sections and active x gaps exceeding 3 dx",
        "section_force": "sum of sigma * cofactor(F) * e_x times reference volume / actual uniform spacing",
        "geometric_radius": "transverse RMS distance about each reference layer centroid; computed from all layer particles, including failed mass after rupture; ratio to initial RMS",
        "intact_radius": "minimum geometric RMS ratio among fully intact gauge layers; value one if none remain, with count reported",
        "reaction_sampling": "next-step grid impulse, including imposed grip inertia; forces are on the tools",
        "energy_scope": "kinetic, plastic and stress work are diagnostics; sampled actuator power is not an energy-closure or fracture-energy claim",
        "rigid_grips": grips,
    }
    return Case((state,), (material,), (config,), end_time, description, observer=observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--end-time", type=float, default=DURATION)
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=Path("outputs/material-engineering"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    try:
        case = make_case(
            grid=args.grid, ppc=args.ppc, frames=args.frames, end_time=args.end_time, cfl=args.cfl
        )
    except ValueError as error:
        parser.error(str(error))
    substeps = round(case.duration / (args.frames * case.configs[0].dt))
    print(
        f"jc-necking: particles={len(case.states[0].mass)}, grid={args.grid}, steps={args.frames * substeps}",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=args.output / "jc-necking" / "johnson-cook",
        label="jc-necking / johnson-cook",
        observer=case.observer,
        metadata=case.description,
        render_output=not args.no_render,
        gif=args.gif,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
