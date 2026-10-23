"""Metal tension, concrete bending and ceramic indentation in SI units."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, block, johnson_cook
from mpm_jax.contact_laws import boundary_reaction
from .benchmarks import select
from .common import run
from .scenarios import Case


SCENES = ("jc-tension", "concrete-bending", "ceramic-indentation")


def tensile_aluminum():
    return johnson_cook(
        young=73.1e9, poisson=0.33, A=265e6, B=426e6, n=0.34, C=0.015, m=1.0,
        reference_rate=1.0, reference_temperature=298.0, melt_temperature=775.15,
        density=2770.0, heat_capacity=875.0, heat_fraction=0.9,
        failure=(0.13, 0.13, -1.5, 0.011, 0.0),
    )


def jc_tension(*, grid=64, ppc=3, frames=60):
    material = tensile_aluminum()
    domain, duration = 0.1, 3e-4
    center, radius = domain / 2, 0.0075
    ends, grip_faces = (0.02, 0.08), (0.03, 0.07)
    gauge_length = grip_faces[1] - grip_faces[0]
    extension = 0.45 * gauge_length
    ramp_time = duration / 3
    speed = 0.5 * extension / (duration - ramp_time / 2)
    left = MovingVelocityBox(
        (0.0, 0.0, 0.0), (grip_faces[0], domain, domain),
        (-speed, 0.0, 0.0), components=(True, False, False), ramp_time=ramp_time,
    )
    right = MovingVelocityBox(
        (grip_faces[1], 0.0, 0.0), (domain, domain, domain),
        (speed, 0.0, 0.0), components=(True, False, False), ramp_time=ramp_time,
    )
    wave_speed = np.sqrt((material.bulk + 4 * material.mu / 3) / material.density)
    substeps = int(np.ceil(duration / (frames * 0.15 * domain / (grid * wave_speed))))
    config = Config(
        dim=3, n_grid=grid, domain_size=domain, dt=duration / (frames * substeps),
        material="johnson-cook", gravity=(0.0, 0.0, 0.0), boundary_cells=0,
        boundaries=(left, right), failure_mode="ballistic",
    )
    state = block(
        config, (ends[0], center - radius, center - radius),
        (ends[1], center + radius, center + radius), particles_per_cell=ppc,
        material=material, density=material.density, dtype=jnp.float64,
    )
    radial = np.asarray(state.x)[:, 1:] - center
    state = select(state, np.sum(radial**2, axis=1) < radius**2)
    reference = np.asarray(state.x).copy()
    imperfection = 0.005
    width = gauge_length / 6
    radial_scale = 1 - imperfection * np.exp(-((reference[:, 0] - center) / width) ** 2)
    reference[:, 1:] = center + radial_scale[:, None] * (reference[:, 1:] - center)
    state = state._replace(
        x=jnp.asarray(reference),
        mass=state.mass * jnp.asarray(radial_scale**2),
        volume=state.volume * jnp.asarray(radial_scale**2),
    )
    axial_levels, sections = np.unique(reference[:, 0], return_inverse=True)
    axial_spacing = float(axial_levels[1] - axial_levels[0])
    middle_sections = np.abs(axial_levels - center) <= 0.51 * axial_spacing
    initial_count = np.bincount(sections)
    gauge_sections = (axial_levels > grip_faces[0] + config.dx) & (
        axial_levels < grip_faces[1] - config.dx
    )
    initial_area = np.pi * radius**2
    initial_mass = float(state.mass.sum())
    center_reference = np.abs(reference[:, 0] - center) <= config.dx

    def observe(states, time):
        current = states[0]
        position, deformation, mass, volume = (
            np.asarray(value) for value in (current.x, current.F, current.mass, current.volume)
        )
        failed = np.asarray(current.history.failed)
        active = ~failed
        stretch = np.linalg.norm(deformation[:, :, 0], axis=1)
        area_ratio = np.linalg.det(deformation) / stretch
        section_count = np.bincount(sections, weights=active, minlength=len(axial_levels))
        section_area_ratio = np.bincount(
            sections, weights=active * area_ratio, minlength=len(axial_levels)
        ) / initial_count
        broken = (section_count == 0) & gauge_sections
        axial_gaps = np.diff(np.sort(position[active, 0]))
        gap = float(axial_gaps.max(initial=0.0))
        component_count = int(active.any()) + int(np.count_nonzero(axial_gaps > 3 * config.dx))
        cofactor_normal = (
            np.linalg.det(deformation)[:, None] * np.swapaxes(np.linalg.inv(deformation), 1, 2)[:, :, 0]
        )
        traction_x = np.einsum(
            "pi,pi->p", np.asarray(current.history.stress)[:, 0, :], cofactor_normal
        )
        section_forces = np.bincount(
            sections, weights=active * traction_x * volume / axial_spacing,
            minlength=len(axial_levels),
        )
        reaction_left = np.asarray(boundary_reaction(current, material, config, boundary_index=0))
        reaction_right = np.asarray(boundary_reaction(current, material, config, boundary_index=1))
        load = 0.5 * (reaction_left[0] - reaction_right[0])
        left_displacement, _ = left.motion(time)
        right_displacement, _ = right.motion(time)
        displacement = float(right_displacement[0] - left_displacement[0])
        return {
            "prescribed_extension_m": displacement,
            "engineering_strain": displacement / gauge_length,
            "mean_grip_tensile_force_N": float(load),
            "nominal_grip_stress_Pa": float(load / initial_area),
            "central_material_section_force_N": float(section_forces[middle_sections].mean()),
            "minimum_absolute_gauge_section_force_N": float(np.abs(section_forces[gauge_sections]).min()),
            "left_grip_reaction_x_N": float(reaction_left[0]),
            "right_grip_reaction_x_N": float(reaction_right[0]),
            "boundary_sample_time_s": float(time + config.dt),
            "minimum_gauge_load_bearing_area_ratio": float(section_area_ratio[gauge_sections].min()),
            "failed_gauge_sections": int(broken.sum()),
            "load_bearing_axial_gap_m": gap,
            "gap_exceeds_grid_support": bool(component_count > 1),
            "support_separated_components": component_count,
            "central_area_ratio_mean": float(area_ratio[center_reference].mean()),
            "maximum_equivalent_plastic_strain": float(current.history.plastic_strain.max()),
            "max_damage": float(current.history.damage.max()),
            "failed_mass_kg": float(mass[failed].sum()),
            "failed_reference_volume_fraction": float(volume[failed].sum() / volume.sum()),
            "total_mass_error_kg": float(mass.sum() - initial_mass),
            "momentum_x_kg_m_per_s": float(np.dot(mass, np.asarray(current.v)[:, 0])),
        }

    grips = []
    for lower_x, upper_x, sign in ((0.017, 0.03, -1), (0.07, 0.083, 1)):
        grips.append({
            "lower": [lower_x, center - 0.011, center - 0.011],
            "upper": [upper_x, center + 0.011, center + 0.011],
            "velocity": [sign * speed, 0.0, 0.0],
            "ramp_time": ramp_time, "start": 0.0, "end": None,
        })
    definition = {
        "purpose": "dynamic tension of a smooth round bar through JC failure and separation",
        "units": "m, s, kg, N, Pa",
        "particles_per_cell_per_axis": ppc,
        "material": "AA2024-T351 Johnson-Cook parameters reproduced from FAA Table 6",
        "material_source": "https://ntlrepository.blob.core.windows.net/lib/17000/17400/17488/PB2001101864.pdf",
        "material_source_table": "DOT/FAA/AR-00/25, Table 6, printed p.17 (PDF p.29)",
        "thermal_source": "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1022/Kaiser_Aluminum_2024_Rod_and_Bar.pdf",
        "parameter_assumptions": "nu=0.33, plastic work heat fraction=0.9, constant cp=875 J/(kg K)",
        "calibration_scope": (
            "published constitutive parameters; specimen, rate and fracture response are an "
            "illustration, not a tensile-test calibration; rate exceeds the reported original torsion fit"
        ),
        "triaxiality": "mean tensile stress / von Mises stress; D3=-1.5",
        "bar_ends_x_m": list(ends), "bar_diameter_m": 2 * radius,
        "gauge_length_m": gauge_length,
        "radial_imperfection_fraction": imperfection,
        "radial_profile": "R(x)=R0*(1-0.005*exp(-((x-center)/(gauge_length/6))**2))",
        "imperfection_scope": "smooth geometric imperfection to select a neck; no cut, notch or initial damage",
        "reference_volume": "particle volumes and masses scaled with the squared radial profile",
        "grips": "opposed translating axial velocity constraints; transverse velocities unconstrained",
        "ramp_time_s": ramp_time, "grip_speed_m_per_s": speed,
        "nominal_strain_rate_plateau_per_s": 2 * speed / gauge_length,
        "final_engineering_strain": extension / gauge_length,
        "duration_s": duration,
        "acoustic_CFL": float(config.dt * wave_speed / config.dx),
        "failure_mode": "ballistic",
        "failure_interpretation": (
            "D=1 removes load carrying capacity and grid participation; mass is retained in "
            "noncontacting ballistic debris; no fracture-energy regularization or crack-face contact"
        ),
        "separation_measure": (
            "empty load-bearing material section; sorted active x-coordinate gaps exceeding "
            "3 grid cells conservatively identify groups without shared quadratic support"
        ),
        "reaction_sampling": "next-step grid constraint force on each grip; same dt as integration",
        "section_force": "sum of sigma * cofactor(F) * e_x times reference volume / reference layer thickness",
        "rigid_grips": grips,
    }
    return Case((state,), (material,), (config,), duration, definition, observer=observe)


def setup(scene, *, grid=64, ppc=3, frames=60, model=None):
    if not isinstance(grid, int) or grid < 24:
        raise ValueError("material engineering cases require grid >= 24")
    if not isinstance(ppc, int) or ppc < 1 or not isinstance(frames, int) or frames < 1:
        raise ValueError("ppc and frames must be positive integers")
    if scene == "jc-tension":
        if model not in (None, "johnson-cook"):
            raise ValueError("jc-tension requires johnson-cook")
        return jc_tension(grid=grid, ppc=ppc, frames=frames)
    from .brittle_engineering import setup as brittle_setup

    return brittle_setup(scene, grid=grid, ppc=ppc, frames=frames, model=model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=SCENES, default="jc-tension")
    parser.add_argument("--model", choices=("johnson-cook", "hjc", "jh2", "rht"))
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("outputs/material-engineering"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    try:
        case = setup(args.case, grid=args.grid, ppc=args.ppc, frames=args.frames, model=args.model)
    except ValueError as error:
        parser.error(str(error))
    model = case.configs[0].material
    substeps = round(case.duration / (args.frames * case.configs[0].dt))
    print(f"{args.case}/{model}: grid={args.grid}, steps={args.frames * substeps}", flush=True)
    run(
        case.states, case.materials, case.configs, frames=args.frames, substeps=substeps,
        output=args.output / args.case / model, label=f"{args.case} / {model}",
        observer=case.observer, metadata=case.description, render_output=not args.no_render,
        gif=args.gif, resume=args.resume,
    )


if __name__ == "__main__":
    main()
