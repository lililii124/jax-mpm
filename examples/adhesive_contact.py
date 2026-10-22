"""Rigid sphere indentation and prescribed Hertz/JKR pressure footprints."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, block, fixed_box, neo_hookean
from mpm_jax.contact_laws import (
    RigidSphereBoundary,
    boundary_reaction,
    hertz_contact,
    jkr_contact,
    jkr_detachment,
    jkr_history,
    reduced_modulus,
    spherical_surface_load,
)
from .common import run
from .scenarios import Case


REFERENCES = [
    "https://doi.org/10.1098/rspa.1971.0141",
    "https://github.com/ContactEngineering/Adhesion/blob/"
    "d4590102abef8e7f392ab7456bf53a6b036d5090/Adhesion/ReferenceSolutions/JKR.py",
]


def make_case(scene="sphere-contact", *, grid=48, ppc=3, frames=40):
    if scene not in ("sphere-contact", "hertz-footprint", "jkr-footprint"):
        raise ValueError("unknown spherical contact example")
    young, poisson, density = 2e4, 0.25, 1000.0
    material = neo_hookean(young, poisson)
    wave_speed = np.sqrt((material.bulk + 4 * material.mu / 3) / density)
    duration = 0.3
    substeps = int(np.ceil(duration / (frames * 0.15 / grid / wave_speed)))
    dt = duration / (frames * substeps)
    lower, upper = (0.2, 0.15, 0.2), (0.8, 0.5, 0.8)
    fixed_base = fixed_box((0.0, 0.0, 0.0), (1.0, 0.15, 1.0))
    sphere = RigidSphereBoundary((0.5, 0.8, 0.5), 0.3, (0.0, -0.04, 0.0), duration)
    cfg = Config(
        dim=3,
        n_grid=grid,
        dt=dt,
        material="neo-hookean",
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        boundaries=(fixed_base, sphere) if scene == "sphere-contact" else (fixed_base,),
    )
    state = block(cfg, lower, upper, particles_per_cell=ppc, density=density, dtype=jnp.float64)
    reference = state.x
    top = reference[:, 1] == reference[:, 1].max()
    r2 = (reference[:, 0] - 0.5) ** 2 + (reference[:, 2] - 0.5) ** 2
    center_patch = top & (r2 < (2 * cfg.dx) ** 2)
    spacing = (upper[0] - lower[0]) / int(np.ceil((upper[0] - lower[0]) * grid * ppc))
    modulus = reduced_modulus(young, poisson)
    definition = {
        "purpose": scene,
        "sources": REFERENCES,
        "particles_per_cell_per_axis": ppc,
        "lower": lower,
        "upper": upper,
        "young_modulus_Pa": young,
        "poisson_ratio": poisson,
        "reduced_modulus_Pa": float(modulus),
        "base": "all velocity components fixed at y <= 0.15 m",
        "other_faces": "traction-free",
        "Hertz_comparison": "elastic half-space reference; finite block, finite strain and inertia remain",
    }
    forces = None
    if scene == "sphere-contact":
        definition.update(
            {
                "contact": "frictionless unilateral grid-node rigid sphere",
                "rigid_sphere": asdict(sphere),
                "rigid_sphere_motion": "center + displacement * (1-cos(2*pi*clip(t/duration,0,1)))/2",
                "reaction_sampling": "next-step P2G boundary impulse divided by dt, at each output frame",
                "peak_Hertz_contact_radius_in_grid_cells": float(np.sqrt(0.3 * 0.04) * grid),
            }
        )

        def observe(states, time):
            center, _ = sphere.motion(time + dt)
            indentation = upper[1] - (center[1] - sphere.radius)
            reference_contact = hertz_contact(indentation, sphere.radius, modulus)
            reaction = boundary_reaction(states[0], material, cfg)
            load = float(reaction[1])
            hertz_load = float(reference_contact.force)
            return {
                "sample_time": float(time + dt),
                "nominal_indentation_m": float(indentation),
                "indenter_reaction_x_N": float(reaction[0]),
                "indenter_reaction_y_N": load,
                "indenter_reaction_z_N": float(reaction[2]),
                "Hertz_reference_load_N": hertz_load,
                "Hertz_reference_contact_radius_m": float(reference_contact.radius),
                "Hertz_patch_grid_cells": float(reference_contact.radius / cfg.dx),
                "load_minus_Hertz_N": load - hertz_load,
                "relative_to_Hertz": (load - hertz_load) / hertz_load if hertz_load > 1.0 else None,
                "central_surface_displacement_y_m": float(
                    jnp.mean((states[0].x - reference)[center_patch, 1])
                ),
                "maximum_surface_displacement_y_m": float(
                    jnp.max(jnp.abs((states[0].x - reference)[top, 1]))
                ),
            }

    else:
        radius, work = 0.4, 3.0 if scene == "jkr-footprint" else 0.0
        _, detach = jkr_detachment(radius, modulus, max(work, 3.0))
        start_gap, peak = float(1.5 * detach), 0.025

        def nominal_indentation(time):
            phase = jnp.clip(time / duration, 0.0, 1.0)
            return start_gap + (peak - start_gap) * (1 - jnp.cos(2 * jnp.pi * phase)) / 2

        def contact_at(time):
            indentation = nominal_indentation(time)
            return (
                jkr_contact(indentation, radius, modulus, work, attached=time >= duration / 2)
                if work
                else hertz_contact(indentation, radius, modulus)
            )

        def load_at(time):
            return spherical_surface_load(
                reference,
                jnp.full(reference.shape[0], spacing**2),
                top,
                (0.5, 0.5, 0.5),
                (0.0, 1.0, 0.0),
                contact_at(time),
                spacing,
            )

        def forces(states, time):
            return (load_at(time).forces,)

        def observe(states, time):
            contact, load = contact_at(time), load_at(time)
            return {
                "nominal_law_indentation_m": float(nominal_indentation(time)),
                "prescribed_law_force_N": float(contact.force),
                "prescribed_contact_radius_m": float(contact.radius),
                "applied_force_y_N": float(-load.reaction[1]),
                "force_assignment_error_N": float(load.reaction[1] - contact.force),
                "footprint_resolved": bool(load.resolved),
                "loaded_surface_particles": int(load.loaded_particles),
                "contact_attached": bool(contact.active),
                "central_surface_displacement_y_m": float(
                    jnp.mean((states[0].x - reference)[center_patch, 1])
                ),
            }

        definition.update(
            {
                "contact": "prescribed reduced-law pressure footprint on the reference top surface",
                "geometric_adhesive_contact": False,
                "work_of_adhesion_J_per_m2": work,
                "indenter_radius_m": radius,
                "nominal_indentation_start_m": start_gap,
                "nominal_indentation_peak_m": peak,
                "pressure_edge_regularization_m": spacing,
                "compliance": "nominal law displacement is prescribed; MPM displacement is an independent response",
                "reaction_sampling": "prescribed external load; base reaction follows total momentum balance",
            }
        )
    case = Case((state,), (material,), (cfg,), duration, definition, forces, observe)
    return case, substeps


def write_law_cycle(output):
    radius, modulus, work = 0.4, float(reduced_modulus(2e4, 0.25)), 3.0
    _, detach = jkr_detachment(radius, modulus, work)
    half = np.linspace(float(1.5 * detach), 0.025, 401)
    indentation = np.r_[half, half[-2::-1]]
    result = jkr_history(jnp.asarray(indentation), radius, modulus, work)
    columns = np.column_stack((indentation, result.radius, result.force, result.active))
    np.savetxt(
        output / "jkr_law_cycle.csv",
        columns,
        delimiter=",",
        header="nominal_indentation_m,contact_radius_m,force_N,attached",
        comments="",
    )
    (output / "jkr_law_parameters.json").write_text(
        json.dumps({"radius_m": radius, "modulus_Pa": modulus, "work_J_per_m2": work}, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("sphere-contact", "hertz-footprint", "jkr-footprint"),
        default="sphere-contact",
    )
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--ppc", type=int, default=3)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("outputs/contact-models"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.grid < 8 or args.ppc < 1 or args.frames < 1:
        parser.error("grid >= 8, ppc >= 1 and frames >= 1 are required")
    jax.config.update("jax_enable_x64", True)
    case, substeps = make_case(args.case, grid=args.grid, ppc=args.ppc, frames=args.frames)
    folder = args.output / args.case
    print(
        f"{args.case}: {case.states[0].x.shape[0]} particles, {args.frames * substeps} steps",
        flush=True,
    )
    run(
        case.states,
        case.materials,
        case.configs,
        frames=args.frames,
        substeps=substeps,
        output=folder,
        label=args.case,
        forces=case.forces,
        observer=case.observer,
        metadata=case.description,
        gif=args.gif,
        render_output=not args.no_render,
        resume=args.resume,
    )
    write_law_cycle(folder)


if __name__ == "__main__":
    main()
