"""Edge-clamped JH-2 ceramic plate under a prescribed spherical impactor."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import block, fixed_box
from mpm_jax.contact_laws import RigidSphereBoundary, boundary_reaction
from .brittle_engineering import _config, _damage_observables, _definition
from .common import run
from .scenarios import Case


def make_case(*, grid=96, ppc=2, frames=60, stroke=.012, failure_strain=.05):
    if grid < 48 or ppc < 1 or frames < 1 or not 0 < stroke < .02:
        raise ValueError("invalid ceramic impact resolution or stroke")
    if not 0 < failure_strain < 1:
        raise ValueError("failure strain must be between zero and one")
    domain, duration = .25, 3e-4
    lower, upper = (.035, .075, .035), (.215, .1, .215)
    edge = .008
    clamps = (
        fixed_box((0., 0., 0.), (lower[0] + edge, domain, domain)),
        fixed_box((upper[0] - edge, 0., 0.), (domain, domain, domain)),
        fixed_box((0., 0., 0.), (domain, domain, lower[2] + edge)),
        fixed_box((0., 0., upper[2] - edge), (domain, domain, domain)),
    )
    sphere = RigidSphereBoundary((.125, .1375, .125), .0375, (0., -stroke, 0.), duration)
    material, density, config, wave_speed = _config(
        "jh2", grid, domain, duration, frames, (*clamps, sphere)
    )
    material = material._replace(failure_strain=failure_strain)
    config = replace(config, failure_mode="ballistic")
    state = block(config, lower, upper, particles_per_cell=ppc,
                  density=density, material=material, dtype=jnp.float64)
    reference = np.asarray(state.x)
    volume = np.asarray(state.volume)
    top = np.isclose(reference[:, 1], reference[:, 1].max(), rtol=0, atol=1e-12)
    central = top & ((reference[:, 0] - .125)**2 + (reference[:, 2] - .125)**2 <= .01**2)
    if not central.any():
        raise ValueError("central surface has no sample particles")
    definition = _definition(
        "ceramic-fragmentation", "jh2", material, config, ppc,
        lower, upper, duration, wave_speed,
    )
    definition.update({
        "loading_regime": "prescribed dynamic spherical impact and withdrawal",
        "boundary_condition": "four 8 mm edge strips fixed in all velocity components; no backing",
        "failure_rule": "JH-2 equivalent plastic strain threshold; failed points coast without grid or tool contact",
        "failure_strain": failure_strain,
        "rigid_sphere": asdict(sphere),
        "rigid_sphere_motion": "one full cosine cycle over 300 microseconds",
        "maximum_nominal_indentation_m": stroke,
        "peak_indenter_speed_m_per_s": float(np.pi * stroke / duration),
        "plate_width_m": .18,
        "plate_thickness_m": .025,
        "edge_clamp_width_m": edge,
        "particle_spacing_m": float((np.asarray(upper) - lower).min() /
                                     np.ceil(.025 / config.dx * ppc)),
        "fragment_interpretation": (
            "ballistic failed material is noncontacting debris; cohesive chunks, "
            "crack-face contact, size-independent fracture energy and measured ceramic calibration are absent"
        ),
    })

    def observe(states, time):
        current = states[0]
        result = _damage_observables(current)
        positions = np.asarray(current.x)
        failed = np.asarray(current.history.failed)
        center, velocity = sphere.motion(time)
        result.update({
            "boundary_sample_time_s": float(time + config.dt),
            "indenter_reaction_y_N": float(boundary_reaction(
                current, material, config, boundary_index=4)[1]),
            "nominal_indentation_m": float(upper[1] - (center[1] - sphere.radius)),
            "indenter_speed_y_m_per_s": float(velocity[1]),
            "central_surface_displacement_y_m": float(
                (positions[central, 1] - reference[central, 1]).mean()),
            "failed_mass_kg": float(np.asarray(current.mass)[failed].sum()),
            "failed_reference_volume_m3": float(volume[failed].sum()),
            "failed_particle_count": int(failed.sum()),
            "maximum_particle_displacement_m": float(np.linalg.norm(positions - reference, axis=1).max()),
        })
        return result

    return Case((state,), (material,), (config,), duration, definition, observer=observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stroke", type=float, default=.012)
    parser.add_argument("--failure-strain", type=float, default=.05)
    parser.add_argument("--stop-after-frame", type=int)
    parser.add_argument("--output", type=Path, default=Path("outputs/ceramic-fragmentation/jh2"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    case = make_case(grid=args.grid, ppc=args.ppc, frames=args.frames,
                     stroke=args.stroke, failure_strain=args.failure_strain)
    substeps = round(case.duration / (args.frames * case.configs[0].dt))
    run(case.states, case.materials, case.configs, frames=args.frames,
        substeps=substeps, output=args.output, label="JH-2 ceramic fragmentation",
        observer=case.observer, metadata=case.description,
        render_output=False, resume=args.resume, stop_after_frame=args.stop_after_frame)


if __name__ == "__main__":
    main()
