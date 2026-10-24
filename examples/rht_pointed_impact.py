"""Deformable pointed steel projectile through a four-edge-clamped RHT plate."""

import argparse
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, block, fixed_box, neo_hookean, rht
from .benchmarks import select
from .common import run
from .scenarios import Case


def make_case(*, grid=96, ppc=2, frames=48, speed=1800., duration=8.e-5,
              erosion_strain=.10):
    if grid < 48 or ppc < 1 or frames < 1 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("invalid impact resolution or projectile speed")
    if not np.isfinite(duration) or duration <= 0 or not 0 < erosion_strain < 1:
        raise ValueError("invalid duration or target erosion strain")

    domain = .36
    plate_lower = np.array((.06, .14, .06))
    plate_upper = np.array((.30, .17, .30))
    center = np.array((.18, .18))
    radius, cone_length, shaft_length = .012, .035, .045
    nose_y = .185
    cone_end = nose_y + cone_length
    tail_y = cone_end + shaft_length
    edge = .01
    concrete = rht(erosion_strain=erosion_strain)
    steel = neo_hookean(210e9, .3)
    target_wave = np.sqrt((concrete.A1 + 4 * concrete.shear / 3) / concrete.density)
    bullet_wave = np.sqrt((steel.bulk + 4 * steel.mu / 3) / 7850.)
    stable = .10 * domain / grid / (max(target_wave, bullet_wave) + speed)
    substeps = int(np.ceil(duration / (frames * stable)))
    dt = duration / (frames * substeps)
    clamps = (
        fixed_box((0., 0., 0.), (plate_lower[0] + edge, domain, domain)),
        fixed_box((plate_upper[0] - edge, 0., 0.), (domain, domain, domain)),
        fixed_box((0., 0., 0.), (domain, domain, plate_lower[2] + edge)),
        fixed_box((0., 0., plate_upper[2] - edge), (domain, domain, domain)),
    )
    target_config = Config(dim=3, n_grid=grid, domain_size=domain, dt=dt,
                           material="rht", gravity=(0., 0., 0.), boundary_cells=0,
                           boundaries=clamps, failure_mode="ballistic")
    projectile_config = replace(target_config, material="neo-hookean",
                                boundaries=(), failure_mode="retain")
    target = block(target_config, plate_lower, plate_upper, particles_per_cell=ppc,
                   density=concrete.density, material=concrete, dtype=jnp.float64)
    projectile = block(projectile_config,
                       (center[0] - radius, nose_y, center[1] - radius),
                       (center[0] + radius, tail_y, center[1] + radius),
                       particles_per_cell=ppc, density=7850., material=steel,
                       dtype=jnp.float64)
    position = np.asarray(projectile.x)
    radial = np.linalg.norm(position[:, (0, 2)] - center, axis=1)
    profile = radius * np.clip((position[:, 1] - nose_y) / cone_length, 0., 1.)
    projectile = select(projectile, radial <= profile)
    if not len(projectile.x) or not len(target.x):
        raise ValueError("impact geometry contains no material points")
    projectile = projectile._replace(v=jnp.broadcast_to(
        jnp.asarray((0., -speed, 0.), dtype=projectile.x.dtype), projectile.x.shape))
    reference = np.asarray(target.x)
    volume = np.asarray(target.volume)
    corridor = np.sum((reference[:, (0, 2)] - center) ** 2, axis=1) <= radius**2

    def observe(states, time):
        plate, bullet = states
        plate_x = np.asarray(plate.x)
        bullet_x = np.asarray(bullet.x)
        failed = np.asarray(plate.history.failed, dtype=bool)
        damage = np.asarray(plate.history.damage)
        mass = np.asarray(bullet.mass)
        return {
            "projectile_tip_y_m": float(bullet_x[:, 1].min()),
            "projectile_center_y_m": float(np.average(bullet_x[:, 1], weights=mass)),
            "projectile_center_vy_m_per_s": float(np.average(np.asarray(bullet.v)[:, 1], weights=mass)),
            "tip_past_front_m": float(plate_upper[1] - bullet_x[:, 1].min()),
            "tip_past_rear_m": float(plate_lower[1] - bullet_x[:, 1].min()),
            "target_failed_reference_volume_fraction": float(volume[failed].sum() / volume.sum()),
            "target_failed_count": int(failed.sum()),
            "target_maximum_damage": float(damage.max()),
            "corridor_intact_count": int(np.count_nonzero(corridor & ~failed)),
            "target_minimum_J": float(np.linalg.det(np.asarray(plate.F)).min()),
            "target_maximum_displacement_m": float(np.linalg.norm(plate_x - reference, axis=1).max()),
        }

    description = {
        "case": "rht-pointed-impact",
        "units": "m, s, kg, Pa",
        "model": "two deformable MLS-MPM bodies with separate-grid frictionless contact",
        "projectile": "elastic steel conical nose and cylindrical shaft",
        "projectile_density_kg_m3": 7850.,
        "projectile_young_Pa": 210e9,
        "projectile_poisson": .3,
        "projectile_speed_m_per_s": speed,
        "projectile_radius_m": radius,
        "projectile_cone_length_m": cone_length,
        "projectile_shaft_length_m": shaft_length,
        "plate_lower_m": plate_lower.tolist(),
        "plate_upper_m": plate_upper.tolist(),
        "four_edge_clamp_width_m": edge,
        "target_erosion_strain": erosion_strain,
        "failure_mode": "ballistic failed target points",
        "duration_s": duration,
        "frames": frames,
        "substeps_per_frame": substeps,
        "scope": "Exploratory high-speed perforation case; no IJIE fit or projectile plasticity claim",
    }
    return Case((target, projectile), (concrete, steel),
                (target_config, projectile_config), duration, description,
                observer=observe)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--speed", type=float, default=1800.)
    parser.add_argument("--duration", type=float, default=8.e-5)
    parser.add_argument("--erosion-strain", type=float, default=.10)
    parser.add_argument("--stop-after-frame", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/rht-pointed-impact"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    case = make_case(grid=args.grid, ppc=args.ppc, frames=args.frames,
                     speed=args.speed, duration=args.duration,
                     erosion_strain=args.erosion_strain)
    substeps = round(case.duration / (args.frames * case.configs[0].dt))
    run(case.states, case.materials, case.configs, frames=args.frames,
        substeps=substeps, output=args.output, label="RHT pointed impact",
        observer=case.observer, metadata=case.description, render_output=False,
        resume=args.resume, stop_after_frame=args.stop_after_frame)


if __name__ == "__main__":
    main()
