"""Edge-cracked plate under symmetric mode-I opening (experimental strength law)."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, block, linear_elastic
from mpm_jax.crack_mpm import TensileTipLaw, advance_straight_tip, step
from mpm_jax.materials import kirchhoff_stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--duration', type=float, default=200e-6)
    parser.add_argument('--grip-speed', type=float, default=2.)
    parser.add_argument('--strength', type=float, default=12e6)
    parser.add_argument('--frames', type=int, default=41)
    args = parser.parse_args()
    if args.duration <= 0 or args.frames < 2 or args.grip_speed < 0:
        parser.error('Positive duration, >= 2 frames and nonnegative speed required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64', True)
    young, poisson, density = 3e9, .2, 1200.
    dx = .06 / args.grid
    wave_speed = np.sqrt(young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson) * density))
    dt = .15 * dx / wave_speed
    total = int(np.ceil(args.duration / dt))
    dt = args.duration / total
    config = Config(
        dim=2, n_grid=args.grid, domain_size=.06, dt=dt, material='linear',
        gravity=(0., 0.), boundary_cells=0,
        boundaries=(
            MovingVelocityBox((.008, .018), (.052, .022), (0., -args.grip_speed), ramp_time=20e-6),
            MovingVelocityBox((.008, .038), (.052, .042), (0., args.grip_speed), ramp_time=20e-6),
        ),
    )
    material = linear_elastic(young, poisson)
    state = block(config, (.01, .02), (.05, .04), particles_per_cell=2, density=density, dtype=jnp.float64)
    reference = np.asarray(state.x)
    segment = jnp.asarray([[.009, .03], [.022, .03]], dtype=state.x.dtype)
    law = TensileTipLaw(args.strength, .0015, .3 * wave_speed, .05 - .009)

    @jax.jit
    def advance(state, segment, exited, count):
        def body(_, carry):
            current, crack, free_end = carry
            current = step(current, material, config, crack, end_exited=free_end)
            new_crack, report = advance_straight_tip(current, material, config, crack, law)
            crack = jnp.where(free_end, crack, new_crack)
            return current, crack, free_end | report.exited
        return jax.lax.fori_loop(0, count, body, (state, segment, exited))

    args.output.mkdir(parents=True)
    snapshots = np.unique(np.linspace(0, total, args.frames).round().astype(int))
    started = time.perf_counter()
    rows = []
    previous = 0
    exited = jnp.asarray(False)
    with (args.output / 'history.csv').open('w', newline='') as stream:
        writer = None
        for frame, target in enumerate(snapshots):
            if target > previous:
                state, segment, exited = advance(state, segment, exited, jnp.asarray(target - previous))
            x, v, F, crack = map(np.asarray, (state.x, state.v, state.F, segment))
            stress = np.asarray(kirchhoff_stress(state.F, material, config.material) / jnp.linalg.det(state.F)[:, None, None])
            _, report = advance_straight_tip(state, material, config, segment, law)
            strain = .5 * (F + np.swapaxes(F, -1, -2)) - np.eye(2)
            near = (reference[:, 0] < .02) & (abs(reference[:, 1] - .03) < .0005)
            upper = near & (reference[:, 1] > .03)
            lower = near & (reference[:, 1] < .03)
            displacement = x - reference
            opening = displacement[upper, 1].mean() - displacement[lower, 1].mean()
            row = dict(step=int(target), time_s=float(state.time), tip_x_m=float(crack[1, 0]),
                       tip_traction_Pa=float(report.traction), opening_m=float(opening),
                       kinetic_J_per_m=float(.5 * np.sum(np.asarray(state.mass)[:, None] * v*v)),
                       elastic_J_per_m=float(.5 * np.sum(np.asarray(state.volume) * np.sum(stress * strain, axis=(1, 2)))),
                       max_strain=float(np.max(np.abs(strain))), min_J=float(np.linalg.det(F).min()),
                       valid=bool(state.valid), exited=bool(exited), elapsed_s=time.perf_counter() - started)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if not row['valid']:
                raise RuntimeError(f'Invalid state at step {target}; stopping without publishing')
            np.savez_compressed(args.output / f'frame-{frame:03d}.npz',
                                x=x, stress_yy=stress[:, 1, 1], displacement=displacement,
                                crack=crack, F=F, time=float(state.time), exited=bool(exited))
            print(json.dumps(row), flush=True)
            previous = target
    summary = dict(model='2D plane-strain elastic MPM; straight explicit crack; experimental tensile tip kinetics',
                   source_method='https://github.com/titoadibaskoro/mpm-fracture',
                   particle_count=len(reference), grid=args.grid, dx_m=dx, dt_s=dt,
                   young_Pa=young, poisson=poisson, density_kg_m3=density, tip_law=asdict(law),
                   grip_speed_m_s=args.grip_speed, frames=len(rows), steps=total,
                   all_states_valid=all(r['valid'] for r in rows),
                   extension_m=rows[-1]['tip_x_m'] - rows[0]['tip_x_m'],
                   final_opening_m=rows[-1]['opening_m'], elapsed_s=time.perf_counter() - started,
                   crack_exited=bool(exited),
                   particle_spacing_m=[float(np.diff(np.unique(reference[:, d])).mean()) for d in (0, 1)],
                   limitations=['Fixed straight crack centreline and small-strain constitutive law',
                                'Strength-driven demonstration, not calibrated fracture-energy validation',
                                'No particle deletion, preassigned opening, or visual displacement scaling'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
