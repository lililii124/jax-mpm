"""Energy-controlled opening of an edge-cracked plate with a cohesive ligament."""

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

from mpm_jax import Config, MovingVelocityBox, block, neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw, Interface, response, step
from mpm_jax.materials import kirchhoff_stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=128)
    parser.add_argument('--duration', type=float, default=200e-6)
    parser.add_argument('--grip-speed', type=float, default=1.)
    parser.add_argument('--strength', type=float, default=12e6)
    parser.add_argument('--energy', type=float, default=200.)
    parser.add_argument('--stiffness', type=float, default=1e13)
    parser.add_argument('--cfl', type=float, default=.1)
    parser.add_argument('--frames', type=int, default=41)
    args = parser.parse_args()
    if args.duration <= 0 or args.frames < 2 or args.grip_speed < 0 or not 0 < args.cfl <= .15:
        parser.error('Require positive duration, >= 2 frames, nonnegative speed and CFL in (0, .15]')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64', True)
    young, poisson, density = 3e9, .2, 1200.
    dx = .06 / args.grid
    speed = np.sqrt(young*(1-poisson)/((1+poisson)*(1-2*poisson)*density))
    total = int(np.ceil(args.duration/(args.cfl*dx/speed)))
    dt = args.duration/total
    config = Config(dim=2, n_grid=args.grid, domain_size=.06, dt=dt, material='neo-hookean',
                    gravity=(0., 0.), boundary_cells=0, boundaries=(
                        MovingVelocityBox((.008, .018), (.052, .022), (0., -args.grip_speed), ramp_time=20e-6),
                        MovingVelocityBox((.008, .038), (.052, .042), (0., args.grip_speed), ramp_time=20e-6)))
    material = neo_hookean(young, poisson)
    state = block(config, (.01, .02), (.05, .04), particles_per_cell=2, density=density, dtype=jnp.float64)
    reference = np.asarray(state.x)
    law = CohesiveLaw(args.strength, args.energy, args.stiffness)
    pieces = []
    for left, right in ((.01, .022), (.022, .05)):
        count = int(np.ceil((right-left)/(dx/2)))
        pieces.append((left+(np.arange(count)+.5)*(right-left)/count, np.full(count, (right-left)/count)))
    qx = np.concatenate([p[0] for p in pieces])
    area = np.concatenate([p[1] for p in pieces])
    q = jnp.asarray(np.column_stack((qx, np.full(len(qx), .03))))
    interface = Interface(q, q, jnp.zeros(len(qx)), jnp.asarray(area), jnp.asarray(qx >= .022), jnp.asarray(True))
    segment = jnp.asarray([[.003, .03], [.057, .03]])

    @jax.jit
    def advance(state, interface, work, count):
        def body(_, carry):
            s, face, accumulated = carry
            s, face, increment = step(s, face, material, config, segment, law)
            summed = accumulated+jnp.stack(increment[:9])
            summed = summed.at[-1].set(jnp.maximum(accumulated[-1], increment.iteration_residual))
            return s, face, summed
        return jax.lax.fori_loop(0, count, body, (state, interface, work))

    args.output.mkdir(parents=True)
    previous, rows = 0, []
    work = jnp.zeros(9)
    started = time.perf_counter()
    with (args.output/'history.csv').open('w', newline='') as stream:
        writer = None
        for frame, target in enumerate(np.unique(np.linspace(0, total, args.frames).round().astype(int))):
            if target > previous:
                state, interface, work = advance(state, interface, work, jnp.asarray(target-previous))
            x, v, F, C = map(np.asarray, (state.x, state.v, state.F, state.C))
            J = np.linalg.det(F)
            tau = np.asarray(kirchhoff_stress(state.F, material, config.material))
            stress = tau/J[:, None, None]
            law_state = response(interface.plus-interface.minus, interface.maximum,
                                 jnp.asarray([0., 1.]), interface.bonded, law)
            damage = np.asarray(law_state.damage)
            free = (qx < .022) | (damage >= 1.-1e-8)
            contiguous = np.cumprod(free)
            tip = .01 + float(np.sum(area*contiguous))
            kinetic = .5*np.sum(np.asarray(state.mass)*(np.sum(v*v, axis=1) + .25*dx**2*np.sum(C*C, axis=(1, 2))))
            W = .5*material.mu*(np.sum(F*F, axis=(1, 2))-2) - material.mu*np.log(J) + .5*material.lam*np.log(J)**2
            elastic_energy = float(np.sum(np.asarray(state.volume)*W))
            stored = float(jnp.sum(interface.area*law_state.stored))
            dissipated = float(jnp.sum(interface.area*law_state.dissipated))
            external = float(work[0]+work[1])
            residual = external-kinetic-elastic_energy-stored-dissipated
            displacement = x-reference
            mouth = (reference[:, 0] < .02) & (abs(reference[:, 1]-.03) < .0005)
            opening = np.mean(displacement[mouth & (reference[:, 1]>.03), 1])-np.mean(displacement[mouth & (reference[:, 1]<.03), 1])
            row = dict(step=int(target), time_s=float(state.time), tip_x_m=tip,
                       failed_ligament_m=float(np.sum(area*(qx>=.022)*(damage>=1.-1e-8))),
                       opening_m=float(opening), external_J_per_m=external, kinetic_J_per_m=float(kinetic),
                       elastic_J_per_m=elastic_energy, cohesive_stored_J_per_m=stored,
                       fracture_J_per_m=dissipated, residual_J_per_m=float(residual),
                       residual_fraction=float(residual/max(abs(external), 1e-12)),
                       cohesive_integration_error_J_per_m=float(work[3]), min_J=float(J.min()),
                       transfer_loss_J_per_m=float(work[4]), bulk_error_J_per_m=float(work[5]),
                       cohesive_grid_error_J_per_m=float(work[6]),
                       unexplained_J_per_m=float(residual-work[4]),
                       mean_iterations=float(work[7]/max(target, 1)), max_iteration_residual=float(work[8]),
                       max_strain=float(np.max(np.abs(.5*(F+F.swapaxes(-1, -2))-np.eye(2)))),
                       valid=bool(state.valid & interface.valid), elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if not row['valid']:
                raise RuntimeError(f'Invalid coupled state at step {target}')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz', x=x, F=F, stress_yy=stress[:, 1, 1],
                                displacement=displacement, crack=np.asarray([[.01, .03], [tip, .03]]),
                                interface_plus=np.asarray(interface.plus), interface_minus=np.asarray(interface.minus),
                                interface_damage=damage, time=float(state.time))
            previous = target
    summary = dict(model='2D neo-Hookean MPM with energy-controlled cohesive ligament',
                   law=asdict(law), young_Pa=young, poisson=poisson, density_kg_m3=density,
                   grid=args.grid, dx_m=dx, dt_s=dt, cfl=args.cfl, particle_count=len(reference),
                   particle_spacing_m=[float(np.diff(np.unique(reference[:, d])).mean()) for d in (0, 1)],
                   grip_speed_m_s=args.grip_speed, initial_crack_m=.012,
                   frames=len(rows), steps=total, all_states_valid=all(r['valid'] for r in rows),
                   final=rows[-1], max_sampled_strain=max(r['max_strain'] for r in rows),
                   elapsed_s=time.perf_counter()-started,
                   limitations=['Straight prescribed potential interface; failure times are computed',
                                'Fixed plane normal and reference interface area',
                                'Equal mode-I/mode-II parameters; no calibrated experimental material'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
