"""Double-cantilever opening with localized grips and a cohesive ligament."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, VelocityBox, block, neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw, Interface, elastic_energy, kinetic_energy, response, step
from mpm_jax.materials import kirchhoff_stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--grid', type=int, default=160)
    parser.add_argument('--duration', type=float, default=.002)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--ramp', type=float, default=.0002)
    parser.add_argument('--energy', type=float, default=2.)
    parser.add_argument('--strength', type=float, default=120e3)
    parser.add_argument('--frames', type=int, default=41)
    args = parser.parse_args()
    if args.grid < 128 or args.duration <= 0 or args.speed < 0 or args.ramp < 0 or args.frames < 2:
        parser.error('Require grid >=128, positive duration, nonnegative speed/ramp and >=2 frames')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64', True)
    young, nu, density = 30e6, .2, 1200.
    dx = .08/args.grid
    wave_speed = np.sqrt(young*(1-nu)/((1+nu)*(1-2*nu)*density))
    total = int(np.ceil(args.duration/(.1*dx/wave_speed)))
    dt = args.duration/total
    upper = MovingVelocityBox((.009, .04125), (.012, .044), (0., args.speed),
                              components=(False, True), ramp_time=args.ramp)
    lower = MovingVelocityBox((.009, .036), (.012, .03875), (0., -args.speed),
                              components=(False, True), ramp_time=args.ramp)
    anchor = VelocityBox((.068, .034), (.072, .046), (0., 0.), components=(True, False))
    config = Config(dim=2, n_grid=args.grid, dt=dt, domain_size=.08, gravity=(0., 0.),
                    boundary_cells=0, material='neo-hookean', boundaries=(upper, lower, anchor))
    material = neo_hookean(young, nu)
    law = CohesiveLaw(args.strength, args.energy, 1e11)
    state = block(config, (.01, .037), (.07, .043), particles_per_cell=2,
                  density=density, dtype=jnp.float64)
    reference = np.asarray(state.x)
    qx, area = [], []
    for left, right in ((.01, .035), (.035, .07)):
        count = int(np.ceil((right-left)/(dx/2)))
        qx.extend(left+(np.arange(count)+.5)*(right-left)/count)
        area.extend(np.full(count, (right-left)/count))
    qx, area = np.asarray(qx), np.asarray(area)
    q = jnp.asarray(np.column_stack((qx, np.full(len(qx), .04))))
    interface = Interface(q, q, jnp.zeros(len(qx)), jnp.asarray(area), jnp.asarray(qx>=.035), jnp.asarray(True))
    segment = jnp.asarray([[.003, .04], [.077, .04]])
    grip_upper = (reference[:, 0]<.012) & (reference[:, 1]>=.04125)
    grip_lower = (reference[:, 0]<.012) & (reference[:, 1]<=.03875)
    metadata = dict(case='Localized double-cantilever opening', grid=args.grid, dx_m=dx, dt_s=dt,
                    young_Pa=young, poisson=nu, density_kg_m3=density, law=asdict(law),
                    lower_m=[.01, .037], upper_m=[.07, .043], initial_tip_x_m=.035,
                    grip_speed_m_s=args.speed, ramp_s=args.ramp, duration_s=args.duration,
                    midpoint_boundaries=True, boundary_conditions=[asdict(b) for b in config.boundaries],
                    particle_count=len(reference),
                    particle_spacing_m=[float(np.diff(np.unique(reference[:, d])).mean()) for d in (0, 1)])
    source_paths = ['examples/cohesive_crack_dcb.py', 'mpm_jax/cohesive_crack.py', 'mpm_jax/crack_mpm.py', 'mpm_jax/core.py', 'mpm_jax/boundaries.py']
    root = Path(__file__).resolve().parents[1]
    metadata['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in source_paths}
    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(metadata, indent=2)+'\n')

    @jax.jit
    def advance(state, interface, accumulated, count):
        def body(_, carry):
            def active(carry):
                s, face, work = carry
                s, face, increment = step(s, face, material, config, segment, law, midpoint_boundaries=True)
                new_work = work+jnp.stack(increment)
                new_work = new_work.at[8].set(jnp.maximum(work[8], increment.iteration_residual))
                return s, face, new_work
            return jax.lax.cond(carry[0].valid & carry[1].valid, active, lambda x: x, carry)
        return jax.lax.fori_loop(0, count, body, (state, interface, accumulated))

    started = time.perf_counter()
    rows, previous = [], 0
    work = jnp.zeros(13)
    previous_impulse = np.zeros(3)
    with (args.output/'history.csv').open('w', newline='') as stream:
        writer = None
        for frame, target in enumerate(np.unique(np.linspace(0, total, args.frames).round().astype(int))):
            if target > previous:
                state, interface, work = advance(state, interface, work, jnp.asarray(target-previous))
            x, F = np.asarray(state.x), np.asarray(state.F)
            stress = np.asarray(kirchhoff_stress(state.F, material, config.material)/jnp.linalg.det(state.F)[:, None, None])
            r = response(interface.plus-interface.minus, interface.maximum, jnp.asarray([0., 1.]), interface.bonded, law)
            damage = np.asarray(r.damage)
            prefix = np.cumprod((qx<.035) | (damage>=1-1e-8))
            tip = .01+np.sum(area*prefix)
            failed_length = np.sum(area*(qx>=.035)*(damage>=1-1e-8))
            softened = (qx>=.035) & (damage>1e-3)
            soft_front = float(np.max(np.where(softened, qx+.5*area, .035)))
            current_time = float(state.time)
            K, U = float(kinetic_energy(state, config)), float(elastic_energy(state, material, config))
            surface = float(jnp.sum(interface.area*r.stored))
            fracture = float(jnp.sum(interface.area*r.dissipated))
            external, numerical = float(work[0]+work[1]), float(work[4]+work[12])
            delta = x-reference
            observed = np.mean(delta[grip_upper, 1])-np.mean(delta[grip_lower, 1])
            prescribed = float(2*upper.motion(state.time)[0][1])
            impulses = np.asarray(work[9:12])
            reactions = (impulses-previous_impulse)/((target-previous)*dt) if target>previous else np.zeros(3)
            momentum = np.asarray(jnp.sum(state.mass[:, None]*state.v, axis=0))
            imbalance = momentum-np.asarray([impulses[2], impulses[0]+impulses[1]])
            row = dict(step=int(round(current_time/dt)), time_s=current_time, tip_x_m=float(tip),
                       failed_ligament_m=float(failed_length), disconnected_failed_m=float(failed_length-(tip-.035)),
                       softening_front_x_m=soft_front, prescribed_opening_m=prescribed, observed_grip_opening_m=float(observed),
                       upper_force_N_per_m=float(reactions[0]), lower_force_N_per_m=float(reactions[1]),
                       anchor_force_N_per_m=float(reactions[2]), momentum_error=float(np.linalg.norm(imbalance)),
                       kinetic_J_per_m=K, elastic_J_per_m=U, cohesive_stored_J_per_m=surface,
                       fracture_J_per_m=fracture, external_J_per_m=external,
                       transfer_loss_J_per_m=float(work[4]), boundary_projection_loss_J_per_m=float(work[12]),
                       unexplained_J_per_m=external-K-U-surface-fracture-numerical,
                       numerical_fraction=numerical/max(abs(external), 1e-12),
                       max_iteration_residual=float(work[8]), mean_iterations=float(work[7]/max(target, 1)),
                       min_J=float(np.linalg.det(F).min()), max_strain=float(np.max(np.abs(.5*(F+F.swapaxes(-1,-2))-np.eye(2)))),
                       min_face_gap_m=float(np.asarray(interface.plus-interface.minus)[:, 1].min()),
                       valid=bool(state.valid & interface.valid), elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if not row['valid']:
                raise RuntimeError(f'Invalid DCB state at {current_time:g} s')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz', x=x, F=F, stress_yy=stress[:,1,1],
                                stress_xx=stress[:,0,0], principal_stress=np.linalg.eigvalsh(stress)[:,-1],
                                interface_plus=np.asarray(interface.plus), interface_minus=np.asarray(interface.minus),
                                interface_damage=damage, time=current_time)
            previous, previous_impulse = target, impulses.copy()
    summary = dict(**metadata, frames=len(rows), all_states_valid=all(r['valid'] for r in rows),
                   final=rows[-1], elapsed_s=time.perf_counter()-started,
                   limitations=['Dynamic loading; static beam theory is a scale reference',
                                'Prescribed straight potential interface; no path turning',
                                'Grid-mask changes and projection losses require refinement'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
