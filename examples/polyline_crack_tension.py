"""Inclined edge crack under opening: advected faces and stress-selected turning."""

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

from mpm_jax import Config, MovingVelocityBox, block, neo_hookean
from mpm_jax.cohesive_crack import elastic_energy, kinetic_energy
from mpm_jax.crack_mpm import TensileTipLaw
from mpm_jax.materials import kirchhoff_stress
from mpm_jax.polyline_crack import advance_tip, make_path, step, vertex_normals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=96)
    parser.add_argument('--duration', type=float, default=120e-6)
    parser.add_argument('--inclination', type=float, default=15.)
    parser.add_argument('--strength', type=float, default=12e6)
    parser.add_argument('--frames', type=int, default=41)
    args = parser.parse_args()
    if args.grid < 64 or args.duration <= 0 or abs(args.inclination) > 25 or args.frames < 2:
        parser.error('Require grid >=64, positive duration, inclination within 25 degrees and >=2 frames')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64', True)
    E, nu, rho = 3e9, .2, 1200.
    dx = .06/args.grid
    wave = np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    total = int(np.ceil(args.duration/(.15*dx/wave)))
    config = Config(dim=2, n_grid=args.grid, dt=args.duration/total, domain_size=.06,
                    gravity=(0., 0.), boundary_cells=0, material='neo-hookean', boundaries=(
                        MovingVelocityBox((.008, .018), (.052, .022), (0., -2.), ramp_time=20e-6),
                        MovingVelocityBox((.008, .038), (.052, .042), (0., 2.), ramp_time=20e-6)))
    material = neo_hookean(E, nu)
    state = block(config, (.01, .02), (.05, .04), particles_per_cell=2, density=rho, dtype=jnp.float64)
    reference = np.asarray(state.x)
    path = make_path([[.01, .03-.012*np.tan(np.deg2rad(args.inclination))], [.022, .03]], capacity=64)
    law = TensileTipLaw(args.strength, .0015, .1*wave, .016)
    insertion_spacing = .0005
    metadata = dict(case='Advected inclined crack with principal-tension turning', grid=args.grid,
                    dx_m=dx, dt_s=config.dt, duration_s=args.duration, inclination_deg=args.inclination,
                    young_Pa=E, poisson=nu, density_kg_m3=rho, particle_count=len(reference),
                    particle_spacing_m=[float(np.diff(np.unique(reference[:, d])).mean()) for d in (0, 1)],
                    tip_law=asdict(law), max_length_means='Accumulated new extension, excluding the initial crack',
                    insertion_spacing_m=insertion_spacing, max_turn_per_insertion_deg=45., capacity=64,
                    initial_path_m=np.asarray(path.vertices[:2]).tolist(),
                    constitutive_model='neo-Hookean', fracture_energy_prescribed=False,
                    contact=False)
    root = Path(__file__).resolve().parents[1]
    sources = ['examples/polyline_crack_tension.py', 'mpm_jax/polyline_crack.py',
               'mpm_jax/crack_mpm.py', 'mpm_jax/core.py', 'mpm_jax/materials/elastic.py', 'mpm_jax/boundaries.py']
    metadata['source_sha256'] = {p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources}

    @jax.jit
    def advance(state, path, count):
        def body(_, carry):
            def active(carry):
                current, crack = step(carry[0], carry[1], material, config)
                crack, _ = advance_tip(current, crack, material, config, law, insertion_spacing)
                return current._replace(valid=current.valid & crack.valid), crack
            return jax.lax.cond(carry[0].valid & carry[1].valid, active, lambda x:x, carry)
        return jax.lax.fori_loop(0, count, body, (state, path))

    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rows, previous = [], 0
    started = time.perf_counter()
    with (args.output/'history.csv').open('w', newline='') as stream:
        writer = None
        for frame, target in enumerate(np.unique(np.linspace(0, total, args.frames).round().astype(int))):
            if target > previous:
                state, path = advance(state, path, jnp.asarray(target-previous))
            count = int(path.count)
            x, F, vertices = np.asarray(state.x), np.asarray(state.F), np.asarray(path.vertices[:count])
            plus, minus = np.asarray(path.plus[:count]), np.asarray(path.minus[:count])
            stress = np.asarray(kirchhoff_stress(state.F, material, config.material)/jnp.linalg.det(state.F)[:, None, None])
            normals = np.asarray(vertex_normals(path.vertices, path.count)[:count])
            gaps = np.sum((plus-minus)*normals, axis=-1)
            _, report = advance_tip(state, path, material, config, law, insertion_spacing)
            row = dict(step=int(round(float(state.time)/config.dt)), time_s=float(state.time),
                       vertices=count, tip_x_m=float(vertices[-1, 0]), tip_y_m=float(vertices[-1, 1]),
                       extension_m=float(path.extension), pending_extension_m=float(path.pending), traction_Pa=float(report.traction),
                       kinetic_J_per_m=float(kinetic_energy(state, config)),
                       elastic_J_per_m=float(elastic_energy(state, material, config)),
                       max_face_gap_m=float(gaps.max()), min_face_gap_m=float(gaps.min()),
                       max_strain=float(np.max(np.abs(.5*(F+F.swapaxes(-1,-2))-np.eye(2)))),
                       min_J=float(np.linalg.det(F).min()), valid=bool(state.valid & path.valid),
                       elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row); writer.writeheader()
            writer.writerow(row); stream.flush()
            print(json.dumps(row), flush=True)
            if not row['valid']:
                raise RuntimeError('Invalid particle or crack state; output is not publishable')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz', x=x, F=F,
                                stress_yy=stress[:,1,1], principal_stress=np.linalg.eigvalsh(stress)[:,-1],
                                crack=vertices, crack_plus=plus, crack_minus=minus, time=float(state.time))
            previous = target
    summary = dict(**metadata, frames=len(rows), steps=total, all_states_valid=all(r['valid'] for r in rows),
                   final=rows[-1], elapsed_s=time.perf_counter()-started,
                   limitations=['Experimental strength kinetics; no calibrated fracture-energy balance',
                                'One nonbranching path; self-intersection and capacity overflow stop the solver',
                                'No crack-face contact; overlapping faces require another formulation'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
