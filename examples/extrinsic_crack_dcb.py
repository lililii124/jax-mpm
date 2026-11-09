"""Diagnostic double-cantilever loading; crack-path validation remains open."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config, MovingVelocityBox, VelocityBox, block, neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw, elastic_energy, kinetic_energy
from mpm_jax.extrinsic_crack import initialize, insert_tip, local_opening, segment_positions, step, surface_response
from mpm_jax.materials import kirchhoff_stress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=160)
    parser.add_argument('--duration',type=float,default=.002)
    parser.add_argument('--inclination',type=float,default=0.)
    parser.add_argument('--energy',type=float,default=2.)
    parser.add_argument('--strength',type=float,default=120e3)
    parser.add_argument('--cfl',type=float,default=.075)
    parser.add_argument('--frames',type=int,default=41)
    parser.add_argument('--legacy-gradients',action='store_true')
    parser.add_argument('--couple-tip-nodes',action='store_true')
    args = parser.parse_args()
    if args.grid<128 or args.duration<=0 or abs(args.inclination)>3 or not 0<args.cfl<=.1 or args.frames<2:
        parser.error('Invalid grid, duration, inclination, CFL or frame count')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64',True)
    E,nu,rho = 30e6,.2,1200.
    dx = .08/args.grid
    wave = np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    total = int(np.ceil(args.duration/(args.cfl*dx/wave)))
    config = Config(dim=2,n_grid=args.grid,dt=args.duration/total,domain_size=.08,material='neo-hookean',
                    gravity=(0.,0.),boundary_cells=0,boundaries=(
                        MovingVelocityBox((.009,.036),(.012,.038),(0.,-1.),components=(False,True),ramp_time=.0002),
                        MovingVelocityBox((.009,.042),(.012,.044),(0.,1.),components=(False,True),ramp_time=.0002),
                        VelocityBox((.068,.034),(.072,.046),(0.,0.),components=(True,False))))
    material = neo_hookean(E,nu)
    law = CohesiveLaw(args.strength,args.energy,1e11)
    state = block(config,(.01,.037),(.07,.043),particles_per_cell=2,density=rho,dtype=jnp.float64)
    reference = np.asarray(state.x)
    path = initialize([[.01,.04-.025*np.tan(np.deg2rad(args.inclination))],[.035,.04]],capacity=64)
    spacing,radius,max_extension = .0005,.0015,.028
    root = Path(__file__).resolve().parents[1]
    sources = ['examples/extrinsic_crack_dcb.py','mpm_jax/extrinsic_crack.py','mpm_jax/cohesive_crack.py',
               'mpm_jax/polyline_crack.py','mpm_jax/core.py','mpm_jax/materials/elastic.py','mpm_jax/boundaries.py']
    metadata = dict(case='Localized double-cantilever cohesive insertion',grid=args.grid,dx_m=dx,dt_s=config.dt,
                    duration_s=args.duration,inclination_deg=args.inclination,young_Pa=E,poisson=nu,density_kg_m3=rho,
                    law=asdict(law),insertion_spacing_m=spacing,averaging_radius_m=radius,max_extension_m=max_extension,
                    max_turn_deg=45.,capacity=64,particle_count=len(reference),consistent_gradients=not args.legacy_gradients,
                    couple_tip_nodes=args.couple_tip_nodes,
                    lower_m=[.01,.037],upper_m=[.07,.043],
                    boundary_conditions=[asdict(b) for b in config.boundaries],
                    initial_tip_x_m=.035,grip_speed_m_s=1.,ramp_s=.0002,
                    particle_spacing_m=[float(np.diff(np.unique(reference[:,d])).mean()) for d in (0,1)],
                    source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})

    @jax.jit
    def advance(state,path,work,insertion_energy,count):
        def body(_,carry):
            def active(carry):
                s,p,w,inserted = carry
                s,p,increment = step(s,p,material,config,law,consistent_gradients=not args.legacy_gradients,
                                     couple_tip_nodes=args.couple_tip_nodes)
                w = w+jnp.stack(increment)
                w = w.at[7].set(jnp.maximum(carry[2][7],increment.residual))
                p,report = insert_tip(s,p,material,config,law,spacing,radius,max_extension)
                s = s._replace(valid=s.valid & p.valid)
                return s,p,w,inserted+report.energy_jump
            return jax.lax.cond(carry[0].valid & carry[1].valid,active,lambda x:x,carry)
        return jax.lax.fori_loop(0,count,body,(state,path,work,insertion_energy))

    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(metadata,indent=2)+'\n')
    work,insertion_energy = jnp.zeros(8),jnp.asarray(0.)
    rows,previous = [],0
    started = time.perf_counter()
    with (args.output/'history.csv').open('w',newline='') as stream:
        writer = None
        for frame,target in enumerate(np.unique(np.linspace(0,total,args.frames).round().astype(int))):
            if target>previous:
                state,path,work,insertion_energy = advance(state,path,work,insertion_energy,jnp.asarray(target-previous))
            count = int(path.count)
            q = segment_positions(path.plus,path.minus)
            response = surface_response(q,path.maximum,path.birth_traction,path.bonded,law)
            K,U = float(kinetic_energy(state,config)),float(elastic_energy(state,material,config))
            surface = float(jnp.sum(path.area*response.stored))
            fracture = float(jnp.sum(path.area*response.dissipated))
            complete = jnp.all(response.softening>=1-1e-8,axis=1)
            connected = jnp.cumprod(~path.bonded|complete)
            failed_length = float(jnp.sum(path.area*path.bonded*connected))
            detached = float(jnp.sum(path.area*path.bonded*complete))-failed_length
            x,F = np.asarray(state.x),np.asarray(state.F)
            stress = np.asarray(kirchhoff_stress(state.F,material,config.material)/jnp.linalg.det(state.F)[:,None,None])
            external = float(work[0]+work[1])
            numerical = float(work[2]+work[3])
            row = dict(step=int(round(float(state.time)/config.dt)),time_s=float(state.time),vertices=count,
                       potential_extension_m=float(path.extension),connected_failed_m=failed_length,detached_failed_m=detached,
                       tip_x_m=float(path.vertices[count-1,0]),tip_y_m=float(path.vertices[count-1,1]),
                       kinetic_J_per_m=K,elastic_J_per_m=U,surface_J_per_m=surface,fracture_J_per_m=fracture,
                       external_J_per_m=external,transfer_loss_J_per_m=float(work[2]),boundary_projection_loss_J_per_m=float(work[3]),
                       insertion_energy_J_per_m=float(insertion_energy),unexplained_J_per_m=external+float(insertion_energy)-K-U-surface-fracture-numerical,
                       numerical_fraction=numerical/max(abs(external),1e-12),max_iteration_residual=float(work[7]),
                       mean_iterations=float(work[6]/max(target,1)),min_J=float(np.linalg.det(F).min()),
                       max_strain=float(np.max(np.abs(.5*(F+F.swapaxes(-1,-2))-np.eye(2)))),
                       min_face_gap_m=float(jnp.min(local_opening(q)[:count-1,:,1])),
                       valid=bool(state.valid & path.valid),elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(stream,fieldnames=row);writer.writeheader()
            writer.writerow(row);stream.flush()
            print(json.dumps(row),flush=True)
            if not row['valid']:
                failure=dict(completed=False,failed=row,last_valid=rows[-2] if len(rows)>1 else None,
                             reason='Coupled state failed validity checks; failed fields are not exported')
                (args.output/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                raise RuntimeError('Invalid coupled state; retained only for diagnosis')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),
                                stress_yy=stress[:,1,1],principal_stress=np.linalg.eigvalsh(stress)[:,-1],
                                plus=np.asarray(path.plus[:count]),minus=np.asarray(path.minus[:count]),
                                maximum=np.asarray(path.maximum[:count-1]),area=np.asarray(path.area[:count-1]),
                                birth_traction=np.asarray(path.birth_traction[:count-1]),bonded=np.asarray(path.bonded[:count-1]),
                                softening=np.asarray(response.softening[:count-1]),time=float(state.time))
            previous = target
    summary = dict(**metadata,frames=len(rows),steps=total,all_states_valid=all(r['valid'] for r in rows),
                   final=rows[-1],elapsed_s=time.perf_counter()-started,
                   limitations=['Regularized shifted-gap unloading; not an initially rigid cohesive law',
                                'Stress averaging does not guarantee exact nodal traction balance at insertion',
                                'One nonbranching crack; no general fragment contact or 3D surfaces'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
