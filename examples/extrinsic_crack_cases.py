"""Cohesive edge cracks under tension, mixed loading and three-point bending."""

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

from mpm_jax import Config,MovingVelocityBox,VelocityBox,block,neo_hookean
from mpm_jax.adaptive_crack import controller,integrate_to
from mpm_jax.cohesive_crack import CohesiveLaw,elastic_energy,kinetic_energy
from mpm_jax.extrinsic_crack import initialize,local_opening,segment_positions,surface_response
from mpm_jax.materials import kirchhoff_stress


def scenario(name):
    if name=='bending':
        return dict(domain=.08,grid=160,duration=160e-6,lower=(.01,.02),upper=(.07,.035),
                    vertices=((.04,.02),(.04,.026)),extension=.0075,center=(.04,.028),half_height=.034,
                    boundaries=(VelocityBox((.014,.019),(.016,.021),(0.,0.)),
                                VelocityBox((.064,.019),(.066,.021),(0.,0.),components=(False,True)),
                                MovingVelocityBox((.038,.034),(.042,.036),(0.,-6.),
                                                  components=(False,True),ramp_time=20e-6)))
    upper_velocity=(0.,2.5) if name=='tension' else (2.,1.5)
    return dict(domain=.06,grid=128,duration=120e-6,lower=(.01,.02),upper=(.05,.04),
                vertices=((.01,.03),(.022,.03)),extension=.025,center=(.03,.03),half_height=.023,
                boundaries=(MovingVelocityBox((.008,.018),(.052,.022),tuple(-v for v in upper_velocity),ramp_time=20e-6),
                            MovingVelocityBox((.008,.038),(.052,.042),upper_velocity,ramp_time=20e-6)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('bending','tension','mixed'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grid',type=int)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--frames',type=int,default=49)
    args=parser.parse_args()
    case=scenario(args.case)
    grid=case['grid'] if args.grid is None else args.grid
    duration=case['duration'] if args.duration is None else args.duration
    if grid<96 or not np.isfinite(duration) or duration<=0 or args.frames<2:
        parser.error('Require grid >=96, finite positive duration and at least two frames')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64',True)
    E,nu,rho=3e9,.2,1200.
    wave=np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    dx=case['domain']/grid
    nominal=int(np.ceil(duration/(.075*dx/wave)))
    config=Config(dim=2,n_grid=grid,dt=duration/nominal,domain_size=case['domain'],material='neo-hookean',
                  gravity=(0.,0.),boundary_cells=0,boundaries=case['boundaries'])
    material=neo_hookean(E,nu)
    law=CohesiveLaw(12e6,200.,1e13)
    state=block(config,case['lower'],case['upper'],particles_per_cell=2,density=rho,dtype=jnp.float64)
    reference=np.asarray(state.x)
    path=initialize(case['vertices'],capacity=96)
    spacing,radius=.0005,.0015
    source_names=['examples/extrinsic_crack_cases.py','mpm_jax/adaptive_crack.py','mpm_jax/extrinsic_crack.py',
                  'mpm_jax/cohesive_crack.py','mpm_jax/polyline_crack.py','mpm_jax/core.py',
                  'mpm_jax/materials/elastic.py','mpm_jax/boundaries.py']
    root=Path(__file__).resolve().parents[1]
    metadata=dict(case=args.case,grid=grid,dx_m=dx,dt_s=config.dt,duration_s=duration,young_Pa=E,poisson=nu,
                  density_kg_m3=rho,law=asdict(law),insertion_spacing_m=spacing,averaging_radius_m=radius,
                  max_extension_m=case['extension'],max_turn_deg=45.,capacity=96,particle_count=len(reference),
                  consistent_gradients=True,couple_tip_nodes=True,adaptive=True,
                  lower_m=case['lower'],upper_m=case['upper'],initial_vertices_m=case['vertices'],
                  boundary_conditions=[dict(type=type(b).__name__,**asdict(b)) for b in config.boundaries],
                  render_center=case['center'],render_half_height=case['half_height'],
                  particle_spacing_m=[float(np.diff(np.unique(reference[:,d])).mean()) for d in (0,1)],
                  source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in source_names})
    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(metadata,indent=2)+'\n')
    ledger=jnp.zeros(9);control=controller(config.dt)
    rows=[];started=time.perf_counter()
    with (args.output/'history.csv').open('w',newline='') as stream:
        writer=None
        for frame,target in enumerate(np.linspace(0,duration,args.frames)):
            if target>0:
                state,path,ledger,control=integrate_to(state,path,material,config,law,ledger,control,
                                                       jnp.asarray(target),spacing,radius,case['extension'])
            count=int(path.count)
            q=segment_positions(path.plus,path.minus)
            response=surface_response(q,path.maximum,path.birth_traction,path.bonded,law)
            K,U=float(kinetic_energy(state,config)),float(elastic_energy(state,material,config))
            surface=float(jnp.sum(path.area*response.stored));fracture=float(jnp.sum(path.area*response.dissipated))
            complete=jnp.all(response.softening>=1-1e-8,axis=1)
            connected=jnp.cumprod(~path.bonded|complete)
            failed=float(jnp.sum(path.area*path.bonded*connected))
            detached=float(jnp.sum(path.area*path.bonded*complete))-failed
            x,F=np.asarray(state.x),np.asarray(state.F)
            stress=np.asarray(kirchhoff_stress(state.F,material,config.material)/jnp.linalg.det(state.F)[:,None,None])
            w=np.asarray(ledger);external=float(w[0]+w[1]);numerical=float(w[2]+w[3])
            row=dict(step=int(control.accepted),time_s=float(state.time),vertices=count,
                     potential_extension_m=float(path.extension),connected_failed_m=failed,detached_failed_m=detached,
                     tip_x_m=float(path.vertices[count-1,0]),tip_y_m=float(path.vertices[count-1,1]),
                     kinetic_J_per_m=K,elastic_J_per_m=U,surface_J_per_m=surface,fracture_J_per_m=fracture,
                     external_J_per_m=external,transfer_loss_J_per_m=float(w[2]),boundary_projection_loss_J_per_m=float(w[3]),
                     insertion_energy_J_per_m=float(w[8]),unexplained_J_per_m=external+w[8]-K-U-surface-fracture-numerical,
                     numerical_fraction=numerical/max(abs(external),1e-12),max_iteration_residual=float(control.maximum_residual),
                     mean_iterations=float(w[6]/max(int(control.accepted),1)),min_J=float(np.linalg.det(F).min()),
                     max_strain=float(np.max(np.abs(.5*(F+F.swapaxes(-1,-2))-np.eye(2)))),
                     min_face_gap_m=float(jnp.min(local_opening(q)[:count-1,:,1])),
                     rejected_attempts=int(control.rejected),minimum_dt_s=float(control.minimum_used),stop_code=int(control.stop_code),
                     valid=bool(state.valid & path.valid & (control.stop_code==0)),elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
            if not row['valid']:
                (args.output/'failure.json').write_text(json.dumps(dict(completed=False,last=row),indent=2)+'\n')
                raise RuntimeError('Stopped at last accepted state; see failure.json')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),
                                stress_yy=stress[:,1,1],principal_stress=np.linalg.eigvalsh(stress)[:,-1],
                                plus=np.asarray(path.plus[:count]),minus=np.asarray(path.minus[:count]),
                                maximum=np.asarray(path.maximum[:count-1]),area=np.asarray(path.area[:count-1]),
                                birth_traction=np.asarray(path.birth_traction[:count-1]),bonded=np.asarray(path.bonded[:count-1]),
                                softening=np.asarray(response.softening[:count-1]),time=float(state.time))
    summary=dict(**metadata,frames=len(rows),steps=int(control.accepted),all_states_valid=True,final=rows[-1],
                 elapsed_s=time.perf_counter()-started,
                 limitations=['Principal-stress direction and point-particle cut integration are not predictive fracture validation',
                              'Geometric insertion cap is not physical crack arrest; no general fragment contact'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
