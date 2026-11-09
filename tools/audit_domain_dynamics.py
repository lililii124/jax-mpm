"""Transient compression audit for moving-domain MPM and its energy ledger."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,VelocityBox,MovingVelocityBox,neo_hookean
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment
from mpm_jax.domain_mpm import step,kinetic_energy,elastic_energy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=32)
    parser.add_argument('--dt-scale',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.grid<32 or args.grid%16 or not 0<args.dt_scale<=1:
        parser.error('Require a grid multiple of 16 >=32 and 0 < dt scale <=1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64',True)
    E,nu,rho=3e6,.2,1200.;duration=.0005;domain=.08;dx=domain/args.grid
    wave=np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    intervals=20;per_frame=int(np.ceil(duration/(.075*dx/wave*args.dt_scale)/intervals))
    steps=intervals*per_frame;dt=duration/steps
    boundaries=(VelocityBox((.019,.034),(.0225,.046),(0.,0.)),
                MovingVelocityBox((.0575,.034),(.061,.046),(-.5,0.),ramp_time=.0001))
    config=Config(dim=2,n_grid=args.grid,dt=dt,domain_size=domain,material='neo-hookean',
                  gravity=(0.,0.),boundary_cells=0,boundaries=boundaries)
    h=dx/2;nx=int(round(.04/h));ny=int(round(.01/h))
    axes=(.02+(np.arange(nx)+.5)*h,.035+(np.arange(ny)+.5)*h)
    points=jnp.asarray(np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,2))
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    state=jax.jit(jax.vmap(lambda x:fragment(reference,rho,x=x)))(points)
    material=neo_hookean(E,nu);initial=float(kinetic_energy(state,config)+elastic_energy(state,material))
    root=Path(__file__).resolve().parents[1]
    sources=['mpm_jax/domain_mpm.py','mpm_jax/domain_transfer.py','mpm_jax/cut_fragments.py',
             'mpm_jax/cut_geometry.py','mpm_jax/cohesive_crack.py','mpm_jax/core.py','mpm_jax/boundaries.py',
             'tools/audit_domain_dynamics.py']
    info=dict(case='Moving-domain transient compression',grid=args.grid,dx_m=dx,dt_s=dt,steps=steps,
              duration_s=duration,domains=len(points),young_Pa=E,poisson=nu,density_kg_m3=rho,
              boundaries=[dict(type=type(b).__name__,**asdict(b)) for b in boundaries],
              source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(info,indent=2)+'\n')

    @jax.jit
    def advance(state,ledger,clock,accepted,geometry_absolute,max_residual,ok):
        def body(_,carry):
            s,w,t,n,g,r,valid=carry
            def attempt(_):
                following,work=step(s,material,config,t)
                accepted=jnp.all(following.valid)&jnp.all(jnp.isfinite(jnp.stack(work)))
                def commit(_):
                    return following,w+jnp.stack(work),t+dt,n+1,g+jnp.abs(work.geometry_energy),jnp.maximum(r,work.residual),accepted
                return jax.lax.cond(accepted,commit,lambda _:(s,w,t,n,g,r,accepted),None)
            return jax.lax.cond(valid,attempt,lambda _:carry,None)
        return jax.lax.fori_loop(0,per_frame,body,(state,ledger,clock,accepted,geometry_absolute,max_residual,ok))

    carry=(state,jnp.zeros(10),jnp.asarray(0.),jnp.asarray(0),jnp.asarray(0.),jnp.asarray(0.),jnp.asarray(True))
    rows=[];started=time.perf_counter()
    with (args.output/'history.csv').open('w',newline='') as stream:
        writer=None
        for frame in range(intervals+1):
            if frame:
                carry=advance(*carry)
            state,ledger,clock,accepted,absolute,residual,ok=carry
            w=np.asarray(ledger);K=float(kinetic_energy(state,config));U=float(elastic_energy(state,material));external=float(w[:3].sum())
            scale=max(abs(external),abs(initial),1e-30)
            row=dict(time_s=float(clock),steps=int(accepted),kinetic_J_per_m=K,elastic_J_per_m=U,
                     external_J_per_m=external,transfer_loss_J_per_m=float(w[3]),boundary_projection_loss_J_per_m=float(w[4]),
                     geometry_energy_J_per_m=float(w[5]),absolute_geometry_energy_J_per_m=float(absolute),
                     numerical_loss_fraction=float((w[3]+w[4])/scale),absolute_geometry_fraction=float(absolute)/scale,
                     balance_error_J_per_m=K+U-initial+float(w[3]+w[4]-w[5])-external,
                     max_iteration_residual=float(residual),mean_iterations=float(w[8]/max(int(accepted),1)),
                     mean_right_displacement_m=float(jnp.mean((state.x-points)[points[:,0]>.0575,0])),
                     min_J=float(jnp.linalg.det(state.F).min()),valid=bool(ok),elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
            if not bool(ok):
                (args.output/'failure.json').write_text(json.dumps(dict(last_accepted=row),indent=2)+'\n')
                raise RuntimeError('Trial step rejected; the last accepted state was preserved')
    np.savez_compressed(args.output/'final.npz',x=np.asarray(state.x),v=np.asarray(state.v),F=np.asarray(state.F),C=np.asarray(state.C),ledger=np.asarray(ledger))
    summary=dict(**info,completed=True,final=rows[-1],max_balance_error_J_per_m=max(abs(r['balance_error_J_per_m']) for r in rows),
                 mesh_convergence_established=False,fracture_front_implemented=False,
                 limitations=['Changing-moment energy is exposed, not removed or assigned to material failure',
                              'This audit advances bulk dynamics; no cohesive/contact evolution is included'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
