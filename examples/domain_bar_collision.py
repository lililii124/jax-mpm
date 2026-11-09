"""Two elastic bars collide across a fully cut planar interface."""

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

from mpm_jax import Config,VelocityBox,neo_hookean
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment
from mpm_jax.domain_mpm import step,kinetic_energy,elastic_energy
from mpm_jax import implicit_contact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=64)
    parser.add_argument('--stiffness',type=float,default=1e11)
    parser.add_argument('--dt-scale',type=float,default=1.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.grid<32 or args.grid%16 or not np.isfinite(args.stiffness) or args.stiffness<=0 or not 0<args.dt_scale<=1:
        parser.error('Require grid multiple of 16 >=32, positive finite stiffness and 0 < dt scale <=1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    jax.config.update('jax_enable_x64',True)
    E,nu,rho=3e6,.2,1200.;domain=.08;duration=.0002;speed=.25;cut=.0403
    dx=domain/args.grid;h=dx/2
    wave=np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    pressure=rho*wave*speed
    per_frame=int(np.ceil(duration/(.075*dx/wave*args.dt_scale)/20));dt=duration/(20*per_frame)
    boundary=VelocityBox((0.,0.),(domain,domain),(0.,0.),components=(False,True))
    config=Config(dim=2,n_grid=args.grid,dt=dt,domain_size=domain,material='neo-hookean',
                  gravity=(0.,0.),boundary_cells=0,boundaries=(boundary,))
    axis=(.02+(np.arange(round(.04/h))+.5)*h,.035+(np.arange(round(.01/h))+.5)*h)
    points=jnp.asarray(np.stack(np.meshgrid(*axis,indexing='ij'),axis=-1).reshape(-1,2))
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    parents=jax.jit(jax.vmap(lambda x:fragment(reference,rho,x=x)))(points)
    split=jax.jit(jax.vmap(lambda s:split_fragment(s,jnp.array([1.,0.]),cut)))(parents)
    state=jax.tree.map(lambda a:a.reshape((-1,)+a.shape[2:]),split)
    fields=jnp.tile(jnp.arange(2),len(points))
    keep=np.flatnonzero(np.asarray(state.active))
    state=jax.tree.map(lambda a:a[keep],state);fields=fields[keep]
    state=state._replace(v=jnp.column_stack((jnp.where(fields==0,-speed,speed),jnp.zeros(len(keep)))))
    interface=implicit_contact.line_interface((cut,.035),(cut,.045),(1.,0.),config)
    material=neo_hookean(E,nu)
    initial=float(kinetic_energy(state,config))
    ref_vertices=np.asarray(state.reference.vertices);ref_center=np.asarray(state.reference_centroid)
    ref_count=np.asarray(state.reference.count);initial_x=np.asarray(state.x)
    initial_vertices=initial_x[:,None,:]+ref_vertices-ref_center[:,None,:]
    face_mask=(np.abs(initial_vertices[:,:,0]-cut)<1e-12)&(np.arange(ref_vertices.shape[1])[None,:]<ref_count[:,None])
    field_array=np.asarray(fields)
    root=Path(__file__).resolve().parents[1]
    sources=['examples/domain_bar_collision.py','mpm_jax/implicit_contact.py','mpm_jax/domain_mpm.py',
             'mpm_jax/domain_transfer.py','mpm_jax/cut_fragments.py','mpm_jax/cut_geometry.py',
             'mpm_jax/cohesive_crack.py','mpm_jax/core.py','mpm_jax/boundaries.py']
    info=dict(case='Planar elastic bar collision',grid=args.grid,dx_m=dx,dt_s=dt,steps=20*per_frame,
              duration_s=duration,young_Pa=E,poisson=nu,density_kg_m3=rho,domains=len(keep),
              contact_stiffness_Pa_per_m=args.stiffness,interface_quadrature=len(interface.area),cut_m=cut,
              initial_speed_m_s=speed,initial_energy_J_per_m=initial,linear_wave_speed_m_s=wave,
              linear_impact_pressure_Pa=pressure,minimum_end_reflection_time_s=min(cut-.02,.06-cut)/wave,
              boundary_conditions=[dict(type=type(boundary).__name__,**asdict(boundary))],
              source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(info,indent=2)+'\n')

    @jax.jit
    def advance(carry):
        def body(_,carry):
            state,contact,w,t,n,metrics,valid=carry
            def attempt(_):
                following,new_contact,work,cwork=step(state,material,config,t,fields=fields,n_fields=2,
                                                     contact=contact,contact_stiffness=args.stiffness)
                accepted=jnp.all(following.valid)&new_contact.valid
                accepted &= jnp.all(jnp.isfinite(jnp.r_[jnp.stack(work),jnp.stack(cwork)]))
                updated=jnp.array([cwork.normal_resultant,metrics[1]+dt*cwork.normal_resultant,
                                   jnp.maximum(metrics[2],jnp.abs(cwork.power_error)),
                                   jnp.maximum(metrics[3],cwork.residual),
                                   jnp.maximum(metrics[4],jnp.abs(cwork.net_moment)),
                                   metrics[5]+cwork.iterations,jnp.maximum(metrics[6],work.residual),
                                   metrics[7]+jnp.abs(work.geometry_energy)])
                return jax.lax.cond(accepted,lambda _:(following,new_contact,w+jnp.stack(work),t+dt,n+1,updated,accepted),
                                    lambda _:(state,contact,w,t,n,metrics,accepted),None)
            return jax.lax.cond(valid,attempt,lambda _:carry,None)
        return jax.lax.fori_loop(0,per_frame,body,carry)

    carry=(state,interface,jnp.zeros(10),jnp.asarray(0.),jnp.asarray(0),jnp.zeros(8),jnp.asarray(True))
    rows=[];started=time.perf_counter()
    with (args.output/'history.csv').open('w',newline='') as stream:
        writer=None
        for frame in range(21):
            if frame:
                carry=advance(carry)
            state,interface,ledger,clock,count,metrics,ok=carry
            x,F=np.asarray(state.x),np.asarray(state.F);w=np.asarray(ledger);a=np.asarray(metrics);t=float(clock)
            K=float(kinetic_energy(state,config));U=float(elastic_energy(state,material))
            surface=float(implicit_contact.stored_energy(interface,args.stiffness))
            vertices=x[:,None,:]+np.einsum('pij,pkj->pki',F,ref_vertices-ref_center[:,None,:])
            contact_faces=[vertices[face_mask&(field_array[:,None]==field)] for field in (0,1)]
            geometry_gap=float(contact_faces[0][:,0].min()-contact_faces[1][:,0].max())
            drift=0.
            for face,samples in zip(contact_faces,(np.asarray(interface.plus),np.asarray(interface.minus))):
                order=np.argsort(face[:,1]);positions=np.interp(samples[:,1],face[order,1],face[order,0])
                drift=max(drift,float(np.abs(positions-samples[:,0]).max()))
            J=np.linalg.det(F);mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
            sigma=(mu*(F@F.swapaxes(-1,-2)-np.eye(2))+lam*np.log(J)[:,None,None]*np.eye(2))/J[:,None,None]
            distance=np.abs(initial_x[:,0]-cut)
            plateau=(distance>.25*wave*t)&(distance<.6*wave*t)&(initial_x[:,1]>.037)&(initial_x[:,1]<.043)
            plateau_pressure=float(np.mean(-sigma[plateau,0,0])) if plateau.any() else 0.
            row=dict(time_s=t,steps=int(count),kinetic_J_per_m=K,elastic_J_per_m=U,contact_energy_J_per_m=surface,
                     external_J_per_m=float(w[:3].sum()),transfer_loss_J_per_m=float(w[3]),
                     boundary_projection_loss_J_per_m=float(w[4]),geometry_energy_J_per_m=float(w[5]),
                     absolute_geometry_energy_J_per_m=float(a[7]),balance_error_J_per_m=K+U+surface-initial+float(w[3]+w[4]-w[5]-w[:3].sum()),
                     numerical_loss_fraction=float((w[3]+w[4])/initial),max_contact_power_error_J_per_m=float(a[2]),
                     max_contact_residual=float(a[3]),max_bulk_residual=float(a[6]),max_contact_net_moment_N=float(a[4]),
                     mean_contact_iterations=float(a[5]/max(int(count),1)),contact_pressure_Pa=float(a[0]/.01),
                     mean_contact_pressure_Pa=float(a[1]/max(t,dt)/.01),plateau_pressure_Pa=plateau_pressure,
                     plateau_linear_reference_relative_error=(plateau_pressure/pressure-1) if plateau.any() else 0.,
                     minimum_trace_gap_m=float(jnp.min(implicit_contact.opening(interface))),
                     minimum_geometry_gap_m=geometry_gap,maximum_trace_geometry_drift_m=drift,
                     min_J=float(J.min()),valid=bool(ok),elapsed_s=time.perf_counter()-started)
            rows.append(row)
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
            if not bool(ok):
                (args.output/'failure.json').write_text(json.dumps(dict(completed=False,last_accepted=row),indent=2)+'\n')
                raise RuntimeError('Contact or bulk solve rejected; the previous accepted state is retained')
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),
                                plus=np.asarray(interface.plus),minus=np.asarray(interface.minus),time_s=t)
    np.savez_compressed(args.output/'final.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),ledger=w,
                        mass=np.asarray(state.mass),reference_vertices=ref_vertices,reference_count=ref_count,
                        reference_centroid=ref_center,reference_area=np.asarray(state.reference_area),
                        reference_covariance=np.asarray(state.reference_covariance),fields=field_array,
                        plus=np.asarray(interface.plus),minus=np.asarray(interface.minus),area=np.asarray(interface.area),normal=np.asarray(interface.normal))
    summary=dict(**info,completed=True,final=rows[-1],max_balance_error_J_per_m=max(abs(r['balance_error_J_per_m']) for r in rows),
                 limitations=['Fixed-normal, fixed tangential quadrature; no finite-sliding contact or fracture growth',
                              'Independent normal traces can drift from affine material-domain boundaries; drift is measured',
                              'Linear acoustic reference is approximate for the nonlinear bulk and does not establish convergence'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
