"""Mode-I edge-notch tension with a prescribed planar cohesive path."""

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

from mpm_jax import Config,MovingVelocityBox,RampDisplacementBox,CyclicVelocityBox,neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment
from mpm_jax.domain_mpm import step,kinetic_energy,elastic_energy
from mpm_jax import implicit_contact,normal_cohesive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=48)
    parser.add_argument('--cfl',type=float,default=.075)
    parser.add_argument('--duration',type=float,default=120e-6)
    parser.add_argument('--loading',choices=('velocity','hold','pulse'),default='velocity')
    parser.add_argument('--notch',choices=('edge','double'),default='edge')
    parser.add_argument('--opening',type=float,default=30e-6)
    parser.add_argument('--ramp-time',type=float,default=40e-6)
    parser.add_argument('--period',type=float,default=80e-6,help='pulse duration from zero displacement back to zero')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.grid<24 or args.grid%24 or not np.isfinite(args.duration) or args.duration<=0 or not 0<args.cfl<=.15:
        parser.error('Require grid multiple of 24 >=24, positive duration and 0 < CFL <=.15')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty')
    if args.batch_size<1 or not np.isfinite(args.opening) or args.opening<=0 or not np.isfinite(args.ramp_time) or args.ramp_time<=0 or not np.isfinite(args.period) or args.period<=0:
        parser.error('Require positive finite opening, ramp time, period and batch size >=1')
    jax.config.update('jax_enable_x64',True)
    E,nu,rho=3e9,.2,1200.;domain=.06;cut=.03;tip=.022
    dx=domain/args.grid;h=dx/2
    wave=np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
    per_frame=int(np.ceil(args.duration/(args.cfl*dx/wave)/40));dt=args.duration/(40*per_frame)
    boundaries=(MovingVelocityBox((.008,.018),(.052,.022),(0.,-2.5),ramp_time=20e-6),
                MovingVelocityBox((.008,.038),(.052,.042),(0.,2.5),ramp_time=20e-6))
    if args.loading=='hold':
        boundaries=(RampDisplacementBox((.008,.018),(.052,.022),(0.,-.5*args.opening),args.ramp_time),
                    RampDisplacementBox((.008,.038),(.052,.042),(0.,.5*args.opening),args.ramp_time))
    if args.loading=='pulse':
        boundaries=(CyclicVelocityBox((.008,.018),(.052,.022),(0.,-.5*args.opening),args.period),
                    CyclicVelocityBox((.008,.038),(.052,.042),(0.,.5*args.opening),args.period))
    config=Config(dim=2,n_grid=args.grid,dt=dt,domain_size=domain,material='neo-hookean',
                  gravity=(0.,0.),boundary_cells=0,boundaries=boundaries)
    axis=(.01+(np.arange(round(.04/h))+.5)*h,.02+(np.arange(round(.02/h))+.5)*h)
    points=jnp.asarray(np.stack(np.meshgrid(*axis,indexing='ij'),axis=-1).reshape(-1,2))
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    parents=jax.jit(jax.vmap(lambda x:fragment(reference,rho,x=x)))(points)
    children=jax.jit(jax.vmap(lambda s:split_fragment(s,jnp.array([0.,1.]),cut)))(parents)
    state=jax.tree.map(lambda a:a.reshape((-1,)+a.shape[2:]),children)
    fields=jnp.tile(jnp.arange(2),len(points));keep=np.flatnonzero(np.asarray(state.active))
    state=jax.tree.map(lambda a:a[keep],state);fields=fields[keep]
    if args.notch=='edge':
        crack_segments=[(.01,tip)]
    else:
        crack_segments=[(.01,.022),(.038,.05)]
    intact_segments=[(crack_segments[0][1],crack_segments[1][0])] if args.notch=='double' else [(tip,.05)]
    cracked_parts=[implicit_contact.line_interface((a,cut),(b,cut),(0.,1.),config) for a,b in crack_segments]
    intact_parts=[implicit_contact.line_interface((a,cut),(b,cut),(0.,1.),config) for a,b in intact_segments]
    parts=cracked_parts+intact_parts
    line=implicit_contact.Interface(jnp.concatenate([p.plus for p in parts]),jnp.concatenate([p.minus for p in parts]),
                                    jnp.concatenate([p.area for p in parts]),parts[-1].normal,
                                    jnp.all(jnp.stack([p.valid for p in parts])))
    bonded=jnp.concatenate([jnp.zeros(len(p.area),dtype=bool) for p in cracked_parts]+
                           [jnp.ones(len(p.area),dtype=bool) for p in intact_parts])
    interface=normal_cohesive.initialize(line,bonded)
    material=neo_hookean(E,nu);law=CohesiveLaw(12e6,200.,1e13)
    ref_vertices=np.asarray(state.reference.vertices);ref_center=np.asarray(state.reference_centroid)
    ref_count=np.asarray(state.reference.count);field_array=np.asarray(fields);initial_x=np.asarray(state.x)
    vertices0=initial_x[:,None,:]+ref_vertices-ref_center[:,None,:]
    face_mask=(np.abs(vertices0[:,:,1]-cut)<1e-12)&(np.arange(ref_vertices.shape[1])[None,:]<ref_count[:,None])
    root=Path(__file__).resolve().parents[1]
    sources=['examples/domain_notched_tension.py','mpm_jax/normal_cohesive.py','mpm_jax/implicit_contact.py',
             'mpm_jax/domain_mpm.py','mpm_jax/domain_transfer.py','mpm_jax/cut_fragments.py','mpm_jax/cut_geometry.py',
             'mpm_jax/cohesive_crack.py','mpm_jax/core.py','mpm_jax/boundaries.py']
    info=dict(case='Domain-integrated Mode-I prescribed-path tension',notch=args.notch,grid=args.grid,dx_m=dx,dt_s=dt,steps=40*per_frame,
              loading=args.loading,batch_size=args.batch_size,prescribed_total_opening_m=args.opening if args.loading=='hold' else None,
              pulse_peak_opening_m=args.opening if args.loading=='pulse' else None,pulse_period_s=args.period if args.loading=='pulse' else None,
              duration_s=args.duration,young_Pa=E,poisson=nu,density_kg_m3=rho,domains=len(keep),law=asdict(law),
              interface_quadrature=len(interface.area),initial_crack_segments_m=[list(s) for s in crack_segments],
              initial_crack_length_m=sum(b-a for a,b in crack_segments),prescribed_path_length_m=.04,
              zero_initial_physical_damage=True,initial_energy_J_per_m=0.,
              boundary_conditions=[dict(type=type(b).__name__,**asdict(b)) for b in boundaries],
              source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.mkdir(parents=True)
    (args.output/'input.json').write_text(json.dumps(info,indent=2)+'\n')

    @jax.jit
    def advance(carry):
        def body(_,carry):
            state,contact,w,t,n,metrics,valid,trial=carry
            def attempt(_):
                following,new_contact,work,cwork=step(state,material,config,t,fields=fields,n_fields=2,
                                                     contact=contact,cohesive_law=law,batch_size=args.batch_size)
                accepted=jnp.all(following.valid)&new_contact.valid
                accepted &= jnp.all(jnp.isfinite(jnp.r_[jnp.stack(work),jnp.stack(cwork)]))
                updated=jnp.array([jnp.maximum(metrics[0],jnp.abs(cwork.power_error)),
                                   jnp.maximum(metrics[1],cwork.residual),jnp.maximum(metrics[2],work.residual),
                                   jnp.maximum(metrics[3],cwork.softening_number),metrics[4]+cwork.iterations,
                                   jnp.maximum(metrics[5],jnp.abs(cwork.net_moment)),metrics[6]+jnp.abs(work.geometry_energy)])
                trial=jnp.array([cwork.residual,work.residual,cwork.softening_number,work.ledger_error,cwork.power_error])
                return jax.lax.cond(accepted,lambda _:(following,new_contact,w+jnp.stack(work),t+dt,n+1,updated,accepted,trial),
                                    lambda _:(state,contact,w,t,n,metrics,accepted,trial),None)
            return jax.lax.cond(valid,attempt,lambda _:carry,None)
        return jax.lax.fori_loop(0,per_frame,body,carry)

    carry=(state,interface,jnp.zeros(10),jnp.asarray(0.),jnp.asarray(0),jnp.zeros(7),jnp.asarray(True),jnp.zeros(5))
    rows=[];started=time.perf_counter()
    with (args.output/'history.csv').open('w',newline='') as stream:
        writer=None
        for frame in range(41):
            if frame:
                carry=advance(carry)
            state,interface,ledger,clock,count,metrics,ok,trial=carry
            x,F=np.asarray(state.x),np.asarray(state.F);w=np.asarray(ledger);a=np.asarray(metrics);t=float(clock)
            K=float(kinetic_energy(state,config));U=float(elastic_energy(state,material))
            stored,dissipated=map(float,normal_cohesive.energy(interface,law))
            max_open=np.asarray(interface.maximum);bound=np.asarray(interface.bonded);area=np.asarray(interface.area)
            damaged=max_open>law.peak_opening*(1+1e-8);failed=max_open>=law.final_opening*(1-1e-8)
            connected_mask=np.zeros(len(bound),dtype=bool);damage_connected_mask=np.zeros(len(bound),dtype=bool)
            offset=0;side_metrics=[]
            for part in cracked_parts:
                n=len(part.area);offset+=n
            for part in intact_parts:
                n=len(part.area);sl=slice(offset,offset+n);offset+=n
                if args.notch=='double':
                    left_failed=np.cumprod(failed[sl]).astype(bool);right_failed=np.cumprod(failed[sl][::-1]).astype(bool)[::-1]
                    left_damaged=np.cumprod(damaged[sl]).astype(bool);right_damaged=np.cumprod(damaged[sl][::-1]).astype(bool)[::-1]
                    connected_mask[sl]=left_failed|right_failed;damage_connected_mask[sl]=left_damaged|right_damaged
                    side_metrics=[(float(np.sum(area[sl]*left_failed)),float(np.sum(area[sl]*left_damaged))),
                                  (float(np.sum(area[sl]*right_failed)),float(np.sum(area[sl]*right_damaged)))]
                else:
                    connected_mask[sl]=np.cumprod(failed[sl]).astype(bool);damage_connected_mask[sl]=np.cumprod(damaged[sl]).astype(bool)
            connected=connected_mask;damage_connected=damage_connected_mask
            vertices=x[:,None,:]+np.einsum('pij,pkj->pki',F,ref_vertices-ref_center[:,None,:])
            drift=0.
            for field,samples in enumerate((np.asarray(interface.plus),np.asarray(interface.minus))):
                face=vertices[face_mask&(field_array[:,None]==field)]
                order=np.argsort(face[:,0]);positions=np.interp(samples[:,0],face[order,0],face[order,1])
                drift=max(drift,float(np.abs(positions-samples[:,1]).max()))
            external=float(w[:3].sum());scale=max(abs(external),1e-30)
            row=dict(time_s=t,steps=int(count),kinetic_J_per_m=K,elastic_J_per_m=U,interface_stored_J_per_m=stored,
                     fracture_J_per_m=dissipated,external_J_per_m=external,transfer_loss_J_per_m=float(w[3]),
                     boundary_projection_loss_J_per_m=float(w[4]),geometry_energy_J_per_m=float(w[5]),
                     absolute_geometry_energy_J_per_m=float(a[6]),balance_error_J_per_m=K+U+stored+dissipated+float(w[3]+w[4]-w[5])-external,
                     numerical_loss_fraction=float((w[3]+w[4])/scale),max_interface_power_error_J_per_m=float(a[0]),
                     max_interface_residual=float(a[1]),max_bulk_residual=float(a[2]),max_softening_number=float(a[3]),
                     mean_interface_iterations=float(a[4]/max(int(count),1)),max_interface_net_moment_N=float(a[5]),
                     connected_failed_m=float(np.sum(area*bound*connected)),detached_failed_m=float(np.sum(area*bound*failed*(~connected))),
                     connected_damaged_m=float(np.sum(area*bound*damage_connected)),minimum_trace_gap_m=float(jnp.min(implicit_contact.opening(interface))),
                     maximum_trace_geometry_drift_m=drift,min_J=float(np.linalg.det(F).min()),valid=bool(ok),elapsed_s=time.perf_counter()-started)
            if args.notch=='double':
                row.update(left_connected_failed_m=side_metrics[0][0],right_connected_failed_m=side_metrics[1][0],
                           left_connected_damaged_m=side_metrics[0][1],right_connected_damaged_m=side_metrics[1][1])
            rows.append(row)
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader()
            writer.writerow(row);stream.flush();print(json.dumps(row),flush=True)
            np.savez_compressed(args.output/f'frame-{frame:03d}.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),
                                plus=np.asarray(interface.plus),minus=np.asarray(interface.minus),maximum=max_open,time_s=t)
            if not bool(ok):
                (args.output/'failure.json').write_text(json.dumps(dict(completed=False,last_accepted=row,last_trial=np.asarray(trial).tolist()),indent=2)+'\n')
                raise RuntimeError('Interface or bulk solve rejected; no failed trial was committed')
    np.savez_compressed(args.output/'final.npz',x=x,F=F,v=np.asarray(state.v),C=np.asarray(state.C),ledger=w,
                        mass=np.asarray(state.mass),reference_vertices=ref_vertices,reference_count=ref_count,
                        reference_centroid=ref_center,reference_area=np.asarray(state.reference_area),
                        reference_covariance=np.asarray(state.reference_covariance),fields=field_array,
                        plus=np.asarray(interface.plus),minus=np.asarray(interface.minus),area=area,normal=np.asarray(interface.normal),
                        maximum=max_open,bonded=bound)
    summary=dict(**info,completed=True,final=rows[-1],max_balance_error_J_per_m=max(abs(r['balance_error_J_per_m']) for r in rows),
                 limitations=['Intrinsic cohesive path is prescribed; this is not freely turning or branching fracture',
                              'Finite intact-interface compliance, fixed Mode-I normal and reference line weights',
                              'Trace/material-boundary drift and numerical transfer loss require refinement checks'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    main()
