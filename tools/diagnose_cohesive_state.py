"""Inspect a saved accepted state at several trial durations without advancing it."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw
from mpm_jax.core import particle_stencil
from mpm_jax.extrinsic_crack import (discrete_surface_gradient,segment_positions,shared_tip,step,
                                    surface_gradient,surface_response,transfers)
from mpm_jax.polyline_crack import link_fields,shifted_nodes,vertex_normals
from tools.recover_extrinsic_dcb import load_saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--frame',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--face-mass-fraction',type=float,default=0.)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    s,p,cfg,info,_=load_saved(args.run,args.frame)
    with np.load(args.checkpoint,allow_pickle=False) as data:
        s=s._replace(**{k:jnp.asarray(data['state_'+k]) for k in s._fields if k!='history'})
        p=p._replace(**{k:jnp.asarray(data['path_'+k]) for k in p._fields})
    material=neo_hookean(info['young_Pa'],info['poisson']);law=CohesiveLaw(**info['law'])
    tied=info.get('couple_tip_nodes',False);consistent=info['consistent_gradients']
    v=p.vertices;first=v[1]-v[0];vis=v.at[0].add(-3*cfg.dx*first/jnp.linalg.norm(first))
    support=particle_stencil(s,cfg);size=(cfg.n_grid+1)**2;index=jnp.arange(size)
    nodes=cfg.dx*jnp.stack((index//(cfg.n_grid+1),index%(cfg.n_grid+1)),axis=-1)
    nodes=shifted_nodes(nodes,vis,p.count)
    field=link_fields(s.x,support,vis,p.count,nodes[support.indices],tip_support_dx=cfg.dx if tied else None)
    remapped=support.indices+field*size
    mass=jnp.zeros(2*size).at[remapped].add(support.weights*s.mass[:,None])
    normal=vertex_normals(v,p.count)
    plus=transfers(p.plus,normal,1.,mass,cfg,vis,p.count,nodes,mass_weighted=not consistent,couple_tip_nodes=tied,minimum_mass_fraction=args.face_mass_fraction)
    minus=transfers(p.minus,normal,-1.,mass,cfg,vis,p.count,nodes,mass_weighted=not consistent,couple_tip_nodes=tied,minimum_mass_fraction=args.face_mass_fraction)
    plus,minus=shared_tip(plus,minus,p.count)
    count=int(p.count);m=np.asarray(mass)
    occupied_face=[];inverse=[]
    for face in (plus,minus):
        weights=np.asarray(face.weights[:count]);local=m[np.asarray(face.indices[:count])]
        used=(np.abs(weights)>1e-12)&(local>0)
        occupied_face.extend(local[used]);inverse.extend(np.sum(weights**2/np.where(local>0,local,np.inf),axis=1))
    old=segment_positions(p.plus,p.minus)

    @jax.jit
    def surface_diagnostic(new):
        before=surface_response(old,p.maximum,p.birth_traction,p.bonded,law)
        after=surface_response(new,before.maximum,p.birth_traction,p.bonded,law)
        average=.5*(surface_gradient(old,before.maximum,p.birth_traction,p.bonded,law)
                     +surface_gradient(new,after.maximum,p.birth_traction,p.bonded,law))
        mid=.5*(old+new);relative=mid-mid.mean(axis=1,keepdims=True)
        rotation=jnp.stack((-relative[:,:,1],relative[:,:,0]),axis=-1)
        norm=jnp.sum(rotation*rotation,axis=(1,2))
        def project(value):
            value=value-value.mean(axis=1,keepdims=True)
            return value-(jnp.sum(value*rotation,axis=(1,2))/jnp.maximum(norm,1e-100))[:,None,None]*rotation
        projected=project(average);direction=project(new-old)
        square=jnp.sum(direction*direction,axis=(1,2))
        energy=after.stored-before.stored+after.dissipated-before.dissipated
        remainder=energy-jnp.sum(projected*(new-old),axis=(1,2))
        resolution=(64*jnp.finfo(old.dtype).eps)**2*jnp.maximum(jnp.maximum(norm,jnp.sum(old*old+new*new,axis=(1,2))),1e-100)
        gradient,_,_,error=discrete_surface_gradient(old,new,p.maximum,p.birth_traction,p.bonded,law)
        return dict(square=square,resolution=resolution,remainder=remainder,energy=energy,
                    stored_change=after.stored-before.stored,dissipated_change=after.dissipated-before.dissipated,
                    correction_norm=jnp.linalg.norm(gradient-projected,axis=(1,2)),gradient_norm=jnp.linalg.norm(gradient,axis=(1,2)),
                    area_work_error=p.area*error)

    rows=[]
    for power in (0,2,4,6,8,9,10,12):
        dt=cfg.dt/(2**power)
        ns,npth,work=step(s,p,material,cfg,law,consistent_gradients=consistent,couple_tip_nodes=tied,dt=jnp.asarray(dt),face_mass_fraction=args.face_mass_fraction)
        values=surface_diagnostic(segment_positions(npth.plus,npth.minus))
        row=dict(dt_s=dt,dt_divisor=2**power,valid=bool(ns.valid & npth.valid),iterations=int(work.iterations),
                 residual=float(work.residual),bulk_work_error=float(work.bulk_error),surface_work_error=float(work.surface_error),
                 minimum_J=float(jnp.linalg.det(ns.F).min()),maximum_particle_speed=float(jnp.linalg.norm(ns.v,axis=1).max()),
                 segments={k:np.asarray(value)[:count-1].tolist() for k,value in values.items()})
        rows.append(row)
        print(json.dumps({k:value for k,value in row.items() if k!='segments'}),flush=True)
    report=dict(time_s=float(s.time),face_mass_fraction=args.face_mass_fraction,support_valid=bool(support.valid),plus_valid=bool(plus.valid),minus_valid=bool(minus.valid),
                minimum_occupied_mass=float(m[m>0].min()),minimum_face_mass=float(min(occupied_face)),
                maximum_face_inverse_mass=float(max(inverse)),rows=rows)
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        return None if isinstance(value,float) and not np.isfinite(value) else value
    args.output.write_text(json.dumps(clean(report),indent=2)+'\n')


if __name__=='__main__':
    main()
