"""Conservative split and compressive traction audit on deformed material domains."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment,invariants
from mpm_jax.domain_transfer import batch_stencils,batch_boundary_integrals,particle_to_grid,internal_force


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.grid<16 or args.grid%4 or args.batch_size<1:
        parser.error('Grid must be a multiple of four >=16; batch size must be positive')
    jax.config.update('jax_enable_x64',True)
    config=Config(dim=2,n_grid=args.grid,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    h=config.dx/2;length=int(round(.5/h));axis=.25+(np.arange(length)+.5)*h
    X=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    center=jnp.array([.5,.5]);angle=.17
    rotation=jnp.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    F=.98*rotation;J=jnp.linalg.det(F)
    points=center+(jnp.asarray(X)-center)@F.T
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    A=jnp.array([[.7,-.3],[1.1,.2]]);b=jnp.array([.6,-.4])
    parents=jax.jit(jax.vmap(lambda x:fragment(reference,1200.,x=x,v=A@x+b,F=F,C=A)))(points)
    parent_support=batch_stencils(parents,config,batch_size=args.batch_size)
    size=(args.grid+1)**2
    mass0,momentum0=particle_to_grid(parents,parent_support,size)
    before=jax.tree.map(lambda v:v.sum(axis=0),jax.vmap(lambda s:invariants(s,config.dx))(parents))
    index=jnp.arange(size);nodes=config.dx*jnp.stack((index//(args.grid+1),index%(args.grid+1)),axis=1)
    box=jnp.array([[.25,.25],[.75,.25],[.75,.75],[.25,.75]])
    box=center+(box-center)@F.T;edge=jnp.roll(box,-1,axis=0)-box
    inward=jnp.stack((-edge[:,1],edge[:,0]),axis=1)
    distance=jnp.einsum('nei,ei->ne',nodes[:,None,:]-box[None,:,:],inward)
    interior=jnp.all(distance>=1.5*config.dx*jnp.sum(jnp.abs(inward),axis=1)[None,:]+1e-13,axis=1)
    if not bool(jnp.all(parent_support.valid)) or not bool(jnp.any(interior)):
        raise RuntimeError('Parent domains or interior-node selection are invalid')
    sigma=-jnp.eye(2);rows=[]

    @jax.jit
    def split_parents(normal,offset):
        children=jax.vmap(lambda s:split_fragment(s,normal,offset))(parents)
        return jax.tree.map(lambda a:a.reshape((-1,)+a.shape[2:]),children)

    for angle in (0.,.31,.68):
        normal=jnp.array([-np.sin(angle),np.cos(angle)])
        for phase in (-.375,0.,.25):
            offset=normal@center+phase*config.dx;started=time.perf_counter()
            children=split_parents(normal,offset)
            support=batch_stencils(children,config,batch_size=args.batch_size)
            boundary=batch_boundary_integrals(children,normal,offset,config,batch_size=args.batch_size)
            if not bool(jnp.all(children.valid&support.valid&boundary.valid)):
                bad=np.flatnonzero(~np.asarray(children.valid&support.valid&boundary.valid))
                print(json.dumps(dict(invalid_indices=bad.tolist(),
                                      child_valid=np.asarray(children.valid)[bad].tolist(),
                                      stencil_valid=np.asarray(support.valid)[bad].tolist(),
                                      boundary_valid=np.asarray(boundary.valid)[bad].tolist(),
                                      reference_area=np.asarray(children.reference_area)[bad].tolist(),
                                      vertex_count=np.asarray(children.reference.count)[bad].tolist())),flush=True)
                raise RuntimeError('Invalid split geometry or quadrature support')
            mass,momentum=particle_to_grid(children,support,size)
            after=jax.tree.map(lambda v:v.sum(axis=0),jax.vmap(lambda s:invariants(s,config.dx))(children))
            errors={name:float(jnp.max(jnp.abs(a-b))) for name,a,b in zip(before._fields,after,before)}
            relative_errors={name:errors[name]/max(float(jnp.max(jnp.abs(value))),1e-100)
                             for name,value in zip(before._fields,before)}
            invariant_error=max(relative_errors.values())
            mass_error=float(jnp.linalg.norm(mass-mass0)/jnp.linalg.norm(mass0))
            momentum_error=float(jnp.linalg.norm(momentum-momentum0)/jnp.linalg.norm(momentum0))
            fields=jnp.tile(jnp.arange(2),len(points))
            separated=support._replace(indices=support.indices+fields[:,None]*size)
            tau=jnp.broadcast_to(J*sigma,children.F.shape)
            bulk=internal_force(tau,children.reference_area,separated,2*size)
            face=jnp.zeros_like(bulk).at[boundary.indices+fields[:,None]*size].add(boundary.vectors@sigma.T)
            mask=jnp.tile(interior,2)
            residual=float(jnp.linalg.norm(jnp.where(mask[:,None],bulk+face,0.))/jnp.linalg.norm(jnp.where(mask[:,None],bulk,0.)))
            resultants=face.reshape(2,size,2).sum(axis=1)
            face_moment=jnp.sum(nodes[None,:,0]*face.reshape(2,size,2)[:,:,1]-nodes[None,:,1]*face.reshape(2,size,2)[:,:,0],axis=1)
            if residual>2e-11 or invariant_error>2e-12 or mass_error>2e-12 or momentum_error>2e-12:
                raise RuntimeError('Conservative transfer or compressive traction balance failed')
            row=dict(angle_rad=angle,phase_over_dx=phase,active_fragments=int(children.active.sum()),
                     relative_force_residual=residual,relative_nodal_mass_error=mass_error,
                     relative_nodal_momentum_error=momentum_error,invariant_absolute_errors=errors,
                     max_relative_invariant_error=invariant_error,
                     contact_resultant_error=float(jnp.linalg.norm(resultants.sum(axis=0))),
                     contact_moment_error=float(jnp.abs(face_moment.sum())),elapsed_s=time.perf_counter()-started)
            rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[1]
    sources=['mpm_jax/cut_fragments.py','mpm_jax/domain_transfer.py','mpm_jax/cut_geometry.py','mpm_jax/core.py','tools/audit_domain_transfer.py']
    report=dict(grid=args.grid,parent_domains=len(points),batch_size=args.batch_size,F=np.asarray(F).tolist(),
                stress=np.asarray(sigma).tolist(),rows=rows,jax_version=jax.__version__,backend=jax.default_backend(),
                bulk_quadrature_points_per_batch=args.batch_size*9*90,
                max_relative_force_residual=max(r['relative_force_residual'] for r in rows),
                max_relative_nodal_mass_error=max(r['relative_nodal_mass_error'] for r in rows),
                max_relative_nodal_momentum_error=max(r['relative_nodal_momentum_error'] for r in rows),
                max_relative_invariant_error=max(r['max_relative_invariant_error'] for r in rows),
                moving_domain_energy_update_validated=False,fracture_front_evolution_validated=False,
                source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
