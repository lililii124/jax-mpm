"""Uniform-stress patch audit for newly inserted cohesive segments."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config,block,neo_hookean
from mpm_jax.cohesive_crack import CohesiveLaw,scatter_face_force
from mpm_jax.core import particle_stencil
from mpm_jax.extrinsic_crack import initialize,segment_positions,shared_tip,surface_gradient,transfers
from mpm_jax.materials import kirchhoff_stress
from mpm_jax.polyline_crack import link_fields,shifted_nodes,vertex_normals
from mpm_jax.apic_projection import force_adjoint


def audit(grid,spacing_ratio,phase,consistent=False,particle_mediated_faces=False):
    config=Config(dim=2,n_grid=grid,dt=1e-6,gravity=(0.,0.),boundary_cells=0)
    material=neo_hookean(1e6,.2)
    state=block(config,(.125,.125),(.875,.875),particles_per_cell=2,density=1000.,dtype=jnp.float64)
    stretch=np.sqrt(1.002)
    state=state._replace(F=jnp.broadcast_to(jnp.diag(jnp.array([1/stretch,stretch])),state.F.shape))
    count=int(round(.25/(spacing_ratio*config.dx)))
    y=.5+phase*config.dx
    vertices=np.column_stack((np.r_[.125,np.linspace(.375,.625,count+1)],np.full(count+2,y)))
    path=initialize(vertices,capacity=len(vertices))
    tau=kirchhoff_stress(state.F,material,config.material)
    traction=float(tau[0,1,1])
    law=CohesiveLaw(500.,1.,1e7)
    path=path._replace(birth_traction=path.birth_traction.at[1:,1].set(traction),
                       maximum=path.maximum.at[1:].set(traction/law.stiffness),
                       bonded=path.bonded.at[1:].set(True))
    support=particle_stencil(state,config)
    size=(grid+1)**2
    index=jnp.arange(size)
    coordinates=config.dx*jnp.stack((index//(grid+1),index%(grid+1)),axis=-1)
    visibility=path.vertices.at[0,0].add(-3*config.dx)
    nodes=shifted_nodes(coordinates,visibility,path.count)
    fields=link_fields(state.x,support,visibility,path.count,nodes[support.indices])
    indices=support.indices+size*fields
    mass=jnp.zeros(2*size).at[indices].add(support.weights*state.mass[:,None])
    gradients=support.gradients if consistent else 4/config.dx**2*support.weights[...,None]*support.dpos
    contributions=-state.volume[:,None,None]*jnp.einsum('pij,pkj->pki',tau,gradients)
    bulk=jnp.zeros((2*size,2)).at[indices].add(contributions)
    normals=vertex_normals(path.vertices,path.count)
    plus=transfers(path.plus,normals,1.,mass,config,visibility,path.count,nodes,mass_weighted=not consistent)
    minus=transfers(path.minus,normals,-1.,mass,config,visibility,path.count,nodes,mass_weighted=not consistent)
    plus,minus=shared_tip(plus,minus,path.count)
    assert bool(plus.valid & minus.valid)
    gradient=surface_gradient(segment_positions(path.plus,path.minus),path.maximum,path.birth_traction,path.bonded,law)
    force=-path.area[:,None,None]*gradient
    fp=jnp.zeros_like(path.plus).at[:-1].add(force[:,0]).at[1:].add(force[:,1])
    fm=jnp.zeros_like(path.minus).at[:-1].add(force[:,2]).at[1:].add(force[:,3])
    cohesive=scatter_face_force(fp,plus,bulk.shape)+scatter_face_force(fm,minus,bulk.shape)
    if particle_mediated_faces:
        cohesive=force_adjoint(cohesive,support._replace(indices=indices),state.mass,mass,config)
    mask=(coordinates[:,0]>.375+2*config.dx)&(coordinates[:,0]<.625-2*config.dx)&(jnp.abs(coordinates[:,1]-y)<2*config.dx)
    mask=jnp.tile(mask,2)&(mass>0)
    assert bool(jnp.any(mask))
    residual=bulk+cohesive
    norm_bulk=jnp.linalg.norm(bulk[mask])
    # Div(sigma)=0 for this homogeneous interior patch; no time stepping is used.
    intact=jnp.zeros((size,2)).at[support.indices].add(contributions)
    intact_error=float(jnp.max(jnp.abs(intact[jnp.tile(mask[:size]|mask[size:],1)])))
    relative=float(jnp.linalg.norm(residual[mask])/norm_bulk)
    fitted=float(-jnp.sum(bulk[mask]*cohesive[mask])/jnp.maximum(jnp.sum(cohesive[mask]**2),1e-100))
    fitted_relative=float(jnp.linalg.norm((bulk+fitted*cohesive)[mask])/norm_bulk)
    return dict(grid=grid,spacing_over_dx=spacing_ratio,phase_over_dx=phase,consistent_gradient=consistent,segments=count,
                particle_mediated_faces=particle_mediated_faces,
                intact_max_force_N_per_m=intact_error,relative_residual=relative,
                best_scalar_traction_multiplier=fitted,best_scalar_relative_residual=fitted_relative,
                maximum_acceleration_m_s2=float(jnp.max(jnp.linalg.norm(residual[mask]/mass[mask,None],axis=1))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--consistent',action='store_true')
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    rows=[]
    for phase in (0.,.125,.375):
        for spacing in (2.,1.,.5,.25):
            result=audit(48,spacing,phase,args.consistent)
            rows.append(result);print(json.dumps(result),flush=True)
    root=Path(__file__).resolve().parents[1]
    sources=['tools/audit_extrinsic_insertion.py','mpm_jax/extrinsic_crack.py','mpm_jax/polyline_crack.py']
    report=dict(case='Homogeneous, area-preserving prestress; interior nodal equilibrium',rows=rows,
                source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
