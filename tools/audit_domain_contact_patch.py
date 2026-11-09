"""Compare an unfiltered contact trace against integrated bulk forces."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from mpm_jax import Config
from mpm_jax.cut_geometry import polygon
from mpm_jax.cut_fragments import fragment,split_fragment
from mpm_jax.domain_transfer import batch_stencils,particle_to_grid,internal_force
from mpm_jax.implicit_contact import line_interface,trace,scatter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    config=Config(dim=2,n_grid=16,dt=1e-4,gravity=(0.,0.),boundary_cells=0)
    size=17**2;h=config.dx/2
    axis=.25+(np.arange(16)+.5)*h
    points=jnp.asarray(np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2))
    reference=polygon(h*jnp.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]))
    parents=jax.jit(jax.vmap(lambda x:fragment(reference,1200.,x=x)))(points)
    fields=jnp.tile(jnp.arange(2),len(points))
    index=jnp.arange(size);nodes=config.dx*jnp.stack((index//17,index%17),axis=1)
    interior=jnp.all((nodes>.25+1.5*config.dx)&(nodes<.75-1.5*config.dx),axis=1)
    rows=[]
    for angle,phase in ((0.,.17),(.31,-.375),(.68,.25)):
        normal=np.array([np.cos(angle),np.sin(angle)]);tangent=np.array([-normal[1],normal[0]])
        center=np.array([.5,.5])+phase*config.dx*normal
        intersections=[]
        for d in (0,1):
            if abs(tangent[d])>1e-12:
                for boundary in (.25,.75):
                    parameter=(boundary-center[d])/tangent[d]
                    p=center+parameter*tangent
                    if np.all(p>=.25-1e-12) and np.all(p<=.75+1e-12):
                        intersections.append(parameter)
        start=center+min(intersections)*tangent;end=center+max(intersections)*tangent
        children=jax.jit(jax.vmap(lambda s:split_fragment(s,jnp.asarray(normal),normal@center)))(parents)
        state=jax.tree.map(lambda a:a.reshape((-1,)+a.shape[2:]),children)
        support=batch_stencils(state,config)
        support=support._replace(indices=support.indices+fields[:,None]*size)
        mass,_=particle_to_grid(state,support,2*size)
        interface=line_interface(start,end,normal,config)
        contact=trace(interface,mass,jnp.ones((2*size,2)),config)
        assert bool(jnp.all(support.valid)&contact.valid)
        sigma=-jnp.outer(normal,normal)
        bulk=internal_force(jnp.broadcast_to(sigma,state.F.shape),state.reference_area,support,2*size)
        force=scatter(-jnp.ones(len(interface.area)),contact,interface.area,(2*size,2))
        mask=jnp.tile(interior,2)
        residual=float(jnp.linalg.norm(jnp.where(mask[:,None],bulk+force,0.))/jnp.linalg.norm(jnp.where(mask[:,None],bulk,0.)))
        resultants=force.reshape(2,size,2).sum(axis=1)
        moment=float(jnp.sum(nodes[None,:,0]*force.reshape(2,size,2)[:,:,1]-nodes[None,:,1]*force.reshape(2,size,2)[:,:,0]))
        relative_resultant=float(jnp.linalg.norm(resultants[0]-np.linalg.norm(end-start)*normal)/np.linalg.norm(end-start))
        assert residual<2e-11 and relative_resultant<2e-12 and abs(moment)<2e-12
        rows.append(dict(angle_rad=angle,offset_over_dx=phase,quadrature_points=len(interface.area),
                         interior_force_relative_residual=residual,relative_resultant_error=relative_resultant,
                         net_force_norm=float(jnp.linalg.norm(resultants.sum(axis=0))),net_moment=moment))
    sources=['tools/audit_domain_contact_patch.py','mpm_jax/implicit_contact.py','mpm_jax/domain_transfer.py',
             'mpm_jax/cut_fragments.py','mpm_jax/cut_geometry.py','mpm_jax/core.py']
    root=Path(__file__).resolve().parents[1]
    report=dict(rows=rows,max_interior_force_relative_residual=max(r['interior_force_relative_residual'] for r in rows),
                scope='Frozen unit compressive traction integration; no claim of an equilibrated zero-gap penalty state',
                source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
