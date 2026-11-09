"""JIT-compiled compatible cut-cell integrals of the actual MPM basis."""

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
from mpm_jax.core import stencil
from mpm_jax.cut_geometry import polygon,split,bulk_quadrature,boundary_quadrature


def operator(grid=12):
    config=Config(dim=2,n_grid=grid,dt=1e-5,gravity=(0.,0.),boundary_cells=0)
    knots=(np.arange(grid)+.5)*config.dx
    cuts=np.unique(np.r_[.125,knots[(knots>.125)&(knots<.875)],.875])
    cells=np.asarray([[[a,c],[b,c],[b,d],[a,d]]
                       for a,b in zip(cuts[:-1],cuts[1:]) for c,d in zip(cuts[:-1],cuts[1:])])
    polygons=jax.vmap(polygon)(jnp.asarray(cells))
    size=(grid+1)**2;idx=jnp.arange(size)
    nodes=config.dx*jnp.stack((idx//(grid+1),idx%(grid+1)),axis=1)
    interior=jnp.all((nodes>=.125+1.5*config.dx-1e-12)&(nodes<=.875-1.5*config.dx+1e-12),axis=1)
    sigma=jnp.array([[2.3,-.7],[-.7,1.1]])

    @jax.jit
    def integrate(normal,offset):
        plus,minus=jax.vmap(lambda p:split(p,normal,offset))(polygons)

        def field(polys):
            points,weights=jax.vmap(bulk_quadrature)(polys)
            points=points.reshape(-1,2);weights=weights.reshape(-1)
            support=stencil(points,config,active=weights>0)
            bulk=jnp.zeros((size,2)).at[support.indices].add(weights[:,None,None]*support.gradients)
            face_points,face_weights,normals=jax.vmap(boundary_quadrature)(polys)
            face_weights=jnp.where((polys.count>=3)[:,None],face_weights,0.)
            face_points=face_points.reshape(-1,2);face_weights=face_weights.reshape(-1);normals=normals.reshape(-1,2)
            on_crack=jnp.abs(face_points@normal-offset)<128*jnp.finfo(points.dtype).eps
            face_weights=jnp.where(on_crack,face_weights,0.)
            trace=stencil(face_points,config,active=face_weights>0)
            face=jnp.zeros((size,2)).at[trace.indices].add(face_weights[:,None,None]*trace.weights[:,:,None]*normals[:,None,:])
            residual=(bulk-face)@sigma.T
            masked=jnp.where(interior[:,None],residual,0.)
            reference=jnp.where(interior[:,None],bulk@sigma.T,0.)
            relative=jnp.linalg.norm(masked)/jnp.maximum(jnp.linalg.norm(reference),1e-100)
            return relative,weights.sum(),jnp.sum(weights[:,None]*points,axis=0),jnp.all(polys.valid)&support.valid&trace.valid

        a,b=field(plus),field(minus)
        return jnp.maximum(a[0],b[0]),a[1]+b[1],a[2]+b[2],a[3]&b[3]
    return integrate,len(cells)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid',type=int,default=12)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.grid<12:
        parser.error('Use at least twelve grid cells')
    jax.config.update('jax_enable_x64',True)
    integrate,cells=operator(args.grid);rows=[]
    for angle in (0.,.31,.68):
        n=jnp.array([-np.sin(angle),np.cos(angle)])
        for phase in (-.375,0.,.25):
            offset=n@jnp.array([.5,.5])+phase/args.grid
            started=time.perf_counter()
            residual,area,moment,valid=integrate(n,offset)
            residual=float(residual);elapsed=time.perf_counter()-started
            np.testing.assert_allclose(area,.75**2,atol=2e-14)
            np.testing.assert_allclose(moment,np.array([.5,.5])*.75**2,atol=2e-14)
            if not bool(valid) or residual>2e-12:
                raise RuntimeError('Cut-cell geometry or traction-balance check failed')
            row=dict(angle_rad=angle,phase_over_dx=phase,relative_nodal_residual=residual,
                     area=float(area),first_moment=np.asarray(moment).tolist(),elapsed_s=elapsed,
                     includes_compilation=not rows)
            rows.append(row);print(json.dumps(row),flush=True)
    root=Path(__file__).resolve().parents[1]
    sources=['mpm_jax/cut_geometry.py','mpm_jax/core.py','tools/audit_jax_cut_geometry.py']
    report=dict(grid=args.grid,knot_cells=cells,jax_version=jax.__version__,backend=jax.default_backend(),
                rows=rows,max_relative_residual=max(r['relative_nodal_residual'] for r in rows),
                compiled_geometry=True,production_dynamics_integrated=False,
                limitations=['Convex pieces and through-cuts; split at spline knots before quadrature',
                             'No evolving fragment constitutive state, APIC moments or fracture-front update in this audit'],
                source_sha256={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
