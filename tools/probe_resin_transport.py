"""Frozen actual-state transport/divergence diagnosis, without advancing time."""
import argparse,json
from pathlib import Path
from itertools import product
import numpy as np
from scipy.ndimage import binary_erosion
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from mpm_jax.resin_apic import MACLayout,component_shape,particle_to_faces,faces_to_particles,cell_mass
from mpm_jax.resin_mac import viscosity_force,pressure_projection,divergence
from examples.three_strand_resin_geometry import obstacle_from_checkpoint,sha256


def sample_gradient(x,velocity,layout,compatible=False):
    values=[];gradients=[]
    for c in range(3):
        q=(x-jnp.asarray(layout.origin))/layout.dx-jnp.asarray([0. if k==c else .5 for k in range(3)])
        quadratics=[not compatible or k==c for k in range(3)];base=jnp.stack([jnp.floor(q[:,k]-(.5 if quadratics[k] else 0.)).astype(jnp.int32) for k in range(3)],axis=1);f=q-base
        offsets=jnp.asarray(list(product(*[range(3 if b else 2) for b in quadratics])));nodes=base[:,None,:]+offsets;shape=component_shape(layout,c);inside=jnp.all((nodes>=0)&(nodes<jnp.asarray(shape)),axis=2);indices=jnp.sum(nodes*jnp.asarray([shape[1]*shape[2],shape[2],1]),axis=2);indices=jnp.where(inside,indices,np.prod(shape));u=jnp.take(velocity[c].ravel(),indices,mode='fill',fill_value=0.)
        w=[];dw=[]
        for k in range(3):
            z=f[:,k]
            if quadratics[k]:a=jnp.stack([.5*(1.5-z)**2,.75-(z-1)**2,.5*(z-.5)**2],axis=1);b=jnp.stack([z-1.5,2*(1-z),z-.5],axis=1)/layout.dx
            else:a=jnp.stack([1-z,z],axis=1);b=jnp.broadcast_to(jnp.asarray([-1.,1.])/layout.dx,a.shape)
            w.append(a[:,offsets[:,k]]);dw.append(b[:,offsets[:,k]])
        values.append(jnp.sum(u*w[0]*w[1]*w[2],axis=1));gradients.append(jnp.stack([jnp.sum(u*dw[k]*w[(k+1)%3]*w[(k+2)%3],axis=1) for k in range(3)],axis=1))
    return jnp.stack(values,axis=1),jnp.stack(gradients,axis=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('flow',type=Path);p.add_argument('forming',type=Path);a=p.parse_args();plan=json.loads((a.flow/'output/plan.json').read_text());g=obstacle_from_checkpoint(a.forming/'checkpoint.npz',a.forming/'plan.json');layout=MACLayout(tuple(g['shape']),tuple(g['origin']),g['dx']);dt=plan['dt_s'];rho=plan['density_kg_m3'];mu=plan['viscosity_Pa_s']
    with np.load(a.flow/'output/checkpoint.npz',allow_pickle=False) as d:x=d['x'].copy();v=d['v'].copy();C=d['C'].copy();m=d['mass'].copy();active=d['active'].copy();step=int(d['step']);time=float(d['time'])
    fixed=tuple(jnp.asarray(z) for z in g['fixed']);target=tuple(jnp.asarray(z) for z in g['prescribed']);open_cells=jnp.asarray(g['open_cells'])
    @jax.jit
    def snapshot(x,v,C,m,active):
        mass,mom,supports=particle_to_faces(x,v,C,m,active,layout);u=tuple(jnp.where(f,b,p/jnp.where(M>0,M,1.)) for M,p,f,b in zip(mass,mom,fixed,target));force,power=viscosity_force(u,mass,fixed,layout.dx,rho,mu);u=tuple(a+dt*jnp.where((M>0)&~f,1/jnp.maximum(M,1e-300),0.)*F for a,M,f,F in zip(u,mass,fixed,force));cm,ok=cell_mass(x,m,active,layout);fluid=open_cells&(cm>.25*rho*layout.dx**3);pressure=pressure_projection(u,mass,fixed,fluid,layout.dx,dt);pic,Cfit=faces_to_particles(pressure.velocity,supports,layout);quad,Gquad=sample_gradient(x,pressure.velocity,layout);comp,Gcomp=sample_gradient(x,pressure.velocity,layout,True);D=divergence(pressure.velocity,layout.dx)/layout.dx**3
        return pic,Cfit,quad,Gquad,comp,Gcomp,D,cm,fluid,pressure.valid,pressure.relative_residual,pressure.velocity
    result=snapshot(*map(jnp.asarray,[x,v,C,m,active]));pic,Cfit,quad,Gquad,comp,Gcomp,D,cm,fluid,valid,residual,velocity=result
    pic,Cfit,quad,Gquad,comp,Gcomp,D,cm,fluid=map(np.asarray,[pic,Cfit,quad,Gquad,comp,Gcomp,D,cm,fluid]);assert bool(valid);assert np.max(np.abs(pic-quad))<1e-13
    cell=np.floor((x-g['origin'])/layout.dx).astype(int);safe=np.clip(cell,0,g['shape']-1);bulk_cells=binary_erosion(fluid,structure=np.ones((3,3,3),dtype=bool),iterations=2);bulk=active&bulk_cells[safe[:,0],safe[:,1],safe[:,2]]
    trace_quad=np.trace(Gquad,axis1=1,axis2=2);trace_comp=np.trace(Gcomp,axis1=1,axis2=2);trace_C=np.trace(Cfit,axis1=1,axis2=2);rt0=[];trace_rt0=np.zeros(len(x))
    frac=(x-g['origin'])/layout.dx-cell
    for c in range(3):
        lo=cell.copy();hi=cell.copy();hi[:,c]+=1;shape=np.array(component_shape(layout,c));lo=np.clip(lo,0,shape-1);hi=np.clip(hi,0,shape-1);field=np.asarray(velocity[c]);L=field[lo[:,0],lo[:,1],lo[:,2]];R=field[hi[:,0],hi[:,1],hi[:,2]];rt0.append((1-frac[:,c])*L+frac[:,c]*R);trace_rt0+=(R-L)/layout.dx
    rt0=np.stack(rt0,axis=1)
    def stats(y):return dict(count=int(y.size),mean=float(np.mean(y)),rms=float(np.sqrt(np.mean(y*y))),quantiles=np.quantile(y,[0,.01,.5,.99,1]).tolist()) if y.size else dict(count=0)
    def measures(G):
        tr=np.trace(G,axis1=1,axis2=2);det=np.linalg.det(np.eye(3)+dt*G)
        return dict(trace_per_s=stats(tr[bulk]),forward_Euler_det_increment=stats((det-1)[bulk]),Euler_nonlinear_remainder_per_s=stats(((det-1)/dt-tr)[bulk]))
    touches=[]
    for axis in range(3):
        for sign in [-1,1]:
            neighbour=cell.copy();neighbour[:,axis]+=sign;inside=np.all((neighbour>=0)&(neighbour<g['shape']),axis=1);q=np.clip(neighbour,0,g['shape']-1);wall=g['collision_solid'][q[:,0],q[:,1],q[:,2]];face=g['origin'][axis]+(cell[:,axis]+(sign>0))*layout.dx;gap=sign*(face-x[:,axis]);mask=active&inside&wall&(np.abs(gap)<1e-10*layout.dx)
            if mask.any():touches.append(dict(axis=axis,sign=sign,count=int(mask.sum()),quad_normal_into_solid=stats(sign*quad[mask,axis]),compatible_normal_into_solid=stats(sign*comp[mask,axis]),RT0_normal_into_solid=stats(sign*rt0[mask,axis])))
    out=dict(step=step,time_s=time,checkpoint_sha256=sha256(a.flow/'output/checkpoint.npz'),physical_steps_executed=0,operator_snapshots=1,pressure_relative_residual=float(residual),quad_matches_production_PIC_max_error=float(np.max(np.abs(pic-quad))),bulk_particles=int(bulk.sum()),MAC_owned_cell_div_per_s=stats(D[safe[bulk,0],safe[bulk,1],safe[bulk,2]]),APIC_fit_Ctrace_per_s=stats(trace_C[bulk]),production_quadratic_transport=measures(Gquad),compatible_B2normal_B1transverse=measures(Gcomp),RT0_trace_per_s=stats(trace_rt0[bulk]),wall_touching_points=touches,reference_volume_m3=float(m[active].sum()/rho),basis_mass_fraction_in_solid=float(cm[g['solid']].sum()/cm.sum()),scope='One frozen projected-grid evaluation on actual CP. No state update, no replay, no mass/detF/position density correction. Instantaneous Jacobian increments, not reconstructed historical particle J.')
    Path('resin-transport-probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
