"""Full-domain 3x3 shared-cell 3-D TL-MPM blast design.

The 3x3 tiles have independent particles and deformation states. Only the
design logits are shared. Sigmoid density is a soft-material relaxation, not
an explicit void/solid topology; thresholded designs require re-simulation.
"""
import argparse
import json
import math
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mpm_jax.shared_cell_design import project_fixed_volume
from mpm_jax.periodic_filter import filter_cell_logits


def build(args):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax import lax
    from mpm_jax.compact_tl import (rectilinear_support, reference_grid,
        point_material, surface_weights, initial_state, step)

    nx, ny, nz = args.cell
    pressure_placement = getattr(args, 'pressure_placement', 'top')
    density_floor = float(getattr(args, 'density_floor', 0.))
    if pressure_placement not in ('top', 'exposed') or not 0 <= density_floor < args.volume_fraction:
        raise ValueError('Invalid pressure placement or density floor')
    if pressure_placement == 'exposed' and args.skin_layers:
        raise ValueError('Exposed pressure requires an open cell without face sheets')
    if min(nx,ny,nz) < 1: raise ValueError('Cell grid counts must be positive')
    if (min(args.cell_size)<=0 or args.skin_layers<0 or args.skin<0 or
            ((args.skin_layers==0) != (args.skin==0))):
        raise ValueError('Cell dimensions must be positive and skin thickness must match its layers')
    if args.steps<1 or args.dt<=0 or args.pulse<=0 or args.blast_radius<=0 or args.pressure<=0:
        raise ValueError('Positive duration, timestep, radius and pressure are required')
    if args.displacement_scale<=0 or args.checkpoint_block<1 or getattr(args,'filter_passes',0)<0:
        raise ValueError('Positive displacement scale and checkpoint block are required')
    # Each voxel has eight integration particles. Each tile has independent
    # dynamics and shares only its material design variables.
    gx, gy, gz = nx*3, ny*3, nz+2*args.skin_layers
    size = np.asarray(args.cell_size)
    h = np.array((size[0]/nx, size[1]/ny, size[2]/nz))
    if not np.isclose(args.skin_layers*h[2],args.skin,rtol=0,atol=1e-12):
        raise ValueError('Core z spacing must tile the requested skin thickness exactly')
    grid = np.stack(np.meshgrid((np.arange(gx)+.5)*h[0],
                                (np.arange(gy)+.5)*h[1],
                                (np.arange(gz)+.5)*h[2], indexing='ij'), -1)
    offsets=np.asarray(list(product((-1/(2*math.sqrt(3)),1/(2*math.sqrt(3))),repeat=3)))*h
    pos=(grid.reshape(-1,3)[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    voxel_ids=np.repeat(np.arange(gx*gy*gz),8)
    zidx=np.broadcast_to(np.arange(gz)[None,None,:],(gx,gy,gz)).reshape(-1)[voxel_ids]
    core = (zidx >= args.skin_layers) & (zidx < gz-args.skin_layers)
    core_ids = np.flatnonzero(core)
    local = np.stack(np.unravel_index(voxel_ids[core_ids], (gx,gy,gz)), axis=-1)
    local[:,2] -= args.skin_layers
    cell_local = local.copy(); cell_local[:,0] %= nx; cell_local[:,1] %= ny
    flat_design = np.ravel_multi_index((cell_local[:,2],cell_local[:,1],cell_local[:,0]), (nz,ny,nx))
    phi_particles = np.zeros(len(pos)); phi_particles[~core] = 1.
    volume = np.full(len(pos), float(np.prod(h))/8); support = rectilinear_support(pos, volume, h)
    target_mass=float(args.solid_density*(np.sum(volume[~core])+np.sum(volume[core])*args.volume_fraction))
    xy = np.stack(np.meshgrid((np.arange(gx)+.5)*h[0],(np.arange(gy)+.5)*h[1],indexing='ij'),-1).reshape(-1,2)
    face_area=np.full(len(xy),h[0]*h[1])
    if pressure_placement == 'exposed':
        xy_layers=np.repeat(xy,nz,axis=0)
        heights=np.tile(np.arange(nz),len(xy))*h[2]
        fi,fw=surface_weights(support,np.column_stack((xy_layers,heights+h[2])))
        bi,bw=surface_weights(support,np.column_stack((xy_layers,heights)))
        fi,fw=fi.reshape(len(xy),nz,-1),fw.reshape(len(xy),nz,-1)
        bi,bw=bi.reshape(len(xy),nz,-1),bw.reshape(len(xy),nz,-1)
    else:
        facepts=np.column_stack((xy,np.full(len(xy),gz*h[2])))
        fi,fw=surface_weights(support,facepts)
        bi,bw=surface_weights(support,np.column_stack((xy,np.zeros(len(xy)))))
    # Clamp a narrow perimeter frame on both sheets.
    xyz=support.nodes
    edge = np.asarray(getattr(args, 'fixed_edge_width', None) if getattr(args, 'fixed_edge_width', None) is not None
                      else [h[0]*1.01, h[1]*1.01], dtype=float)
    if edge.shape != (2,) or not np.isfinite(edge).all() or np.any(edge <= 0) or np.any(edge >= (gx*h[0]/2, gy*h[1]/2)):
        raise ValueError('Fixed edge widths must be positive and less than half the domain')
    fixed=((xyz[:,0]<edge[0])|(xyz[:,0]>gx*h[0]-edge[0])|
           (xyz[:,1]<edge[1])|(xyz[:,1]>gy*h[1]-edge[1]))[:,None]
    fixed=np.broadcast_to(fixed,xyz.shape)
    center=np.array((gx*h[0]/2,gy*h[1]/2)); radius=args.blast_radius
    footprint=np.exp(-.5*np.sum(((xy-center)/radius)**2,axis=1)); footprint/=np.sum(footprint*face_area)/np.sum(face_area)
    face_scale=((fw*face_area[:,None,None]*footprint[:,None,None]) if pressure_placement=='exposed'
                else (fw*face_area[:,None]*footprint[:,None])).reshape(-1)
    node_ids=fi.reshape(-1)
    dt=args.dt; nsteps=args.steps
    wave=math.sqrt(max(args.skin_young,args.core_young)*(1-.28)/(args.solid_density*(1+.28)*(1-2*.28)))
    cfl=dt*wave*math.sqrt(float(np.sum(1/h**2)))
    if cfl > args.cfl_limit: raise ValueError(f'CFL {cfl:g} exceeds limit {args.cfl_limit:g}')
    def objective(logits):
        filtered=filter_cell_logits(logits.reshape(nz,ny,nx),getattr(args,'filter_passes',0))
        rho_core=project_fixed_volume(filtered,args.volume_fraction,beta=args.beta).reshape(-1)
        rho=jnp.asarray(phi_particles).at[jnp.asarray(core_ids)].set(rho_core[jnp.asarray(flat_design)])
        skin_mask=jnp.asarray(~core)
        effective=density_floor+(1-density_floor/args.volume_fraction)*rho
        young=jnp.where(skin_mask,args.skin_young,args.core_young*effective**2)
        density=jnp.where(skin_mask,args.solid_density,args.solid_density*effective)
        ref=reference_grid(support,density); mat=point_material(young,.28)
        mass_ok=jnp.abs(jnp.sum(ref.particle_mass)/target_mass-1.)<=1e-10
        if pressure_placement=='exposed':
            from mpm_jax.exposed_pressure import exposed_weights
            columns=jnp.tile(rho_core.reshape(nz,ny,nx),(1,3,3)).transpose(2,1,0).reshape(len(xy),nz)
            front=exposed_weights(columns,from_top=True)
            back_visibility=exposed_weights(columns,from_top=False)
            scale=jnp.asarray(face_scale).reshape(len(xy),nz,-1)*front[...,None]
        else:
            scale=jnp.asarray(face_scale)
        load=jnp.zeros((len(xyz),3),dtype=jnp.float64).at[jnp.asarray(node_ids),2].add(-args.pressure*scale.reshape(-1))
        zero=jnp.zeros_like(load)
        state=initial_state(ref,mat,zero)
        def body(carry,i):
            st,scale,power=carry
            t0=i*dt; t1=(i+1)*dt
            p0=jnp.sin(jnp.pi*jnp.minimum(t0/args.pulse,1.))**2*(t0<args.pulse)
            p1=jnp.sin(jnp.pi*jnp.minimum(t1/args.pulse,1.))**2*(t1<args.pulse)
            nxt,_,_=step(st,ref,mat,dt,load*p0,load*p1,fixed)
            back=jnp.sum(nxt.displacement[jnp.asarray(bi),2]*jnp.asarray(bw),axis=-1)
            if pressure_placement=='exposed': back=jnp.sum(back*back_visibility,axis=1)
            following=lax.stop_gradient(jnp.maximum(scale,jnp.max(jnp.abs(back))))
            power=power*(scale/following)**16+jnp.sum((back/following)**16)
            return (nxt,following,power),None
        block=max(1,args.checkpoint_block)
        def chunk(carry,start):
            def inner(c,i):
                return lax.cond(i<nsteps,lambda x:body(x,i)[0],lambda x:x,c),None
            return lax.scan(inner,carry,start*block+jnp.arange(block))[0],None
        (state,scale,power),_=lax.scan(jax.checkpoint(chunk),
            (state,jnp.asarray(1e-30,dtype=jnp.float64),jnp.asarray(0.,dtype=jnp.float64)),
            jnp.arange(math.ceil(nsteps/block)))
        # p-norm is a smooth upper bound on the sampled space-time maximum.
        merit=scale*jnp.maximum(power,1e-250)**(1/16)/args.displacement_scale
        return jnp.where(state.valid & mass_ok,merit,jnp.asarray(jnp.inf,dtype=merit.dtype))
    return objective, dict(particles=len(pos),nodes=len(xyz),design_variables=nx*ny*nz,
        cells=(gx,gy,gz),dt=dt,steps=nsteps,tiles=9,full_domain=True,
        cell_size_m=size.tolist(),plate_size_m=[float(gx*h[0]),float(gy*h[1]),float(gz*h[2])],
        fixed_edge_width_m=edge.tolist(),
        fixed_face_sheets=args.skin_layers>0,
        blast_center_m=center.tolist(),blast_gaussian_sigma_m=radius,
        adjacent_cell_center_pressure_ratio=float(np.exp(-.5*(size[0]/radius)**2)),
        objective='16-norm upper bound of absolute back-face displacement over space and time',
        fixed_mass_fraction=args.volume_fraction,filter_passes=getattr(args,'filter_passes',0),soft_density=True,cfl_number=cfl,
        pressure_placement=pressure_placement,density_floor=density_floor,
        nominal_total_mass_kg=target_mass,
        proxy_material=('Soft-material relaxation: effective density fraction is floor + '
                        '(1 - floor / volume_fraction) * rho; Young modulus scales with its square. No true void.'),
        pressure_qualification=('Smoothed visibility of column surfaces; no cavity-side pressure or fluid dynamics.'
                                if pressure_placement=='exposed' else 'Pressure acts on the fixed top plane.'),
        uncheckpointed_state_bytes_lower_bound=8*nsteps*(9*len(xyz)+9*len(pos)),
        checkpoint_block=args.checkpoint_block,
        physical_horizon_s=nsteps*dt,validity='nonfinite objective marks TL state invalid')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',type=int,nargs=3,default=(16,16,20))
    p.add_argument('--cell-size',type=float,nargs=3,default=(.1,.1,.02))
    p.add_argument('--fixed-edge-width',type=float,nargs=2)
    p.add_argument('--skin',type=float,default=.002);p.add_argument('--skin-layers',type=int,default=2);p.add_argument('--volume-fraction',type=float,default=.3)
    p.add_argument('--beta',type=float,default=5.);p.add_argument('--skin-young',type=float,default=70e9)
    p.add_argument('--filter-passes',type=int,default=0)
    p.add_argument('--pressure-placement',choices=('top','exposed'),default='top')
    p.add_argument('--density-floor',type=float,default=0.)
    p.add_argument('--core-young',type=float,default=20e9);p.add_argument('--solid-density',type=float,default=2700.)
    p.add_argument('--pressure',type=float,default=2e6);p.add_argument('--pulse',type=float,default=2e-5)
    p.add_argument('--blast-radius',type=float,default=.08);p.add_argument('--dt',type=float,default=1e-8);p.add_argument('--cfl-limit',type=float,default=.1)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--checkpoint-block',type=int,default=32)
    p.add_argument('--displacement-scale',type=float,default=.005)
    p.add_argument('--init',choices=('uniform','random'),default='uniform');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--check',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    objective,plan=build(a);n=math.prod(a.cell);rng=np.random.default_rng(a.seed)
    if a.plan:
        print(json.dumps(plan,indent=2));return
    q=np.zeros(n) if a.init=='uniform' else rng.normal(0,.1,n)
    import jax, jax.numpy as jnp
    val,grad=jax.value_and_grad(objective)(jnp.asarray(q))
    result=dict(plan=plan,objective=float(val),gradient_norm=float(jnp.linalg.norm(grad)),gradient_finite=bool(jnp.all(jnp.isfinite(grad))),init=a.init)
    if a.check:
        d=jnp.asarray(rng.normal(size=n));d/=jnp.linalg.norm(d);eps=1e-4
        fd=(objective(jnp.asarray(q)+eps*d)-objective(jnp.asarray(q)-eps*d))/(2*eps)
        ad=jnp.vdot(grad,d); scale=jnp.maximum(jnp.maximum(jnp.abs(fd),jnp.abs(ad)),1e-30)
        measurable=bool(scale>1e-15)
        result.update(directional_fd=float(fd),directional_ad=float(ad),absolute_error=float(abs(fd-ad)),
                      relative_error=float(abs(fd-ad)/scale) if measurable else None,
                      gradient_check_informative=measurable)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
