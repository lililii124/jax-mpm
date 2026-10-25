"""Actual formed-particle obstacle and inlet data for independent resin filling.

The fixed obstacle is voxelized from F-advected particle quadrature cells, never
from a prescribed helix. Staircase boundary uncertainty is grid scale. This is
not a reconstructed smooth fibre surface or a fluid/structure coupling model.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):value.update(block)
    return value.hexdigest()


def material_cells(points,centres,F,spacing,*,batch=2048):
    """Union of actual advected local material cubes with a conservative query.

    Each parallelepiped has bounding sphere .5*sqrt(3)*spacing*||F||_2.
    query_ball_point includes every possible cell, with no fixed nearest-k cap.
    """
    tree=cKDTree(centres);inverse=np.linalg.inv(F)
    radius=.5*np.sqrt(3)*spacing*np.linalg.svd(F,compute_uv=False)[:,0].max()
    inside=np.zeros(len(points),dtype=bool)
    for lo in range(0,len(points),batch):
        group=points[lo:lo+batch];candidates=tree.query_ball_point(group,radius,workers=1)
        for j,(point,indices) in enumerate(zip(group,candidates)):
            if indices:
                indices=np.asarray(indices);local=np.einsum('nij,nj->ni',inverse[indices],point-centres[indices]);inside[lo+j]=np.any(np.max(np.abs(local),axis=1)<=spacing*.5)
    return inside


def obstacle_from_checkpoint(checkpoint,plan,*,allow_partial=False,dx=.00025):
    """Return grid-cell obstacle only after source state provenance is recorded."""
    plan_sha=sha256(plan)
    plan=json.loads(Path(plan).read_text())
    with np.load(checkpoint,allow_pickle=False) as d:
        step=int(d['step']);x=d['state_x'].copy();F=d['state_F'].copy();ids=d['state_fibre_id'].copy()
        if not bool(d['state_valid']) or not np.all(d['history_valid']) or np.any(d['history_damage']!=0):raise ValueError('Invalid or damaged forming checkpoint')
        if step!=plan['steps'] and not allow_partial:raise ValueError('Production fill requires completed actual forming; partial only for operator preparation')
        if not np.isfinite(F).all() or np.linalg.det(F).min()<=0:raise ValueError('Invalid obstacle geometry')
    radius=.0032;left=-.001;right=.025
    spacing=plan['geometry']['spacing_m']
    radial_bound=np.linalg.norm(x[:,1:],axis=1)+.5*np.sqrt(3)*spacing*np.linalg.svd(F,compute_uv=False)[:,0]
    axial_bound=.5*spacing*np.sum(np.abs(F[:,0,:]),axis=1)
    if radial_bound.max()>=radius or np.any(x[:,0]-axial_bound<=left) or np.any(x[:,0]+axial_bound>=right):
        raise ValueError('Actual formed material-cell envelope intersects proposed mold; revise documented mold dimensions, never clip fibres')
    origin=np.array([left-3*dx,-radius-3*dx,-radius-3*dx]);shape=np.ceil((np.array([right+3*dx,radius+3*dx,radius+3*dx])-origin)/dx).astype(int)
    axes=[origin[k]+(np.arange(shape[k])+.5)*dx for k in range(3)];grid=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);flat=grid.reshape(-1,3)
    cavity=(flat[:,0]>left)&(flat[:,0]<right)&(np.sum(flat[:,1:]**2,axis=1)<radius**2)
    fibre=np.zeros(len(flat),dtype=bool);fibre[cavity]=material_cells(flat[cavity],x,F,plan['geometry']['spacing_m'])
    solid=(~cavity|fibre).reshape(tuple(shape));open_cells=(cavity&~fibre).reshape(tuple(shape))
    fixed=[];target=[]
    for component in range(3):
        before=np.pad(solid,[(int(k==component),0) for k in range(3)],constant_values=True)
        after=np.pad(solid,[(0,int(k==component)) for k in range(3)],constant_values=True)
        mask=before|after
        face_axes=[origin[k]+(np.arange(mask.shape[k])+(0. if k==component else .5))*dx for k in range(3)]
        face=np.stack(np.meshgrid(*face_axes,indexing='ij'),axis=-1)
        radial=np.sum(face[...,1:]**2,axis=-1)
        inlet=(face[...,0]<=left+dx*.1)&(radial<.0025**2)
        outlet=(face[...,0]>=right-dx*.1)&(radial<.0025**2)
        # Prescribed physical inlet extends through support ghost layers. Outlet
        # is open to zero gauge pressure, never a closed-end volume source.
        mask=(mask|inlet)&~outlet
        velocity=np.zeros(mask.shape)
        if component==0:velocity[inlet]=.2*(1-radial[inlet]/.0025**2)
        fixed.append(mask);target.append(velocity)
    collision_solid=solid.copy()
    outlet_cells=(grid[...,0]>=right)&(np.sum(grid[...,1:]**2,axis=-1)<.0025**2)
    collision_solid[outlet_cells]=False
    receipt=dict(checkpoint_sha256=sha256(checkpoint),forming_plan_sha256=plan_sha,actual_forming_step=step,full_turn=step==plan['steps'],fixed_fibres=True,grid_dx_m=dx,grid_origin_m=origin.tolist(),grid_shape=shape.tolist(),conservative_radial_clearance_m=float(radius-radial_bound.max()),cylinder_radius_m=radius,cylinder_limits_m=[left,right],voxelized_fibre_volume_m3=float(fibre.sum()*dx**3),cavity_void_volume_m3=float(open_cells.sum()*dx**3),boundary_scope='Union of actual F-advected quadrature cells evaluated at cell centres; staircase fixed boundary, grid-scale geometry error, not smooth surface/FSI/cure')
    return dict(origin=origin,shape=shape,dx=dx,solid=solid,collision_solid=collision_solid,open_cells=open_cells,fixed=tuple(fixed),prescribed=tuple(target),fibre_x=x,fibre_id=ids,receipt=receipt)


def inlet_schedule(*,radius=.0025,spacing=.0002,maximum_speed=.2,duration=.32,left=-.001,density=1100.):
    """Finite actual inlet births: parabolic profile, mass enters only at birth.

    A line emits successive material particles spacing/u apart. No particles are
    preplaced inside a final cylinder. The schedule alone does not add mass to
    the simulation; runner must log every activation and its kinetic energy.
    """
    a=(np.arange(int(np.ceil(2*radius/spacing)))+.5)*spacing-radius;y,z=np.meshgrid(a,a,indexing='ij');yz=np.column_stack((y.ravel(),z.ravel()));yz=yz[np.sum(yz*yz,axis=1)<radius**2]
    u=maximum_speed*(1-np.sum(yz*yz,axis=1)/radius**2);counts=np.floor(duration*u/spacing).astype(int)
    stream=np.repeat(np.arange(len(yz)),counts);layer=np.concatenate([np.arange(n) for n in counts]);birth=(layer+.5)*spacing/u[stream]
    x=np.column_stack((np.full(len(stream),left+spacing*.5),yz[stream]));v=np.column_stack((u[stream],np.zeros((len(stream),2))))
    order=np.argsort(birth,kind='stable');return dict(x=x[order],v=v[order],birth_time=birth[order],mass=np.full(len(order),density*spacing**3),nominal_flow_m3_per_s=float(np.sum(u)*spacing**2),capacity=len(order))


def continued_inlet_schedule(horizons,**kwargs):
    """Append future physical births, preserving every previous particle ID."""
    horizons=list(horizons);assert horizons and all(b>a for a,b in zip(horizons,horizons[1:]))
    source=inlet_schedule(duration=horizons[0],**kwargs)
    for previous,end in zip(horizons,horizons[1:]):
        extra=inlet_schedule(duration=end,**kwargs);future=extra['birth_time']>previous+1e-12
        for name in ['x','v','mass','birth_time']:source[name]=np.concatenate((source[name],extra[name][future]),axis=0)
        source['capacity']=len(source['mass'])
    return source
