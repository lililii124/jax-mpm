"""Grid-resolution actual particle geometry diagnostics; NumPy only."""
import numpy as np
from scipy.spatial import cKDTree


def inspect(state,plan,geometry,*,collar_target=None):
    x=np.asarray(state.x);F=np.asarray(state.F);ids=np.asarray(state.fibre_id);dx=plan['dx'];spacing=plan['geometry']['spacing_m']
    bounds=np.asarray(plan['bounds']);lower=np.floor(bounds[:,0]/dx).astype(int)-3;upper=np.ceil(bounds[:,1]/dx).astype(int)+3
    base=np.floor(x/dx-.5).astype(int);support=bool(np.all(base>=lower)&np.all(base+2<=upper))
    t=float(state.time);duration=plan['duration'];ramp=plan.get('protocol',{}).get('rotation_ramp_s',.001);z=min(max(t,0),duration)
    primitive=z*z/(2*ramp) if z<ramp else z-ramp/2 if z<=duration-ramp else duration-ramp-(duration-z)**2/(2*ramp)
    theta=2*np.pi*primitive/(duration-ramp);c,s=np.cos(theta),np.sin(theta);Q=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    z=np.clip(t/.002,0,1);extension=.000105*z*z*(3-2*z);errors=[]
    for side,mask in enumerate(geometry['masks']):
        target=geometry['x'][mask]@(Q.T if side else np.eye(3));target[:,0]+=(1 if side else -1)*extension/2
        if collar_target is not None:target=collar_target(side,mask,t)
        errors.append(float(np.linalg.norm(x[mask]-target,axis=1).max()))
    pairs=[]
    # Frobenius norm bounds each advected Cartesian cell's circumradius.
    radius_bound=.5*np.sqrt(3)*spacing*np.linalg.norm(F,axis=(1,2)).max()
    trees={}
    for a,b in ((0,1),(0,2),(1,2)):
        xa=x[ids==a];xb=x[ids==b];Fa=F[ids==a]
        if a not in trees:trees[a]=cKDTree(xa)
        tree=trees[a]
        # If any bounded query is finite, its minimum is the exact global
        # minimum: excluded distances exceed this upper bound. The bound
        # also contains every possible advected-cell intrusion candidate.
        bound=max(2*dx,radius_bound)
        distance,index=tree.query(xb,k=1,workers=1,distance_upper_bound=bound)
        if not np.isfinite(distance).any():distance,index=tree.query(xb,k=1,workers=1)
        candidate=np.flatnonzero(distance<=radius_bound)
        inside=deep=0;worst=0.
        for first in range(0,len(candidate),2048):
            selection=candidate[first:first+2048];_,near=tree.query(xb[selection],k=8,workers=1)
            local=np.linalg.solve(Fa[near],(xb[selection,None,:]-xa[near])[...,None])[...,0]
            normalized=np.max(np.abs(local),axis=-1)/(.5*spacing)
            minimum=normalized.min(axis=1);inside+=int(np.sum(minimum<1));deep+=int(np.sum(minimum<.8))
            worst=max(worst,float(np.maximum(1-minimum,0).max(initial=0))*.5*spacing)
        pairs.append(dict(strands=[a,b],minimum_particle_distance_m=float(distance.min()),other_centres_inside_advected_cells=inside,deep_cell_intrusions=deep,maximum_material_cell_intrusion_m=worst))
    return dict(time_s=t,right_angle_rad=theta,full_support=support,collar_error_m=max(errors),pairs=pairs,geometry_valid=support and max(errors)<1e-9 and all(p['deep_cell_intrusions']==0 for p in pairs),
        scope=f'Actual point distances plus local F-advected quadrature-cell intrusion diagnostic at{spacing*1000:g}mm spacing/{dx*1000:g}mm grid; not exact reconstructed surface or sub-grid nonpenetration proof. Deep intrusion means a foreign centre inside0.8 of the cell half-width; rejection is diagnostic, not an altered force law.')
