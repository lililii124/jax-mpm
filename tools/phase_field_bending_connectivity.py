"""Reference-cell face connectivity of actual exported AT2 damage; no renderer."""
from collections import deque
import numpy as np


def analyze(frame,threshold=.9,dx=.025):
    bounds=np.asarray(frame['reference_cell_bounds']);damage=np.asarray(frame['damage'])
    centers=bounds.mean(axis=-1)
    assert centers.shape==(len(damage),3) and np.isfinite(damage).all()
    axes=[np.unique(centers[:,axis]) for axis in range(3)]
    indices=np.stack([np.searchsorted(axes[axis],centers[:,axis]) for axis in range(3)],axis=-1)
    shape=tuple(len(axis) for axis in axes)
    grid=np.full(shape,-1,dtype=np.int32)
    grid[tuple(indices.T)]=np.arange(len(damage),dtype=np.int32)
    assert np.count_nonzero(grid>=0)==len(damage)
    selected=damage>=threshold
    tip=np.isclose(bounds[:,1,0],.16,rtol=0,atol=1e-12)&(centers[:,0]>.72)&(centers[:,0]<.88)
    left=np.isclose(bounds[:,0,1],.72,rtol=0,atol=1e-12)&(centers[:,1]<.16)
    right=np.isclose(bounds[:,0,0],.88,rtol=0,atol=1e-12)&(centers[:,1]<.16)
    seed=selected&(tip|left|right)
    visited=np.zeros(len(damage),bool);components=[]
    for initial in np.flatnonzero(seed):
        if visited[initial]:continue
        queue=deque([int(initial)]);visited[initial]=True;component=[]
        while queue:
            particle=queue.popleft();component.append(particle)
            ijk=indices[particle]
            for axis in range(3):
                for direction in (-1,1):
                    neighbor=ijk.copy();neighbor[axis]+=direction
                    if neighbor[axis]<0 or neighbor[axis]>=shape[axis]:continue
                    candidate=int(grid[tuple(neighbor)])
                    if candidate<0 or visited[candidate] or not selected[candidate]:continue
                    # Check actual shared face, including the variable cell widths.
                    face_a=bounds[particle,axis,1 if direction>0 else 0]
                    face_b=bounds[candidate,axis,0 if direction>0 else 1]
                    assert abs(face_a-face_b)<1e-12
                    visited[candidate]=True;queue.append(candidate)
        component=np.asarray(component,dtype=int)
        advance=max(0.,float(bounds[component,1,1].max())-.16)
        top=component[np.isclose(bounds[component,1,1],.4,rtol=0,atol=1e-12)]
        z_indices=np.unique(indices[component,2]);top_z=np.unique(indices[top,2])
        widths={int(k):float(bounds[component[np.flatnonzero(indices[component,2]==k)[0]],2,1]-
                             bounds[component[np.flatnonzero(indices[component,2]==k)[0]],2,0]) for k in z_indices}
        components.append(dict(particles=len(component),advance=advance,advance_over_h=advance/dx,
            connected_thickness_fraction=sum(widths[int(k)] for k in z_indices)/.16,
            top_reaching_thickness_fraction=sum(widths[int(k)] for k in top_z)/.16,
            advances_at_least2h=advance>=2*dx-1e-12,
            reference_bounds=[bounds[component,:,0].min(axis=0).tolist(),bounds[component,:,1].max(axis=0).tolist()]))
    return dict(threshold=threshold,maximum_particle_damage=float(damage.max()),
        particles_at_or_above_threshold=int(selected.sum()),
        notch_attached_particles=int(visited.sum()),notch_attached_components=len(components),
        unattached_high_damage_particles=int(selected.sum()-visited.sum()),
        maximum_notch_attached_advance=max((item['advance'] for item in components),default=0.),
        maximum_top_reaching_thickness_fraction=max((item['top_reaching_thickness_fraction'] for item in components),default=0.),
        adjacency='six shared reference-cell faces; no diagonal or void bridging',
        components=components,crack_validated=False)
