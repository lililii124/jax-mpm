"""Full plate integration; crack is explicitly a phase-field seed, not void."""
import numpy as np


def reference_cells(dx):
    extent = np.array([.6, .6, .025])
    counts = np.rint(extent/(dx/2)).astype(int)
    if np.max(abs(counts*dx/2-extent)) > 1e-12:
        raise ValueError('Geometry must be exactly tiled by h/2 cells')
    axes = [(np.arange(n)+.5)*dx/2 for n in counts]
    x = np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    bounds = np.stack([x-dx/4,x+dx/4],axis=1)
    return x,np.full(len(x),(dx/2)**3),bounds


def preflight(dx):
    x,v,b = reference_cells(dx)
    return dict(particles=len(x), volume=float(v.sum()),extent=[.6,.6,.025],
        seed_length=.3,void_notch_width=0.,discrete_seed_coefficient_width=2*dx,
        particle_spacing=dx/2,cell_bounds_shape=list(b.shape))
