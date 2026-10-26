"""Exact-boundary reference cells for AT2 notched-beam bending.

Every particle is a true cell centroid with its actual cell volume. Segment
boundaries include all beam/notch faces, independently of mechanics grid h.
"""
import numpy as np

BEAM = (1.6, .4, .16)
NOTCH_X = (.72, .88)
NOTCH_DEPTH = .16
EXACT_VOLUME = (1.6*.4-.16*.16)*.16


def _segments(faces, maximum_spacing):
    pieces = []
    for lower, upper in zip(faces[:-1], faces[1:]):
        count = int(np.ceil((upper-lower)/maximum_spacing-1e-12))
        edge = np.linspace(lower, upper, count+1)
        pieces.append(np.stack((edge[:-1],edge[1:]),axis=-1))
    return np.concatenate(pieces)


def reference_cells(dx):
    if dx not in (.04,.025,.02):
        raise ValueError('Candidate grids are .04, .025 and .02')
    axes = [_segments(faces,dx/2) for faces in
            ((0.,.72,.88,1.6),(0.,.16,.4),(0.,.16))]
    indices = np.stack(np.meshgrid(*[np.arange(len(a)) for a in axes],indexing='ij'),-1).reshape(-1,3)
    bounds = np.stack([axes[i][indices[:,i]] for i in range(3)],axis=1)
    centroid = bounds.mean(axis=-1)
    inside_notch = ((centroid[:,0]>.72)&(centroid[:,0]<.88)&(centroid[:,1]<.16))
    bounds,centroid = bounds[~inside_notch],centroid[~inside_notch]
    volume = np.prod(bounds[:,:,1]-bounds[:,:,0],axis=1)
    # Geometric set is exact; only quadrature and B-spline grid vary with h.
    if not np.isclose(volume.sum(),EXACT_VOLUME,rtol=0,atol=1e-14):
        raise RuntimeError('Beam-minus-notch volume mismatch')
    if np.any(volume<=0) or np.any(bounds[:,:,0]<-1e-14) or np.any(bounds[:,:,1]>np.asarray(BEAM)+1e-14):
        raise RuntimeError('Invalid reference cell')
    return centroid,volume,bounds


def preflight(dx):
    x,volume,bounds = reference_cells(dx)
    return dict(dx=dx,particles=len(x),beam_dimensions=list(BEAM),
        notch_x=list(NOTCH_X),notch_depth=NOTCH_DEPTH,
        exact_volume=EXACT_VOLUME,quadrature_volume=float(volume.sum()),
        minimum_cell_width=float(np.min(bounds[:,:,1]-bounds[:,:,0])),
        maximum_cell_width=float(np.max(bounds[:,:,1]-bounds[:,:,0])),
        cell_centroid_volume_rule=True,executed_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps([preflight(h) for h in (.04,.025,.02)],indent=2))
