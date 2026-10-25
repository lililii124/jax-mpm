"""NumPy evaluation of stored quadratic tensor-product B-spline coefficients."""
from itertools import product
import numpy as np


def evaluate(points, coordinates, coefficients, dx, gradient=False):
    points = np.asarray(points, dtype=float)
    origin = coordinates.min(axis=0)
    integer = np.rint((coordinates-origin)/dx).astype(int)
    shape = tuple(integer.max(axis=0)+1)
    table = np.full(shape, -1, dtype=np.int64)
    table[tuple(integer.T)] = np.arange(len(coordinates))
    if not np.allclose(coordinates, origin+integer*dx, rtol=0, atol=1e-12):
        raise RuntimeError('Stored nodes are not an exact Cartesian grid')
    q = (points-origin)/dx
    base = np.floor(q-.5).astype(int)
    f = q-base
    w = np.stack((.5*(1.5-f)**2, .75-(f-1)**2, .5*(f-.5)**2), axis=1)
    dw = np.stack((f-1.5, -2*(f-1), f-.5), axis=1)/dx
    result = np.zeros(len(points))
    derivative = np.zeros((len(points), 3))
    for offset in product(range(3), repeat=3):
        node = base+offset
        if np.any(node < 0) or np.any(node >= np.asarray(shape)):
            raise RuntimeError('Sample outside complete stored B-spline support')
        ids = table[tuple(node.T)]
        if np.any(ids < 0):
            raise RuntimeError('Missing node in stored B-spline support')
        selected = w[:, offset, np.arange(3)]
        local = coefficients[ids]
        result += local*np.prod(selected, axis=1)
        for axis in range(3):
            derivative[:, axis] += local*dw[:, offset[axis], axis]*selected[:, (axis+1)%3]*selected[:, (axis+2)%3]
    return (result, derivative) if gradient else result


def sampling_grid(dx):
    spacing = dx/4
    axes = [np.linspace(0, extent, round(extent/spacing)+1) for extent in (.6, 1., .1)]
    shape = tuple(map(len, axes))
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    ids = np.arange(len(points)).reshape(shape)
    cells = np.stack([ids[i:shape[0]-1+i, j:shape[1]-1+j, k:shape[2]-1+k].ravel()
        for i,j,k in ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))], axis=1)
    # Remove cells intersecting the physical pre-existing notch, including
    # boundary-straddling cells. Damage never controls geometry membership.
    corners = points[cells]
    intersects = ((corners[:,:,0].min(axis=1) < .25) &
        (corners[:,:,1].min(axis=1) < .58) & (corners[:,:,1].max(axis=1) > .42))
    return points, cells[~intersects], spacing
