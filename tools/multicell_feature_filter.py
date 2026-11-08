"""Measure axis-aligned single-voxel runs in a periodic cell."""
import numpy as np


def single_voxel_run_fraction(mask):
    cell = np.asarray(mask, dtype=bool)
    if cell.ndim != 3 or any(size < 1 for size in cell.shape) or not cell.any():
        raise ValueError('A nonempty 3D cell is required')
    before_z = np.pad(cell, ((1, 0), (0, 0), (0, 0)))[:-1]
    after_z = np.pad(cell, ((0, 1), (0, 0), (0, 0)))[1:]
    neighbors = ((before_z, after_z),
                 (np.roll(cell, 1, axis=1), np.roll(cell, -1, axis=1)),
                 (np.roll(cell, 1, axis=2), np.roll(cell, -1, axis=2)))
    thin = np.logical_or.reduce([cell & ~before & ~after for before, after in neighbors])
    return float(thin.sum()/cell.sum())
