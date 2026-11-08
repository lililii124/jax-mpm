"""Discrete projected-pressure allocation on an occupied column grid."""
import numpy as np


def allocate_projected_pressure(weights, occupied, mode='omit'):
    """Allocate a nonnegative projected footprint to occupied columns.

    Redistribution preserves the total normal force, but may alter its first
    moment. It is an equal-impulse approximation, not a gas-flow solution.
    """
    weights=np.asarray(weights,dtype=float);occupied=np.asarray(occupied,dtype=bool)
    if (weights.ndim!=2 or weights.shape!=occupied.shape or not occupied.any() or
            not np.isfinite(weights).all() or np.any(weights<0)):
        raise ValueError('A finite nonnegative footprint and occupied columns are required')
    if mode not in ('omit','redistribute'):
        raise ValueError('Unknown projected-pressure allocation')
    result=np.where(occupied,weights,0.)
    if mode=='redistribute':
        targets=np.argwhere(occupied)
        for point in np.argwhere(~occupied):
            distance=np.sum((targets-point)**2,axis=1)
            nearest=targets[distance==distance.min()]
            result[nearest[:,0],nearest[:,1]]+=weights[tuple(point)]/len(nearest)
    return result
