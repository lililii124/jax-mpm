"""Separate locally disconnected material patches on shared grid nodes."""
from collections import deque
from typing import NamedTuple

import numpy as np

from .compact_tl import Support


class SplitSupport(NamedTuple):
    support: Support
    voxel_coordinates: np.ndarray
    pair_keys: np.ndarray
    pair_fields: np.ndarray
    node_count: int


def split_disconnected_node_support(support):
    positions = np.asarray(support.positions)
    spacing = np.asarray(support.spacing)
    voxel = np.floor(positions/spacing).astype(np.int64)
    coordinates, particle_voxels = np.unique(voxel, axis=0, return_inverse=True)
    voxel_count = len(coordinates)
    if voxel_count == 0:
        raise ValueError('Cannot split empty material')
    old_indices = np.asarray(support.indices)
    if old_indices.ndim != 2 or old_indices.shape[0] != len(positions):
        raise ValueError('Support indices do not match particles')
    if np.any(np.asarray(support.weights) <= 0):
        raise ValueError('Split support requires positive quadrature weights')
    flat_keys = old_indices.astype(np.int64).reshape(-1)*voxel_count + np.repeat(particle_voxels, old_indices.shape[1])
    pair_keys, inverse = np.unique(flat_keys, return_inverse=True)
    nodes = pair_keys//voxel_count
    voxels = pair_keys % voxel_count
    starts = np.r_[0, np.flatnonzero(np.diff(nodes))+1, len(nodes)]
    pair_fields = np.empty(len(pair_keys), dtype=np.int64)
    original_nodes = []
    directions = ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))
    for left, right in zip(starts[:-1], starts[1:]):
        local_voxels = voxels[left:right]
        coords = coordinates[local_voxels]
        lookup = {tuple(c): i for i, c in enumerate(coords)}
        unseen = np.ones(len(local_voxels), dtype=bool)
        for seed in range(len(local_voxels)):
            if not unseen[seed]: continue
            field = len(original_nodes); original_nodes.append(int(nodes[left]))
            unseen[seed] = False; queue = deque([seed])
            while queue:
                current = queue.popleft()
                pair_fields[left+current] = field
                x, y, z = coords[current]
                for dx, dy, dz in directions:
                    neighbor = lookup.get((x+dx, y+dy, z+dz))
                    if neighbor is not None and unseen[neighbor]:
                        unseen[neighbor] = False; queue.append(neighbor)
    remapped = pair_fields[inverse].reshape(old_indices.shape)
    split = Support(support.positions, support.volume, np.asarray(support.nodes)[original_nodes],
                    remapped.astype(np.int32), support.weights, support.gradients,
                    support.spacing, support.shape)
    return SplitSupport(split, coordinates, pair_keys, pair_fields, len(support.nodes))


def surface_fields(split, original_indices, owner_voxels, weights=None):
    indices = np.asarray(original_indices)
    owners = np.asarray(owner_voxels, dtype=np.int64)
    if indices.ndim != 2 or owners.shape != (len(indices), 3):
        raise ValueError('Surface points and voxel owners do not match')
    lookup = {tuple(c): i for i, c in enumerate(split.voxel_coordinates)}
    try:
        voxel_ids = np.asarray([lookup[tuple(c)] for c in owners], dtype=np.int64)
    except KeyError as error:
        raise ValueError('Surface owner has no material particles') from error
    keys = indices.astype(np.int64)*len(split.voxel_coordinates) + voxel_ids[:, None]
    at = np.searchsorted(split.pair_keys, keys)
    safe = np.minimum(at, len(split.pair_keys)-1)
    present = (at < len(split.pair_keys)) & (split.pair_keys[safe] == keys)
    required = np.ones_like(present) if weights is None else np.abs(np.asarray(weights)) > 1e-14
    if required.shape != present.shape:
        raise ValueError('Surface weights do not match indices')
    if np.any(required & ~present):
        raise ValueError('Surface support is absent from its owning voxel')
    return np.where(present, split.pair_fields[safe], 0).astype(np.int32)
