"""Compare displacement on common material points of two binary bottom faces."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.compare_multicell_binary_blast import compare, read_run
from tools.reconstruct_multicell_blast import mask_digest


def compare_common_bottom(reference, reference_geometry, candidate, candidate_geometry):
    reference, reference_geometry = Path(reference), Path(reference_geometry)
    candidate, candidate_geometry = Path(candidate), Path(candidate_geometry)
    comparison = compare(reference, candidate)
    meta_a, rows_a, _ = read_run(reference, .02)
    meta_b, rows_b, _ = read_run(candidate, .02)
    masks = []
    for geometry, meta in ((reference_geometry, meta_a), (candidate_geometry, meta_b)):
        if hashlib.sha256(geometry.read_bytes()).hexdigest() != meta['geometry_sha256']:
            raise ValueError('Common-bottom geometry differs from its replay')
        with np.load(geometry, allow_pickle=False) as data:
            solid = np.asarray(data['solid_mask'], dtype=bool)
        if mask_digest(solid) != meta['solid_mask_sha256']:
            raise ValueError('Common-bottom solid mask differs from its replay')
        masks.append(solid)
    if masks[0].shape != masks[1].shape:
        raise ValueError('Common-bottom grids differ')
    common = masks[0][0] & masks[1][0]
    yx = np.argwhere(common)
    if not len(yx): raise ValueError('No common material points on the bottom plane')
    if [row['step'] for row in rows_a] != [row['step'] for row in rows_b]:
        raise ValueError('Common-bottom frame schedules differ')
    from mpm_jax.compact_tl import surface_weights
    from mpm_jax.multifield_support import surface_fields
    peaks = []
    for folder, meta, rows in ((reference, meta_a, rows_a), (candidate, meta_b, rows_b)):
        with np.load(folder/'reference.npz', allow_pickle=False) as data:
            spacing, shape = np.asarray(data['spacing']), tuple(data['shape'])
            split_mode = meta.get('support_mode', 'single') == 'split-disconnected'
            grid = SimpleNamespace(nodes=data['grid_nodes'] if split_mode else data['nodes'],
                                   spacing=spacing, shape=shape)
            if split_mode:
                split = SimpleNamespace(voxel_coordinates=data['voxel_coordinates'], pair_keys=data['pair_keys'],
                                        pair_fields=data['pair_fields'])
        points = np.column_stack(((yx[:, 1]+.5)*spacing[0], (yx[:, 0]+.5)*spacing[1], np.zeros(len(yx))))
        owners = np.column_stack((yx[:, 1], yx[:, 0], np.zeros(len(yx), dtype=int)))
        ids, weights = surface_weights(grid, points)
        if split_mode: ids = surface_fields(split, ids, owners, weights)
        maximum, p90 = 0., 0.
        for row in rows:
            path = folder/f"frame-{row['step']:07d}.npz"
            with np.load(path, allow_pickle=False) as data:
                nodal = data['nodal_displacement']
                values = np.abs(np.sum(nodal[ids, 2]*weights, axis=1))
            if not np.isfinite(values).all(): raise ValueError('Non-finite common-bottom displacement')
            maximum = max(maximum, float(values.max()))
            p90 = max(p90, float(np.percentile(values, 90)))
        peaks.append((maximum, p90))
    if min(peaks[0]) <= 0: raise ValueError('Reference common-bottom response is zero')
    return {'sampling': 'saved-frames', 'common_columns': len(yx),
            'common_bottom_coverage': float(np.mean(common)),
            'reference_peak_m': peaks[0][0], 'candidate_peak_m': peaks[1][0],
            'peak_ratio': peaks[1][0]/peaks[0][0],
            'reference_p90_peak_m': peaks[0][1], 'candidate_p90_peak_m': peaks[1][1],
            'p90_peak_ratio': peaks[1][1]/peaks[0][1],
            'physical_horizon_s': comparison['horizon_s']}
