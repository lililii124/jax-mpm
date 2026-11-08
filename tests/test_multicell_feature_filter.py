import numpy as np

from tools.multicell_feature_filter import single_voxel_run_fraction


def test_single_voxel_runs_respect_periodic_cell_edges():
    solid = np.ones((3, 4, 4), dtype=bool)
    assert single_voxel_run_fraction(solid) == 0.
    solid[1] = False
    assert single_voxel_run_fraction(solid) == 1.
    bridge = np.ones((3, 4, 4), dtype=bool)
    bridge[:, :, 1:3] = False
    assert single_voxel_run_fraction(bridge) == 0.
