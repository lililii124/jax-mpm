import hashlib
import sys
import types

import numpy as np
import pytest

from tools import multicell_common_surface as surface
from tools.reconstruct_multicell_blast import mask_digest


def test_common_peak_includes_unsaved_steps(tmp_path, monkeypatch):
    mask = np.ones((1, 2, 2), dtype=bool)
    runs = []
    metadata = {}
    for name, saved, peak in [('initial', 1.0, 2.0), ('candidate', 0.5, 3.0)]:
        folder = tmp_path / name
        folder.mkdir()
        geometry = folder / 'geometry.npz'
        np.savez_compressed(geometry, solid_mask=mask)
        metadata[str(folder)] = {
            'geometry_sha256': hashlib.sha256(geometry.read_bytes()).hexdigest(),
            'solid_mask_sha256': mask_digest(mask),
            'bottom_point_peak_sampling': 'every-step',
        }
        columns = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        np.savez_compressed(folder / 'reference.npz', spacing=np.ones(3), shape=np.array([2, 2, 1]),
                            nodes=np.zeros((4, 3)), bottom_sample_columns=columns)
        for step in (0, 1):
            nodal = np.zeros((4, 3))
            nodal[:, 2] = saved if step else 0
            intervals = np.full(4, peak if step else 0)
            np.savez_compressed(folder / f'frame-{step:07d}.npz', nodal_displacement=nodal,
                                bottom_interval_peak_m=intervals)
        runs.append((folder, geometry))

    def read_run(folder, tolerance):
        peak = 2.0 if folder.name == 'initial' else 3.0
        return metadata[str(folder)], [{'step': 0, 'back_displacement_interval_peak_m': 0.0},
                                      {'step': 1, 'back_displacement_interval_peak_m': peak}], None

    def surface_weights(grid, points):
        return np.arange(len(points))[:, None], np.ones((len(points), 1))

    monkeypatch.setattr(surface, 'compare', lambda a, b: {'horizon_s': 1.0})
    monkeypatch.setattr(surface, 'read_run', read_run)
    monkeypatch.setitem(sys.modules, 'mpm_jax', types.ModuleType('mpm_jax'))
    compact = types.ModuleType('mpm_jax.compact_tl')
    compact.surface_weights = surface_weights
    monkeypatch.setitem(sys.modules, 'mpm_jax.compact_tl', compact)
    fields = types.ModuleType('mpm_jax.multifield_support')
    fields.surface_fields = lambda *args: None
    monkeypatch.setitem(sys.modules, 'mpm_jax.multifield_support', fields)

    result = surface.compare_common_bottom(*runs[0], *runs[1], require_every_step=True)
    assert result['sampling'] == 'every-step'
    assert result['peak_ratio'] == pytest.approx(1.5)
    assert result['candidate_saved_peak_m'] / result['reference_saved_peak_m'] == pytest.approx(0.5)
    assert result['p90_sampling'] == 'saved-frames'

    bad_frame = runs[1][0] / 'frame-0000001.npz'
    with np.load(bad_frame) as frame:
        values = {key: frame[key] for key in frame.files}
    values['bottom_interval_peak_m'][:] = 4.0
    np.savez_compressed(bad_frame, **values)
    with pytest.raises(ValueError, match='disagree'):
        surface.compare_common_bottom(*runs[0], *runs[1], require_every_step=True)

    metadata[str(runs[1][0])]['bottom_point_peak_sampling'] = 'not-recorded'
    with pytest.raises(ValueError, match='sampling modes differ'):
        surface.compare_common_bottom(*runs[0], *runs[1], require_every_step=True)

    metadata[str(runs[0][0])]['bottom_point_peak_sampling'] = 'not-recorded'
    with pytest.raises(ValueError, match='not recorded'):
        surface.compare_common_bottom(*runs[0], *runs[1], require_every_step=True)
