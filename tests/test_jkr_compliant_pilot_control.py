import json
from pathlib import Path
import numpy as np
import pytest
from tools.run_jkr_compliant_pilot import guarded_run, restore_parent
from tools.review_jkr_compliant_pilot import projected_pixels


def test_restore_rejects_different_physical_request_before_copy(tmp_path):
    current, parent = tmp_path/'current', tmp_path/'parent'
    current.mkdir()
    parent.mkdir()
    for root, request in ((current,'new'), (parent,'old')):
        (root/'preflight.json').write_text(json.dumps(dict(source_files={}, common_request_sha256=request)))
    with pytest.raises(AssertionError, match='physical request'):
        restore_parent(current, parent)
    assert not (current/'data').exists()


@pytest.mark.parametrize('timeout', [float('nan'), float('inf'), 0., -1., 7201.])
def test_unbounded_guard_rejected_before_any_launch(tmp_path, timeout):
    with pytest.raises(ValueError, match='guard'):
        guarded_run(tmp_path, 'jkr', 'prefix', timeout, 2)


def test_explicit_home_camera_projection_uses_fixed_scale():
    camera = dict(direction=[1.4,.2,1.8], resolution=400,
        focus_bounds_m=[-.003,.003,-.0018,.0042,-.003,.003], parallel_scale=.00471506391179459)
    assert projected_pixels(0., camera) == 0
    assert 19 < projected_pixels(.000464, camera) < 20
    changed_resolution = dict(camera, resolution=1200)
    np.testing.assert_allclose(projected_pixels(.000464, changed_resolution),
                               3*projected_pixels(.000464, camera), rtol=1e-14)
