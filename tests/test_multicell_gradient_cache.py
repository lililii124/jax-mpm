import numpy as np
import pytest

from tools.optimize_multicell_binary_loop import publish_cached_gradient, sha256, validate_cached_gradient


def test_gradient_cache_checks_design_and_rejects_corruption(tmp_path):
    inputs = tmp_path/'inputs.json'
    inputs.write_text('{}')
    design = tmp_path/'design.npz'
    np.savez_compressed(design, logits=np.array([1., 2.]), rho_cell=np.array([.2, .8]))
    source = tmp_path/'gradient.npz'
    np.savez_compressed(source, inputs_sha256=sha256(inputs), design_sha256=sha256(design),
                        logits=np.array([1., 2.]), rho_cell=np.array([.2, .8]),
                        gradient=np.array([.1, -.1]), objective=1.)
    cached = tmp_path/'cache'/'entry.npz'
    publish_cached_gradient(source, cached, inputs, design)
    validate_cached_gradient(cached, inputs, design)
    np.savez_compressed(cached, inputs_sha256=sha256(inputs), design_sha256=sha256(design),
                        logits=np.array([1., 2.]), rho_cell=np.array([.2, .8]),
                        gradient=np.array([np.nan, -.1]), objective=1.)
    with pytest.raises(ValueError, match='Cached gradient is invalid'):
        validate_cached_gradient(cached, inputs, design)
