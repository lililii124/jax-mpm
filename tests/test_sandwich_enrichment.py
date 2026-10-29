import hashlib
import json

import numpy as np
import pytest

from tools.export_sandwich_fullmpm_face import apply_axis, spline_matrix
from tools.prepare_sandwich_enrichment import prepare


def fixture(tmp_path):
    spacing = np.asarray([.01, .02, .001])
    axes = [((np.arange(6)[:, None] + [.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)]).ravel())*h
            for h in spacing]
    node_axes = [np.arange(-1, 8)*h for h in spacing]
    nodes = np.stack(np.meshgrid(*node_axes, indexing='ij'), axis=-1).reshape(-1, 3)
    reference = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    u = np.random.default_rng(7).normal(size=(9, 9, 9, 3))*1e-4
    sample = u
    for axis in range(3):
        sample = apply_axis(spline_matrix(axes[axis], node_axes[axis], spacing[axis]), sample, axis)
    directory = tmp_path/'physical'; directory.mkdir()
    controls = np.zeros(1)
    binding = dict(case='eccentric', controls=controls.tolist())
    (directory/'status.json').write_text(json.dumps(dict(binding=binding, valid=True)))
    for step in (1000, 2000):
        np.savez(directory/f'frame_{step:07d}.npz', reference=reference,
                 displacement=sample.reshape(-1, 3), time=step*1e-8,
                 phi=np.full(len(reference), .18), young=np.ones(len(reference)),
                 density=np.ones(len(reference)), volume=np.ones(len(reference)),
                 skin=np.zeros(len(reference), dtype=bool))
    spec = dict(version=1, spacing_m=spacing.tolist(), sources=[dict(
        name='selected', directory='physical',
        controls_sha256=hashlib.sha256(controls.tobytes()).hexdigest(),
        training_steps=[1000], development_steps=[2000], validation_steps=[],
        reserved_validation_from_step=3000)])
    path = tmp_path/'spec.json'; path.write_text(json.dumps(spec))
    return path, spec, nodes, u.reshape(-1, 3)


def test_roundtrip_preserves_fields_and_partition_hashes(tmp_path):
    path, _, nodes, expected = fixture(tmp_path)
    output = tmp_path/'snapshots.npz'
    prepare(path, output)
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['nodes'], nodes)
        # Random ghost-node values exercise the least-conditioned boundary
        # directions; their inversion accumulates more roundoff than clamped
        # physical states. This is a field recovery check, not a ROM error gate.
        np.testing.assert_allclose(saved['displacement'][0], expected, rtol=0, atol=2e-14)
        np.testing.assert_array_equal(saved['training'], [True, False])
        np.testing.assert_array_equal(saved['development'], [False, True])
        assert not saved['validation'].any()
    receipt = json.loads(output.with_suffix('.json').read_text())
    assert receipt['training_fields_sha256']
    assert receipt['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['overlap', 'unfrozen_validation', 'reserved', 'controls'])
def test_rejects_invalid_provenance_or_data_allocation(tmp_path, change):
    path, spec, _, _ = fixture(tmp_path)
    source = spec['sources'][0]
    if change == 'overlap': source['development_steps'] = [1000]
    elif change == 'unfrozen_validation':
        source['development_steps'] = []; source['validation_steps'] = [2000]
    elif change == 'reserved': source['reserved_validation_from_step'] = 2000
    else: source['controls_sha256'] = 'invalid'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError): prepare(path, tmp_path/'snapshots.npz')
