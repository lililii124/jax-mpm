import json
from collections import namedtuple

import numpy as np
import pytest

from mpm_jax.export import export_series, write_vtp
from mpm_jax.particle_fields import (
    array_descriptor, checked_particle_fields, material_metadata,
)
from mpm_jax.visualization import trajectory_arrays


def test_material_metadata_preserves_small_parameters_and_fingerprints_orientations():
    Small = namedtuple("Small", "modulus active stiffness")
    material = Small(2.0, True, np.eye(6))
    old = {name: np.asarray(value).tolist() for name, value in material._asdict().items()}
    assert json.dumps(material_metadata(material), sort_keys=True) == json.dumps(old, sort_keys=True)
    Large = namedtuple("Large", "orientation phase")
    orientations = np.broadcast_to(np.eye(3), (1000, 3, 3))
    fields = Large(orientations, np.zeros(1000, dtype=np.uint8))
    record = material_metadata(fields)
    assert len(json.dumps(record)) < 512
    assert record["orientation"] == array_descriptor(orientations.copy())
    changed = orientations.copy()
    changed[999, 0, 0] = 0.5
    assert array_descriptor(changed) != record["orientation"]


@pytest.mark.parametrize("fields", [
    {"bad-name": [0, 1]}, {"damage": [0, 1]}, {"phase": [0]},
    {"phase": [[0], [1]]}, {"phase": [0, np.inf]}, {"phase": ["a", "b"]},
])
def test_particle_labels_reject_invalid_fields(fields):
    with pytest.raises(ValueError):
        checked_particle_fields(fields, 2)


def test_static_labels_survive_trajectory_and_vtk_export_and_detect_changes(tmp_path):
    positions = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]] * 3)
    positions[1:, :, 0] += 0.01
    labels = {"phase": np.array([0, 1], dtype=np.uint8), "family_id": np.array([-1, 0], dtype=np.int8)}
    np.savez_compressed(tmp_path / "trajectory.npz", x=positions, time=[0.0, 0.5, 1.0])
    np.savez_compressed(tmp_path / "particle_fields.npz", **labels)
    summary = {"particle_fields": {
        "file": "particle_fields.npz",
        "fields": {name: array_descriptor(value) for name, value in labels.items()},
    }}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with trajectory_arrays(tmp_path) as data:
        for name, value in labels.items():
            np.testing.assert_array_equal(data[name], value)
        export_series(data["x"], {name: data[name] for name in labels}, data["time"], tmp_path)
    metadata = json.loads((tmp_path / "vtk_metadata.json").read_text())
    assert metadata["fields"] == list(labels)
    assert metadata["field_ranges"]["family_id"] == [-1.0, 0.0]
    assert len(list((tmp_path / "vtk").glob("*.vtp"))) == 3
    labels["family_id"][1] = 1
    np.savez_compressed(tmp_path / "particle_fields.npz", **labels)
    with pytest.raises(ValueError, match="integrity check"):
        with trajectory_arrays(tmp_path):
            pass


@pytest.mark.parametrize("dtype, values, vtk_type", [
    (np.int64, [-2**40, 2**40], "Int64"),
    (np.uint64, [0, 2**63 + 1], "UInt64"),
])
def test_vtk_preserves_large_integer_labels(tmp_path, dtype, values, vtk_type):
    import struct
    import zlib

    path = tmp_path / "labels.vtp"
    write_vtp(path, np.zeros((2, 3)), {"tag": np.array(values, dtype=dtype)})
    header, payload = path.read_bytes().split(b'<AppendedData encoding="raw">_', 1)
    assert f'type="{vtk_type}" Name="tag"'.encode() in header
    blocks, block_size, last_size, compressed_size = struct.unpack_from("<4Q", payload)
    assert blocks == 1 and last_size == 16
    decoded = np.frombuffer(zlib.decompress(payload[32:32 + compressed_size]), dtype=dtype)
    np.testing.assert_array_equal(decoded, values)
