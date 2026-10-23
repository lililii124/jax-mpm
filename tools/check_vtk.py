"""Run with pvpython to verify exported particle data against the NPZ trajectory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectory_samples(output, summary):
    """Read first/last saved fields, including completed mmap-only scalars."""
    output = Path(output)

    def mapped_field(name):
        return summary.get("completed") is True and (output / f"trajectory_{name}.npy").is_file()

    def samples(name, archive, indices):
        if mapped_field(name):
            mapped = np.load(output / f"trajectory_{name}.npy", mmap_mode="r", allow_pickle=False)
            try:
                return mapped[indices]
            finally:
                mapped._mmap.close()
        return archive[name][indices]

    with np.load(output / "trajectory.npz", allow_pickle=False) as data:
        indices = [0, len(data["time"]) - 1]
        positions = samples("x", data, indices)
        fields = {
            key: samples(key, data, indices)
            for key in ("speed", "damage", "temperature", "failed")
            if key in data or mapped_field(key)
        }
        if "body" in data:
            fields["body"] = data["body"]
    return indices, positions, fields


def main():
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    indices, positions, fields = trajectory_samples(args.output, summary)
    static = {"body"}
    if "particle_fields" in summary:
        record = summary["particle_fields"]
        if record["file"] != "particle_fields.npz":
            raise ValueError("unexpected static particle field filename")
        with np.load(args.output / record["file"]) as labels:
            if set(labels.files) != set(record["fields"]):
                raise ValueError("static particle fields differ from the recorded field set")
            for key in labels.files:
                values = np.ascontiguousarray(labels[key])
                descriptor = {
                    "shape": list(values.shape), "dtype": str(values.dtype),
                    "sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
                }
                if descriptor != record["fields"][key]:
                    raise ValueError(f"static particle field {key} failed its integrity check")
                fields[key] = values
                static.add(key)
    metadata_path = args.output / "vtk_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "displacement" in metadata["fields"]:
            fields["displacement"] = np.linalg.norm(positions - positions[0], axis=-1)
    for sample, i in enumerate(indices):
        reader = vtkXMLPolyDataReader()
        reader.SetFileName(str(args.output / "vtk" / f"particles_{i:05d}.vtp"))
        reader.Update()
        result = reader.GetOutput()
        assert result.GetNumberOfPoints() == len(positions[sample])
        points = vtk_to_numpy(result.GetPoints().GetData())
        np.testing.assert_allclose(
            points[:, : positions.shape[-1]], positions[sample], rtol=1e-6, atol=1e-7
        )
        for key, values in fields.items():
            expected = values if key in static else values[sample]
            actual = vtk_to_numpy(result.GetPointData().GetArray(key))
            if expected.dtype.kind in "biu":
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual, expected,
                    rtol=1e-6,
                    atol=1e-7,
                )
    print("VTK particle count, coordinates and scalar fields match the saved trajectory.")


if __name__ == "__main__":
    main()
