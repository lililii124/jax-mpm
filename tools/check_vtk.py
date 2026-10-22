"""Run with pvpython to verify exported particle data against the NPZ trajectory."""

import argparse
from pathlib import Path

import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with np.load(args.output / "trajectory.npz") as data:
        indices = [0, len(data["time"]) - 1]
        positions = data["x"][indices]
        fields = {
            key: data[key] if key == "body" else data[key][indices]
            for key in ("speed", "damage", "temperature", "body")
            if key in data
        }
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
        for key in ("speed", "damage", "temperature", "body"):
            if key in fields:
                expected = fields[key] if key == "body" else fields[key][sample]
                np.testing.assert_allclose(
                    vtk_to_numpy(result.GetPointData().GetArray(key)),
                    expected,
                    rtol=1e-6,
                    atol=1e-7,
                )
    print("VTK particle count, coordinates and scalar fields match the saved trajectory.")


if __name__ == "__main__":
    main()
