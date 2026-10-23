"""Run with pvpython to check mixed failure-mode particle displays."""

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
from paraview.simple import ResetSession


def write_case(output, configs):
    output.mkdir(parents=True, exist_ok=True)
    points = np.array(
        [[x, y, 0.5] for x in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8) for y in (0.45, 0.55)]
    )
    body = np.repeat([0, 1], 6)
    failures = np.array(
        [[0] * 12, [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], [1] * 12], dtype=np.uint8
    )
    files = []
    for frame, failed in enumerate(failures):
        arrays = []
        for name, values, kind in (
            ("body", body, "Int32"),
            ("failed", failed, "UInt8"),
            ("damage", failed, "Float32"),
        ):
            values = " ".join(map(str, values))
            arrays.append(
                f'<DataArray type="{kind}" Name="{name}" format="ascii">{values}</DataArray>'
            )
        coordinates = " ".join(map(str, points.ravel()))
        connectivity = " ".join(map(str, range(12)))
        offsets = " ".join(map(str, range(1, 13)))
        text = (
            '<?xml version="1.0"?><VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">'
            '<PolyData><Piece NumberOfPoints="12" NumberOfVerts="12" NumberOfLines="0" '
            'NumberOfStrips="0" NumberOfPolys="0"><PointData>'
            + "".join(arrays)
            + '</PointData><Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">'
            + coordinates
            + '</DataArray></Points><Verts><DataArray type="Int32" '
            'Name="connectivity" format="ascii">'
            + connectivity
            + '</DataArray><DataArray type="Int32" Name="offsets" format="ascii">'
            + offsets
            + "</DataArray></Verts></Piece></PolyData></VTKFile>"
        )
        filename = f"frame{frame}.vtp"
        (output / filename).write_text(text, encoding="utf-8")
        files.append(f'<DataSet timestep="{frame}" file="{filename}"/>')
    (output / "particles.pvd").write_text(
        '<?xml version="1.0"?><VTKFile type="Collection"><Collection>'
        + "".join(files)
        + "</Collection></VTKFile>",
        encoding="utf-8",
    )
    metadata = {
        "dim": 3,
        "bounds": [0.2, 0.8, 0.45, 0.55, 0.5, 0.5],
        "spacing": 0.08,
        "frames": 3,
        "particles": 12,
        "field_ranges": {"body": [0, 1], "failed": [0, 1], "damage": [0, 1]},
    }
    (output / "vtk_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = {} if configs is None else {"configs": configs}
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".local/paraview-failure-smoke")
    )
    args = parser.parse_args()
    renderer = Path(__file__).resolve().parents[1] / "mpm_jax/paraview_scene.py"
    cases = (
        (
            "mixed",
            [{"failure_mode": "ballistic"}, {"failure_mode": "retain"}],
            [(12, 0), (10, 2), (6, 6)],
            ["ballistic", "retain"],
        ),
        ("missing-configs", None, [(12, 0)], ["retain", "retain"]),
    )
    results = {}
    for name, configs, expected_counts, expected_modes in cases:
        output = (args.output / name).resolve()
        write_case(output, configs)
        sys.argv = [
            str(renderer),
            str(output),
            "--field",
            "body",
            "--resolution",
            "240",
        ]
        if configs is None:
            sys.argv.append("--final-only")
        namespace = runpy.run_path(str(renderer), run_name="__main__")
        assert namespace["body_failure_modes"]({"configs": [{}]}, 2) == [
            "retain",
            "retain",
        ]
        record = json.loads((output / "render.json").read_text(encoding="utf-8"))
        display = record["failure_display"]
        actual_counts = [
            (r["active_particles"], r["debris_particles"])
            for r in display["displayed_counts"]
        ]
        assert actual_counts == expected_counts, (name, actual_counts)
        assert display["body_failure_modes"] == expected_modes
        assert all(sum(counts) == 12 for counts in actual_counts)
        results[name] = {
            "body_failure_modes": expected_modes,
            "displayed_counts": actual_counts,
        }
        ResetSession()
    (args.output / "check.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
