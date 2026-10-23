"""Compressed VTK particle output for ParaView."""

from html import escape
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def _compressed(array):
    raw = memoryview(np.ascontiguousarray(array)).cast("B")
    block_size = 32768
    blocks = [
        zlib.compress(raw[i : i + block_size], level=3) for i in range(0, len(raw), block_size)
    ]
    last = len(raw) - (len(blocks) - 1) * block_size if blocks else 0
    header = np.asarray([len(blocks), block_size, last, *map(len, blocks)], dtype="<u8").tobytes()
    return header + b"".join(blocks)


def write_vtp(path, points, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype="<f4")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points), dtype="<f4")))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 2) or (N, 3)")
    n = len(points)
    arrays, offset = [], 0

    def data_array(name, data, vtk_type, components=1):
        nonlocal offset
        encoded = _compressed(data)
        element = f'<DataArray type="{vtk_type}" Name="{escape(name, quote=True)}" NumberOfComponents="{components}" format="appended" offset="{offset}"/>'
        offset += len(encoded)
        arrays.append(encoded)
        return element

    point_data = []
    for name, values in fields.items():
        values = np.asarray(values)
        if values.shape[0] != n:
            raise ValueError(f"field {name} does not match the particle count")
        integer = np.issubdtype(values.dtype, np.integer)
        vtk_type, storage_type = "Float32", "<f4"
        if integer:
            vtk_type, storage_type = "Int32", "<i4"
            limits = np.iinfo(np.int32)
            if values.size and (values.min() < limits.min or values.max() > limits.max):
                vtk_type, storage_type = ("UInt64", "<u8") if values.dtype.kind == "u" else ("Int64", "<i8")
        values = values.astype(storage_type)
        components = 1 if values.ndim == 1 else values.shape[1]
        point_data.append(data_array(name, values, vtk_type, components))
    coordinates = data_array("Position", points, "Float32", 3)
    connectivity = data_array("connectivity", np.arange(n, dtype="<i8"), "Int64")
    offsets = data_array("offsets", np.arange(1, n + 1, dtype="<i8"), "Int64")
    header = (
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64" compressor="vtkZLibDataCompressor">\n<PolyData>\n'
        f'<Piece NumberOfPoints="{n}" NumberOfVerts="{n}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n'
        f'<PointData>{"".join(point_data)}</PointData>\n<Points>{coordinates}</Points>\n'
        f'<Verts>{connectivity}{offsets}</Verts>\n</Piece></PolyData>\n<AppendedData encoding="raw">_'
    )
    with path.open("wb") as stream:
        stream.write(header.encode("utf-8"))
        for array in arrays:
            stream.write(array)
        stream.write(b"</AppendedData>\n</VTKFile>\n")


def displacement_magnitude(positions, reference, *, chunk_size=65536):
    """One frame of |x(t)-x(0)|, without a whole-history difference array."""
    if positions.shape != reference.shape or positions.ndim != 2 or chunk_size < 1:
        raise ValueError("matching (particles, dimensions) arrays and a positive chunk size are required")
    magnitude = np.empty(len(positions), dtype=np.float64)
    for first in range(0, len(positions), chunk_size):
        last = min(first + chunk_size, len(positions))
        change = np.asarray(positions[first:last], dtype=np.float64) - reference[first:last]
        magnitude[first:last] = np.linalg.norm(change, axis=-1)
    return magnitude


def export_series(positions, fields, times, output, *, spacing=None, dim=3, displacement=False):
    output = Path(output)
    vtk = output / "vtk"
    vtk.mkdir(parents=True, exist_ok=True)
    datasets = []
    displacement_range = [float("inf"), -float("inf")]
    if displacement and "displacement" in fields:
        raise ValueError("displacement is derived from positions; do not also supply it as a field")
    for i, time in enumerate(times):
        current = {
            name: values[i] if np.ndim(values) > 1 and np.shape(values)[0] == len(times) else values
            for name, values in fields.items()
        }
        if displacement:
            magnitude = displacement_magnitude(positions[i], positions[0])
            current["displacement"] = magnitude
            displacement_range[0] = min(displacement_range[0], float(magnitude.min()))
            displacement_range[1] = max(displacement_range[1], float(magnitude.max()))
        name = f"particles_{i:05d}.vtp"
        write_vtp(vtk / name, positions[i], current)
        datasets.append(
            f'<DataSet timestep="{float(time):.17g}" group="" part="0" file="vtk/{name}"/>'
        )
    (output / "particles.pvd").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n'
        + "\n".join(datasets)
        + "\n</Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    lo, hi = np.min(positions, axis=(0, 1)), np.max(positions, axis=(0, 1))
    if dim == 2:
        lo, hi = np.append(lo, 0.0), np.append(hi, 0.0)
    metadata = {
        "bounds": [float(v) for pair in zip(lo, hi) for v in pair],
        "dim": dim,
        "spacing": spacing,
        "frames": len(times),
        "particles": len(positions[0]),
        "fields": list(fields),
        "visualization_dtype": "float32",
        "source_dtype": str(np.asarray(positions).dtype),
        "field_ranges": {
            key: [float(np.min(value)), float(np.max(value))] for key, value in fields.items()
        },
    }
    if displacement:
        metadata["fields"].append("displacement")
        metadata["field_ranges"]["displacement"] = displacement_range
        metadata["derived_fields"] = {
            "displacement": {
                "definition": "Euclidean norm of current minus first saved particle position",
                "units": "m",
                "reference_time": float(times[0]),
                "geometry_scale": 1.0,
                "evaluation": "one frame at a time, in chunks of at most 65536 particles",
            }
        }
    (output / "vtk_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return output / "particles.pvd"
