"""ParaView batch rendering and animation assembly."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .export import export_series
from .particle_fields import array_descriptor, checked_particle_fields


def _checked_sphere_opacity(value):
    if value is None:
        return None
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("--sphere-opacity must be finite and satisfy 0 < opacity <= 1")
    return float(value)


def _checked_matrix_opacity(value):
    if value is None:
        return None
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("--matrix-opacity must be finite and satisfy 0 < opacity <= 1")
    return float(value)


def find_pvpython():
    configured = os.environ.get("PARAVIEW_PYTHON")
    if configured and Path(configured).is_file():
        return configured
    installed = shutil.which("pvpython")
    if installed:
        return installed
    root = Path(__file__).resolve().parents[1]
    candidates = list((root / ".local/ParaView").glob("**/pvpython.exe"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "ParaView pvpython was not found. Set PARAVIEW_PYTHON or use --no-render."
    )


def render_directory(
    output, *, gif=False, field="body", resolution=1200, final_only=False,
    color_range=None, scene_bounds=None, sphere_opacity=None, focus_bounds=None,
    matrix_opacity=None,
):
    if gif and final_only:
        raise ValueError("--gif and --final-only are mutually exclusive")
    if scene_bounds is not None and focus_bounds is not None:
        raise ValueError("scene_bounds and focus_bounds are mutually exclusive")
    sphere_opacity = _checked_sphere_opacity(sphere_opacity)
    matrix_opacity = _checked_matrix_opacity(matrix_opacity)
    if matrix_opacity is not None and field != "family_id":
        raise ValueError("matrix_opacity requires field='family_id'")
    output = Path(output).resolve()
    pvpython = find_pvpython()
    command = [
        pvpython,
        "--force-offscreen-rendering",
        str(Path(__file__).with_name("paraview_scene.py")),
        str(output),
        "--field",
        field,
        "--resolution",
        str(resolution),
    ]
    if color_range is not None:
        command.extend(["--color-range", *(str(value) for value in color_range)])
    if scene_bounds is not None:
        command.extend(["--scene-bounds", *(str(value) for value in scene_bounds)])
    if focus_bounds is not None:
        command.extend(["--focus-bounds", *(str(value) for value in focus_bounds)])
    if sphere_opacity is not None:
        command.extend(["--sphere-opacity", str(sphere_opacity)])
    if matrix_opacity is not None:
        command.extend(["--matrix-opacity", str(matrix_opacity)])
    if final_only or not gif:
        command.append("--final-only")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as process:
        with (output / "render.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        if process.wait():
            raise RuntimeError(f"ParaView rendering failed; see {output / 'render.log'}")
    if gif:
        from PIL import Image

        frame_count = json.loads((output / "vtk_metadata.json").read_text(encoding="utf-8"))[
            "frames"
        ]
        paths = [output / "frames" / f"frame_{i:05d}.png" for i in range(frame_count)]
        frames = [Image.open(path).convert("RGB") for path in paths]
        frames[0].save(
            output / "simulation.gif",
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0,
            optimize=False,
            disposal=2,
        )
        for frame in frames:
            frame.close()
        record_path = output / "render.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["animation"] = "simulation.gif"
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


@contextmanager
def trajectory_arrays(output):
    """Read completed file-backed histories without inflating the whole archive."""
    output = Path(output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    names = ("x", "speed", "damage", "temperature")
    mapped = summary.get("completed", False) and all(
        (output / f"trajectory_{name}.npy").is_file() for name in names
    )
    arrays = {}
    try:
        with np.load(output / "trajectory.npz") as data:
            if mapped:
                arrays.update(
                    {
                        name: np.load(output / f"trajectory_{name}.npy", mmap_mode="r")
                        for name in names
                    }
                )
            else:
                arrays.update({name: data[name] for name in names if name in data})
            arrays.update({name: data[name] for name in ("time", "body") if name in data})
            failed_file = output / "trajectory_failed.npy"
            if "failed" in data or (mapped and failed_file.is_file()):
                arrays["failed"] = (
                    np.load(failed_file, mmap_mode="r") if mapped and failed_file.is_file()
                    else data["failed"]
                )
        if "particle_fields" in summary:
            record = summary["particle_fields"]
            if record.get("file") != "particle_fields.npz":
                raise ValueError("static particle labels must use particle_fields.npz")
            with np.load(output / record["file"]) as labels:
                if set(labels.files) != set(record["fields"]):
                    raise ValueError("static particle labels differ from the recorded fields")
                fields = checked_particle_fields(
                    {name: labels[name] for name in labels.files}, arrays["x"].shape[1]
                )
            for name, value in fields.items():
                if array_descriptor(value) != record["fields"][name]:
                    raise ValueError(f"static particle field {name} failed its integrity check")
            arrays.update(fields)
        yield arrays
    finally:
        for value in arrays.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        arrays.clear()


def render_trajectory(
    output, *, gif=False, field="body", resolution=1200, final_only=False,
    color_range=None, scene_bounds=None, sphere_opacity=None, focus_bounds=None,
    matrix_opacity=None,
):
    if gif and final_only:
        raise ValueError("--gif and --final-only are mutually exclusive")
    if scene_bounds is not None and focus_bounds is not None:
        raise ValueError("scene_bounds and focus_bounds are mutually exclusive")
    sphere_opacity = _checked_sphere_opacity(sphere_opacity)
    matrix_opacity = _checked_matrix_opacity(matrix_opacity)
    if matrix_opacity is not None and field != "family_id":
        raise ValueError("matrix_opacity requires field='family_id'")
    output = Path(output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    config = summary.get("configs", [summary.get("config", {})])[0]
    spacing = config.get("domain_size", 1.0) / config.get("n_grid", 48)
    spacing /= summary.get("definition", {}).get(
        "particles_per_cell_per_axis", summary.get("particles_per_cell_per_axis", 2)
    )
    spacing = summary.get("definition", {}).get("particle_spacing_m", spacing)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("particle display spacing must be finite and positive")
    with trajectory_arrays(output) as data:
        positions = data["x"]
        fields = {
            key: data[key] for key in ("speed", "damage", "temperature", "body", "failed")
            if key in data
        }
        fields.update({name: data[name] for name in summary.get("particle_fields", {}).get("fields", {})})
        if field == "family_id" and not {"phase", "family_id"}.issubset(fields):
            raise ValueError("family_id display requires static phase and family_id fields")
        if "body" not in fields:
            fields["body"] = np.zeros(positions.shape[1], dtype=np.int32)
        export_series(
            positions,
            fields,
            data["time"],
            output,
            spacing=spacing,
            dim=config.get("dim", positions.shape[-1]),
            displacement=field == "displacement",
        )
    del positions, fields
    render_directory(
        output, gif=gif, field=field, resolution=resolution, final_only=final_only,
        color_range=color_range, scene_bounds=scene_bounds, sphere_opacity=sphere_opacity,
        focus_bounds=focus_bounds,
        matrix_opacity=matrix_opacity,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument(
        "--field", choices=("body", "speed", "displacement", "damage", "temperature", "phase", "family_id"), default="body"
    )
    parser.add_argument("--resolution", type=int, default=1200)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--color-range", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--scene-bounds", type=float, nargs=6,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--sphere-opacity", type=float, help="rigid-sphere CAD opacity (default: 1)")
    parser.add_argument("--matrix-opacity", type=float, help="matrix opacity when coloring woven yarn families (default: 0.008)")
    parser.add_argument("--focus-bounds", type=float, nargs=6, help="explicit close-up bounds; may crop geometry outside the view")
    args = parser.parse_args()
    if args.scene_bounds is not None and args.focus_bounds is not None:
        parser.error("--scene-bounds and --focus-bounds are mutually exclusive")
    try:
        args.sphere_opacity = _checked_sphere_opacity(args.sphere_opacity)
        args.matrix_opacity = _checked_matrix_opacity(args.matrix_opacity)
    except ValueError as error:
        parser.error(str(error))
    if args.color_range is not None and args.field in ("body", "family_id"):
        parser.error("--color-range requires a scalar field, not categorical colors")
    render_trajectory(
        args.output,
        gif=args.gif,
        field=args.field,
        resolution=args.resolution,
        final_only=args.final_only,
        color_range=args.color_range,
        scene_bounds=args.scene_bounds,
        sphere_opacity=args.sphere_opacity,
        focus_bounds=args.focus_bounds,
        matrix_opacity=args.matrix_opacity,
    )


if __name__ == "__main__":
    main()
