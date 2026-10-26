"""Render saved differentiable MPM results through ParaView 6.2/OSMesa.

No solver is imported or rerun. The 800 x 400 GIF places two exact 400 x 400
views side by side and uses unscaled saved particle coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from paraview import servermanager
from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateView,
    Delete,
    GetColorTransferFunction,
    Hide,
    Render,
    SaveScreenshot,
    Show,
    TrivialProducer,
    _DisableFirstRenderCameraReset,
)
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData


SCALE_BOUNDS = (0.429, 1.313)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_cloud(coordinates: np.ndarray, volume_ratio: np.ndarray) -> vtkPolyData:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    volume_ratio = np.asarray(volume_ratio, dtype=np.float32)
    assert coordinates.ndim == 2 and coordinates.shape[1] == 3
    assert volume_ratio.shape == (len(coordinates),)
    assert np.isfinite(coordinates).all() and np.isfinite(volume_ratio).all()
    poly = vtkPolyData()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(coordinates, deep=True))
    poly.SetPoints(points)
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(len(coordinates) + 1, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.arange(len(coordinates), dtype=np.int64), deep=True),
    )
    poly.SetVerts(cells)
    values = numpy_to_vtk(volume_ratio, deep=True)
    values.SetName("volume_ratio")
    poly.GetPointData().AddArray(values)
    return poly


def configure_view(view):
    _DisableFirstRenderCameraReset()
    view.ViewSize = [400, 400]
    view.UseColorPaletteForBackground = 0
    view.Background = [0.985, 0.99, 1.0]
    view.OrientationAxesVisibility = 0
    if hasattr(view, "CenterAxesVisibility"):
        view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [0.55, 0.50, 0.50]
    view.CameraPosition = [1.10, 1.36, 1.23]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.40


def style(display, view, lookup):
    display.Representation = "Point Gaussian"
    display.GaussianRadius = 0.0034
    display.ShaderPreset = "Sphere"
    display.Emissive = 0
    display.Ambient = 0.36
    display.Diffuse = 0.64
    display.Specular = 0.18
    display.SpecularPower = 24
    ColorBy(display, ("POINTS", "volume_ratio"))
    display.LookupTable = lookup
    display.SetScalarBarVisibility(view, False)


def add_frame(producer_name, coordinates, ratio, view, lookup):
    source = TrivialProducer(registrationName=producer_name)
    source.GetClientSideObject().SetOutput(point_cloud(coordinates, ratio))
    source.UpdatePipeline()
    fetched = servermanager.Fetch(source)
    fetched_points = vtk_to_numpy(fetched.GetPoints().GetData())
    assert fetched_points.shape == coordinates.shape
    assert np.array_equal(fetched_points, coordinates), f"ParaView changed {producer_name} coordinates"
    display = Show(source, view)
    style(display, view, lookup)
    return source, display


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="result directory containing summary.json and trajectory.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--case", choices=("y", "z", "both"), default="y",
                        help="transverse load to render; defaults to the representative y case")
    args = parser.parse_args(argv)
    summary_path = args.input / "summary.json"
    trajectory_path = args.input / "trajectory.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data = np.load(trajectory_path, allow_pickle=False)
    ratio_initial = data["volume_ratio_initial"].astype(np.float32, copy=False)
    ratio_optimized = data["volume_ratio_optimized"].astype(np.float32, copy=False)
    reference = data["points"].astype(np.float64, copy=False)
    frame_count = int(data["initial_y_positions"].shape[0])
    assert frame_count == 21, f"expected 21 actual production frames, found {frame_count}"
    assert reference.ndim == 2 and reference.shape[1] == 3
    assert ratio_initial.shape == ratio_optimized.shape == (len(reference),)
    for design in ("initial", "optimized"):
        for case in ("y", "z"):
            positions = data[f"{design}_{case}_positions"]
            assert positions.shape == (frame_count, len(reference), 3)
            assert np.isfinite(positions).all()
    horizon = float(summary["optimization"]["time_horizon_s"])
    times = np.linspace(0.0, horizon, frame_count)
    args.output.mkdir(parents=True, exist_ok=False)
    png_root = args.output / "png"
    png_root.mkdir()

    _DisableFirstRenderCameraReset()
    layout = CreateLayout("JAX MPM structural material layout")
    initial_view = CreateView("RenderView")
    optimized_view = CreateView("RenderView")
    AssignViewToLayout(initial_view, layout, 0)
    layout.SplitHorizontal(0, 0.5)
    AssignViewToLayout(optimized_view, layout, 2)
    layout.SetSize(800, 400)
    configure_view(initial_view)
    configure_view(optimized_view)

    lookup = GetColorTransferFunction("volume_ratio")
    lookup.ColorSpace = "RGB"
    low, mid, high = SCALE_BOUNDS[0], 0.75, SCALE_BOUNDS[1]
    lookup.RGBPoints = [
        low, 0.105, 0.22, 0.63,
        mid, 0.08, 0.66, 0.62,
        high, 0.96, 0.61, 0.19,
    ]
    lookup.RescaleTransferFunction(*SCALE_BOUNDS)
    lookup.AutomaticRescaleRangeMode = "Never"

    manifest = {
        "render_engine": "ParaView 6.2 OSMesa",
        "source_summary_sha256": sha256(summary_path),
        "source_trajectory_sha256": sha256(trajectory_path),
        "source_simulation_precision": summary["simulation_precision"],
        "source_grid": summary["optimization"]["production_grid"],
        "particle_count": int(len(reference)),
        "frame_count": frame_count,
        "time_s": times.tolist(),
        "retained_all_particles": True,
        "fetched_coordinates_exact": True,
        "deformation_scale": 1.0,
        "axes_text_and_labels": False,
        "view_size_px_each": [400, 400],
        "comparison_layout_px": [800, 400],
        "camera": {
            "position": [1.10, 1.36, 1.23],
            "focal_point": [0.55, 0.50, 0.50],
            "view_up": [0.0, 0.0, 1.0],
            "parallel_scale": 0.40,
        },
        "cases": {},
    }
    cases = ("y", "z") if args.case == "both" else (args.case,)
    for case in cases:
        records = []
        design_records = {"initial": [], "optimized": []}
        case_dir = png_root / f"load-{case}"
        case_dir.mkdir()
        for design in design_records:
            (case_dir / design).mkdir()
        for frame in range(frame_count):
            sources, displays = [], []
            for design, view, ratio in (
                ("initial", initial_view, ratio_initial),
                ("optimized", optimized_view, ratio_optimized),
            ):
                coordinates = data[f"{design}_{case}_positions"][frame].astype(np.float64, copy=False)
                source, display = add_frame(
                    f"{case} {design} t={times[frame]:.3f}", coordinates, ratio,
                    view, lookup,
                )
                sources.append(source)
                displays.append(display)
            Render(initial_view)
            Render(optimized_view)
            path = case_dir / f"frame-{frame:03d}.png"
            from paraview.simple import SaveScreenshot
            SaveScreenshot(str(path), layout, ImageResolution=[800, 400])
            for design, view in (("initial", initial_view), ("optimized", optimized_view)):
                panel_path = case_dir / design / f"frame-{frame:03d}.png"
                SaveScreenshot(str(panel_path), view, ImageResolution=[400, 400])
                design_records[design].append({
                    "frame": frame,
                    "time_s": float(times[frame]),
                    "png": panel_path.name,
                    "png_sha256": sha256(panel_path),
                })
            record = {
                "frame": frame,
                "time_s": float(times[frame]),
                "point_count_each": int(len(reference)),
                "initial_coordinates_exact": True,
                "optimized_coordinates_exact": True,
                "png": path.name,
                "png_sha256": sha256(path),
            }
            records.append(record)
            print(json.dumps({"rendered": case, **record}), flush=True)
            for display in displays:
                Hide(display)
                Delete(display)
            for source in sources:
                Delete(source)

        gif_path = args.output / f"material-design-load-{case}.gif"
        duration_ms = int(round(1000 / args.fps))
        frames = [Image.open(case_dir / f"frame-{i:03d}.png").convert("RGB")
                  for i in range(frame_count)]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        with Image.open(gif_path) as check:
            decoded = [frame.convert("RGB").tobytes() for frame in ImageSequence.Iterator(check)]
        distinct = len(set(decoded))
        assert len(decoded) == frame_count and distinct >= 2, (len(decoded), distinct)
        manifest["cases"][case] = {
            "gif": gif_path.name,
            "gif_sha256": sha256(gif_path),
            "png_frames": records,
            "decoded_gif_frames": len(decoded),
            "distinct_decoded_frames": distinct,
            "fps": args.fps,
            "playback_duration_ms": frame_count * duration_ms,
            "source_span_s": horizon,
            "individual_views": {},
        }
        for design, panel_records in design_records.items():
            panel_frames = [
                Image.open(case_dir / design / record["png"]).convert("RGB")
                for record in panel_records
            ]
            panel_path = args.output / f"material-design-{case}-{design}.gif"
            panel_frames[0].save(
                panel_path,
                save_all=True,
                append_images=panel_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
                disposal=2,
            )
            with Image.open(panel_path) as check:
                assert check.size == (400, 400)
                decoded_count = 0
                distinct = set()
                for image in ImageSequence.Iterator(check):
                    decoded_count += 1
                    distinct.add(image.convert("RGB").tobytes())
                assert decoded_count == frame_count and len(distinct) >= 2
            manifest["cases"][case]["individual_views"][design] = {
                "gif": panel_path.name,
                "gif_sha256": sha256(panel_path),
                "size_px": [400, 400],
                "decoded_frames": decoded_count,
                "distinct_frames": len(distinct),
                "png_frames": panel_records,
            }
            print(json.dumps({"individual_gif_written": str(panel_path),
                              "sha256": manifest["cases"][case]["individual_views"][design]["gif_sha256"],
                              "frames": decoded_count, "distinct": len(distinct)}), flush=True)
        print(json.dumps({"gif_written": str(gif_path), "sha256": manifest["cases"][case]["gif_sha256"],
                          "frames": frame_count, "distinct": len(distinct)}), flush=True)

    (args.output / "render-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"render_complete": True, "output": str(args.output),
                      "gif_paths": [manifest["cases"][case]["gif"] for case in cases]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
