"""Executed by ParaView's pvpython, not the solver's Python environment."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from paraview.simple import (
    _DisableFirstRenderCameraReset,
    ColorBy,
    CompositeKeyFrame,
    CreateView,
    GetAnimationScene,
    GetAnimationTrack,
    GetColorTransferFunction,
    GetParaViewVersion,
    PVDReader,
    Render,
    SaveScreenshot,
    SaveState,
    Show,
    Sphere,
    Threshold,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--field", default="body")
    parser.add_argument("--resolution", type=int, default=1200)
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()
    _DisableFirstRenderCameraReset()
    output = args.output.resolve()
    metadata = json.loads((output / "vtk_metadata.json").read_text(encoding="utf-8"))
    reader = PVDReader(FileName=str(output / "particles.pvd"))
    reader.UpdatePipeline()
    times = list(reader.TimestepValues)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    sphere_definition = summary.get("definition", {}).get("rigid_sphere")
    view = CreateView("RenderView")
    view.ViewSize = [args.resolution, args.resolution]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.AxesGrid.Visibility = 0
    view.UseColorPaletteForBackground = 0
    view.Background = [0.965, 0.975, 0.985]
    view.CameraParallelProjection = 1
    bounds = list(metadata["bounds"])
    auxiliary_geometry = []
    if sphere_definition:
        center0 = np.asarray(sphere_definition["center"])
        stroke = np.asarray(sphere_definition["displacement"])
        radius = sphere_definition["radius"]
        duration = sphere_definition["duration"]
        sphere = Sphere(registrationName="Prescribed rigid sphere")
        sphere.Radius = radius
        sphere.ThetaResolution = 128
        sphere.PhiResolution = 96
        sphere.Center = center0
        surface = Show(sphere, view)
        surface.ColorArrayName = ["POINTS", ""]
        surface.DiffuseColor = [0.56, 0.62, 0.68]
        surface.Ambient = 0.2
        surface.Diffuse = 0.8
        surface.Specular = 0.65
        surface.SpecularPower = 45
        phase = np.clip(np.asarray(times) / duration, 0, 1)
        centers = center0 + 0.5 * (1 - np.cos(2 * np.pi * phase[:, None])) * stroke
        span = times[-1] - times[0]
        for axis in range(3):
            track = GetAnimationTrack("Center", axis, sphere)
            keys = []
            for time, position in zip(times, centers):
                key = CompositeKeyFrame()
                key.KeyTime = (time - times[0]) / span if span else 0.0
                key.KeyValues = [float(position[axis])]
                key.Interpolation = "Ramp"
                keys.append(key)
            track.KeyFrames = keys
            bounds[2 * axis] = min(bounds[2 * axis], center0[axis] + min(0, stroke[axis]) - radius)
            bounds[2 * axis + 1] = max(
                bounds[2 * axis + 1], center0[axis] + max(0, stroke[axis]) + radius
            )
        auxiliary_geometry.append(
            {
                "type": "prescribed_rigid_sphere",
                "parameters": sphere_definition,
                "included_in_particle_count": False,
                "motion": "cosine loading/unloading cycle",
            }
        )
    center = [(bounds[2 * i] + bounds[2 * i + 1]) / 2 for i in range(3)]
    length = max(bounds[2 * i + 1] - bounds[2 * i] for i in range(3))
    view.CameraFocalPoint = center
    view.CameraPosition = [
        center[0] + 1.4 * length,
        center[1] + 0.85 * length,
        center[2] + 1.8 * length,
    ]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    direction = np.array([1.4, 0.85, 1.8])
    direction /= np.linalg.norm(direction)
    right = np.cross([0.0, 1.0, 0.0], direction)
    right /= np.linalg.norm(right)
    upward = np.cross(direction, right)
    corners = np.array(list(product(*[(bounds[2 * i], bounds[2 * i + 1]) for i in range(3)])))
    view.CameraParallelScale = 0.56 * max(np.ptp(corners @ right), np.ptp(corners @ upward))
    if metadata["dim"] == 2:
        view.CameraPosition = [center[0], center[1], center[2] + 3 * length]
        view.CameraParallelScale = 0.58 * length
    spacing = metadata["spacing"] or length / 100
    colors = ([0.045, 0.36, 0.67], [0.94, 0.36, 0.08], [0.32, 0.66, 0.48])
    body_array = reader.PointData["body"]
    count = int(body_array.GetRange()[1]) + 1 if body_array else 1
    displays = []
    for body in range(count):
        selected = Threshold(Input=reader)
        selected.Scalars = ["POINTS", "body"]
        selected.LowerThreshold = body
        selected.UpperThreshold = body
        selected.UpdatePipeline()
        display = Show(selected, view)
        display.Representation = "Point Gaussian"
        display.GaussianRadius = 0.62 * spacing
        display.ShaderPreset = "Sphere"
        display.Emissive = 0
        display.Ambient = 0.3
        display.Diffuse = 0.7
        display.Specular = 0.35
        display.SpecularPower = 35
        display.ColorArrayName = ["POINTS", ""]
        display.DiffuseColor = colors[body % len(colors)]
        display.AmbientColor = colors[body % len(colors)]
        if args.field != "body":
            ColorBy(display, ("POINTS", args.field))
            lut = GetColorTransferFunction(args.field)
            lo, hi = metadata["field_ranges"][args.field]
            if hi <= lo:
                hi = lo + 1.0
            palette = [
                (0.0, 0.267, 0.005, 0.329),
                (0.25, 0.230, 0.322, 0.546),
                (0.5, 0.128, 0.567, 0.551),
                (0.75, 0.369, 0.789, 0.383),
                (1.0, 0.993, 0.906, 0.144),
            ]
            lut.RGBPoints = [
                value for t, r, g, b in palette for value in (lo + t * (hi - lo), r, g, b)
            ]
            lut.ColorSpace = "RGB"
            lut.RescaleTransferFunction(lo, hi)
            lut.AutomaticRescaleRangeMode = "Never"
        if args.field != "body":
            display.SetScalarBarVisibility(view, False)
        displays.append(display)
    indices = [len(times) - 1] if args.final_only else range(len(times))
    (output / "frames").mkdir(exist_ok=True)
    for i in indices:
        scene.AnimationTime = times[i]
        view.ViewTime = times[i]
        reader.UpdatePipeline(times[i])
        Render(view)
        SaveScreenshot(
            str(output / "frames" / f"frame_{i:05d}.png"),
            view,
            ImageResolution=[args.resolution] * 2,
        )
        if i == len(times) - 1:
            SaveScreenshot(str(output / "final.png"), view, ImageResolution=[args.resolution] * 2)
        if i % 10 == 0 or i == len(times) - 1:
            print(f"ParaView: frame {i + 1}/{len(times)}", flush=True)
    SaveState(str(output / "scene.pvsm"))
    provenance = {
        "renderer": "ParaView",
        "version": str(GetParaViewVersion()),
        "style_revision": 3,
        "frames_rendered": len(indices),
        "animation": None,
        "field": args.field,
        "resolution": [args.resolution] * 2,
        "axes": False,
        "annotations": False,
        "particle_radius": 0.62 * spacing,
        "camera_position": list(view.CameraPosition),
        "camera_focal_point": list(view.CameraFocalPoint),
        "camera_parallel_scale": view.CameraParallelScale,
        "particles": metadata["particles"],
        "auxiliary_geometry": auxiliary_geometry,
    }
    (output / "render.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
