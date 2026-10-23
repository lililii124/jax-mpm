"""Executed by ParaView's pvpython, not the solver's Python environment."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from paraview.simple import (
    _DisableFirstRenderCameraReset,
    Box,
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


def grip_displacement(definition, time):
    """MovingVelocityBox translation, evaluated without importing the solver."""
    start = definition.get("start", 0.0)
    end = definition.get("end")
    end = np.inf if end is None else end
    ramp = definition.get("ramp_time", 0.0)
    elapsed = np.clip(np.asarray(time) - start, 0.0, end - start)
    if ramp == 0:
        distance = elapsed
    else:
        during_ramp = np.minimum(elapsed, ramp)
        distance = 0.5 * (
            during_ramp - ramp / np.pi * np.sin(np.pi * during_ramp / ramp)
        )
        distance += np.maximum(elapsed - ramp, 0.0)
    return np.asarray(definition["velocity"]) * np.asarray(distance)[..., None]


def waypoint_sphere_motion(definition, time):
    """SpherePath position evaluated independently of JAX."""
    times, centers = np.asarray(definition["times"]), np.asarray(definition["centers"])
    phase = np.clip(
        (np.asarray(time)[..., None] - times[:-1]) / np.diff(times), 0.0, 1.0
    )
    scale = 0.5 * (1.0 - np.cos(np.pi * phase))
    return centers[0] + np.sum(scale[..., None] * np.diff(centers, axis=0), axis=-2)


def body_failure_modes(summary, count):
    configs = summary.get("configs") or []
    return [
        "ballistic"
        if body < len(configs) and configs[body].get("failure_mode") == "ballistic"
        else "retain"
        for body in range(count)
    ]


def velocity_patch_fixtures(definition):
    required = (
        "lower",
        "upper",
        "support_centers_x_m",
        "support_patch_width_m",
        "loading_boundary",
    )
    if definition.get("purpose") != "concrete-bending" or any(
        key not in definition for key in required
    ):
        return []
    lower, upper = definition["lower"], definition["upper"]
    width = definition["support_patch_width_m"]
    fixtures = [
        {
            "lower": [center - width / 2, lower[1] - width, lower[2]],
            "upper": [center + width / 2, lower[1], upper[2]],
            "velocity": [0.0, 0.0, 0.0],
            "role": "support",
        }
        for center in definition["support_centers_x_m"]
    ]
    loading = definition["loading_boundary"]
    fixtures.append(
        {
            "lower": [loading["lower"][0], upper[1], lower[2]],
            "upper": [loading["upper"][0], loading["upper"][1], upper[2]],
            "velocity": loading["velocity"],
            "ramp_time": loading.get("ramp_time", 0.0),
            "start": loading.get("start", 0.0),
            "end": loading.get("end"),
            "role": "loading",
        }
    )
    return fixtures


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
    has_failed = reader.PointData["failed"] is not None
    times = list(reader.TimestepValues)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    summary_path = output / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    sphere_definition = summary.get("definition", {}).get("rigid_sphere")
    waypoint_sphere_definitions = (
        summary.get("definition", {}).get("rigid_spheres") or []
    )
    grip_definitions = summary.get("definition", {}).get("rigid_grips") or []
    fixture_definitions = velocity_patch_fixtures(summary.get("definition", {}))
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
    grip_sources = []
    waypoint_sphere_sources = []
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
            bounds[2 * axis] = min(
                bounds[2 * axis], center0[axis] + min(0, stroke[axis]) - radius
            )
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
    for index, definition in enumerate(waypoint_sphere_definitions):
        sphere = Sphere(registrationName=f"Adhesive rigid sphere {index + 1}")
        radius = definition["radius"]
        sphere.Radius = radius
        sphere.ThetaResolution = 128
        sphere.PhiResolution = 96
        sphere.Center = definition["centers"][0]
        surface = Show(sphere, view)
        surface.ColorArrayName = ["POINTS", ""]
        surface.DiffuseColor = [0.59, 0.65, 0.71]
        surface.Ambient = 0.2
        surface.Diffuse = 0.8
        surface.Specular = 0.65
        surface.SpecularPower = 45
        centers = waypoint_sphere_motion(definition, times)
        all_centers = np.asarray(definition["centers"])
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
            bounds[2 * axis] = min(
                bounds[2 * axis], all_centers[:, axis].min() - radius
            )
            bounds[2 * axis + 1] = max(
                bounds[2 * axis + 1], all_centers[:, axis].max() + radius
            )
        waypoint_sphere_sources.append((sphere, definition))
        auxiliary_geometry.append(
            {
                "type": "prescribed_adhesive_rigid_sphere",
                "parameters": definition,
                "included_in_particle_count": False,
                "motion": "SpherePath cosine waypoint interpolation, evaluated at every output time",
                "state_file_interpolation": "linear between output-time keyframes",
                "interpretation": "Prescribed contact geometry; the deformable layer alone is represented by MPM particles",
            }
        )
    boxes = [
        (f"Prescribed rigid grip {index + 1}", "prescribed_rigid_grip", definition)
        for index, definition in enumerate(grip_definitions)
    ]
    boxes += [
        (
            f"Velocity patch {definition['role']} {index + 1}",
            "velocity_patch_fixture",
            definition,
        )
        for index, definition in enumerate(fixture_definitions)
    ]
    for name, kind, definition in boxes:
        lower, upper = np.asarray(definition["lower"]), np.asarray(definition["upper"])
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError(
                "auxiliary boxes require positive three-dimensional extents"
            )
        center0 = 0.5 * (lower + upper)
        grip = Box(registrationName=name)
        grip.Center = center0
        grip.XLength, grip.YLength, grip.ZLength = upper - lower
        surface = Show(grip, view)
        surface.ColorArrayName = ["POINTS", ""]
        surface.DiffuseColor = [0.66, 0.70, 0.74]
        surface.Ambient = 0.25
        surface.Diffuse = 0.75
        surface.Specular = 0.65
        surface.SpecularPower = 45
        translations = grip_displacement(definition, times)
        centers = center0 + translations
        span = times[-1] - times[0]
        for axis in range(3):
            track = GetAnimationTrack("Center", axis, grip)
            keys = []
            for time, position in zip(times, centers):
                key = CompositeKeyFrame()
                key.KeyTime = (time - times[0]) / span if span else 0.0
                key.KeyValues = [float(position[axis])]
                key.Interpolation = "Ramp"
                keys.append(key)
            track.KeyFrames = keys
            bounds[2 * axis] = min(
                bounds[2 * axis], lower[axis] + translations[:, axis].min()
            )
            bounds[2 * axis + 1] = max(
                bounds[2 * axis + 1], upper[axis] + translations[:, axis].max()
            )
        grip_sources.append((grip, center0, definition))
        record = {
            "type": kind,
            "parameters": definition,
            "included_in_particle_count": False,
            "motion": "MovingVelocityBox analytic translation at output times",
            "state_file_interpolation": "linear between output-time keyframes",
        }
        if kind == "velocity_patch_fixture":
            record[
                "interpretation"
            ] = "Bilateral velocity-patch representation; no separately solved contact"
            record["display_geometry"] = (
                "Support height equals patch width; loading block starts at nominal top face; "
                "depth is clipped to the specimen"
            )
        auxiliary_geometry.append(record)
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
    corners = np.array(
        list(product(*[(bounds[2 * i], bounds[2 * i + 1]) for i in range(3)]))
    )
    view.CameraParallelScale = 0.56 * max(
        np.ptp(corners @ right), np.ptp(corners @ upward)
    )
    if metadata["dim"] == 2:
        view.CameraPosition = [center[0], center[1], center[2] + 3 * length]
        view.CameraParallelScale = 0.58 * length
    spacing = metadata["spacing"] or length / 100
    scalar_color_range = None
    if args.field != "body":
        lo, hi = metadata["field_ranges"][args.field]
        if args.field in ("speed", "displacement"):
            lo = 0.0
        if hi <= lo:
            hi = lo + 1.0
        scalar_color_range = [lo, hi]
    colors = ([0.045, 0.36, 0.67], [0.94, 0.36, 0.08], [0.32, 0.66, 0.48])
    body_array = reader.PointData["body"]
    count = int(body_array.GetRange()[1]) + 1 if body_array else 1
    failure_modes = body_failure_modes(summary, count)
    displays = []
    active_sources = []
    debris_sources = []
    for body in range(count):
        selected = Threshold(Input=reader)
        selected.Scalars = ["POINTS", "body"]
        selected.LowerThreshold = body
        selected.UpperThreshold = body
        selected.UpdatePipeline()
        if has_failed and failure_modes[body] == "ballistic":
            failed = Threshold(
                registrationName=f"Ballistic debris in body {body}", Input=selected
            )
            failed.Scalars = ["POINTS", "failed"]
            failed.LowerThreshold = 1
            failed.UpperThreshold = 255
            failed.UpdatePipeline()
            debris_sources.append(failed)
            active = Threshold(
                registrationName=f"Active particles in body {body}", Input=selected
            )
            active.Scalars = ["POINTS", "failed"]
            active.LowerThreshold = 0
            active.UpperThreshold = 0
            active.UpdatePipeline()
            selected = active
        active_sources.append(selected)
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
            lo, hi = scalar_color_range
            palette = [
                (0.0, 0.267, 0.005, 0.329),
                (0.25, 0.230, 0.322, 0.546),
                (0.5, 0.128, 0.567, 0.551),
                (0.75, 0.369, 0.789, 0.383),
                (1.0, 0.993, 0.906, 0.144),
            ]
            lut.RGBPoints = [
                value
                for t, r, g, b in palette
                for value in (lo + t * (hi - lo), r, g, b)
            ]
            lut.ColorSpace = "RGB"
            lut.RescaleTransferFunction(lo, hi)
            lut.AutomaticRescaleRangeMode = "Never"
        if args.field != "body":
            display.SetScalarBarVisibility(view, False)
        displays.append(display)
    for failed_selection in debris_sources:
        debris = Show(failed_selection, view)
        debris.Representation = "Point Gaussian"
        debris.GaussianRadius = 0.38 * spacing
        debris.ShaderPreset = "Sphere"
        debris.Emissive = 0
        debris.Ambient = 0.4
        debris.Diffuse = 0.6
        debris.Specular = 0.15
        debris.Opacity = 0.28
        debris.ColorArrayName = ["POINTS", ""]
        debris.DiffuseColor = [0.46, 0.49, 0.53]
        debris.AmbientColor = [0.46, 0.49, 0.53]
        displays.append(debris)
    indices = [len(times) - 1] if args.final_only else range(len(times))
    displayed_counts = []
    (output / "frames").mkdir(exist_ok=True)
    for i in indices:
        scene.AnimationTime = times[i]
        view.ViewTime = times[i]
        reader.UpdatePipeline(times[i])
        for grip, center0, definition in grip_sources:
            grip.Center = center0 + grip_displacement(definition, times[i])
        for sphere, definition in waypoint_sphere_sources:
            sphere.Center = waypoint_sphere_motion(definition, times[i])
        Render(view)
        if has_failed:
            active_count = sum(
                source.GetDataInformation().GetNumberOfPoints()
                for source in active_sources
            )
            debris_count = sum(
                source.GetDataInformation().GetNumberOfPoints()
                for source in debris_sources
            )
            if active_count + debris_count != metadata["particles"]:
                raise RuntimeError("failure display did not retain every particle")
            displayed_counts.append(
                {
                    "time": times[i],
                    "active_particles": active_count,
                    "debris_particles": debris_count,
                }
            )
        SaveScreenshot(
            str(output / "frames" / f"frame_{i:05d}.png"),
            view,
            ImageResolution=[args.resolution] * 2,
        )
        if i == len(times) - 1:
            SaveScreenshot(
                str(output / "final.png"), view, ImageResolution=[args.resolution] * 2
            )
        if i % 10 == 0 or i == len(times) - 1:
            print(f"ParaView: frame {i + 1}/{len(times)}", flush=True)
    SaveState(str(output / "scene.pvsm"))
    provenance = {
        "renderer": "ParaView",
        "version": str(GetParaViewVersion()),
        "style_revision": 7
        if waypoint_sphere_definitions
        else 6
        if fixture_definitions
        else 5
        if has_failed
        else 4
        if grip_definitions
        else 3,
        "frames_rendered": len(indices),
        "animation": None,
        "field": args.field,
        "color_range": scalar_color_range,
        "color_range_policy": "fixed over the complete saved time history",
        "magnitude_color_range_includes_zero": args.field in ("speed", "displacement"),
        "geometry_scale": 1.0,
        "resolution": [args.resolution] * 2,
        "axes": False,
        "annotations": False,
        "particle_radius": 0.62 * spacing,
        "scene_bounds": bounds,
        "camera_position": list(view.CameraPosition),
        "camera_focal_point": list(view.CameraFocalPoint),
        "camera_parallel_scale": view.CameraParallelScale,
        "particles": metadata["particles"],
        "auxiliary_geometry": auxiliary_geometry,
    }
    if args.field == "displacement":
        provenance["color_field_definition"] = metadata.get("derived_fields", {}).get("displacement")
    if has_failed:
        provenance["failure_display"] = {
            "field": "failed",
            "body_failure_modes": failure_modes,
            "missing_config_failure_mode": "retain",
            "active_selection": "all retain-body particles and ballistic-body particles with failed == 0",
            "failed_selection": "failed != 0 in ballistic bodies only",
            "failed_representation": "Point Gaussian",
            "failed_marker_radius": 0.38 * spacing,
            "failed_marker_opacity": 0.28,
            "failed_marker_rgb": [0.46, 0.49, 0.53],
            "all_particles_retained": True,
            "displayed_counts": displayed_counts,
            "interpretation": "Non-load-bearing particle markers, not resolved fragment surfaces",
        }
    (output / "render.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
