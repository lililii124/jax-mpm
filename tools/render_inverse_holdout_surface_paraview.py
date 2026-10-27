"""Render real structured-particle holdout frames as an antialiased outer surface."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def surface_topology(counts: tuple[int, int, int]):
    nx, ny, nz = counts
    grid = np.arange(nx * ny * nz, dtype=np.int64).reshape(nx, ny, nz)
    boundary = np.zeros((nx, ny, nz), dtype=bool)
    boundary[[0, -1], :, :] = True
    boundary[:, [0, -1], :] = True
    boundary[:, :, [0, -1]] = True
    particle_ids = grid[boundary]
    remap = np.full(nx * ny * nz, -1, dtype=np.int64)
    remap[particle_ids] = np.arange(len(particle_ids), dtype=np.int64)
    faces: list[tuple[int, int, int, int]] = []

    # Quads are connected by integer reference-grid adjacency, with outward winding.
    for j in range(ny - 1):
        for k in range(nz - 1):
            q0 = [grid[0, j, k], grid[0, j, k + 1], grid[0, j + 1, k + 1], grid[0, j + 1, k]]
            faces.append(tuple(remap[q0]))
            q1 = [grid[-1, j, k], grid[-1, j + 1, k], grid[-1, j + 1, k + 1], grid[-1, j, k + 1]]
            faces.append(tuple(remap[q1]))
    for i in range(nx - 1):
        for k in range(nz - 1):
            q0 = [grid[i, 0, k], grid[i + 1, 0, k], grid[i + 1, 0, k + 1], grid[i, 0, k + 1]]
            faces.append(tuple(remap[q0]))
            q1 = [grid[i, -1, k], grid[i, -1, k + 1], grid[i + 1, -1, k + 1], grid[i + 1, -1, k]]
            faces.append(tuple(remap[q1]))
    for i in range(nx - 1):
        for j in range(ny - 1):
            q0 = [grid[i, j, 0], grid[i, j + 1, 0], grid[i + 1, j + 1, 0], grid[i + 1, j, 0]]
            faces.append(tuple(remap[q0]))
            q1 = [grid[i, j, -1], grid[i + 1, j, -1], grid[i + 1, j + 1, -1], grid[i, j + 1, -1]]
            faces.append(tuple(remap[q1]))
    return particle_ids, np.asarray(faces, dtype=np.int64), (nx, ny, nz)


def write_surface_vtk(path: Path, points: np.ndarray, scalars: np.ndarray,
                      quads: np.ndarray) -> None:
    n = len(points)
    polygon_data = np.empty((len(quads), 5), dtype=">i4")
    polygon_data[:, 0] = 4
    polygon_data[:, 1:] = quads.astype(np.int32, copy=False)
    # Area-weighted smooth point normals from the deformed exterior quads.
    tri_a = quads[:, [0, 1, 2]]
    tri_b = quads[:, [0, 2, 3]]
    cross_a = np.cross(points[tri_a[:, 1]] - points[tri_a[:, 0]],
                       points[tri_a[:, 2]] - points[tri_a[:, 0]])
    cross_b = np.cross(points[tri_b[:, 1]] - points[tri_b[:, 0]],
                       points[tri_b[:, 2]] - points[tri_b[:, 0]])
    normals = np.zeros_like(points, dtype=np.float64)
    np.add.at(normals, tri_a.reshape(-1), np.repeat(cross_a, 3, axis=0))
    np.add.at(normals, tri_b.reshape(-1), np.repeat(cross_b, 3, axis=0))
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm <= 0):
        raise ValueError("degenerate deformed exterior-surface vertex normal")
    normals /= norm[:, None]
    with path.open("wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nactual MPM exterior particle-adjacency surface\nBINARY\n")
        f.write(f"DATASET POLYDATA\nPOINTS {n} float\n".encode("ascii"))
        f.write(np.asarray(points, dtype=">f4", order="C").tobytes(order="C")); f.write(b"\n")
        f.write(f"POLYGONS {len(quads)} {5*len(quads)}\n".encode("ascii"))
        f.write(polygon_data.tobytes(order="C")); f.write(b"\n")
        f.write(f"POINT_DATA {n}\nNORMALS Normals float\n".encode("ascii"))
        f.write(np.asarray(normals, dtype=">f4", order="C").tobytes(order="C")); f.write(b"\n")
        f.write(b"SCALARS displacement_magnitude_m float 1\nLOOKUP_TABLE default\n")
        f.write(np.asarray(scalars, dtype=">f4").tobytes(order="C")); f.write(b"\n")


def write_scene_script(path: Path, vtk_root: Path, png_prefix: Path,
                       vmax: float, fps: int, render_size: int) -> None:
    truth = sorted((vtk_root / "truth").glob("frame_*.vtk"))
    inferred = sorted((vtk_root / "inferred").glob("frame_*.vtk"))
    if len(truth) < 2 or len(truth) != len(inferred):
        raise ValueError("truth and inferred surface series must have matching frames")
    lines = [
        "import paraview.simple as pvs",
        "from paraview.simple import *",
        "import json",
        "pvs._DisableFirstRenderCameraReset()",
        f"truth_files={json.dumps([str(p.resolve()) for p in truth])}",
        f"inferred_files={json.dumps([str(p.resolve()) for p in inferred])}",
        f"output={json.dumps(str(png_prefix.resolve()))}",
        f"vmax={float(vmax):.17g}", f"fps={int(fps)}", f"render_size={int(render_size)}",
        "layout=CreateLayout('Holdout smooth exterior comparison')",
        "views=[]",
        "for idx,files in enumerate((truth_files,inferred_files)):",
        "    reader=LegacyVTKReader(registrationName=('truth' if idx==0 else 'inferred'),FileNames=files)",
        "    view=CreateView('RenderView')",
        "    AssignViewToLayout(view,layout,idx)",
        "    view.UseColorPaletteForBackground=0",
        "    view.Background=[0.96,0.96,0.96]",
        "    view.OrientationAxesVisibility=0",
        "    view.UseFXAA=1",
        "    view.MultiSamples=8",
        "    view.CameraParallelProjection=1",
        "    view.CameraPosition=[1.45,1.40,1.30]",
        "    view.CameraFocalPoint=[0.5,0.5,0.5]",
        "    view.CameraViewUp=[0,0,1]",
        "    view.CameraParallelScale=0.50",
        "    rep=Show(reader,view)",
        "    rep.Representation='Surface'",
        "    rep.InterpolateScalarsBeforeMapping=1",
        "    ColorBy(rep,('POINTS','displacement_magnitude_m'))",
        "    lut=GetColorTransferFunction('displacement_magnitude_m')",
        "    lut.ApplyPreset('Cool to Warm (Extended)',True)",
        "    lut.RescaleTransferFunction(0.0,vmax)",
        "    rep.SetScalarBarVisibility(view,False)",
        "    view.Update()",
        "    views.append(view)",
        f"layout.SetSize({render_size},{render_size})",
        "scene=GetAnimationScene()",
        "scene.UpdateAnimationUsingDataTimeSteps()",
        "scene.PlayMode='Sequence'",
        "scene.NumberOfFrames=len(truth_files)",
        "scene.StartTime=0.0",
        "scene.EndTime=float(len(truth_files)-1)",
        f"SaveAnimation(output,layout,ImageResolution=[{render_size},{render_size}],FrameRate=fps)",
        "print(json.dumps({'output':output,'frames':len(truth_files),'resolution':[render_size,render_size],'fps':fps,'vmax_m':vmax}))",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--pvpython", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--supersample", type=int, default=4)
    args = ap.parse_args()
    receipt = json.loads((args.frames / "particle_frames.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise ValueError("source MPM frames are not complete")
    if args.supersample < 2:
        raise ValueError("supersample factor must be at least two")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    grid = tuple(np.maximum(4, np.rint(np.asarray(receipt["lengths_m"]) *
                                      int(receipt["grid"]) * float(receipt["ppc"])).astype(int)))
    if int(np.prod(grid)) != int(receipt["particles"]):
        raise ValueError("structured reference particle grid does not match the saved particle count")
    surface_ids, quads, grid_shape = surface_topology(grid)
    vtk_root = out / "vtk_surface_series"
    max_mag = 0.0
    for label in ("truth", "inferred"):
        with np.load(args.frames / f"{label}_particle_positions.npz", allow_pickle=False) as data:
            positions = data["positions_m"].copy()
            reference = data["reference_positions_m"].copy()
            steps = data["frame_step_indices"].copy()
        if not np.array_equal(steps, np.asarray(receipt["frame_step_indices"])):
            raise ValueError(f"{label} frame times differ from the MPM receipt")
        if len(reference) != int(np.prod(grid)):
            raise ValueError("reference coordinates do not match the structured particle lattice")
        displacement = positions - reference[None, :, :]
        magnitudes = np.linalg.norm(displacement, axis=2)
        max_mag = max(max_mag, float(magnitudes.max()))
        folder = vtk_root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i, (x, mag) in enumerate(zip(positions, magnitudes)):
            write_surface_vtk(folder / f"frame_{i:04d}.vtk", x[surface_ids], mag[surface_ids], quads)

    high = out / "paraview_supersampled"
    high.mkdir(parents=True, exist_ok=True)
    render_size = 400 * args.supersample
    prefix = high / "holdout_surface_frame.png"
    scene = out / "render_surface_scene.py"
    write_scene_script(scene, vtk_root, prefix, max_mag, args.fps, render_size)
    proc = subprocess.run([str(args.pvpython), str(scene)], capture_output=True, text=True, timeout=600)
    (out / "paraview_surface_render.log").write_text("STDOUT\n"+proc.stdout+"\nSTDERR\n"+proc.stderr+"\n", encoding="utf-8")
    pngs = sorted(high.glob("holdout_surface_frame*.png"))
    frame_steps = receipt["frame_step_indices"]
    if proc.returncode != 0 or len(pngs) != len(frame_steps):
        raise RuntimeError(f"ParaView surface render failed (exit={proc.returncode}, pngs={len(pngs)}); see paraview_surface_render.log")

    # LANCZOS downsampling is spatial antialiasing only; each PNG is one exact
    # saved simulation state, and the sequence length/time order is unchanged.
    reduced = []
    for p in pngs:
        with Image.open(p) as im:
            if im.size != (render_size, render_size):
                raise ValueError(f"unexpected ParaView image dimensions for {p.name}: {im.size}")
            reduced.append(im.convert("RGB").resize((400, 400), Image.Resampling.LANCZOS))
    palette_source = Image.new("RGB", (400, 400 * len(reduced)))
    for i, im in enumerate(reduced):
        palette_source.paste(im, (0, 400 * i))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    gif_frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in reduced]
    gif = out / "orthotropic_holdout_particles_surface_400.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:],
                       duration=round(1000 / args.fps), loop=0, optimize=False, disposal=2)

    # Unlabeled review comparison, using the same final MPM frame and camera:
    # left is the original point-cloud rendering; right is this smooth surface.
    old_png = args.frames / "paraview_particle_frame.0025.png"
    if old_png.exists():
        with Image.open(old_png) as old:
            before = old.convert("RGB").resize((400, 400), Image.Resampling.LANCZOS)
        comparison = Image.new("RGB", (800, 400), "white")
        comparison.paste(before, (0, 0)); comparison.paste(reduced[-1], (400, 0))
        comparison.save(out / "surface_before_after.png", optimize=True)

    render = {
        "status": "complete", "gif": gif.name, "gif_bytes": gif.stat().st_size,
        "resolution_px": [400, 400], "paraview_supersampled_resolution_px": [render_size, render_size],
        "supersample_factor": args.supersample, "fxaa": True, "multisamples": 8,
        "spatial_downsampling": "Pillow LANCZOS; temporal frames unchanged",
        "frame_rate_fps": args.fps, "frame_count": len(frame_steps),
        "frame_step_indices": frame_steps, "panel_order": ["truth", "inferred"],
        "outer_surface": {"reference_grid_particle_counts_xyz": list(map(int, grid_shape)),
            "surface_vertices": int(len(surface_ids)), "quad_faces": int(len(quads)),
            "construction": "six structured reference-lattice boundary faces connected by nearest material-particle adjacency; deformed coordinates sampled from actual MPM states"},
        "scalar": "actual per-particle displacement magnitude; same fixed range in both panels",
        "maximum_particle_displacement_m": max_mag,
        "text_overlays": False, "scalar_bar": False, "orientation_axes": False,
        "coordinate_source": "float64 actual TL-MPM particle state; VTK display coordinates/scalars stored float32; no temporal interpolation or deformation scaling",
        "original_point_cloud_gif_preserved": str(args.frames / "orthotropic_holdout_particles.gif"),
    }
    (out / "surface_render_receipt.json").write_text(json.dumps(render, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(render, indent=2))


if __name__ == "__main__":
    main()
