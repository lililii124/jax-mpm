"""Render actual moving reference polygons and Neo-Hookean principal stress."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from paraview import simple as pv, servermanager, vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=400)
    parser.add_argument('--stress-max', type=float, default=40000.)
    parser.add_argument('--center', nargs=2, type=float, default=(.04, .04))
    parser.add_argument('--parallel-scale', type=float, default=.023)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.png').exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('GIF, final PNG, and receipt paths must not already exist')
    if args.size < 128 or not np.isfinite(args.stress_max) or args.stress_max <= 0 or not np.isfinite(args.parallel_scale) or args.parallel_scale <= 0:
        raise ValueError('Require size >= 128 and finite positive stress and camera ranges')

    input_data = json.loads((args.input / 'input.json').read_text())
    summary = json.loads((args.input / 'summary.json').read_text())
    if not summary['completed'] or not summary['final']['valid']:
        raise ValueError('A completed, valid trajectory is required')
    E, nu = input_data['young_Pa'], input_data['poisson']
    mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
    frames = sorted(args.input.glob('frame-*.npz'))
    if len(frames) < 2:
        raise ValueError('At least two saved frames are required')
    with np.load(args.input / 'final.npz', allow_pickle=False) as data:
        reference_vertices = data['reference_vertices'].copy()
        reference_count = data['reference_count'].copy()
        reference_centroid = data['reference_centroid'].copy()
        mass = data['mass'].copy()
    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError('Mass must be finite and nonnegative')
    include = (mass != 0) & (reference_count >= 3)
    if not include.any():
        raise ValueError('No nonzero-mass polygon domains found')

    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView('RenderView')
    view.ViewSize = [args.size * 4, args.size * 4]
    view.UseColorPaletteForBackground = 0
    view.Background = [1., 1., 1.]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [*args.center, 1.]
    view.CameraFocalPoint = [*args.center, 0.]
    view.CameraViewUp = [0., 1., 0.]
    view.CameraParallelScale = args.parallel_scale
    view.UseFXAA = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.with_suffix('.render-temp.png')
    images, distinct = [], set()
    source_hash = hashlib.sha256()
    checked_frames, max_vertex_error = 0, 0.
    for path in frames:
        raw_bytes = path.read_bytes()
        source_hash.update(raw_bytes)
        with np.load(path, allow_pickle=False) as data:
            x, F = data['x'], data['F']
            xy_list, offsets, conn = [], [0], []
            stresses = []
            for i in np.flatnonzero(include):
                count = int(reference_count[i])
                vertices = reference_vertices[i, :count]
                current = x[i] + (F[i] @ (vertices - reference_centroid[i]).T).T
                base = offsets[-1]
                xy_list.append(current)
                conn.extend(range(base, base + count))
                offsets.append(len(conn))
                J = np.linalg.det(F[i])
                if J <= 0:
                    raise ValueError(f'Nonpositive deformation determinant in {path.name}')
                B = F[i] @ F[i].T
                sigma = (mu / J) * (B - np.eye(2)) + (lam * np.log(J) / J) * np.eye(2)
                stresses.append(float(np.linalg.eigvalsh(sigma)[-1]))
            xy = np.concatenate(xy_list, axis=0)
            values = np.asarray(stresses)
        xyz = np.column_stack((xy, np.zeros(len(xy))))
        connectivity = []
        for j, value in enumerate(values):
            start, end = offsets[j], offsets[j + 1]
            connectivity.extend([end - start, *conn[start:end]])
        if not np.isfinite(xyz).all() or not np.isfinite(values).all():
            raise ValueError('Nonfinite vertices or stress values')

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(xyz, deep=True))
        cells = vtk.vtkCellArray()
        cells.SetCells(len(values), numpy_to_vtkIdTypeArray(np.asarray(connectivity, dtype=np.int64), deep=True))
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(cells)
        field = numpy_to_vtk(values, deep=True)
        field.SetName('principal_stress')
        mesh.GetCellData().SetScalars(field)
        source = pv.TrivialProducer()
        source.GetClientSideObject().SetOutput(mesh)
        source.UpdatePipeline()
        fetched = servermanager.Fetch(source)
        actual = vtk_to_numpy(fetched.GetPoints().GetData())
        if actual.shape != xyz.shape:
            raise ValueError('Rendered vertex count differs from computed polygons')
        error = float(np.max(np.abs(actual - xyz)))
        if error > 1e-12:
            raise ValueError(f'Rendered polygon vertices differ from computed positions: {error}')
        max_vertex_error = max(max_vertex_error, error)
        checked_frames += 1

        display = pv.Show(source, view)
        display.Representation = 'Surface'
        display.Ambient = 1.
        display.Diffuse = 0.
        pv.ColorBy(display, ('CELLS', 'principal_stress'))
        lut = pv.GetColorTransferFunction('principal_stress')
        lut.RGBPoints = [0., .19, .36, .59,
                         .5 * args.stress_max, .88, .91, .94,
                         args.stress_max, .77, .16, .10]
        lut.ColorSpace = 'RGB'
        lut.RescaleTransferFunction(0., args.stress_max)
        display.SetScalarBarVisibility(view, False)
        pv.Render(view)
        pv.SaveScreenshot(str(scratch), view, ImageResolution=[args.size * 4] * 2)
        with Image.open(scratch) as raw:
            image = raw.convert('RGB').resize((args.size, args.size), Image.Resampling.LANCZOS)
        distinct.add(hashlib.sha256(image.tobytes()).hexdigest())
        images.append(image)
        pv.Delete(display)
        pv.Delete(source)

    scratch.unlink()
    if len(distinct) < 2:
        raise ValueError('Static rendering; no animated GIF produced')
    images[0].save(args.output, save_all=True, append_images=images[1:],
                   duration=[100] * (len(images) - 1) + [900], loop=0, disposal=2)
    images[-1].save(args.output.with_suffix('.png'))
    gif_hash = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = dict(renderer='ParaView', frames=len(images), distinct_frames=len(distinct),
                   size=[args.size, args.size], supersample=4, deformation_scale=1.,
                   stress_field='maximum in-plane Cauchy principal stress (compressible Neo-Hookean)',
                   stress_range_Pa=[0., args.stress_max], camera_center=args.center,
                   camera_parallel_scale=args.parallel_scale,
                   geometry='Current polygons from reference_vertices/count/centroid mapped by frame F and x',
                   included_domains=int(include.sum()), checked_frames=checked_frames,
                   max_rendered_vertex_error_m=max_vertex_error,
                   source_sha256=source_hash.hexdigest(), final_state_sha256=hashlib.sha256((args.input / 'final.npz').read_bytes()).hexdigest(),
                   gif_sha256=gif_hash)
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(gif=str(args.output), **receipt)), flush=True)


if __name__ == '__main__':
    main()
