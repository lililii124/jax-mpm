"""Render material-point domains at their computed positions with ParaView."""

import argparse
import csv
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
    parser.add_argument('--stress-max', type=float, default=60e6)
    parser.add_argument('--field', default='stress_yy')
    parser.add_argument('--center', nargs=2, type=float, default=(.03, .03))
    parser.add_argument('--half-height', type=float, default=.023)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    completed = (args.input / 'summary.json').exists()
    if completed:
        summary = json.loads((args.input / 'summary.json').read_text())
    elif args.allow_partial:
        summary = json.loads((args.input / 'input.json').read_text())
        rows = list(csv.DictReader((args.input / 'history.csv').open()))
        summary.update(frames=len(rows), all_states_valid=all(row['valid']=='True' for row in rows))
    else:
        raise ValueError('Incomplete trajectory; use --allow-partial for explicit development review')
    if not summary['all_states_valid'] or args.size < 128 or args.stress_max <= 0:
        raise ValueError('A valid trajectory, positive stress limit and size >= 128 are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    frames = sorted(args.input.glob('frame-*.npz'))
    if len(frames) < 2 or len(frames) != summary['frames']:
        raise ValueError('Saved frames must match the accepted history')
    rows = list(csv.DictReader((args.input / 'history.csv').open()))
    if len(rows)!=len(frames) or any(row['valid']!='True' for row in rows):
        raise ValueError('Saved frames require matching valid history rows')
    spacing = np.asarray(summary['particle_spacing_m'])
    corners = .5 * spacing * np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    pv._DisableFirstRenderCameraReset()
    view = pv.CreateView('RenderView')
    view.ViewSize = [args.size * 4] * 2
    view.UseColorPaletteForBackground = 0
    view.Background = [1., 1., 1.]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [*args.center, 1.]
    view.CameraFocalPoint = [*args.center, 0.]
    view.CameraViewUp = [0., 1., 0.]
    view.CameraParallelScale = args.half_height
    view.UseFXAA = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.with_suffix('.frame.png')
    images, distinct = [], set()
    source_hash = hashlib.sha256()
    for index, path in enumerate(frames):
        source_hash.update(path.read_bytes())
        with np.load(path, allow_pickle=False) as data:
            if not np.isclose(float(data['time']),float(rows[index]['time_s']),atol=1e-15,rtol=1e-12):
                raise ValueError('Frame time differs from its history row')
            xy = data['x'][:, None, :] + np.einsum('pij,kj->pki', data['F'], corners)
            xy = xy.reshape(-1, 2)
            xyz = np.column_stack((xy, np.zeros(len(xy))))
            values = data[args.field]
        if not np.isfinite(xyz).all() or not np.isfinite(values).all():
            raise ValueError('Nonfinite coordinates or stresses')
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(xyz, deep=True))
        cells = vtk.vtkCellArray()
        connectivity = np.column_stack((np.full(len(values), 4), np.arange(len(xyz)).reshape(-1, 4)))
        cells.SetCells(len(values), numpy_to_vtkIdTypeArray(connectivity.ravel().astype(np.int64), deep=True))
        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(cells)
        field = numpy_to_vtk(values, deep=True)
        field.SetName(args.field)
        mesh.GetCellData().SetScalars(field)
        source = pv.TrivialProducer()
        source.GetClientSideObject().SetOutput(mesh)
        source.UpdatePipeline()
        fetched = servermanager.Fetch(source)
        if not np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()), xyz):
            raise ValueError('Rendered coordinates differ from computed domains')
        display = pv.Show(source, view)
        display.Representation = 'Surface'
        display.Ambient = 1.
        display.Diffuse = 0.
        pv.ColorBy(display, ('CELLS', args.field))
        lut = pv.GetColorTransferFunction(args.field)
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
        raise ValueError('Static rendering; no GIF produced')
    images[0].save(args.output, save_all=True, append_images=images[1:],
                   duration=[100] * (len(images) - 1) + [900], loop=0, disposal=2)
    images[-1].save(args.output.with_suffix('.png'))
    receipt = dict(renderer='ParaView', frames=len(images), distinct_frames=len(distinct),
                   completed_schedule=completed, rendered_end_time_s=float(rows[-1]['time_s']),
                   requested_end_time_s=summary['duration_s'],
                   size=[args.size, args.size], deformation_scale=1., stress_range_Pa=[0., args.stress_max],
                   field=args.field, camera_center=args.center, camera_half_height=args.half_height,
                   geometry='Particle reference rectangles mapped by computed F and translated to computed x',
                   artificial_crack_lines=False, source_sha256=source_hash.hexdigest(),
                   gif_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(gif=str(args.output), **receipt)), flush=True)


if __name__ == '__main__':
    main()
