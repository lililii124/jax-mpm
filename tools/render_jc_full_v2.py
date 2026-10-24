"""Render verified original coordinates with a fixed orthographic ParaView camera."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(folder, output, review_path):
    import numpy as np
    from paraview.simple import CreateView, Show, SaveScreenshot, SaveState, TrivialProducer
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    review = json.loads(review_path.read_text())
    if not review['passed'] or not review['coordinate_neck_audit']['passed']:
        raise ValueError('Original raw audit did not pass neck gate')
    output.mkdir(parents=True, exist_ok=False)
    frames = output / 'frames'
    frames.mkdir()
    trajectory = np.load(folder / 'trajectory_x.npy', mmap_mode='r', allow_pickle=False)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for coordinates in trajectory:
        low = np.minimum(low, coordinates.min(axis=0))
        high = np.maximum(high, coordinates.max(axis=0))
    center = (low+high)/2
    scale = max((high[1]-low[1])/2, (high[0]-low[0])/2)*1.12
    view = CreateView('RenderView')
    view.ViewSize = [1200, 1200]
    view.Background = [1, 1, 1]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [float(center[0]), float(center[1]), float(high[2]+.1)]
    view.CameraFocalPoint = center.tolist()
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = float(scale)
    source = TrivialProducer()
    cells = vtkCellArray()
    cells.SetCells(341584, numpy_to_vtkIdTypeArray(
        np.column_stack((np.ones(341584, dtype=np.int64),
                         np.arange(341584, dtype=np.int64))).ravel(), deep=True))
    initial = vtkPolyData()
    initial_points = vtkPoints()
    initial_points.SetData(numpy_to_vtk(np.array(trajectory[0]), deep=True))
    initial.SetPoints(initial_points)
    initial.SetVerts(cells)
    source.GetClientSideObject().SetOutput(initial)
    source.UpdatePipeline()
    display = Show(source, view)
    display.Representation = 'Point Gaussian'
    display.GaussianRadius = 2.5e-5
    display.DiffuseColor = [.12, .30, .48]
    display.AmbientColor = [.12, .30, .48]
    display.ColorArrayName = ['POINTS', '']
    inputs, exports = {}, []
    for index in range(61):
        raw = folder / 'frames_data' / f'frame_{index:05d}.npz'
        with np.load(raw, allow_pickle=False) as frame:
            coordinates = np.array(frame['x'], copy=True)
            np.testing.assert_array_equal(coordinates, trajectory[index])
            points = vtkPoints()
            points.SetData(numpy_to_vtk(coordinates, deep=True))
            poly = vtkPolyData()
            poly.SetPoints(points)
            poly.SetVerts(cells)
            np.testing.assert_array_equal(vtk_to_numpy(poly.GetPoints().GetData()), coordinates)
            source.GetClientSideObject().SetOutput(poly)
            source.UpdatePipeline()
            if source.GetDataInformation().GetNumberOfPoints() != 341584:
                raise ValueError('ParaView point count differs')
            target = frames / f'frame_{index:05d}.png'
            SaveScreenshot(str(target), view, ImageResolution=[1200, 1200])
            inputs[raw.name] = sha(raw)
            exports.append({'frame': index, 'time_s': float(frame['time']), 'points': len(coordinates),
                            'png_sha256': sha(target)})
    trajectory._mmap.close()
    SaveState(str(output / 'scene.pvsm'))
    report = {'renderer': 'ParaView', 'frames_rendered': 61, 'geometry_scale': 1,
              'camera': 'fixed orthographic along positive z', 'axes': False, 'text': False,
              'all_particles_retained': True, 'resolution': [1200, 1200],
              'raw_frames_sha256': inputs, 'frames': exports, 'review_sha256': sha(review_path),
              'source_sha256': sha(__file__)}
    (output / 'render.json').write_text(json.dumps(report, indent=2)+'\n')


def encode(output):
    from PIL import Image, GifImagePlugin
    report = json.loads((output / 'render.json').read_text())
    images = []
    durations = []
    for item in report['frames']:
        path = output / 'frames' / f"frame_{item['frame']:05d}.png"
        if sha(path) != item['png_sha256']:
            raise ValueError('Rendered frame changed')
        with Image.open(path) as image:
            images.append(image.convert('RGB').resize((400, 400), Image.Resampling.LANCZOS).quantize(colors=128))
        durations.append(250 if 25e-6-1e-15 <= item['time_s'] <= 45e-6+1e-15 else 70)
    target = output / 'jc-full61-necking.gif'
    with target.open('wb') as stream:
        header, _ = GifImagePlugin.getheader(images[0], info={'loop': 0})
        for block in header:
            stream.write(block)
        for image, duration in zip(images, durations):
            for block in GifImagePlugin.getdata(image, duration=duration, disposal=2,
                                                include_color_table=True):
                stream.write(block)
        stream.write(b';')
    with Image.open(target) as image:
        if image.n_frames != 61 or image.size != (400, 400):
            raise ValueError('GIF frame count/dimensions differ')
        for i, expected in enumerate(durations):
            image.seek(i)
            image.load()
            if image.info['duration'] != expected:
                raise ValueError('GIF timing differs')
    if target.stat().st_size >= 9_000_000:
        raise ValueError('GIF exceeds 9 MB')
    report['gif'] = {'path': target.name, 'bytes': target.stat().st_size,
                     'sha256': sha(target), 'durations_ms': durations, 'frames': 61}
    (output / 'media-audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['render', 'encode'])
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    render(args.folder, args.output, args.review) if args.action == 'render' else encode(args.output)
