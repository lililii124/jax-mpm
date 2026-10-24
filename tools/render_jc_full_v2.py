"""Render verified original coordinates with a fixed orthographic ParaView camera."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(folder, output, review_path, smoke=False):
    import numpy as np
    from paraview.simple import CreateView, Show, SaveScreenshot, SaveState, XMLPolyDataReader, Render
    from paraview.simple import _DisableFirstRenderCameraReset
    from paraview import servermanager
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    review = json.loads(review_path.read_text())
    if not review['passed'] or not review['coordinate_neck_audit']['passed']:
        raise ValueError('Original raw audit did not pass neck gate')
    output.mkdir(parents=True, exist_ok=False)
    frames = output / 'frames'
    frames.mkdir()
    indices = [0, 17, 60] if smoke else list(range(61))
    trajectory = None if smoke else np.load(folder / 'trajectory_x.npy', mmap_mode='r', allow_pickle=False)
    vtk_dir = output / 'vtk'
    vtk_dir.mkdir()
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for index in indices:
        with np.load(folder / 'frames_data' / f'frame_{index:05d}.npz', allow_pickle=False) as frame:
            coordinates = frame['x']
            low = np.minimum(low, coordinates.min(axis=0))
            high = np.maximum(high, coordinates.max(axis=0))
    center = (low+high)/2
    scale = max((high[1]-low[1])/2, (high[0]-low[0])/2)*1.12
    _DisableFirstRenderCameraReset()
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
    cells = vtkCellArray()
    cells.SetCells(341584, numpy_to_vtkIdTypeArray(
        np.column_stack((np.ones(341584, dtype=np.int64),
                         np.arange(341584, dtype=np.int64))).ravel(), deep=True))
    source, display = None, None
    inputs, exports = {}, []
    for index in indices:
        raw = folder / 'frames_data' / f'frame_{index:05d}.npz'
        with np.load(raw, allow_pickle=False) as frame:
            coordinates = np.array(frame['x'], copy=True)
            if trajectory is not None:
                np.testing.assert_array_equal(coordinates, trajectory[index])
            points = vtkPoints()
            points.SetData(numpy_to_vtk(coordinates, deep=True))
            poly = vtkPolyData()
            poly.SetPoints(points)
            poly.SetVerts(cells)
            np.testing.assert_array_equal(vtk_to_numpy(poly.GetPoints().GetData()), coordinates)
            vtp = vtk_dir / f'particles_{index:05d}.vtp'
            writer = vtkXMLPolyDataWriter()
            writer.SetFileName(str(vtp))
            writer.SetInputData(poly)
            if writer.Write() != 1:
                raise ValueError('VTP export failed')
            if source is None:
                source = XMLPolyDataReader(FileName=[str(vtp.resolve())])
                source.UpdatePipeline()
                display = Show(source, view)
                display.Representation = 'Point Gaussian'
                display.GaussianRadius = 2.5e-5
                display.DiffuseColor = [.12, .30, .48]
                display.AmbientColor = [.12, .30, .48]
                display.ColorArrayName = ['POINTS', '']
            else:
                source.FileName = [str(vtp.resolve())]
                source.UpdateVTKObjects()
                source.UpdatePipeline()
            actual = servermanager.Fetch(source)
            np.testing.assert_array_equal(vtk_to_numpy(actual.GetPoints().GetData()), coordinates)
            if actual.GetNumberOfPoints() != 341584:
                raise ValueError('ParaView pipeline point count differs')
            view.CameraParallelProjection = 1
            view.CameraPosition = [float(center[0]), float(center[1]), float(high[2]+.1)]
            view.CameraFocalPoint = center.tolist()
            view.CameraViewUp = [0, 1, 0]
            view.CameraParallelScale = float(scale)
            view.Update()
            Render(view)
            target = frames / f'frame_{index:05d}.png'
            SaveScreenshot(str(target), view, ImageResolution=[1200, 1200])
            inputs[raw.name] = sha(raw)
            exports.append({'frame': index, 'time_s': float(frame['time']), 'points': len(coordinates),
                            'fetched_coordinates_sha256': hashlib.sha256(
                                vtk_to_numpy(actual.GetPoints().GetData()).tobytes()).hexdigest(),
                            'png_sha256': sha(target)})
    if trajectory is not None:
        trajectory._mmap.close()
    anchors = [row['png_sha256'] for row in exports if row['frame'] in (0, 17, 60)]
    if len(set(anchors)) != 3:
        raise ValueError('ParaView anchor images failed to change with geometry')
    SaveState(str(output / 'scene.pvsm'))
    report = {'renderer': 'ParaView', 'frames_rendered': len(indices), 'smoke_only': smoke, 'pipeline_coordinates_verified': True, 'geometry_scale': 1,
              'unique_png_sha256_count': len({row['png_sha256'] for row in exports}),
              'camera': 'fixed orthographic along positive z', 'axes': False, 'text': False,
              'camera_bounds_m': [low.tolist(), high.tolist()], 'camera_parallel_scale': float(scale),
              'all_particles_retained': True, 'resolution': [1200, 1200],
              'raw_frames_sha256': inputs, 'frames': exports, 'review_sha256': sha(review_path),
              'source_sha256': sha(__file__)}
    (output / 'render.json').write_text(json.dumps(report, indent=2)+'\n')


def encode(output):
    from PIL import Image, GifImagePlugin
    report = json.loads((output / 'render.json').read_text())
    if report.get('smoke_only') or len(report['frames']) != 61:
        raise ValueError('Full original 61 frames required for GIF')
    if len({r['png_sha256'] for r in report['frames']}) <= 1:
        raise ValueError('Rendered animation contains one repeated image')
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
        pixel_states = []
        for i, expected in enumerate(durations):
            image.seek(i)
            image.load()
            pixel_states.append(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
            if image.info['duration'] != expected:
                raise ValueError('GIF timing differs')
    with Image.open(target) as gif:
        anchors = []
        for index in (0, 17, 60):
            gif.seek(index)
            anchors.append(hashlib.sha256(gif.convert('RGB').tobytes()).hexdigest())
        if len(set(anchors)) != 3:
            raise ValueError('GIF anchor pixels do not reflect changing geometry')
    if target.stat().st_size >= 9_000_000:
        raise ValueError('GIF exceeds 9 MB')
    report['gif'] = {'path': target.name, 'bytes': target.stat().st_size,
                     'sha256': sha(target), 'durations_ms': durations, 'frames': 61,
                     'unique_decoded_pixel_states': len(set(pixel_states)),
                     'anchor_pixel_states': {str(i): pixel_states[i] for i in (0, 17, 60)}}
    (output / 'media-audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['render', 'encode'])
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    render(args.folder, args.output, args.review, args.smoke) if args.action == 'render' else encode(args.output)
