"""Render a collected C2 accepted-state history with ParaView's pvpython."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_vtp(path: Path, state: dict) -> dict:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    visible = np.asarray(state['damage']) < 1.0
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(state['x'][visible], dtype=np.float64), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(points)
    point_count = int(np.count_nonzero(visible))
    vertex_cells = np.empty((point_count, 2), dtype=np.int64)
    vertex_cells[:, 0] = 1
    vertex_cells[:, 1] = np.arange(point_count, dtype=np.int64)
    verts = vtkCellArray()
    verts.SetCells(point_count, numpy_to_vtkIdTypeArray(vertex_cells.ravel(), deep=True))
    poly.SetVerts(verts)
    for name, array in (('damage', state['damage']), ('body_id', state['body_id']),
                        ('particle_mass', state['particle_mass'])):
        vtk_array = numpy_to_vtk(np.ascontiguousarray(array[visible]), deep=True)
        vtk_array.SetName(name)
        poly.GetPointData().AddArray(vtk_array)
    poly.GetPointData().SetActiveScalars('damage')
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOn()
    if writer.Write() != 1:
        raise RuntimeError(f'ParaView could not write {path}')
    return dict(physical_particle_count=int(len(visible)),
                hidden_fully_failed_count=int(np.count_nonzero(~visible)),
                displayed_particle_count=int(np.count_nonzero(visible)),
                physical_mass_kg=float(np.asarray(state['particle_mass']).sum()),
                displayed_mass_kg=float(np.asarray(state['particle_mass'])[visible].sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collected_history', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--allow-prefix', action='store_true',
                        help='render an explicitly marked incomplete actual-state prefix')
    parser.add_argument('--size', type=int, default=400)
    args = parser.parse_args()
    manifest_path = args.collected_history / 'history-manifest.json'
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('complete') and not args.allow_prefix:
        parser.error('history is a prefix; pass --allow-prefix to render and label it')
    if args.size < 100:
        parser.error('--size must be at least 100')

    from paraview import simple as p
    from PIL import Image

    args.output.mkdir(parents=True)
    data_dir = args.output / 'data'
    data_dir.mkdir()
    pvd = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(pvd, 'Collection')
    times = []
    visibility_rows = []
    for index, record in enumerate(manifest['frames']):
        src = args.collected_history / record['file']
        if not src.resolve().is_relative_to(args.collected_history.resolve()):
            raise ValueError(f'{src} escapes the collected history directory')
        if sha256(src) != record['source_sha256']:
            raise ValueError(f'{src} differs from the collector hash')
        with np.load(src, allow_pickle=False) as saved:
            state = {name: np.asarray(saved[name]) for name in
                     ('x', 'v', 'damage', 'body_id', 'particle_mass')}
            time_s = float(saved['time_s'])
            step = int(saved['step'])
        if time_s != float(record['time_s']) or step != int(record['step']):
            raise ValueError(f'{src} does not match the collected manifest')
        vtp = data_dir / f'frame-{index:06d}.vtp'
        visibility = write_vtp(vtp, state)
        if visibility['physical_particle_count'] != int(record['particle_count']):
            raise ValueError(f'{src} particle count differs from the collected manifest')
        if not np.isclose(visibility['physical_mass_kg'], float(record['physical_mass_kg']),
                          rtol=1e-13, atol=0):
            raise ValueError(f'{src} particle mass differs from the collected manifest')
        visibility_rows.append(dict(step=step, time_s=time_s, **visibility))
        ET.SubElement(collection, 'DataSet', timestep=format(time_s, '.17g'),
                      group='', part='0', file=f'data/{vtp.name}')
        times.append(time_s)
    pvd_path = args.output / 'history.pvd'
    ET.ElementTree(pvd).write(pvd_path, encoding='utf-8', xml_declaration=True)

    p._DisableFirstRenderCameraReset()
    reader = p.PVDReader(FileName=str(pvd_path))
    view = p.CreateView('RenderView')
    view.ViewSize = [args.size, args.size]
    view.Background = [1., 1., 1.]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [0.014, 0.012, 0.020]
    view.CameraFocalPoint = [0., 0., 0.]
    view.CameraViewUp = [0., 1., 0.]
    view.CameraParallelScale = .015
    display = p.Show(reader, view)
    display.Representation = 'Points'
    display.PointSize = 4.0
    display.RenderPointsAsSpheres = 1
    p.ColorBy(display, ('POINTS', 'damage'))
    lut = p.GetColorTransferFunction('damage')
    lut.ColorSpace = 'RGB'
    lut.RGBPoints = [0., .50, .57, .62, .25, .23, .48, .62,
                     .65, .92, .66, .28, 1., .77, .19, .13]
    lut.RescaleTransferFunction(0., 1.)
    lut.AutomaticRescaleRangeMode = 'Never'
    display.LookupTable = lut
    display.SetScalarBarVisibility(view, False)
    images = []
    image_hashes = []
    for index, time_s in enumerate(times):
        view.ViewTime = time_s
        reader.UpdatePipeline(time_s)
        p.Render(view)
        screenshot = args.output / f'frame-{index:06d}.png'
        p.SaveScreenshot(str(screenshot), view, ImageResolution=[args.size, args.size])
        image_hashes.append(sha256(screenshot))
        images.append(Image.open(screenshot).convert('RGB'))
    distinct_images = len(set(image_hashes))
    if len(times) > 1 and distinct_images < 2:
        raise RuntimeError('ParaView produced identical pixels for every distinct physical frame time')
    span = max(times[-1] - times[0], 1e-30)
    durations = [max(10, round((b - a) / span * 10000)) for a, b in zip(times, times[1:])]
    durations.append(durations[-1] if durations else 100)
    gif_name = 'metal-fragment-cloud.gif' if manifest.get('complete') else 'metal-fragment-cloud-prefix.gif'
    gif = args.output / gif_name
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, disposal=2, optimize=False)
    receipt = dict(schema='domain-impact-paraview-render-v1', engine='ParaView',
                   size=[args.size, args.size], source_history_manifest_sha256=sha256(manifest_path),
                   complete=bool(manifest.get('complete')), actual_frame_count=len(times),
                   first_time_s=times[0], last_time_s=times[-1],
                   interpolation=False, synthetic_frames=0,
                   visibility=('Rendering only: particles with damage >= 1 are hidden; '
                               'complete physical particles, mass, and source frames remain in history NPZs.'),
                   frame_visibility=visibility_rows,
                   frame_png_sha256=image_hashes, distinct_rendered_frames=distinct_images,
                   camera_parallel_scale_m=.015, gif_sha256=sha256(gif), duration_ms=durations)
    (args.output / 'render-audit.json').write_text(json.dumps(receipt, indent=2) + '\n',
                                                  encoding='utf-8')
    print(json.dumps(dict(gif=str(gif), frames=len(times), complete=receipt['complete'],
                          first_time_s=times[0], last_time_s=times[-1])))


if __name__ == '__main__':
    main()
