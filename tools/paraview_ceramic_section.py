"""Real ParaView rendering of a Lagrangian central material slab, without JAX."""
import json
import sys
from pathlib import Path
import numpy as np
from paraview.simple import (TrivialProducer, CreateView, Show, ColorBy,
    GetColorTransferFunction, SaveScreenshot, SaveState, Render,
    GetParaViewVersion, _DisableFirstRenderCameraReset)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def main(folder):
    folder = Path(folder)
    frames = sorted((folder/'frames_data').glob('frame_*.npz'))
    if [p.name for p in frames] != [f'frame_{i:05d}.npz' for i in range(61)]:
        raise ValueError('Exactly 61 original states required')
    with np.load(frames[0], allow_pickle=False) as first:
        reference = first['x'].copy()
    # Two central particle layers at original 1.25 mm particle spacing.
    mask = np.abs(reference[:, 0] - .125) <= .00125 + 1e-12
    ids = np.flatnonzero(mask)
    if len(ids) == 0:
        raise ValueError('Empty material cross section')
    bounds = np.array([np.inf, -np.inf, np.inf, -np.inf])
    counts = []
    for path in frames:
        with np.load(path, allow_pickle=False) as f:
            x = f['x'][ids]
            bounds = [min(bounds[0], x[:, 2].min()), max(bounds[1], x[:, 2].max()),
                      min(bounds[2], x[:, 1].min()), max(bounds[3], x[:, 1].max())]
            failed = f['failed'][ids].astype(bool)
            counts.append(dict(active_particles=int((~failed).sum()), debris_particles=int(failed.sum())))
    _DisableFirstRenderCameraReset()
    view = CreateView('RenderView')
    view.ViewSize = [1200, 1200]
    view.UseColorPaletteForBackground = 0
    view.Background = [1., 1., 1.]
    view.OrientationAxesVisibility = 0
    view.CenterAxesVisibility = 0
    view.CameraParallelProjection = 1
    cz, cy = (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2
    view.CameraPosition = [1.125, cy, cz]
    view.CameraFocalPoint = [.125, cy, cz]
    view.CameraViewUp = [0, 1, 0]
    view.CameraParallelScale = .55*max(bounds[1]-bounds[0], bounds[3]-bounds[2])
    producers = [TrivialProducer(), TrivialProducer()]
    # Establish the field schema before ParaView configures the lookup table.
    for producer in producers:
        initial = vtkPolyData()
        initial.SetPoints(vtkPoints())
        damage = numpy_to_vtk(np.empty(0, dtype=np.float32), deep=True)
        damage.SetName('damage')
        initial.GetPointData().AddArray(damage)
        producer.GetClientSideObject().SetOutput(initial)
        producer.UpdatePipeline()
    displays = [Show(p, view) for p in producers]
    for display in displays:
        display.Representation = 'Point Gaussian'
        display.GaussianRadius = .00065
    lut = GetColorTransferFunction('damage')
    lut.RGBPoints = [0, .12, .32, .64, .5, .95, .8, .15, 1, .75, .05, .05]
    ColorBy(displays[0], ('POINTS', 'damage'))
    displays[0].LookupTable = lut
    lut.RescaleTransferFunction(0, 1)
    displays[0].SetScalarBarVisibility(view, False)
    displays[1].ColorArrayName = ['POINTS', '']
    displays[1].DiffuseColor = [1., .32, .02]
    displays[1].AmbientColor = [1., .32, .02]
    out = folder/'frames'
    out.mkdir(exist_ok=True)
    for i, path in enumerate(frames):
        with np.load(path, allow_pickle=False) as f:
            failed = f['failed'][ids].astype(bool)
            for producer, selected in zip(producers, (ids[~failed], ids[failed])):
                poly = vtkPolyData()
                points = vtkPoints()
                points.SetData(numpy_to_vtk(np.ascontiguousarray(f['x'][selected], dtype=np.float32), deep=True))
                poly.SetPoints(points)
                cells = vtkCellArray()
                cells.SetCells(len(selected), numpy_to_vtkIdTypeArray(np.column_stack((np.ones(len(selected), dtype=np.int64), np.arange(len(selected), dtype=np.int64))).ravel(), deep=True))
                poly.SetVerts(cells)
                damage = numpy_to_vtk(np.ascontiguousarray(f['damage'][selected], dtype=np.float32), deep=True)
                damage.SetName('damage')
                poly.GetPointData().AddArray(damage)
                producer.GetClientSideObject().SetOutput(poly)
                producer.UpdatePipeline()
        Render(view)
        SaveScreenshot(str(out/f'frame_{i:05d}.png'), view, ImageResolution=[1200,1200])
    SaveState(str(folder/'section-scene.pvsm'))
    record = dict(renderer='ParaView', version=str(GetParaViewVersion()), frames_rendered=61,
        geometry_scale=1, axes=False, annotations=False, field='damage', color_range=[0,1],
        resolution=[1200,1200], view_scope='central material cross section',
        reference_plane_x_m=.125, reference_half_thickness_m=.00125,
        selection='fixed particle IDs selected from frame_00000 reference x',
        camera_direction='along +x to -x; z horizontal, y vertical',
        selected_particle_ids=ids.tolist(), camera_projected_bounds=list(map(float,bounds)),
        selected_particles=len(ids), displayed_counts=counts,
        failed_debris_color=[1.,.32,.02], solver_executed=False)
    (folder/'section-render.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main(sys.argv[1])
