"""Run with ParaView pvpython after the guarded producer completed."""
import json
import hashlib
import argparse
from pathlib import Path
import sys
import numpy as np
from paraview.simple import (XMLPolyDataReader, _DisableFirstRenderCameraReset, CreateView, Show, ColorBy,
    GetColorTransferFunction, SaveScreenshot, Render, Delaunay3D, Contour)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from paraview import servermanager
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
parser=argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
out = Path(args.output)
checks = {} if args.smoke else json.loads((out / 'complete-checks.json').read_text())
if not args.smoke and (checks['completed'] is not True or checks['frames'] != 81):
    raise RuntimeError('Full completion certificate required')
if args.smoke:
    raise RuntimeError('Crack renderer has no qualification bypass')
crack = json.loads((out/'crack-checks.json').read_text())
if crack['passed'] is not True:
    raise RuntimeError('Actual .99 connected crack admission required')
frames = sorted(out.glob('frame_*.npz'))
if (len(frames) != 81 and not args.smoke) or (args.smoke and len(frames) < 2):
    raise RuntimeError('81 real snapshots required')
images = out / 'png'
images.mkdir(exist_ok=True)
_DisableFirstRenderCameraReset()
view = CreateView('RenderView')
view.ViewSize = [400, 400]
view.Background = [1., 1., 1.]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
view.CameraPosition = [1.65, .6, 3.8]
view.CameraFocalPoint = [.3, .5, .05]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = .62
source = None
records=[]
vtp=out/'vtp'
vtp.mkdir(exist_ok=True)
for index, path in enumerate(frames):
    with np.load(path) as data:
        points = vtkPoints()
        points.SetData(numpy_to_vtk(data['position'].copy(), deep=True))
        poly = vtkPolyData()
        poly.SetPoints(points)
        vertices = vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(i)
        poly.SetVerts(vertices)
        field = numpy_to_vtk(data['damage'].copy(), deep=True)
        field.SetName('damage')
        poly.GetPointData().AddArray(field)
    vtp_path=vtp/f'{index:03d}.vtp'
    writer=vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path.resolve()))
    writer.SetInputData(poly)
    writer.SetDataModeToAppended()
    if writer.Write() != 1:
        raise RuntimeError('VTP write failed')
    if source is None:
        source=XMLPolyDataReader(FileName=[str(vtp_path.resolve())])
        display=Show(source,view)
        display.Representation='Point Gaussian'
        display.GaussianRadius=.00625
        display.Opacity=.18
        tetra = Delaunay3D(Input=source)
        tetra.Alpha=.025
        tetra.AlphaTets=1
        tetra.AlphaTris=0
        tetra.AlphaLines=0
        tetra.AlphaVerts=0
        contour = Contour(Input=tetra)
        contour.ContourBy=['POINTS', 'damage']
        contour.Isosurfaces=[.99]
        crack_display=Show(contour, view)
        crack_display.ColorArrayName=['POINTS','']
        crack_display.DiffuseColor=[.02,.02,.02]
    else:
        source.FileName=[str(vtp_path.resolve())]
    source.UpdateVTKObjects()
    source.UpdatePipeline()
    fetched=servermanager.Fetch(source)
    with np.load(path) as raw:
        if not np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()),raw['position']):
            raise RuntimeError('ParaView fetched coordinates differ from raw NPZ')
        if not np.array_equal(vtk_to_numpy(fetched.GetPointData().GetArray('damage')),raw['damage']):
            raise RuntimeError('ParaView fetched damage differs from raw NPZ')
    ColorBy(display, ('POINTS', 'damage'))
    lut = GetColorTransferFunction('damage')
    lut.RGBPoints = [0., .12, .27, .65, .5, .98, .85, .35, 1., .7, .04, .03]
    lut.RescaleTransferFunction(0., 1.)
    display.SetScalarBarVisibility(view, False)
    contour.UpdatePipeline()
    iso_points = servermanager.Fetch(contour).GetNumberOfPoints()
    if index == len(frames)-1 and iso_points == 0:
        raise RuntimeError('Final genuine .99 isosurface is empty; no media admission')
    Render(view)
    image=images/f'{index:03d}.png'
    SaveScreenshot(str(image), view, ImageResolution=[400, 400])
    records.append(dict(npz=path.name,png=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),fetched_exact=True, isosurface_points=iso_points))
if len(set(r['sha256'] for r in records)) < 2:
    raise RuntimeError('Static PNG sequence rejected')
(out / 'render.json').write_text(json.dumps(dict(renderer='ParaView', frames=len(frames),smoke=args.smoke,records=records,
    resolution=[400, 400], field='damage', scalar_range=[0, 1], annotations=False,
    geometry='saved deformed particles at true scale; black .99 damage isosurface of alpha .025 tetrahedral interpolation; translucent full particles; no deletion'), indent=2))
