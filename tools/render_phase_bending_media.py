"""Run with ParaView pvpython after the guarded producer completed."""
import json
import hashlib
import argparse
from pathlib import Path
import sys
import numpy as np
from paraview.simple import (XMLPolyDataReader, _DisableFirstRenderCameraReset, CreateView, Show, ColorBy,
    GetColorTransferFunction, SaveScreenshot, Render, Box)
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
checks = {} if args.smoke else json.loads((out / 'qualification.json').read_text())
if not args.smoke and (checks['numerical_qualification_passed'] is not True or checks['mode'] != 'medium' or checks['actual_saved_states'] != 13):
    raise RuntimeError('Full completion certificate required')
frames = sorted(out.glob('frame_*.npz'))
if (len(frames) != 13 and not args.smoke) or (args.smoke and len(frames) < 2):
    raise RuntimeError('13 real saved snapshots required')
images = out / 'png'
images.mkdir(exist_ok=True)
_DisableFirstRenderCameraReset()
view = CreateView('RenderView')
view.ViewSize = [400, 400]
view.Background = [1., 1., 1.]
view.OrientationAxesVisibility = 0
view.CameraParallelProjection = 1
view.CameraPosition = [.8, .2, 3.8]
view.CameraFocalPoint = [.8, .2, .08]
view.CameraViewUp = [0, 1, 0]
view.CameraParallelScale = .9
import csv
rows={int(float(r['step'])):r for r in csv.DictReader((out/'history.csv').open())}
shoes=[]
for center in ([.2,-.01,.08],[1.4,-.01,.08],[.8,.41,.08]):
    box=Box();box.XLength=.12;box.YLength=.02;box.ZLength=.16;box.Center=center
    display_box=Show(box,view);display_box.ColorArrayName=['POINTS',''];display_box.DiffuseColor=[.4,.4,.4]
    shoes.append(box)
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
        display.GaussianRadius=.004
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
    step=int(path.stem.split('_')[1]);stroke=0. if step==0 else float(rows[step]['punch_displacement'])
    shoes[2].Center=[.8,.41-stroke,.08];shoes[2].UpdateVTKObjects();shoes[2].UpdatePipeline()
    Render(view)
    image=images/f'{index:03d}.png'
    SaveScreenshot(str(image), view, ImageResolution=[400, 400])
    records.append(dict(npz=path.name,png=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),fetched_exact=True))
if len(set(r['sha256'] for r in records)) < 2:
    raise RuntimeError('Static PNG sequence rejected')
(out / 'render.json').write_text(json.dumps(dict(renderer='ParaView', frames=len(frames),smoke=args.smoke,records=records,
    resolution=[400, 400], field='damage', scalar_range=[0, 1], annotations=False,
    geometry='saved deformed particle positions; no displacement scaling', fixtures='Flat shoe upper faces y0, punch lower face y.4 minus actual CSV stroke; reference-grid footprints, not rollers'), indent=2))
