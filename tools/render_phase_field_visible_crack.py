"""Run with ParaView pvpython after the guarded producer completed."""
import json
import hashlib
import argparse
from pathlib import Path
import sys
import numpy as np
from paraview.simple import (XMLPolyDataReader, XMLUnstructuredGridReader, _DisableFirstRenderCameraReset, CreateView, Show, ColorBy,
    GetColorTransferFunction, SaveScreenshot, Render, Contour, Text)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkUnstructuredGrid, VTK_HEXAHEDRON
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray

from paraview import servermanager
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter, vtkXMLUnstructuredGridWriter
from phase_field_bspline_sampling import evaluate, sampling_grid
parser=argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('--smoke',action='store_true')
args=parser.parse_args()
out = Path(args.output)
checks = {} if args.smoke else json.loads((out / 'complete-checks.json').read_text())
if not args.smoke and (checks['completed'] is not True or checks['frames'] < 81):
    raise RuntimeError('Full completion certificate required')
if args.smoke:
    raise RuntimeError('Crack renderer has no qualification bypass')
crack = json.loads((out/'crack-checks.json').read_text())
if crack['passed'] is not True:
    raise RuntimeError('Actual .99 connected crack admission required')
frames = sorted(out.glob('frame_*.npz'))
if not args.smoke and len(frames) != checks['frames']:
    raise RuntimeError('Exact certified real snapshot series required')
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
label = Text()
label.Text = 'Black: native d=0.99, reference coordinates\nParticles: actual deformed positions'
label_display = Show(label, view)
label_display.Color = [.1, .1, .1]
label_display.FontSize = 9
label_display.WindowLocation = 'Upper Left Corner'
source = None
summary = json.loads((out/'summary.json').read_text())
sample_points, sample_cells, sample_spacing = sampling_grid(summary['dx'])
fixed_coordinates = None
def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
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
    with np.load(path) as data:
        coordinates = data['nodal_coordinates']
        coefficients = data['nodal_damage']
        if fixed_coordinates is None:
            fixed_coordinates = coordinates.copy()
        if not np.array_equal(coordinates, fixed_coordinates):
            raise RuntimeError('Reference nodes changed across frames')
        checked = evaluate(data['reference'], coordinates, coefficients, summary['dx'])
        if not np.allclose(checked, data['damage'], rtol=0, atol=1e-12):
            raise RuntimeError('Stored particle field differs from native B-spline evaluation')
        sampled = evaluate(sample_points, coordinates, coefficients, summary['dx'])
        if not np.isfinite(sampled).all() or sampled.min() < -1e-12 or sampled.max() > 1+1e-12:
            raise RuntimeError('Invalid native sampled damage')
        provenance = dict(npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            coefficient_sha256=digest(coefficients), coordinates_sha256=digest(coordinates),
            sampled_field_sha256=digest(sampled), particle_max_error=float(np.max(np.abs(checked-data['damage']))))
    grid = vtkUnstructuredGrid()
    grid_points = vtkPoints()
    grid_points.SetData(numpy_to_vtk(sample_points, deep=True))
    grid.SetPoints(grid_points)
    cells = vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(np.arange(len(sample_cells)+1, dtype=np.int64)*8, deep=True),
        numpy_to_vtkIdTypeArray(sample_cells.ravel().astype(np.int64), deep=True))
    grid.SetCells(VTK_HEXAHEDRON, cells)
    sampled_vtk = numpy_to_vtk(sampled, deep=True)
    sampled_vtk.SetName('native_damage')
    grid.GetPointData().AddArray(sampled_vtk)
    grid_path = vtp/f'{index:03d}-native.vtu'
    grid_writer = vtkXMLUnstructuredGridWriter()
    grid_writer.SetFileName(str(grid_path.resolve()))
    grid_writer.SetInputData(grid)
    if grid_writer.Write() != 1:
        raise RuntimeError('Native sampled field write failed')
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
        native_source = XMLUnstructuredGridReader(FileName=[str(grid_path.resolve())])
        contour = Contour(Input=native_source)
        contour.ContourBy=['POINTS', 'native_damage']
        contour.Isosurfaces=[.99]
        crack_display=Show(contour, view)
        crack_display.ColorArrayName=['POINTS','']
        crack_display.DiffuseColor=[.02,.02,.02]
    else:
        source.FileName=[str(vtp_path.resolve())]
        native_source.FileName=[str(grid_path.resolve())]
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
    native_source.UpdatePipeline()
    fetched_native = servermanager.Fetch(native_source)
    if not np.array_equal(vtk_to_numpy(fetched_native.GetPointData().GetArray('native_damage')), sampled):
        raise RuntimeError('ParaView native sampled field differs')
    if not np.array_equal(vtk_to_numpy(fetched_native.GetPoints().GetData()), sample_points):
        raise RuntimeError('ParaView reference sampling coordinates differ')
    contour.UpdatePipeline()
    iso_points = servermanager.Fetch(contour).GetNumberOfPoints()
    if sampled.max() < .99 and iso_points != 0:
        raise RuntimeError('Unexpected isosurface below native threshold')
    crack_display.Visibility = int(iso_points > 0)
    if index == len(frames)-1 and iso_points == 0:
        raise RuntimeError('Final genuine .99 isosurface is empty; no media admission')
    Render(view)
    image=images/f'{index:03d}.png'
    SaveScreenshot(str(image), view, ImageResolution=[400, 400])
    records.append(dict(npz=path.name,png=image.name,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),fetched_exact=True, isosurface_points=iso_points, native_field_fetched_exact=True, provenance=provenance))
if len(set(r['sha256'] for r in records)) < 2:
    raise RuntimeError('Static PNG sequence rejected')
(out / 'render.json').write_text(json.dumps(dict(renderer='ParaView', frames=len(frames),smoke=args.smoke,records=records,
    resolution=[400, 400], field='damage', scalar_range=[0, 1], annotations=True,
    geometry='saved deformed particles at true scale; black .99 native quadratic B-spline field sampled in REFERENCE coordinates; no particle deletion',
    field_evaluation=dict(method='stored exact nodal coefficients, quadratic tensor-product B-splines', sampling_spacing=sample_spacing, reference_points_sha256=digest(sample_points), cells_sha256=digest(sample_cells),
        limitation='Contour is piecewise linear on fixed dx/4 samples; physical notch-straddling cells excluded independently of damage; reference contour is not a deformed crack surface')), indent=2))
