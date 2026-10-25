"""Private ParaView view of actual persistent-grid damage: full d range, no cutoffs.

Usage: pvpython --force-offscreen-rendering this.py FOLDER [DX].
FOLDER/raw contains authenticated actual frame NPZs. Initial finite seed remains visible.
Rendering is evidence review and does not itself qualify mesh/time convergence.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from phase_field_bspline_sampling import evaluate
from paraview.simple import *
from paraview.simple import _DisableFirstRenderCameraReset
from paraview import servermanager
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkCellArray
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path.cwd();h=float(sys.argv[2]) if len(sys.argv)>2 else .0125;out=root/'media';out.mkdir();(out/'png').mkdir();(out/'vtk').mkdir()
_DisableFirstRenderCameraReset()
view=CreateView('RenderView');view.ViewSize=[400,400];view.UseColorPaletteForBackground=0;view.Background=[1,1,1];view.OrientationAxesVisibility=0;view.CenterAxesVisibility=0
view.CameraParallelProjection=1;view.CameraPosition=[.3,.3,2];view.CameraFocalPoint=[.3,.3,.0125];view.CameraViewUp=[0,1,0];view.CameraParallelScale=.327
axis=np.linspace(0,.6,round(.6/(h/4))+1);X,Y=np.meshgrid(axis,axis,indexing='ij');points=np.column_stack([X.ravel(),Y.ravel(),np.full(X.size,.025)])
ids=np.arange(len(points)).reshape(X.shape);cells=np.stack([ids[:-1,:-1].ravel(),ids[1:,:-1].ravel(),ids[1:,1:].ravel(),ids[:-1,1:].ravel()],axis=1)
reader=None;records=[]
for index,path in enumerate(sorted((root/'raw').glob('frame_*.npz'))):
 with np.load(path) as f:
  coords=f['grid_coordinates']
  checked=evaluate(f['reference'],coords,f['nodal_damage'],h);error=float(abs(checked-f['damage']).max());assert error<1e-12
  damage=evaluate(points,coords,f['nodal_damage'],h);assert damage.min()>-1e-12 and damage.max()<1+1e-12
  displacement=np.column_stack([evaluate(points,coords,f['nodal_displacement'][:,k],h) for k in range(3)]);actual=points+displacement
 poly=vtkPolyData();vp=vtkPoints();vp.SetData(numpy_to_vtk(actual,deep=True));poly.SetPoints(vp)
 vc=vtkCellArray();vc.SetData(numpy_to_vtkIdTypeArray(np.arange(len(cells)+1,dtype=np.int64)*4,deep=True),numpy_to_vtkIdTypeArray(cells.ravel().astype(np.int64),deep=True));poly.SetPolys(vc)
 array=numpy_to_vtk(damage,deep=True);array.SetName('native_damage');poly.GetPointData().AddArray(array)
 vtp=out/'vtk'/f'{index:03}.vtp';writer=vtkXMLPolyDataWriter();writer.SetFileName(str(vtp));writer.SetInputData(poly);assert writer.Write()==1
 if reader is None:
  reader=XMLPolyDataReader(FileName=[str(vtp)]);display=Show(reader,view);display.Representation='Surface';ColorBy(display,('POINTS','native_damage'));display.SetScalarBarVisibility(view,False);display.Ambient=1;display.Diffuse=0;display.Opacity=1
  lut=GetColorTransferFunction('native_damage');lut.RGBPoints=[0,.32,.67,.86,1,0,0,0];lut.ColorSpace='RGB';lut.RescaleTransferFunction(0,1);lut.AutomaticRescaleRangeMode='Never'
 else:reader.FileName=[str(vtp)]
 reader.UpdatePipeline();fetched=servermanager.Fetch(reader);assert np.array_equal(vtk_to_numpy(fetched.GetPoints().GetData()),actual);assert np.array_equal(vtk_to_numpy(fetched.GetPointData().GetArray('native_damage')),damage)
 png=out/'png'/f'{index:03}.png';Render(view);SaveScreenshot(str(png),view,ImageResolution=[400,400])
 records.append(dict(raw=path.name,raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),png=png.name,png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),native_particle_error=error,range=[float(damage.min()),float(damage.max())],geometry_scale=1,fetched_exact=True))
 print(path.name,flush=True)
SaveState(str(out/'scene.pvsm'))
(out/'render.json').write_text(json.dumps(dict(paraview_version=str(GetParaViewVersion()),scope='private dynamic prefix review, no convergence claim',frames=len(records),damage_map=[0,1],opacity=1,threshold=False,geometry_scale=1,view='actual deformed front material surface',records=records),indent=2))
