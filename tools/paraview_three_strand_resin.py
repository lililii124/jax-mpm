"""Private actual resin particle render around authenticated fixed formed fibres."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from paraview.simple import TrivialProducer,Threshold,Show,Hide,Delete,GetActiveViewOrCreate,Render,SaveScreenshot,_DisableFirstRenderCameraReset
from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray


def polydata(x,ids=None):
    assert x.ndim==2 and x.shape[1]==3 and np.isfinite(x).all()
    poly=vtkPolyData();points=vtkPoints();points.SetData(numpy_to_vtk(x,deep=True));poly.SetPoints(points);verts=vtkCellArray();verts.SetData(numpy_to_vtkIdTypeArray(np.arange(len(x)+1,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(np.arange(len(x),dtype=np.int64),deep=True));poly.SetVerts(verts)
    if ids is not None:
        array=numpy_to_vtk(ids.astype(np.int32),deep=True);array.SetName('family_id');poly.GetPointData().AddArray(array)
    return poly


def appearance(display,radius,color,opacity=1.):
    display.Representation='Point Gaussian';display.GaussianRadius=radius;display.ShaderPreset='Sphere';display.Emissive=0;display.Ambient=.3;display.Diffuse=.7;display.Specular=.35;display.SpecularPower=35;display.Opacity=opacity;display.ColorArrayName=['POINTS',''];display.DiffuseColor=color;display.AmbientColor=color


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();m=json.loads((a.input/'manifest.json').read_text());frames=a.output/'frames';frames.mkdir(parents=True,exist_ok=True)
    _DisableFirstRenderCameraReset();view=GetActiveViewOrCreate('RenderView');view.ViewSize=[400,400];view.Background=[1,1,1];view.UseColorPaletteForBackground=0;view.OrientationAxesVisibility=0;view.CameraParallelProjection=1;view.CameraPosition=[-.020,.016,.012];view.CameraFocalPoint=[.012,0.,0.];view.CameraViewUp=[0.,0.,1.];view.CameraParallelScale=.0105
    if hasattr(view,'UseFXAA'):view.UseFXAA=1
    with np.load(a.input/'fixed-fibres.npz',allow_pickle=False) as d:x=d['x'];ids=d['fibre_id']
    assert x.shape==(227520,3) and np.array_equal(np.unique(ids),[0,1,2])
    fibre=TrivialProducer();fibre.GetClientSideObject().SetOutput(polydata(x,ids));colors=([.045,.36,.67],[.94,.36,.08],[.32,.66,.48]);filters=[]
    for family,color in enumerate(colors):
        selection=Threshold(Input=fibre);selection.Scalars=['POINTS','family_id'];selection.LowerThreshold=family;selection.UpperThreshold=family;selection.UpdatePipeline();display=Show(selection,view);appearance(display,.000062,color);filters.append(selection)
    hashes=[];counts=[]
    for i,item in enumerate(m['frames']):
        with np.load(a.input/item['file'],allow_pickle=False) as d:x=d['x']
        counts.append(len(x));producer=None;display=None
        if len(x):
            producer=TrivialProducer();producer.GetClientSideObject().SetOutput(polydata(x));display=Show(producer,view);appearance(display,.000124,[.72,.79,.84],.32)
        Render(view);path=frames/f'frame_{i:05d}.png';SaveScreenshot(str(path),view,ImageResolution=[400,400]);hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        if producer is not None:Hide(producer,view);Delete(display);Delete(producer)
        print(json.dumps(dict(frame=i,step=item['step'],time_s=item['time_s'],actual_resin_particles=len(x))),flush=True)
    assert len(set(hashes))==len(hashes) and hashes[0]!=hashes[-1]
    record=dict(actual_frames=len(hashes),actual_particle_counts=counts,distinct_images=len(set(hashes)),image_sha256=hashes,resolution=[400,400],camera_position=list(view.CameraPosition),camera_focal_point=list(view.CameraFocalPoint),camera_view_up=list(view.CameraViewUp),camera_parallel_scale=float(view.CameraParallelScale),fiber_gaussian_radius_m=.000062,resin_gaussian_radius_m=.000124,resin_opacity=.32,predefined_resin_cylinder=False,mold_surface_rendered=False,actual_flow_only=True,full_filling_certified=False,private_only=True,published=False,scope='Actual finite inlet-flow prefix around fixed actual formed fibres; no cylinder fade, no particle-position interpolation, no cure/fullFSI')
    (a.output/'render.json').write_text(json.dumps(record,indent=2))

if __name__=='__main__':main()
