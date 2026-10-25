"""Private fixed-camera render of actual accepted forming particle snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
from paraview.simple import TrivialProducer,Threshold,Show,GetActiveViewOrCreate,Render,SaveScreenshot,Delete,_DisableFirstRenderCameraReset
from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray


def polydata(path):
    with np.load(path,allow_pickle=False) as data:x=data['x'];ids=data['fibre_id']
    assert x.shape==(227520,3) and np.isfinite(x).all() and np.array_equal(np.unique(ids),[0,1,2])
    poly=vtkPolyData();points=vtkPoints();points.SetData(numpy_to_vtk(x,deep=True));poly.SetPoints(points)
    verts=vtkCellArray();verts.SetData(numpy_to_vtkIdTypeArray(np.arange(len(x)+1,dtype=np.int64),deep=True),numpy_to_vtkIdTypeArray(np.arange(len(x),dtype=np.int64),deep=True));poly.SetVerts(verts)
    array=numpy_to_vtk(ids.astype(np.int32),deep=True);array.SetName('family_id');poly.GetPointData().AddArray(array)
    return poly


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    manifest=json.loads((a.input/'manifest.json').read_text());a.output.mkdir(exist_ok=True);frames=a.output/'frames';frames.mkdir(exist_ok=True)
    _DisableFirstRenderCameraReset();view=GetActiveViewOrCreate('RenderView');view.ViewSize=[400,400];view.Background=[1,1,1];view.UseColorPaletteForBackground=0;view.OrientationAxesVisibility=0
    if hasattr(view,'UseFXAA'):view.UseFXAA=1
    view.CameraParallelProjection=1;view.CameraPosition=[-.020,.016,.012];view.CameraFocalPoint=[.012,0.,0.];view.CameraViewUp=[0.,0.,1.];view.CameraParallelScale=.0105
    colors=([.045,.36,.67],[.94,.36,.08],[.32,.66,.48])
    image_hashes=[]
    import hashlib
    for i,item in enumerate(manifest['frames']):
        producer=TrivialProducer();producer.GetClientSideObject().SetOutput(polydata(a.input/item['prepared_file']))
        displays=[];filters=[]
        for family,color in enumerate(colors):
            selection=Threshold(Input=producer);selection.Scalars=['POINTS','family_id'];selection.LowerThreshold=family;selection.UpperThreshold=family;selection.UpdatePipeline()
            display=Show(selection,view);display.Representation='Point Gaussian';display.GaussianRadius=.000062;display.ShaderPreset='Sphere';display.Emissive=0;display.Ambient=.3;display.Diffuse=.7;display.Specular=.35;display.SpecularPower=35;display.Opacity=1.;display.ColorArrayName=['POINTS',''];display.DiffuseColor=color;display.AmbientColor=color
            displays.append(display);filters.append(selection)
        Render(view);SaveScreenshot(str(frames/f'frame_{i:05d}.png'),view,ImageResolution=[400,400])
        image_hashes.append(hashlib.sha256((frames/f'frame_{i:05d}.png').read_bytes()).hexdigest())
        for selection in filters:Delete(selection)
        Delete(producer)
        print(json.dumps(dict(frame=i,actual_step=item['step'],actual_time_s=item['time_s'])),flush=True)
    assert len(set(image_hashes))>1 and image_hashes[0]!=image_hashes[-1], 'render input failed to update'
    record=dict(distinct_image_count=len(set(image_hashes)),image_sha256=image_hashes,frames_rendered=len(manifest['frames']),particles=227520,resolution=[400,400],geometry_scale=1,axes=False,annotations=False,colors_rgb=colors,gaussian_radius_m=.000062,camera_position=list(view.CameraPosition),camera_focal_point=list(view.CameraFocalPoint),camera_view_up=list(view.CameraViewUp),camera_parallel_scale=float(view.CameraParallelScale),actual_steps=[f['step'] for f in manifest['frames']],actual_times_s=[f['time_s'] for f in manifest['frames']],physical_steps_executed=0,full_turn_completed=bool(manifest.get("full_turn_completed",False)),published=False,scope='Private actual partial formation, no interpolated positions, no full-turn/energy/time-convergence claim.')
    (a.output/'render.json').write_text(json.dumps(record,indent=2))

if __name__=='__main__':main()
