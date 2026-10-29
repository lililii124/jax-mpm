"""Render example NPZ output with pvpython (NumPy and Pillow required).

pvpython tools/render_mat173_toe.py --input outputs/mat173-toe --output outputs/mat173-toe-render
The fixed camera, scalar ranges, LUT, 800-to-400 reduction and GIF settings
match the published toe-excavation animations. No solver is invoked.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageSequence
from paraview.simple import _DisableFirstRenderCameraReset,GetActiveViewOrCreate,TrivialProducer,Show,ColorBy,GetColorTransferFunction,Render,SaveScreenshot,Delete
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkCellArray
from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def polydata(points,cells,arrays=None):
 points=np.asarray(points,dtype=np.float64); points3=np.column_stack((points,np.zeros(len(points),dtype=np.float64)));vtkp=vtkPoints();vtkp.SetData(numpy_to_vtk(points3,deep=True));data=vtkPolyData();data.SetPoints(vtkp);polys=vtkCellArray()
 for q in ([] if cells is None else cells):
  polys.InsertNextCell(4)
  for i in q:polys.InsertCellPoint(int(i))
 data.SetPolys(polys)
 if cells is None:
  verts=vtkCellArray()
  for i in range(len(points)):verts.InsertNextCell(1);verts.InsertCellPoint(i)
  data.SetVerts(verts)
 if arrays:
  for name,val in arrays.items():
   a=np.asarray(val,dtype=np.float64);v=numpy_to_vtk(a,deep=True);v.SetName(name);data.GetPointData().AddArray(v)
  data.GetPointData().SetActiveScalars(next(iter(arrays)))
 for name,val in (arrays or {}).items():
  assert np.array_equal(vtk_to_numpy(data.GetPointData().GetArray(name)),val)
 assert np.array_equal(vtk_to_numpy(data.GetPoints().GetData()),points3)
 return data

def affine_domain(x,F,volume,failed):
    active=~failed.astype(bool)
    side=np.sqrt(volume);corners=x[:,None,:]+np.einsum('pij,pkj->pki',F[:,:2,:2],np.array([[-1,-1],[1,-1],[1,1],[-1,1]])[None,:,:]*side[:,None,None]*.5)
    edge=np.roll(corners,-1,axis=1)-corners;cross=edge[:,:,0]*np.roll(edge,-1,axis=1)[:,:,1]-edge[:,:,1]*np.roll(edge,-1,axis=1)[:,:,0]
    area=.5*np.sum(corners[:,:,0]*np.roll(corners,-1,axis=1)[:,:,1]-corners[:,:,1]*np.roll(corners,-1,axis=1)[:,:,0],axis=1)
    assert np.isfinite(corners).all() and (area>0).all() and (cross>0).all();np.testing.assert_allclose(area,volume*np.linalg.det(F[:,:2,:2]),rtol=1e-8,atol=1e-18)
    points=corners[active].reshape(-1,2);return points,np.arange(len(points)).reshape(-1,4),active

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);args=ap.parse_args();inp=args.input;out=args.output;out.mkdir(parents=True,exist_ok=False)
 files=sorted(inp.glob('frame_*.npz'))
 if not files:raise ValueError('No frame_*.npz files in input directory')
 seed=np.load(inp/'post_cut_initial_checkpoint.npz',allow_pickle=False)
 terminal=np.load(inp/'checkpoint.npz',allow_pickle=False)
 for body in ('soil','base'):
  np.testing.assert_array_equal(seed[body+'_volume'],terminal[body+'_volume'])
 _DisableFirstRenderCameraReset();view=GetActiveViewOrCreate('RenderView');view.ViewSize=[800,800];view.UseFXAA=1;view.UseColorPaletteForBackground=0;view.Background=[.985,.985,.985];view.OrientationAxesVisibility=0
 if hasattr(view,'CenterAxesVisibility'):view.CenterAxesVisibility=0
 view.CameraParallelProjection=1;view.CameraParallelScale=1.75;view.CameraFocalPoint=[1.747,1.0,0.0];view.CameraPosition=[1.747,1.0,20.0];view.CameraViewUp=[0.0,1.0,0.0]
 lut_damage=GetColorTransferFunction('soil_damage');lut_damage.ColorSpace='RGB';lut_damage.RGBPoints=[0.,.06,.20,.62,.25,.12,.52,.78,.5,.45,.75,.58,.75,.91,.62,.32,1.,.72,.12,.18];lut_damage.RescaleTransferFunction(0.,1.);lut_damage.AutomaticRescaleRangeMode='Never'
 increment_max=0.0060235601776985614;increment_max=max(increment_max,1e-12)
 lut_inc=GetColorTransferFunction('soil_plastic_strain_increment');lut_inc.ColorSpace='RGB';lut_inc.RGBPoints=[0.,.06,.20,.62,.25*increment_max,.12,.52,.78,.5*increment_max,.45,.75,.58,.75*increment_max,.91,.62,.32,increment_max,.72,.12,.18];lut_inc.RescaleTransferFunction(0.,increment_max);lut_inc.AutomaticRescaleRangeMode='Never'
 displacement_max=0.013226548820548363;displacement_max=max(displacement_max,1e-12)
 lut_disp=GetColorTransferFunction('soil_displacement_m');lut_disp.ColorSpace='RGB';lut_disp.RGBPoints=[0.,.06,.20,.62,.25*displacement_max,.12,.52,.78,.5*displacement_max,.45,.75,.58,.75*displacement_max,.91,.62,.32,displacement_max,.72,.12,.18];lut_disp.RescaleTransferFunction(0.,displacement_max);lut_disp.AutomaticRescaleRangeMode='Never'
 pngdir=out/'png';pngdir.mkdir();records=[]
 for pos,path in enumerate(files):
  with np.load(path,allow_pickle=False) as saved:f={k:saved[k] for k in saved.files}
  f['soil_displacement_m']=np.linalg.norm(f['soil_x']-seed['soil_x'],axis=1)
  for body in ('soil','base'):
   if body+'_reference_volume' not in f:f[body+'_reference_volume']=terminal[body+'_volume']
   if body+'_failed' not in f:
    if not np.array_equal(seed[body+'_history_failed'],terminal[body+'_history_failed']):
     raise ValueError('Failure flags changed: per-frame failure records are required')
    f[body+'_failed']=terminal[body+'_history_failed']
  sx=np.asarray(f['soil_x']);bx=np.asarray(f['base_x']);damage=np.asarray(f['soil_damage']);inc=np.asarray(f['soil_plastic_strain_increment'])
  assert np.array_equal(sx,f['soil_reference']+f['soil_displacement']);assert np.array_equal(bx,f['base_reference']+f['base_displacement'])
  soil_points,soil_quads,soil_active=affine_domain(sx,f['soil_F'],f['soil_reference_volume'],f['soil_failed']);base_points,base_quads,base_active=affine_domain(bx,f['base_F'],f['base_reference_volume'],f['base_failed']);soil_source=TrivialProducer(registrationName=f'soil {pos}');soil_source.GetClientSideObject().SetOutput(polydata(soil_points,soil_quads,{'soil_damage':np.repeat(damage[soil_active],4),'soil_plastic_strain_increment':np.repeat(inc[soil_active],4),'soil_displacement_m':np.repeat(np.asarray(f['soil_displacement_m'])[soil_active],4)}));soil_source.UpdatePipeline()
  soil_display=Show(soil_source,view);soil_display.Representation='Surface';soil_display.PointSize=3;soil_display.Ambient=.35;soil_display.Diffuse=.65;soil_display.Specular=0.;soil_display.SetScalarBarVisibility(view,False)
  base_source=TrivialProducer(registrationName=f'base {pos}');base_source.GetClientSideObject().SetOutput(polydata(base_points,base_quads));base_source.UpdatePipeline();base_display=Show(base_source,view);base_display.Representation='Surface';base_display.PointSize=3;base_display.DiffuseColor=[.43,.45,.47];base_display.Ambient=.35;base_display.Diffuse=.65;base_display.Specular=0.;base_display.ColorArrayName=['POINTS',''];base_display.SetScalarBarVisibility(view,False)
  step=int(f['step']);tm=float(f['time_s']);fields={}
  for name,lut,lo,hi in [('soil_damage',lut_damage,0.,1.),('soil_plastic_strain_increment',lut_inc,0.,increment_max),('soil_displacement_m',lut_disp,0.,displacement_max)]:
   ColorBy(soil_display,('POINTS',name));soil_display.LookupTable=lut;lut.AutomaticRescaleRangeMode='Never';lut.RescaleTransferFunction(lo,hi);Render(view)
   image=pngdir/f'{name}_{pos:03d}_800.png';SaveScreenshot(str(image),view,ImageResolution=[800,800]);reduced=pngdir/f'{name}_{pos:03d}_400.png'
   with Image.open(image) as im:im.convert('RGB').resize((400,400),Image.Resampling.LANCZOS).save(reduced)
   image.unlink();fields[name]={'png':reduced.name,'png_sha256':sha(reduced),'min':float(np.min(f[name])),'max':float(np.max(f[name])),'range':[lo,hi]}
  records.append({'position':pos,'source_file':path.name,'step':step,'time_s':tm,'source_npz_sha256':sha(path),'soil_geometry':'independent_affine_domains','base_geometry':'independent_affine_domains','fields':fields,'particle_coordinates_used_without_transform':True})
  for obj in [soil_display,base_display,soil_source,base_source]:Delete(obj)
  if pos%6==0:print(json.dumps({'frame':pos,'step':step,'time_s':tm,'damage_max':fields['soil_damage']['max'],'increment_max':fields['soil_plastic_strain_increment']['max']}),flush=True)
 outputs={}
 for name in ['soil_damage','soil_plastic_strain_increment','soil_displacement_m']:
  imgs=[Image.open(pngdir/f'{name}_{i:03d}_400.png').convert('RGB') for i in range(len(files))];gif=out/f'{name}.gif';imgs[0].save(gif,save_all=True,append_images=imgs[1:],duration=100,loop=0,optimize=False,disposal=2)
  for im in imgs:im.close()
  decoded=[]
  with Image.open(gif) as im:decoded=[f.convert('RGB').tobytes() for f in ImageSequence.Iterator(im)]
  assert len(decoded)>0
  outputs[name]={'gif':gif.name,'sha256':sha(gif),'bytes':gif.stat().st_size,'decoded_frames':len(decoded),'distinct_frames':len(set(decoded)),'pixels':[400,400]}
  (out/f'{name}_final.png').write_bytes((pngdir/f'{name}_{len(files)-1:03d}_400.png').read_bytes())
 receipt={'schema':'mat173-strict2d-render-v1','source_task':'mat173-separated-mechanism-toe-v1','scope':'Strict 2D plane-strain toe excavation.','view':{'canvas':[400,400],'projection':'parallel','parallel_scale':1.75,'focal_point':[1.747,1.0,0.0],'axes_title_text_scalarbar':False,'displacement_scaling':1.0,'geometry_smoothing':False},'soil_geometry':'independentownreferencerectanglesfromreferencevolume,transformedbyactualFxy;noadjacentIDmeshing,gapfill,shrink,orexaggerateddeformation;failedTrueparticleshidden','scalars':{'soil_damage':{'range':[0.,1.],'lookup_table_rgb_points':list(lut_damage.RGBPoints)},'soil_plastic_strain_increment':{'range':[0.,increment_max],'lookup_table_rgb_points':list(lut_inc.RGBPoints)}},'frames':records,'outputs':outputs,'derived_displacement_scalar':'Euclidean norm of actual current minus post-excavation initial checkpoint  coordinates; geometry uses current coordinates at scale 1','renderer_sha256':sha(Path(__file__).resolve())}
 (out/'render-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'complete':True,'outputs':outputs}),flush=True)
if __name__=='__main__':main()
