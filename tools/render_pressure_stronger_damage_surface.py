import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageSequence
from paraview import servermanager
from paraview.simple import (GetActiveViewOrCreate,Show,Hide,Render,SaveScreenshot,ColorBy,Delete,TrivialProducer,_DisableFirstRenderCameraReset)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkStructuredGrid
from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def hexa_jacobian_audit(x,nx,ny,nz):
 p=np.asarray(x,dtype=np.float64).reshape(nx,ny,nz,3)
 n=np.stack([p[:-1,:-1,:-1],p[1:,:-1,:-1],p[:-1,1:,:-1],p[1:,1:,:-1],p[:-1,:-1,1:],p[1:,:-1,1:],p[:-1,1:,1:],p[1:,1:,1:]],axis=-2).reshape(-1,8,3)
 sg=np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]],dtype=np.float64)
 evals=[(0.,0.,0.)]+[(float(i),float(j),float(k)) for i in [-1,1] for j in [-1,1] for k in [-1,1]]
 g=1./np.sqrt(3.);evals += [(i*g,j*g,k*g) for i in [-1,1] for j in [-1,1] for k in [-1,1]]
 minimum=float('inf');maximum=-float('inf');nonpositive=0;count=0
 for xi,eta,zeta in evals:
  dx=sg[:,0]*(1+sg[:,1]*eta)*(1+sg[:,2]*zeta)/8;dy=sg[:,1]*(1+sg[:,0]*xi)*(1+sg[:,2]*zeta)/8;dz=sg[:,2]*(1+sg[:,0]*xi)*(1+sg[:,1]*eta)/8
  ax=np.einsum('nac,a->nc',n,dx);ay=np.einsum('nac,a->nc',n,dy);az=np.einsum('nac,a->nc',n,dz)
  det=np.einsum('nc,nc->n',np.cross(ax,ay),az);minimum=min(minimum,float(det.min()));maximum=max(maximum,float(det.max()));nonpositive+=int(np.count_nonzero(det<=0));count+=len(det)
 del n
 return {'sampling':'cell center, 8 corners, and 8 2x2x2 Gauss points','cells':(nx-1)*(ny-1)*(nz-1),'min_signed_jacobian_m3':minimum,'max_signed_jacobian_m3':maximum,'nonpositive_jacobian_samples':nonpositive,'sample_count':count,'all_sampled_jacobians_positive':nonpositive==0}

def make_grid(x,d,nx,ny,nz):
 l=np.asarray(x).reshape(nx,ny,nz,3);xyz=l.transpose(2,1,0,3).reshape(-1,3);damage=np.asarray(d).reshape(nx,ny,nz).transpose(2,1,0).reshape(-1)
 pts=vtkPoints();pts.SetData(numpy_to_vtk(xyz,deep=True));g=vtkStructuredGrid();g.SetDimensions(nx,ny,nz);g.SetPoints(pts)
 a=numpy_to_vtk(damage,deep=True);a.SetName('damage');g.GetPointData().AddArray(a);g.GetPointData().SetActiveScalars('damage')
 assert np.array_equal(vtk_to_numpy(g.GetPoints().GetData()),xyz)
 assert np.array_equal(vtk_to_numpy(g.GetPointData().GetArray('damage')),damage)
 return g,xyz,damage

def build_gif(rows,pngdir,path):
 times=np.array([r['time_s'] for r in rows],dtype=float);d=np.diff(times);median=float(np.median(d));base_ms=90
 durations=[max(10,int(round((float(x)/median*base_ms)/10)*10)) for x in np.r_[d,median]]
 ims=[Image.open(pngdir/r['png']).convert('RGB') for r in rows]
 ims[0].save(path,save_all=True,append_images=ims[1:],duration=durations,loop=0,optimize=False,disposal=2)
 with Image.open(path) as check:decoded=[f.convert('RGB').tobytes() for f in ImageSequence.Iterator(check)]
 return {'source_png_frames':len(rows),'decoded_gif_frames':len(decoded),'distinct_decoded_frames':len(set(decoded)),'frame_durations_ms':durations,'simulation_span_s':float(times[-1]-times[0]),'gif_sha256':sha(path)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();inputs=json.loads(Path(args.inputs).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 nx,ny,nz=192,13,192;_DisableFirstRenderCameraReset();view=GetActiveViewOrCreate('RenderView');view.ViewSize=[1600,1600];view.UseFXAA=1;view.UseColorPaletteForBackground=0;view.Background=[1.,1.,1.];view.OrientationAxesVisibility=0
 if hasattr(view,'CenterAxesVisibility'):view.CenterAxesVisibility=0
 view.CameraParallelProjection=1;view.CameraViewUp=[0.,0.,1.];view.CameraParallelScale=.14
 final=[]
 for model,item in inputs.items():
  idir=Path(item['path']);case=json.loads((idir/'case.json').read_text());frames=int(case['frames']);assert int(case['particle_count'])==nx*ny*nz
  cpman=json.loads((idir/'checkpoint.json').read_text());checkpoint=idir/cpman['states'][0];assert checkpoint.is_file()
  fd=idir/'frames_data';paths=[fd/f'frame_{i:05d}.npz' for i in range(frames+1)];assert all(p.is_file() for p in paths)
  with np.load(paths[0],allow_pickle=False) as ar:ref=ar['x'].astype(np.float64,copy=True)
  ux,uy,uz=(np.unique(ref[:,j]) for j in range(3));assert (len(ux),len(uy),len(uz))==(nx,ny,nz)
  lattice=ref.reshape(nx,ny,nz,3);expected=np.stack(np.meshgrid(ux,uy,uz,indexing='ij'),axis=-1);lattice_error=float(np.max(np.abs(lattice-expected)));assert lattice_error==0.0
  back_y=float(uy[0]);back_mask=(ref[:,1]==back_y);assert int(np.count_nonzero(back_mask))==nx*nz
  spacing=[float(np.median(np.diff(u))) for u in (ux,uy,uz)];sel=np.unique(np.rint(np.linspace(0,frames,61)).astype(np.int32));assert len(sel)==61 and sel[0]==0 and sel[-1]==frames
  modelout=out/model;pngdir=modelout/'png';checks=modelout/'checks';pngdir.mkdir(parents=True);checks.mkdir();rows=[]
  for pos,index in enumerate(sel):
   path=paths[int(index)]
   with np.load(path,allow_pickle=False) as ar:x=ar['x'].astype(np.float64,copy=True);d=ar['damage'].astype(np.float64,copy=True);t=float(np.asarray(ar['time']).reshape(-1)[0])
   assert x.shape==ref.shape and d.shape==(len(x),) and np.isfinite(x).all() and np.isfinite(d).all() and float(d.min())>=-1e-8 and float(d.max())<=1+1e-8
   back_d=d[back_mask];assert back_d.shape==(nx*nz,) and np.isfinite(back_d).all() and float(back_d.min())>=-1e-8 and float(back_d.max())<=1+1e-8
   jac=hexa_jacobian_audit(x,nx,ny,nz);assert jac['all_sampled_jacobians_positive'],(model,int(index),jac)
   grid,ordered_x,ordered_d=make_grid(x,d,nx,ny,nz);gf=vtkGeometryFilter();gf.SetInputData(grid);gf.PassThroughPointIdsOn();gf.Update();surface=gf.GetOutput();expected_quads=2*((ny-1)*(nz-1)+(nx-1)*(nz-1)+(nx-1)*(ny-1));expected_points=nx*ny*nz-(nx-2)*(ny-2)*(nz-2)
   assert surface.GetNumberOfPolys()==expected_quads and surface.GetNumberOfPoints()==expected_points
   ids=vtk_to_numpy(surface.GetPointData().GetArray('vtkOriginalPointIds'));sx=vtk_to_numpy(surface.GetPoints().GetData());sd=vtk_to_numpy(surface.GetPointData().GetArray('damage'));assert np.array_equal(sx,ordered_x[ids]) and np.array_equal(sd,ordered_d[ids])
   producer=TrivialProducer(registrationName=f'{model} material surface frame {index}');producer.GetClientSideObject().SetOutput(surface);producer.UpdatePipeline();fetched=servermanager.Fetch(producer);fx=vtk_to_numpy(fetched.GetPoints().GetData());fids=vtk_to_numpy(fetched.GetPointData().GetArray('vtkOriginalPointIds'));fdmg=vtk_to_numpy(fetched.GetPointData().GetArray('damage'));assert np.array_equal(fx,sx) and np.array_equal(fids,ids) and np.array_equal(fdmg,sd)
   display=Show(producer,view);display.Representation='Surface';display.Specular=.08;display.SpecularPower=20;display.Ambient=.22;display.Diffuse=.78;ColorBy(display,('POINTS','damage'));display.SetScalarBarVisibility(view,False)
   interp=hasattr(display,'InterpolateScalarsBeforeMapping');
   if interp:display.InterpolateScalarsBeforeMapping=1
   from paraview.simple import GetColorTransferFunction
   lut=GetColorTransferFunction('damage');lut.ColorSpace='RGB';lut.RGBPoints=[0.,.08,.24,.68,.25,.12,.68,.82,.6,.98,.82,.22,1.,.78,.08,.10];lut.RescaleTransferFunction(0.,1.);lut.AutomaticRescaleRangeMode='Never';display.LookupTable=lut
   centre=[float((a+b)/2) for a,b in zip(case['geometry']['lower_m'],case['geometry']['upper_m'])];view.CameraFocalPoint=centre;view.CameraPosition=[centre[0]+.025,centre[1]-.50,centre[2]+.018];Render(view)
   high=pngdir/f'frame_{index:05d}_1600.png';SaveScreenshot(str(high),view,ImageResolution=[1600,1600]);low=pngdir/f'frame_{index:05d}.png'
   with Image.open(high) as im:im.convert('RGB').resize((400,400),Image.Resampling.LANCZOS).save(low)
   high_sha=sha(high);high.unlink()
   row={'index':int(index),'file':path.name,'source_frame_sha256':sha(path),'time_s':t,'damage_min':float(d.min()),'damage_max':float(d.max()),'damage_q50':float(np.quantile(d,.5)),'damage_q90':float(np.quantile(d,.9)),'backside_reference_y_m':back_y,'backside_particle_count':int(len(back_d)),'backside_damage_max':float(back_d.max()),'backside_damage_q99':float(np.quantile(back_d,.99)),'particle_count':len(d),'surface_points':int(surface.GetNumberOfPoints()),'surface_quads':int(surface.GetNumberOfPolys()),'jacobian_audit':jac,'all_structured_points_exact':True,'all_structured_damage_exact':True,'surface_point_ids_exact':True,'surface_coordinate_and_damage_values_exact':True,'damage_interpolate_before_lut':interp,'source_png_sha256':sha(low),'temporary_1600_png_sha256':high_sha,'png':low.name}
   rows.append(row);Hide(producer,view);Delete(display);Delete(producer)
   if pos%10==0:print(json.dumps({'model':model,'rendered_position':pos,'source_frame':int(index),'Dmax':row['damage_max'],'minJ':jac['min_signed_jacobian_m3']}),flush=True)
  (pngdir/'final.png').write_bytes((pngdir/rows[-1]['png']).read_bytes());gif=modelout/'simulation.gif';gifchecks=build_gif(rows,pngdir,gif);(checks/'gif-checks.json').write_text(json.dumps(gifchecks,indent=2)+'\n')
  record={'schema':'pressure-damage-structured-surface-render-v1','model':model,'source_task':item['task'],'source_case_sha256':sha(idir/'case.json'),'source_checkpoint_sha256':sha(checkpoint),'source_checkpoint_frame':frames,'source_frame_count':frames+1,'selected_frame_indices':sel.tolist(),'rendered_frame_count':len(rows),'particles_per_frame':nx*ny*nz,'grid_dimensions':[nx,ny,nz],'archive_order':'C-order (x,y,z), z fastest; transposed (z,y,x) before VTK structured-grid input, including damage array.','initial_lattice_max_abs_coordinate_error_m':lattice_error,'reference_particle_spacing_m':spacing,'surface_method':'vtkGeometryFilter extracts exterior quadrilateral faces from a vtkStructuredGrid whose points and scalar damage are the original per-frame arrays.','surface_boundary':'Connects material-point centres; nominal plate outline is inset by half a particle spacing. Damage is interpolated linearly over connected structured cells; no geometry scaling, damage smoothing, or scalar remapping. The camera is on the minus-Y backside opposite the loaded plus-Y face.','jacobian_check':'Each cell sampled at center, 8 corners and 8 two-by-two Gauss locations; all sampled signed Jacobians must be positive. This is a finite sample inversion check, not an exhaustive continuous proof.','scalar':'damage','damage_range':[0.,1.],'rendered_view':'minus-Y backside, looking toward plus Y','backside_reference_y_m':back_y,'backside_reference_particle_count':int(np.count_nonzero(back_mask)),'selected_frame_backside_damage_max':float(max(r['backside_damage_max'] for r in rows)),'selected_frame_backside_damage_max_is_not_interior_max':True,'all_particle_damage_mapped':True,'clamp_area_not_greyed':True,'damage_interpolate_before_lut_available':rows[-1]['damage_interpolate_before_lut'],'camera':{'position':list(view.CameraPosition),'focal_point':list(view.CameraFocalPoint),'view_up':list(view.CameraViewUp),'parallel_projection':True,'parallel_scale':float(view.CameraParallelScale),'output_size_px':[400,400],'rendered_size_px':[1600,1600],'downsample':'PIL Lanczos to400px','fxaa':True,'axes_and_text':False},'background':'white','frame_records':rows,'gif_checks':gifchecks,'gif_sha256':sha(gif),'final_png_sha256':sha(pngdir/'final.png')}
  (modelout/'render.json').write_text(json.dumps(record,indent=2)+'\n');final.append({'model':model,'gif_sha256':record['gif_sha256'],'frames':len(rows),'distinct_gif_frames':gifchecks['distinct_decoded_frames']});print(json.dumps({'model_complete':model,**final[-1]}),flush=True)
 (out/'render-complete.json').write_text(json.dumps({'models':final},indent=2)+'\n');print(json.dumps({'complete':True,'models':final}),flush=True)
if __name__=='__main__':main()
