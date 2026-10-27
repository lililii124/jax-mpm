"""Render the actual particle-supported exterior surface for MAT173 review."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageSequence
from paraview import servermanager
from paraview.simple import (_DisableFirstRenderCameraReset,GetActiveViewOrCreate,
    TrivialProducer,Show,ColorBy,Glyph,GetColorTransferFunction,Render,SaveScreenshot,
    Hide,Delete)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray,vtkHexahedron,vtkUnstructuredGrid
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray,vtk_to_numpy


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def lattice_cells(reference):
 ref=np.asarray(reference,dtype=np.float64)
 axes=[np.unique(ref[:,d]) for d in range(3)]
 dims=tuple(map(len,axes))
 index=np.column_stack([np.searchsorted(axes[d],ref[:,d]) for d in range(3)])
 grid=np.full(dims,-1,dtype=np.int64)
 grid[index[:,0],index[:,1],index[:,2]]=np.arange(len(ref),dtype=np.int64)
 corners=np.stack((grid[:-1,:-1,:-1],grid[1:,:-1,:-1],grid[1:,1:,:-1],grid[:-1,1:,:-1],
                   grid[:-1,:-1,1:],grid[1:,:-1,1:],grid[1:,1:,1:],grid[:-1,1:,1:]),axis=-1)
 cells=corners[np.all(corners>=0,axis=-1)]
 spacing=[float(np.median(np.diff(a))) for a in axes]
 return dims,spacing,cells


def jacobian_review(x,cells):
 x=np.asarray(x,dtype=np.float64); nodes=x[cells]
 signs=np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float64)
 points=[(0.,0.,0.)]+[(float(a),float(b),float(c)) for a in (-1.,1.) for b in (-1.,1.) for c in (-1.,1.)]
 minimum=float('inf');maximum=-float('inf');bad=0;sample_count=0
 for xi,eta,zeta in points:
  dxi=signs[:,0]*(1+signs[:,1]*eta)*(1+signs[:,2]*zeta)/8
  deta=signs[:,1]*(1+signs[:,0]*xi)*(1+signs[:,2]*zeta)/8
  dzeta=signs[:,2]*(1+signs[:,0]*xi)*(1+signs[:,1]*eta)/8
  ax=np.einsum('nac,a->nc',nodes,dxi);ay=np.einsum('nac,a->nc',nodes,deta);az=np.einsum('nac,a->nc',nodes,dzeta)
  det=np.einsum('nc,nc->n',np.cross(ax,ay),az)
  minimum=min(minimum,float(det.min()));maximum=max(maximum,float(det.max()))
  bad+=int(np.count_nonzero(det<=0));sample_count+=len(det)
 return {'cells':int(len(cells)),'samples_per_cell':len(points),'min_signed_jacobian_m3':minimum,
         'max_signed_jacobian_m3':maximum,'nonpositive_samples':bad,'all_sampled_positive':bad==0,
         'sample_count':sample_count}


def make_surface(x,cells,scalar=None):
 x=np.asarray(x,dtype=np.float64)
 pts=vtkPoints();pts.SetData(numpy_to_vtk(x,deep=True))
 grid=vtkUnstructuredGrid();grid.SetPoints(pts)
 legacy=np.empty((len(cells),9),dtype=np.int64);legacy[:,0]=8;legacy[:,1:]=cells
 cell_array=vtkCellArray();cell_array.SetCells(len(cells),numpy_to_vtkIdTypeArray(legacy.reshape(-1),deep=True))
 grid.SetCells(vtkHexahedron().GetCellType(),cell_array)
 ids=numpy_to_vtkIdTypeArray(np.arange(len(x),dtype=np.int64),deep=True);ids.SetName('source_particle_id')
 grid.GetPointData().AddArray(ids)
 if scalar is not None:
  values=numpy_to_vtk(np.asarray(scalar,dtype=np.float64),deep=True);values.SetName('plastic_strain')
  grid.GetPointData().AddArray(values);grid.GetPointData().SetActiveScalars('plastic_strain')
 gf=vtkGeometryFilter();gf.SetInputData(grid);gf.PassThroughPointIdsOn();gf.Update()
 normals=vtkPolyDataNormals();normals.SetInputData(gf.GetOutput());normals.ConsistencyOn()
 normals.AutoOrientNormalsOn();normals.SplittingOff();normals.ComputePointNormalsOn();normals.Update()
 surface=vtkPolyData();surface.ShallowCopy(normals.GetOutput())
 src_ids=vtk_to_numpy(surface.GetPointData().GetArray('source_particle_id')).astype(np.int64,copy=False)
 coords=vtk_to_numpy(surface.GetPoints().GetData())
 assert np.array_equal(coords,x[src_ids])
 if scalar is not None:
  val=vtk_to_numpy(surface.GetPointData().GetArray('plastic_strain'))
  assert np.array_equal(val,np.asarray(scalar)[src_ids])
 return surface,src_ids


def set_output(producer,data):
 producer.GetClientSideObject().SetOutput(data);producer.UpdatePipeline()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 args=ap.parse_args();inp=args.input.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 paths=sorted((inp/'frames').glob('frame_*.npz'));manifest=json.loads((inp/'render-input.json').read_text())
 assert len(paths)==len(manifest['frames']) and len(paths)>=2
 first=np.load(paths[0],allow_pickle=False);soil_ref=first['soil_reference_x'].astype(np.float64,copy=True)
 base_ref=first['foundation_reference_x'].astype(np.float64,copy=True)
 sdims,sspacing,scells=lattice_cells(soil_ref);bdims,bspacing,bcells=lattice_cells(base_ref)
 assert sdims==(55,26,40) and bdims==(31,3,24),(sdims,bdims)
 soil0=first['soil_x'].astype(np.float64,copy=True);base0=first['foundation_x'].astype(np.float64,copy=True)
 assert len(scells)>20000 and len(bcells)>1000
 _DisableFirstRenderCameraReset();view=GetActiveViewOrCreate('RenderView')
 view.ViewSize=[1600,1600];view.UseFXAA=1;view.UseColorPaletteForBackground=0;view.Background=[.985,.985,.985]
 view.OrientationAxesVisibility=0
 if hasattr(view,'CenterAxesVisibility'):view.CenterAxesVisibility=0
 view.CameraParallelProjection=1;view.CameraParallelScale=1.85
 view.CameraFocalPoint=[1.62,1.,1.5];view.CameraPosition=[4.55,3.2,4.28];view.CameraViewUp=[0.,1.,0.]
 lut=GetColorTransferFunction('plastic_strain');lut.ColorSpace='RGB'
 lut.RGBPoints=[0.,.07,.22,.63,.01,.1,.66,.8,.025,.96,.77,.2,.05,.72,.07,.08]
 lut.RescaleTransferFunction(0.,.05);lut.AutomaticRescaleRangeMode='Never'
 rows=[];pngdir=out/'png';pngdir.mkdir()
 for pos,path in enumerate(paths):
  with np.load(path,allow_pickle=False) as f:
   x=f['soil_x'].astype(np.float64,copy=True);b=f['foundation_x'].astype(np.float64,copy=True)
   d=f['soil_plastic_strain'].astype(np.float64,copy=True);step=int(f['step']);tm=float(f['time_s'])
  assert x.shape==soil_ref.shape and b.shape==base_ref.shape and d.shape==(len(x),)
  assert np.isfinite(x).all() and np.isfinite(b).all() and np.isfinite(d).all()
  sj=jacobian_review(x,scells);bj=jacobian_review(b,bcells)
  assert sj['all_sampled_positive'] and bj['all_sampled_positive'],(pos,sj,bj)
  soil_surface,soil_ids=make_surface(x,scells,d);base_surface,base_ids=make_surface(b,bcells)
  soil_prod=TrivialProducer(registrationName=f'MAT173 connected soil surface {pos}');set_output(soil_prod,soil_surface)
  soil_display=Show(soil_prod,view);soil_display.Representation='Surface'
  soil_display.Specular=.08;soil_display.SpecularPower=22;soil_display.Ambient=.2;soil_display.Diffuse=.8
  ColorBy(soil_display,('POINTS','plastic_strain'));soil_display.LookupTable=lut;soil_display.SetScalarBarVisibility(view,False)
  lut.AutomaticRescaleRangeMode='Never';lut.RescaleTransferFunction(0.,.05)
  base_prod=TrivialProducer(registrationName=f'MAT173 connected foundation surface {pos}');set_output(base_prod,base_surface)
  base_display=Show(base_prod,view);base_display.Representation='Surface'
  base_display.ColorArrayName=['POINTS',''];base_display.DiffuseColor=[.38,.4,.42]
  base_display.Ambient=.25;base_display.Diffuse=.75;base_display.Specular=.04
  base_display.SetScalarBarVisibility(view,False)
  Render(view)
  fetched_soil=servermanager.Fetch(soil_prod);fx=vtk_to_numpy(fetched_soil.GetPoints().GetData())
  fid=vtk_to_numpy(fetched_soil.GetPointData().GetArray('source_particle_id')).astype(np.int64)
  fd=vtk_to_numpy(fetched_soil.GetPointData().GetArray('plastic_strain'))
  assert np.array_equal(fx,x[fid]) and np.array_equal(fd,d[fid])
  image=pngdir/f'frame_{pos:03d}_1600.png';SaveScreenshot(str(image),view,ImageResolution=[1600,1600])
  reduced=pngdir/f'frame_{pos:03d}.png'
  with Image.open(image) as im:im.convert('RGB').resize((400,400),Image.Resampling.LANCZOS).save(reduced)
  highsha=sha(image);image.unlink()
  rec={'position':pos,'source_file':path.name,'source_npz_sha256':sha(path),'step':step,'time_s':tm,
       'soil_particle_count':len(x),'soil_connected_hex_cells':sj['cells'],'soil_exterior_surface_points':len(soil_ids),
       'soil_exterior_quads':int(soil_surface.GetNumberOfPolys()),'foundation_particle_count':len(b),
       'foundation_connected_hex_cells':bj['cells'],'foundation_exterior_surface_points':len(base_ids),
       'foundation_exterior_quads':int(base_surface.GetNumberOfPolys()),
       'soil_jacobian_review':sj,'foundation_jacobian_review':bj,
       'source_particle_coordinate_ids_exact':True,'source_plastic_scalar_at_surface_nodes_exact':True,
       'fetched_server_data_coordinates_and_plastic_exact':True,'plastic_min':float(d.min()),
       'plastic_p50':float(np.quantile(d,.5)),'plastic_p95':float(np.quantile(d,.95)),
       'plastic_max':float(d.max()),'source_png_sha256':sha(reduced),'temporary_1600_sha256':highsha,
       'png':reduced.name}
  rows.append(rec)
  Hide(soil_prod,view);Hide(base_prod,view);Delete(soil_display);Delete(base_display);Delete(soil_prod);Delete(base_prod)
  if pos%10==0:print(json.dumps({'position':pos,'step':step,'time_s':tm,'Dmax':rec['plastic_max'],'soil_min_jacobian':sj['min_signed_jacobian_m3']}),flush=True)

 times=np.asarray([r['time_s'] for r in rows]);assert np.all(np.diff(times)>0)
 median=float(np.median(np.diff(times)));durations=[max(40,int(round(dt/median*90/10)*10)) for dt in np.r_[np.diff(times),median]]
 images=[Image.open(pngdir/r['png']).convert('RGB') for r in rows];gif=out/'simulation.gif'
 images[0].save(gif,save_all=True,append_images=images[1:],duration=durations,loop=0,optimize=False,disposal=2)
 for im in images:im.close()
 with Image.open(gif) as im:decoded=[x.convert('RGB').tobytes() for x in ImageSequence.Iterator(im)]
 assert len(decoded)==len(rows) and len(set(decoded))>1
 (out/'final.png').write_bytes((pngdir/rows[-1]['png']).read_bytes())
 checks={'source_frames':len(rows),'decoded_gif_frames':len(decoded),'distinct_decoded_frames':len(set(decoded)),
  'times_strictly_increasing':True,'time_start_s':float(times[0]),'time_end_s':float(times[-1]),
  'time_span_s':float(times[-1]-times[0]),'frame_durations_ms':durations,'gif_sha256':sha(gif)}
 receipt={'schema':'mat173-strength-reduction-slope-connected-surface-review-v1',
  'claim_scope':'strength-reduced slope response through controlled 5% small-strain stop; not a complete landslide claim or native MAT173 validation.',
  'input_manifest':manifest,'renderer_sha256':sha(Path(__file__).resolve()),
  'surface_method':'Build hexahedral cells only when all 8 corners are original neighboring material points from the initial regular x-y-z lattice. vtkGeometryFilter extracts the connected exterior faces; no new coordinates are added. Partial boundary cells are omitted, so the connected boundary is inset by up to one particle spacing.',
  'scalar_method':'Equivalent plastic strain is read from original MAT173 particle history and linearly interpolated over connected exterior cells. No scalar smoothing, clipping, or rescaling is applied.',
  'geometry_checks':'Each connected hexa is sampled at center and 8 corners for signed Jacobian. Every surface point coordinate and plastic value is verified against its original particle id, including servermanager.Fetch.',
  'soil_particle_spacing_m':sspacing,'soil_lattice_dimensions':sdims,'foundation_particle_spacing_m':bspacing,
  'foundation_lattice_dimensions':bdims,'scalar_range':[0.,.05],'scalar':'plastic_strain',
  'view':{'projection':'parallel','parallel_scale_m':float(view.CameraParallelScale),'position':list(view.CameraPosition),
   'focal_point':list(view.CameraFocalPoint),'up':list(view.CameraViewUp),'resolution_px':[400,400],
   'render_size_px':[1600,1600],'downsample':'PIL Lanczos','FXAA':True,'axes_text_scalarbar':False,
   'no_coordinate_scaling':True},'frame_records':rows,'gif_checks':checks,'gif_sha256':checks['gif_sha256'],
  'final_png_sha256':sha(out/'final.png')}
 (out/'render.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'complete':True,'frames':len(rows),'distinct':checks['distinct_decoded_frames'],'gif_sha256':checks['gif_sha256']}),flush=True)


if __name__=='__main__':main()
