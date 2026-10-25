"""Review-only ParaView scene: exact authenticated f34..60 particles and sphere."""
import hashlib,json
from pathlib import Path
import numpy as np
from paraview.simple import *
from PIL import Image,GifImagePlugin
root=Path('/parent/output/plain/deep-sphere-impact');old=Path('/saved/media');out=Path('/work/media');out.mkdir()
digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
expected='ef4dcb98bb77108ec8404f210361c2c52a22dd9271cf8a2aad8579a7e103791e'
assert digest(root/'checkpoint.npz')==expected
with np.load(root/'checkpoint.npz',allow_pickle=False) as cp:
 assert int(cp['frame'])==60 and bool(cp['state_valid']);tables=json.loads(str(cp['tables']));radius=float(cp['sphere_radius']);mass=float(cp['sphere_mass']);last_center=cp['sphere_center'].copy()
plan=json.loads((root/'plan.json').read_text());assert radius==plan['tool']['diameter_m']/2 and mass==plan['tool']['mass_kg']
poses=[dict(frame=i,time_s=tables['times'][i],center=[tables['rows'][i]['ball_center_'+a+'_m'] for a in 'xyz'],velocity=[tables['rows'][i]['ball_velocity_'+a+'_m_s'] for a in 'xyz']) for i in range(61)]
assert np.array_equal(last_center,poses[-1]['center'])
source=json.loads((old/'saved-frame-review.json').read_text());assert source['checkpoint_sha256']==expected
reader=PVDReader(FileName='/work/saved/particles.pvd');reader.UpdatePipeline();assert np.array_equal(np.asarray(reader.TimestepValues),[p['time_s'] for p in poses])
view=CreateView('RenderView');view.ViewSize=[400,400];view.OrientationAxesVisibility=0;view.CenterAxesVisibility=0;view.AxesGrid.Visibility=0;view.UseColorPaletteForBackground=0;view.Background=[.99,.99,.99];view.CameraParallelProjection=1;view.CameraViewUp=[0,1,0]
parts=[]
for field,value,color,opacity in [('phase',0,[.70,.76,.81],.025),('family_id',0,[.025,.29,.61],1),('family_id',1,[.91,.29,.035],1)]:
 part=Threshold(Input=reader);part.Scalars=['POINTS',field];part.LowerThreshold=value;part.UpperThreshold=value
 display=Show(part,view);display.Representation='Point Gaussian';display.GaussianRadius=.00020;display.ShaderPreset='Sphere';display.Emissive=0;display.Ambient=.4;display.Diffuse=.6;display.Specular=.15;display.Opacity=opacity;display.ColorArrayName=['POINTS',''];display.DiffuseColor=color;display.AmbientColor=color;parts.append(part)
sphere=Sphere();sphere.Radius=radius;sphere.ThetaResolution=192;sphere.PhiResolution=128
display=Show(sphere,view);display.ColorArrayName=['POINTS',''];display.DiffuseColor=[.35,.43,.51];display.Opacity=.20;display.Ambient=.35;display.Diffuse=.65;display.Specular=.4
receipt=dict(source_checkpoint_sha256=expected,source_run=40113303961,radius_m=radius,mass_kg=mass,poses=poses,source_particle_frames=json.loads(Path('/work/saved/full-source-review.json').read_text())['source_frames'],renderer='ParaView6.2',geometry_scale=1,axes=False,annotations=False,particle_radius_m=.00020,particle_spacing_m=.00025,sphere_opacity=.20,particle_selection='all matrix and both yarn families; no damage filtering',history_scope='Exact recovered saved frames0..33 plus authenticated resumed frames34..60; no invented/interpolated state',views={})
for name,focal,position,scale in [('overview',[0,.024,0],[.21,.115,.27],.112),('detail',[0,.008,0],[.11,.055,.14],.047)]:
 folder=out/name;(folder/'frames').mkdir(parents=True);view.CameraFocalPoint=focal;view.CameraPosition=position;view.CameraParallelScale=scale;records=[];images=[]
 for j,pose in enumerate(poses):
  reader.UpdatePipeline(pose['time_s']);view.ViewTime=pose['time_s'];sphere.Center=pose['center'];sphere.UpdatePipeline();assert np.array_equal(sphere.Center,pose['center']);Render(view)
  png=folder/'frames'/f'frame_{j:05d}.png';SaveScreenshot(str(png),view,ImageResolution=[400,400])
  with Image.open(png) as im:
   rgb=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest();images.append(im.convert('RGB').quantize(colors=256))
  records.append(dict(source_frame=pose['frame'],time_s=pose['time_s'],sphere_center=list(sphere.Center),png_sha256=digest(png),png_rgb_sha256=rgb))
  print(name,j+1,61,flush=True)
 with (folder/'simulation.gif').open('wb') as stream:
  header,_=GifImagePlugin.getheader(images[0],info={'loop':0})
  for b in header:stream.write(b)
  for j,im in enumerate(images):
   for b in GifImagePlugin.getdata(im,duration=80,disposal=2,include_color_table=j>0):stream.write(b)
  stream.write(b';')
 with Image.open(folder/'simulation.gif') as gif:
  assert gif.n_frames==61 and gif.info['loop']==0
  for j in range(61):
   gif.seek(j);assert gif.info['duration']==80 and gif.convert('RGB').tobytes()==images[j].convert('RGB').tobytes();records[j]['decoded_gif_rgb_sha256']=hashlib.sha256(gif.convert('RGB').tobytes()).hexdigest()
 with Image.open(folder/'frames/frame_00060.png') as im:im.save(folder/'last.png')
 SaveState(str(folder/'scene.pvsm'))
 receipt['views'][name]=dict(frames=records,camera_position=position,camera_focal_point=focal,camera_parallel_scale=scale,scope='Complete scene' if name=='overview' else 'Explicit central detail crop; true sphere and plate extend outside view',gif_sha256=digest(folder/'simulation.gif'),gif_bytes=(folder/'simulation.gif').stat().st_size)
assert digest(root/'checkpoint.npz')==expected
(out/'render-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
