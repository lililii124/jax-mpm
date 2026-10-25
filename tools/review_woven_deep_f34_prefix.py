"""Cloud-only copy admission and actual 35-state particle-only ParaView review.

The incomplete prefix is never represented as a completed numerical result.
The integrated tool is omitted because its canonical reader requires completion.
"""
import json
import hashlib
import os
import subprocess
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def pixel_sha(image):
 return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()

def encode_frames(output):
 from PIL import Image,GifImagePlugin
 images=[];png_hashes=[]
 for index in range(35):
  with Image.open(output/'frames'/f'frame_{index:05d}.png') as source:
   assert source.size==(400,400)
   png_hashes.append(pixel_sha(source))
   images.append(source.convert('RGB').quantize(colors=256))
 # Write one image descriptor per actual PNG; save_all coalesces equal frames.
 with (output/'simulation.gif').open('xb') as stream:
  header,_=GifImagePlugin.getheader(images[0],info={'loop':0})
  for block in header:stream.write(block)
  for index,frame in enumerate(images):
   for block in GifImagePlugin.getdata(frame,duration=60,disposal=2,include_color_table=index>0):stream.write(block)
  stream.write(b';')
 decoded=[]
 with Image.open(output/'simulation.gif') as gif:
  assert gif.n_frames==35
  for index in range(35):
   gif.seek(index);decoded.append(pixel_sha(gif))
   assert gif.info.get('duration')==60
   assert gif.convert('RGB').tobytes()==images[index].convert('RGB').tobytes(), 'Decoded image differs from its actual source PNG'
 record=dict(frames=35,png_pixel_sha256=png_hashes,decoded_pixel_sha256=decoded,
  distinct_png_frames=len(set(png_hashes)),distinct_decoded_frames=len(set(decoded)),
  identical_adjacent_png_pairs=[[i-1,i] for i in range(1,35) if png_hashes[i-1]==png_hashes[i]],
  scope='Exactly one encoded frame per saved-state PNG. Equal rendered pixels, if present, reflect the actual rendering; no pixels or states are fabricated.')
 (output/'gif-check.json').write_text(json.dumps(record,indent=2)+'\n')
 assert len(set(decoded))>=2, 'Static full sequence refused; genuine stationary early PNGs remain encoded'
 return record

def pv_check(raw,output):
 import numpy as np
 from paraview.simple import XMLPolyDataReader,UpdatePipeline,Delete
 from paraview import servermanager
 from vtkmodules.util.numpy_support import vtk_to_numpy
 positions=np.load(raw/'trajectory_x.npy',mmap_mode='r',allow_pickle=False)
 frames=[]
 for index in range(35):
  reader=XMLPolyDataReader(FileName=[str(output/'vtk'/f'particles_{index:05d}.vtp')])
  UpdatePipeline(proxy=reader)
  fetched=servermanager.Fetch(reader)
  actual=vtk_to_numpy(fetched.GetPoints().GetData())
  expected=np.asarray(positions[index],dtype=np.float32)
  assert fetched.GetNumberOfPoints()==819200
  assert np.array_equal(actual,expected),index
  frames.append(dict(frame=index,points=len(actual),coordinate_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
  Delete(reader)
 (output/'paraview-readback.json').write_text(json.dumps(dict(passed=True,actual_paraview_fetch=True,frames=frames),indent=2)+'\n')

def main():
 for name in ('tools/review_woven_deep_f34_prefix.py','mpm_jax/visualization.py','mpm_jax/export.py','mpm_jax/paraview_scene.py'):
  assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),name
 import numpy as np
 from tools import run_woven_deep_f34_resume as recovery
 from tools import rht_impact_prefix as transport
 from mpm_jax.export import export_series
 from mpm_jax.visualization import render_directory
 recovery.main(certify_only=True)
 raw=recovery.DEST/'output/plain/deep-sphere-impact'
 output=ROOT/'.local/woven-deep-f34-prefix-review'
 assert not output.exists()
 output.mkdir(parents=True)
 before={p.relative_to(recovery.DEST).as_posix():transport.digest(p) for p in recovery.DEST.rglob('*') if p.is_file()}
 with np.load(raw/'checkpoint.npz',allow_pickle=False) as cp:
  tables=json.loads(str(cp['tables']))
  assert int(cp['frame'])==34 and len(tables['times'])==35
 arrays={name:np.load(raw/f'trajectory_{name}.npy',mmap_mode='r',allow_pickle=False)[:35] for name in ('x','speed','damage','temperature')}
 with np.load(raw/'particle_fields.npz',allow_pickle=False) as labels:
  fields={name:labels[name] for name in labels.files}
 fields.update({name:arrays[name] for name in ('speed','damage','temperature')})
 from examples import woven_large_plate as base
 _,plan=base.load_saved_case(raw)
 spacing=json.loads((raw/'summary.json').read_text())['definition']['particle_spacing_m']
 export_series(arrays['x'],fields,np.asarray(tables['times']),output,spacing=spacing,dim=3)
 transport.write(output/'summary.json',dict(completed=False,production_eligible=False,
  definition=dict(particle_spacing_m=spacing),
  scope='Admitted actual saved frames 0..34 only; incomplete 40J run; particle-only review; integrated tool omitted; no depth acceptance.'))
 subprocess.run([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(Path(__file__).resolve()),'--pv-check',str(raw),str(output)],check=True)
 render_directory(output,gif=True,field='family_id',resolution=400,matrix_opacity=.008)
 (output/'simulation.gif').unlink()
 gif_check=encode_frames(output)
 render_record=json.loads((output/'render.json').read_text())
 render_record.update(animation='simulation.gif',animation_frames=35,distinct_decoded_frames=gif_check['distinct_decoded_frames'])
 transport.write(output/'render.json',render_record)
 assert before=={name:transport.digest(recovery.DEST/name) for name in before}
 transport.write(output/'prefix-review-admission.json',dict(passed=True,frames=35,raw_unchanged=True,
  solver_executed=False,full_protocol_completed=False,publication_eligible=False,
  tool_rendered=False,actual_paraview_fetch_all35=True,distinct_decoded_frames=gif_check['distinct_decoded_frames'],parent_transport_sha256=recovery.PARENT_TRANSPORT,
  gif_sha256=transport.digest(output/'simulation.gif'),scope='Actual ParaView particle-only prefix review; no interpolated states or tool motion.'))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--pv-check':pv_check(Path(sys.argv[2]),Path(sys.argv[3]))
 else:main()
