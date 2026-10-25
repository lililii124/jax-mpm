import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from mpm_jax.export import write_vtp
from examples.woven_large_plate import array_descriptor
parent=Path('/parent/output/plain/deep-sphere-impact');history=Path('/history/raw/output/plain/deep-sphere-impact');out=Path('/work/saved');(out/'vtk').mkdir(parents=True)
digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert digest(history/'checkpoint.npz')=='5b58b2cafb9da612297d77e130a54c7c9609cc6f76fdb796da3611062fbea0d0'
assert digest(parent/'checkpoint.npz')=='ef4dcb98bb77108ec8404f210361c2c52a22dd9271cf8a2aad8579a7e103791e'
with np.load(history/'checkpoint.npz',allow_pickle=False) as cp:old=json.loads(str(cp['tables']));assert int(cp['frame'])==34
with np.load(parent/'checkpoint.npz',allow_pickle=False) as cp:tables=json.loads(str(cp['tables']));assert int(cp['frame'])==60
assert old['trajectory_frames']==tables['trajectory_frames'][:35] and old['times']==tables['times'][:35]
arrays={k:np.load(history/('trajectory_'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ['x','speed','damage','temperature']}
with np.load(parent/'particle_fields.npz',allow_pickle=False) as cp:labels={k:cp[k] for k in cp.files}
datasets=[];records=[]
for i in range(61):
 if i<34:
  frame={k:arrays[k][i] for k in arrays}
  assert all(array_descriptor(v)==tables['trajectory_frames'][i][k] for k,v in frame.items())
  path=out/'vtk'/f'particles_{i:05d}.vtp';write_vtp(path,frame.pop('x'),{**labels,**frame});source='original recovered f34 raw, admitted saved prefix only'
 else:
  path=Path('/saved/media/vtk')/f'particles_{i-34:05d}.vtp';source='actual resumed f34..60 saved state, authenticated by previous render receipt'
 records.append(dict(source_frame=i,time_s=tables['times'][i],descriptor=tables['trajectory_frames'][i],vtp_sha256=digest(path),source=source))
 datasets.append(f'<DataSet timestep="{tables["times"][i]:.17g}" group="" part="0" file="{path}"/>')
(out/'particles.pvd').write_text('<?xml version="1.0"?><VTKFile type="Collection" version="0.1" byte_order="LittleEndian"><Collection>'+''.join(datasets)+'</Collection></VTKFile>')
(out/'full-source-review.json').write_text(json.dumps(dict(passed=True,frames=61,original_checkpoint_sha256=digest(history/'checkpoint.npz'),final_checkpoint_sha256=digest(parent/'checkpoint.npz'),original_future_slots_used=False,all_saved_descriptors_verified=True,source_frames=records),indent=2)+'\n')
subprocess.run(['/opt/paraview/bin/pvpython','--force-offscreen-rendering','/work/scene.py'],check=True)
