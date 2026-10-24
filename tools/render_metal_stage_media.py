"""Read-only media from the exact historical f16 cache; never advance a solver."""
import argparse
import hashlib
import json
import os
import math
import re
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.local/metal-corrected-continuation-cache'
OUT = ROOT/'.local/metal-stage-media'
DELIVERY = ROOT/'.local/metal-stage-delivery'
FROZEN = ROOT/'.local/metal-stage-source'
KEY = 'metal-corrected-continuation-v1-2df4e0ac39937b36e972f473b1db5fd33cebaca2-40076211097-1-f16'
RECEIPT_SHA = '9aef7177cb7aecf18eb2bb202077693c901a1b84a29519fb20ff6d7f6bb8c55f'
SCOPE = ('Historical 5 km/s old shock-model stage only: 17 saved states, 0 to 266.667 ns. '
         'Phase/EOS extrapolation; no physical validation or full-impact claim. '
         'Damage/failed fields are zero bookkeeping, not fracture predictions.')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2)+'\n')
def record(path): return dict(bytes=Path(path).stat().st_size, sha256=sha(path))

def media_identity():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert commit==os.environ['GITHUB_SHA']
    assert os.environ['GITHUB_REF']=='refs/heads/work/metal-return-adaptive-prefix'
    assert os.environ['GITHUB_WORKFLOW_REF'].split('@')[0]=='lililii124/jax-mpm/.github/workflows/engineering-previews.yml'
    names=('tools/render_metal_stage_media.py','.github/workflows/metal-stage-media.yml',
           '.github/workflows/engineering-previews.yml','tools/run_woven_preview.py',
           'mpm_jax/export.py','mpm_jax/visualization.py','mpm_jax/paraview_scene.py')
    for name in names:
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT),name
    return commit,{name:record(ROOT/name) for name in names}

def requirements():
    verify()
    pins=[]
    for line in (CACHE/'requirements.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith(('#','-e ','jax-mpm @ ')): continue
        assert re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9.!+_-]+',line),'Unpinned renderer dependency'
        if line.split('==')[0].lower().replace('_','-')!='jax-mpm': pins.append(line)
    assert {'jax','jaxlib','numpy','pillow','psutil'} <= {line.split('==')[0].lower() for line in pins}
    (OUT/'renderer-requirements.txt').write_text('\n'.join(pins)+'\n')

def verify():
    assert sha(CACHE/'receipt.json') == RECEIPT_SHA, 'Exact historical receipt required'
    r = read(CACHE/'receipt.json')
    assert r['cache_key'] == KEY and r['source']['run_id']=='40076211097'
    assert r['requested_stop_frame']==16 and r['requested_speed_m_s']==5000
    assert r['physical_validation_passed'] is False and r['production_eligible'] is False
    for name, expected in r['files'].items():
        assert record(CACHE/name)==expected, name
    assert read(CACHE/'source.json')==r['source']
    for revision, expected in (('HEAD',r['source']['commit']),('HEAD^{tree}',r['source']['tree'])):
        assert subprocess.check_output(['git','rev-parse',revision],cwd=FROZEN,text=True).strip()==expected
    assert read(CACHE/'continuation.json')==r['lineage']
    assert sha(CACHE/'parent-receipt.json')==r['lineage']['parent_receipt_sha256']
    assert r['lineage']['start_frame']==8 and r['lineage']['stop_frame']==16
    assert read(CACHE/'runtime.json')==dict(python='3.12.13', numpy='2.5.4', jax='0.12.0.1', jaxlib='0.12.0.1', backend='cpu', x64=True)
    for name, expected in r['source']['files'].items():
        assert record(FROZEN/name)=={k:expected[k] for k in ('bytes','sha256')}, name
    return r

def prepare():
    assert os.environ['CACHE_HIT']=='true' and os.environ['CACHE_MATCH']==KEY
    r = verify()
    media_commit,media_sources=media_identity()
    assert not OUT.exists(), 'Fresh output required'
    OUT.mkdir(parents=True)
    save(OUT/'admission.json', dict(source_receipt_sha256=RECEIPT_SHA, cache_key=KEY,
         source=r['source'], lineage=r['lineage'], original_runtime=read(CACHE/'runtime.json'),
         media_commit=media_commit,media_sources=media_sources,
         solver_executed=False, physical_validation_passed=False, production_eligible=False, scope=SCOPE))

def install():
    import run_woven_preview as pv
    pv.LOGS=OUT
    pv.install()
    r=read(OUT/'paraview-install.json')
    r['executable_sha256']=sha(r['pvpython'])
    save(OUT/'paraview-install.json',r)

def render():
    verify()
    for n,v in read(OUT/'admission.json')['media_sources'].items(): assert record(ROOT/n)==v,n
    import numpy as np
    sys.path.insert(0,str(ROOT))
    from mpm_jax.export import export_series
    from mpm_jax.visualization import render_directory
    folder=OUT/'result'
    folder.mkdir()
    shutil.copyfile(CACHE/'result/summary.json',folder/'summary.json')
    with np.load(CACHE/'result/trajectory.npz',allow_pickle=False) as z:
        times=z['time'].copy(); body=z['body'].copy()
    assert times.shape==(17,) and len(body)==409504
    np.testing.assert_allclose(times,np.arange(17)/60*1e-6,rtol=1e-14,atol=0)
    arrays={k:np.load(CACHE/f'result/trajectory_{k}.npy',mmap_mode='r') for k in ('x','speed','temperature','damage','failed')}
    assert arrays['x'].shape[1:]==(409504,3)
    assert all(a.shape[0]>=17 for a in arrays.values())
    assert np.isfinite(arrays['x'][:17]).all()
    assert np.isfinite(arrays['speed'][:17]).all()
    actual_max_speed=float(np.max(arrays['speed'][:17]))
    speed_range=[0,max(5000,math.ceil(actual_max_speed/1000)*1000)]
    summary=read(folder/'summary.json')
    export_series(arrays['x'][:17],dict(body=body,**{k:v[:17] for k,v in arrays.items() if k!='x'}),
                  times,folder,spacing=float(np.cbrt(np.prod(summary['particle_spacing_m']))),dim=3)
    pv=read(OUT/'paraview-install.json')
    assert sha(pv['pvpython'])==pv['executable_sha256']
    os.environ['PARAVIEW_PYTHON']=pv['pvpython']
    render_directory(folder,gif=True,field='speed',color_range=speed_range,resolution=1200)
    save(OUT/'saved-state-proof.json',dict(times_s=times.tolist(),saved_states=17,
        raw_source_files=read(CACHE/'receipt.json')['files'],speed_range_m_s=speed_range,
        actual_maximum_saved_speed_m_s=actual_max_speed,scope=SCOPE,solver_executed=False))
    subprocess.run([pv['pvpython'],'--force-offscreen-rendering',str(Path(__file__).resolve()),'vtk'],
                   cwd=ROOT,check=True,timeout=600)
    audit()

def vtk():
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
    verify()
    with np.load(CACHE/'result/trajectory.npz',allow_pickle=False) as archive: body=archive['body'].copy()
    arrays={k:np.load(CACHE/f'result/trajectory_{k}.npy',mmap_mode='r') for k in ('x','speed','temperature','damage','failed')}
    hashes={}
    for i in range(17):
        path=OUT/f'result/vtk/particles_{i:05d}.vtp';reader=vtkXMLPolyDataReader();reader.SetFileName(str(path));reader.Update();data=reader.GetOutput()
        assert data.GetNumberOfPoints()==409504
        np.testing.assert_array_equal(vtk_to_numpy(data.GetPoints().GetData()),arrays['x'][i].astype(np.float32))
        for name in ('speed','temperature','damage','failed','body'):
            actual=data.GetPointData().GetArray(name);assert actual is not None,name
            expected=body if name=='body' else arrays[name][i]
            if name not in ('body','failed'):expected=expected.astype(np.float32)
            np.testing.assert_array_equal(vtk_to_numpy(actual),expected)
        hashes[path.name]=sha(path)
    save(OUT/'vtk-audit.json',dict(passed=True,saved_states_checked=17,geometry_scale=1,
        vtk_files=hashes,scope='Actual VTK readback of all17saved coordinates and scalar/body fields; no phase-physics validation.'))

def audit():
    verify()
    from PIL import Image
    r=read(OUT/'result/render.json')
    assert r['renderer']=='ParaView' and r['frames_rendered']==17 and r['particles']==409504
    assert r['axes'] is False and r['annotations'] is False and r['geometry_scale']==1
    assert r['field']=='speed' and r['color_range']==read(OUT/'saved-state-proof.json')['speed_range_m_s'] and r['view_scope']=='complete scene'
    assert read(OUT/'vtk-audit.json')['passed'] and read(OUT/'vtk-audit.json')['saved_states_checked']==17
    for i in range(17):
        with Image.open(OUT/f'result/frames/frame_{i:05d}.png') as im:
            im.load()
            assert im.size==(1200,1200) and im.convert('RGB').getextrema()!=((255,255),)*3
    with Image.open(OUT/'result/simulation.gif') as im:
        assert im.n_frames==17 and im.size==(1200,1200)
        for i in range(17): im.seek(i); im.load()
    save(OUT/'media-audit.json',dict(passed=True,saved_states=17,raw_unchanged=True,
        no_axes_or_text=True,solver_executed=False,physical_validation_passed=False,scope=SCOPE))

def package():
    audit()
    assert not DELIVERY.exists()
    names=['admission.json','saved-state-proof.json','media-audit.json','vtk-audit.json','paraview-install.json',
           'result/summary.json','result/render.json','result/simulation.gif']
    names += [f'result/frames/frame_{i:05d}.png' for i in (0,4,8,12,16)]
    for n in names:
        p=DELIVERY/n; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(OUT/n,p)
    limits=read(OUT/'saved-state-proof.json')['speed_range_m_s']
    (DELIVERY/'README.txt').write_text(SCOPE+f'\nSpeed colors: fixed {limits[0]} to {limits[1]} m/s across all17states. Playback60ms per saved state.\n')
    files={p.relative_to(DELIVERY).as_posix():record(p) for p in DELIVERY.rglob('*') if p.is_file()}
    total=sum(v['bytes'] for v in files.values())
    assert total+100000<100_000_000
    r=dict(files=files,bytes=total,bound_bytes=100_000_000,scope=SCOPE,solver_executed=False,
           raw_preserved_in_exact_cache=True,permanent_raw_backup_verified=False)
    save(DELIVERY/'delivery-receipt.json',r); save(OUT/'delivery-receipt.json',r)

def retrieved(path):
    r=read(path/'delivery-receipt.json')
    assert r==read(OUT/'delivery-receipt.json')
    for n,v in r['files'].items(): assert record(path/n)==v,n
    save(OUT/'retrieval-receipt.json',dict(passed=True,artifact_id=os.environ['METAL_ARTIFACT_ID'],
        delivery_receipt_sha256=sha(path/'delivery-receipt.json'),files_checked=len(r['files']),
        run_id=os.environ['GITHUB_RUN_ID'],run_attempt=os.environ['GITHUB_RUN_ATTEMPT']))
    verify()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','requirements','install','render','vtk','audit','package','retrieved'))
    p.add_argument('--retrieved',type=Path)
    a=p.parse_args()
    if a.action=='retrieved': retrieved(a.retrieved)
    else: globals()[a.action]()
