"""Verify raw actual flow snapshots, privately render and encode their cadence."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--pvpython',default=os.environ.get('PVPYTHON','/opt/paraview/bin/pvpython'));a=p.parse_args();a.output.mkdir(exist_ok=False);prepared=a.output/'prepared';prepared.mkdir();m=json.loads((a.input/'manifest.json').read_text());manifest=dict(m);manifest['frames']=[]
    fixed=a.input/'fixed-fibres.npz';assert digest(fixed)==m['fixed_fibres_sha256']
    with np.load(fixed,allow_pickle=False) as d:
        assert np.isfinite(d['x']).all() and d['x'].shape==(227520,3);np.savez(prepared/'fixed-fibres.npz',x=d['x'],fibre_id=d['fibre_id'])
    for i,item in enumerate(m['frames']):
        if item['kind']=='declared-empty-initial-state':x=np.empty((0,3))
        else:
            source=a.input/item['file'];assert digest(source)==item['sha256']
            with np.load(source,allow_pickle=False) as d:x=d['x'][d['active']] if item['kind']=='checkpoint' else d['x']
        assert x.ndim==2 and x.shape[1]==3 and np.isfinite(x).all();target=f'actual-{i:05d}.npz';np.savez(prepared/target,x=x);manifest['frames'].append(dict(item,file=target,actual_particles=len(x)))
    (prepared/'manifest.json').write_text(json.dumps(manifest,indent=2));command=[a.pvpython,'--force-offscreen-rendering',str(Path(__file__).with_name('paraview_three_strand_resin.py')),str(prepared.resolve()),str(a.output.resolve())]
    with (a.output/'paraview.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    render=json.loads((a.output/'render.json').read_text());assert render['distinct_images']==len(m['frames'])
    times=np.asarray([f['time_s'] for f in m['frames']]);assert np.all(np.diff(times)>0)
    duration=[max(10,round(float(v/(times[-1]-times[0])*6000)/10)*10) for v in np.diff(times)]+[400];images=[]
    for i in range(len(times)):
        with Image.open(a.output/'frames'/f'frame_{i:05d}.png') as image:assert image.size==(400,400);images.append(image.convert('RGB'))
    gif=a.output/'three-strand-resin-private.gif';images[0].save(gif,save_all=True,append_images=images[1:],duration=duration,loop=0,optimize=False,disposal=2)
    for image in images:image.close()
    receipt=dict(passed=True,physical_steps_executed=0,actual_snapshots=len(times),last_actual_step=m['frames'][-1]['step'],last_flow_time_s=float(times[-1]),forming_checkpoint_sha256=m['forming_checkpoint_sha256'],source_manifest_sha256=digest(a.input/'manifest.json'),gif_sha256=digest(gif),resolution=[400,400],position_interpolation=False,predefined_resin_cylinder=False,full_filling_certified=False,curing=False,full_FSI=False,private_only=True,published=False)
    (a.output/'private-render-review.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
