"""Verify/extract actual snapshot leaves, render in ParaView, encode private GIF."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False)
    original=json.loads((a.input/'manifest.json').read_text());prepared=a.output/'prepared';prepared.mkdir();manifest=dict(original);manifest['frames']=[];reference_ids=None
    for i,item in enumerate(original['frames']):
        source=a.input/item['file'];assert hashlib.sha256(source.read_bytes()).hexdigest()==item['sha256']
        with np.load(source,allow_pickle=False) as data:
            x=data['state_x'] if item['kind']=='checkpoint' else data['x'];ids=data['state_fibre_id'] if item['kind']=='checkpoint' else data['fibre_id']
            assert x.shape==(227520,3) and np.isfinite(x).all()
            if reference_ids is None:reference_ids=ids.copy()
            else:assert np.array_equal(ids,reference_ids)
            target=f'actual-{i:05d}.npz';np.savez(prepared/target,x=x,fibre_id=ids)
        manifest['frames'].append(dict(item,prepared_file=target))
    if manifest.get('full_turn_completed',False):
        assert manifest['frames'][-1]['step']==manifest['planned_steps']
    (prepared/'manifest.json').write_text(json.dumps(manifest,indent=2))
    command=['/opt/paraview/bin/pvpython','--force-offscreen-rendering','/work/tools/paraview_three_strand_forming.py',str(prepared.resolve()),str(a.output.resolve())]
    with (a.output/'paraview.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    record=json.loads((a.output/'render.json').read_text());assert record['resolution']==[400,400] and record['frames_rendered']==len(manifest['frames'])
    times=np.array([x['time_s'] for x in manifest['frames']]);interval=np.diff(times);assert np.all(interval>0)
    durations=[max(10,round(float(dt/(times[-1]-times[0])*6000)/10)*10) for dt in interval]+[400]
    pictures=[]
    for i in range(len(times)):
        with Image.open(a.output/'frames'/f'frame_{i:05d}.png') as image:
            assert image.size==(400,400);pictures.append(image.convert('RGB'))
    gif=a.output/'three-strand-forming-private.gif';pictures[0].save(gif,save_all=True,append_images=pictures[1:],duration=durations,loop=0,optimize=False,disposal=2)
    for image in pictures:image.close()
    review=dict(passed=True,physical_steps_executed=0,source_manifest_sha256=hashlib.sha256((a.input/'manifest.json').read_bytes()).hexdigest(),actual_snapshots=len(times),last_actual_step=manifest['frames'][-1]['step'],last_actual_time_s=float(times[-1]),frame_durations_ms=durations,resolution=[400,400],three_fixed_colors=True,fixed_camera=True,position_interpolation=False,full_turn_completed=bool(manifest.get("full_turn_completed",False)),private_only=True,published=False,energy_temporal_spatial_convergence_certified=False,gif_sha256=hashlib.sha256(gif.read_bytes()).hexdigest())
    (a.output/'private-render-review.json').write_text(json.dumps(review,indent=2));print(json.dumps(review,indent=2),flush=True)

if __name__=='__main__':main()
