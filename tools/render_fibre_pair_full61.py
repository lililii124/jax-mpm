"""Render all61 admitted real MPM states with the common400px low camera."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mpm_jax.export import export_series
from mpm_jax.visualization import find_pvpython
from examples.single_fibre_tension import descriptor,save_json


def render(source,output):
    if output.exists():raise ValueError('Use a separate new read-only preview output')
    summary_path=source/'summary.json';summary_raw=summary_path.read_bytes() if summary_path.exists() else None
    if summary_raw is None or not json.loads(summary_raw)['completed']:raise ValueError('Complete actual full61 summary required')
    plan=json.loads((source/'plan.json').read_text())
    if summary_raw is not None:summary=json.loads(summary_raw)
    else:
        with np.load(source/'particle_fields.npz',allow_pickle=False) as labels:static_record={k:descriptor(labels[k]) for k in labels.files}
        summary={'case':'fibre-pair-live-saved-frame-snapshot','completed':False,'definition':plan,'source':plan['source'],'particle_fields':{'file':'particle_fields.npz','fields':static_record},'configs':[{'dim':3,'domain_size':plan['geometry']['length_m'],'n_grid':round(plan['geometry']['length_m']/plan['grid']['dx_m'])}]}
    checkpoint_raw=(source/'checkpoint.npz').read_bytes()
    with np.load(io.BytesIO(checkpoint_raw),allow_pickle=False) as cp:
        signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if str(cp['signature'])!=signature:raise ValueError('Actual saved-frame checkpoint differs from source/physical plan')
        frame=int(cp['frame']);tables=json.loads(str(cp['tables']));times=np.asarray(tables['times']);records=tables['records']
    if frame!=60:raise ValueError('Exactly61 complete actual states required')
    if len(times)!=frame+1 or len(records)!=frame+1:raise ValueError('Incomplete saved prefix')
    arrays={};fields={}
    for name in ('x','speed','damage','temperature','failed'):
        a=np.load(source/f'trajectory_{name}.npy',mmap_mode='r');selected=np.asarray(a[:frame+1]).copy()
        for i in range(frame+1):
            if descriptor(selected[i])!=records[i][name]:raise ValueError('Changed actual raw saved frame')
        arrays[name]=selected
    with np.load(source/'particle_fields.npz',allow_pickle=False) as static:
        for name in static.files:
            if descriptor(static[name])!=summary['particle_fields']['fields'][name]:raise ValueError('Changed actual particle labels')
            fields[name]=np.array(static[name])
    fields.update({k:arrays[k] for k in ('speed','damage','temperature','failed')});fields['body']=np.zeros(plan['particles'],dtype=np.int32)
    output.mkdir(parents=True);save_json(output/'summary.json',summary)
    export_series(arrays['x'],fields,times,output,spacing=plan['geometry']['particle_spacing_m'],dim=3)
    command=[find_pvpython(),'--force-offscreen-rendering',str(ROOT/'tools/paraview_fibre_pair_scene.py'),str(output),'--field','family_id','--resolution','400','--focus-bounds',*map(str,plan['camera']['focus_bounds_m'])]
    environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None)
    with (output/'render.log').open('w') as log:subprocess.run(command,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    record=json.loads((output/'render.json').read_text())
    if record['renderer']!='ParaView' or record['frames_rendered']!=frame+1 or record['axes'] or record['annotations'] or record['camera_elevation_policy']!=.2:raise ValueError('Actual common low-angle ParaView policy failed')
    from PIL import Image,GifImagePlugin
    gif=output/'simulation.gif'
    with gif.open('wb') as stream:
        for i in range(61):
            with Image.open(output/'frames'/f'frame_{i:05d}.png') as picture:
                frame_image=picture.convert('RGB').convert('P',palette=Image.Palette.WEB,dither=Image.Dither.NONE)
            if i==0:
                header,_=GifImagePlugin.getheader(frame_image,info={'loop':0})
                for block in header:stream.write(block)
            for block in GifImagePlugin.getdata(frame_image,duration=120,disposal=2,include_color_table=True):stream.write(block)
            frame_image.close()
        stream.write(b';')
    with Image.open(gif) as animation:
        if animation.n_frames!=61 or animation.size!=(400,400):raise ValueError('Exactly61 real400px GIF frames required')
        for i in range(61):
            animation.seek(i)
            if animation.info.get('duration')!=120:raise ValueError('Same fixed frame timing required')
    if gif.stat().st_size>=9*1024**2:raise ValueError('GIF exceeds9MiB; no frames omitted to fit')
    save_json(output/'preview-receipt.json',{'schema':'fibre-pair-real-full61-low-camera-media-v1','original_frame_count':frame+1,'live_snapshot_without_completed_job_summary':summary_raw is None,'source_plan_sha256':hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),'source_summary_sha256':hashlib.sha256(summary_raw).hexdigest() if summary_raw is not None else None,'source_checkpoint_sha256':hashlib.sha256(checkpoint_raw).hexdigest(),'actual_renderer':'ParaView','camera_direction':[1.4,.2,1.8],'geometry_scale':1,'axes':False,'annotations':False,'production_eligible':False,'complete61frame_result':frame==60,'display_gif':'simulation.gif','display_resolution':[400,400],'display_palette':'fixed web palette, no dithering','all_original_states_retained':True,'scope':'All61 actual MPM saved states; original raw input remains read-only. Pair-level measured hold admission is required separately before rendering/public replacement.'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();render(a.source.resolve(),a.output.resolve())


if __name__=='__main__':main()
