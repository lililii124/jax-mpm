"""Synchronize the five actual ParaView frame sequences in one local viewer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
import webbrowser
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--open',action='store_true');p.add_argument('--assemble-gifs',action='store_true');args=p.parse_args()
    root=Path(args.root).resolve();dest=Path(args.out).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
    cases=[('square','四方'),('rectangular','长方'),('hexagonal','六方'),('octagonal','八方'),('isotropic','各向同性')]
    records=[];times=None;color_range=None;case_paths={}
    for case,label in cases:
        folder=root/case;receipt=json.loads((folder/'render_receipt.json').read_text())
        paths=sorted(folder.glob('*.png'));current=[row['time_s'] for row in receipt['frames']]
        if len(paths)!=len(current):raise ValueError('Frame/receipt count mismatch: '+case)
        if times is None:times=current
        if len(times)!=len(current) or any(abs(a-b)>1.e-12 for a,b in zip(times,current)):raise ValueError('Physical times differ: '+case)
        if color_range is None:color_range=receipt['color_range_Pa']
        if receipt['color_range_Pa']!=color_range:raise ValueError('Comparison color ranges differ')
        case_paths[case]=paths
        for path in paths:
            with Image.open(path) as im:
                if im.size!=(400,400):raise ValueError('Expected clean 400x400 frame')
        records.append(dict(id=case,label=label,frames=[quote(os.path.relpath(path,dest.parent).replace('\\','/'),safe='/.:') for path in paths]))
    if args.assemble_gifs:
        count=sum(len(paths) for paths in case_paths.values());sheet=Image.new('RGB',(100,100*count));offset=0
        for paths in case_paths.values():
            for path in paths:
                with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((100,100)),(0,100*offset))
                offset+=1
        palette=sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
        for case,paths in case_paths.items():
            frames=[]
            for path in paths:
                with Image.open(path) as im:frames.append(im.convert('RGB').quantize(palette=palette,dither=Image.Dither.NONE))
            gif=root/case/'stress.gif';frames[0].save(gif,save_all=True,append_images=frames[1:],duration=80,loop=0,optimize=False,disposal=2)
            unique=set()
            with Image.open(gif) as image:
                gif_count=image.n_frames
                for i in range(gif_count):image.seek(i);unique.add(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
            (root/case/'gif_receipt.json').write_text(json.dumps(dict(frames=gif_count,distinct_frames=len(unique),dimensions=[400,400],sha256=hashlib.sha256(gif.read_bytes()).hexdigest(),shared_palette=True,color_range_Pa=color_range),indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>相同爆炸压力下的材料对称性对比</title><style>
body{font:16px system-ui,sans-serif;background:#f5f6f8;margin:0;color:#20242b}main{max-width:1264px;margin:auto;padding:18px}h1{font-size:21px;margin:0 0 8px}p{margin:8px 0}nav{display:flex;align-items:center;gap:14px;margin:16px 0}input{flex:1}button{padding:7px 18px}output{min-width:130px;font-variant-numeric:tabular-nums}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0;background:white;padding:8px;border-radius:8px;text-align:center}figcaption{padding:5px;font-weight:600}img{width:100%;max-width:400px;height:auto;display:block;margin:auto}
</style><main><h1>相同爆炸压力下的材料对称性对比</h1><p>共同应力色标：__COLOR_RANGE__ MPa。五幅云图显示同一物理时刻。</p><nav><button id="play">暂停</button><input id="seek" type="range" min="0" value="0" aria-label="共同帧"><output id="time"></output></nav><section id="grid"></section></main><script>
const cases=__CASES__,times=__TIMES__;let index=0,running=true,timer;
const grid=document.getElementById('grid'),seek=document.getElementById('seek'),play=document.getElementById('play'),time=document.getElementById('time');
for(const c of cases){const figure=document.createElement('figure'),caption=document.createElement('figcaption'),image=document.createElement('img');caption.textContent=c.label;image.id=c.id;image.width=400;image.height=400;image.alt=c.label+'实际应力云图';figure.append(caption,image);grid.append(figure);}
seek.max=times.length-1;
function show(){for(const c of cases)document.getElementById(c.id).src=c.frames[index];seek.value=index;time.textContent=(times[index]*1000).toFixed(2)+' ms · '+(index+1)+'/'+times.length;}
function tick(){if(!running)return;index=(index+1)%times.length;show();timer=setTimeout(tick,80);}
function setRunning(value){running=value;clearTimeout(timer);play.textContent=value?'暂停':'播放';if(value)timer=setTimeout(tick,80);}
play.onclick=()=>setRunning(!running);seek.oninput=()=>{setRunning(false);index=Number(seek.value);show();};
document.addEventListener('keydown',e=>{if(e.key===' '){e.preventDefault();setRunning(!running);}else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();setRunning(false);index=Math.max(0,Math.min(times.length-1,index+(e.key==='ArrowRight'?1:-1)));show();}});
show();timer=setTimeout(tick,80);
</script></html>'''
    page=page.replace('__COLOR_RANGE__',f'{color_range[0]/1.e6:.6g}–{color_range[1]/1.e6:.6g}').replace('__CASES__',json.dumps(records,ensure_ascii=False)).replace('__TIMES__',json.dumps(times));dest.write_text(page,encoding='utf-8')
    print(json.dumps(dict(viewer=str(dest),cases=len(cases),frames=len(times),horizon_s=times[-1]),ensure_ascii=False))
    if args.open:webbrowser.open(dest.as_uri())

if __name__=='__main__':main()
