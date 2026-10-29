"""Create a standalone GIF viewer with frame count, seeking and playback."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import webbrowser

from PIL import Image


def create_review(source, destination=None):
    source = Path(source).resolve()
    destination = (Path(destination).resolve() if destination else
                   source.with_suffix('.review.html'))
    if source == destination:
        raise ValueError('The viewer must not overwrite its source GIF')
    frames, durations, distinct = [], [], set()
    with Image.open(source) as gif:
        if gif.format != 'GIF':
            raise ValueError('Expected a GIF file')
        size = gif.size
        for index in range(gif.n_frames):
            gif.seek(index)
            frame = gif.convert('RGBA')
            distinct.add(hashlib.sha256(frame.tobytes()).hexdigest())
            buffer = io.BytesIO()
            frame.save(buffer, format='PNG')
            frames.append('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii'))
            durations.append(max(10, int(gif.info.get('duration', 100) or 100)))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f5f6f8;font:16px system-ui,sans-serif}
main{max-width:95vw;padding:16px;box-sizing:border-box;background:white;border-radius:10px}
img{display:block;max-width:100%;max-height:80vh;width:auto;height:auto;margin:auto}
nav{display:flex;gap:12px;align-items:center;margin-top:14px}
input{flex:1;min-width:40px}button{padding:6px 14px;cursor:pointer}
output{min-width:70px;text-align:right;font-variant-numeric:tabular-nums}
</style><main><img id="picture" alt="__TITLE__"><nav>
<button id="play" type="button">暂停</button>
<input id="seek" aria-label="帧" type="range" min="0" value="0" step="1">
<output id="count" aria-live="off"></output></nav></main><script>
const frames=__FRAMES__,durations=__DURATIONS__;
const picture=document.getElementById('picture'),seek=document.getElementById('seek');
const count=document.getElementById('count'),play=document.getElementById('play');
let current=0,running=true,timer;
seek.max=frames.length-1;
function show(){picture.src=frames[current];seek.value=current;count.textContent=`${current+1} / ${frames.length}`;}
function tick(){if(!running)return;current=(current+1)%frames.length;show();timer=setTimeout(tick,durations[current]);}
function setRunning(value){running=value;clearTimeout(timer);play.textContent=running?'暂停':'播放';if(running)timer=setTimeout(tick,durations[current]);}
play.onclick=()=>setRunning(!running);
seek.oninput=()=>{setRunning(false);current=Number(seek.value);show();};
document.addEventListener('keydown',event=>{
  if(event.key===' '){event.preventDefault();setRunning(!running);}
  else if(event.key==='ArrowLeft'||event.key==='ArrowRight'){
    event.preventDefault();setRunning(false);
    current=Math.max(0,Math.min(frames.length-1,current+(event.key==='ArrowRight'?1:-1)));show();
  }
});
show();timer=setTimeout(tick,durations[0]);
</script></html>'''
    page = page.replace('__TITLE__', html.escape(source.name, quote=True))
    page = page.replace('__FRAMES__', json.dumps(frames)).replace('__DURATIONS__', json.dumps(durations))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(page, encoding='utf-8')
    return dict(source=str(source), viewer=str(destination), frames=len(frames),
                distinct_frames=len(distinct), size=list(size),
                duration_ms=sum(durations), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gif', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--open', action='store_true', help='Open the viewer in the default browser')
    args = parser.parse_args()
    result = create_review(args.gif, args.output)
    print(json.dumps(result, ensure_ascii=False))
    if args.open:
        webbrowser.open(Path(result['viewer']).as_uri())


if __name__ == '__main__':
    main()
