"""Render only authenticated true zero-degree frames20,21,22 for local review."""
import hashlib,json,os,shutil,subprocess,sys,time,ctypes,signal
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
SEED=ROOT/'review-seeds/triaxial-zero-f20-f22'
OUT=ROOT/'.local/triaxial-zero-short-review'
MANIFEST_SHA='fa880bbefddb6b868a84536e8f4e7956bcde60538bb3e209caaae8b39a41792e'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def bind():
 head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
 if head!=os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF')!='refs/heads/work/triaxial-zero-review-media' or os.environ.get('GITHUB_REPOSITORY')!='lililii124/jax-mpm':raise ValueError('Wrong exact source/ref')
 template='.github/workflows/triaxial-review-media.yml';active='.github/workflows/engineering-previews.yml'
 if (ROOT/template).read_bytes()!=(ROOT/active).read_bytes():raise ValueError('Registered workflow differs')
 if sha(SEED/'vtk-payload-manifest.json')!=MANIFEST_SHA:raise ValueError('Exact authenticated payload required')
 manifest=json.loads((SEED/'vtk-payload-manifest.json').read_text());names=['tools/render_triaxial_review_payload.py','mpm_jax/paraview_scene.py','tools/run_woven_preview.py',template,active]+['review-seeds/triaxial-zero-f20-f22/'+n for n in [*manifest['files'],'vtk-payload-manifest.json','.gitattributes']];files={}
 for n in names:
  if (ROOT/n).read_bytes()!=subprocess.check_output(['git','show',head+':'+n]):raise ValueError('Git source/payload changed: '+n)
  files[n]=sha(ROOT/n)
 for n,r in manifest['files'].items():
  if (SEED/n).stat().st_size!=r['bytes'] or sha(SEED/n)!=r['sha256']:raise ValueError('Authenticated payload changed')
 if manifest['frames']!=[20,21,22] or manifest['production_eligible'] is not False:raise ValueError('Wrong actual frame scope')
 return {'commit':head,'files':files,'frames':[20,21,22],'mechanical_steps_executed':0,'production_eligible':False}
def owned_session_uss(group_id, psutil, attempts=3):
    """Two stable session inventories; never accept a partial live-tree sum."""

    def inventory():
        members = {}
        for process in psutil.process_iter():
            if os.getsid(process.pid) == group_id:
                members[process.pid] = process.create_time()
        owner = psutil.Process(os.getpid())
        members[owner.pid] = owner.create_time()
        return members
    for attempt in range(attempts):
        try:
            before = inventory()
            total = 0
            for pid, created in before.items():
                process = psutil.Process(pid)
                if process.create_time() != created:
                    raise RuntimeError('PID identity changed during USS observation')
                total += process.memory_full_info().uss
            if before == inventory():
                return total
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        if attempt + 1 < attempts:
            time.sleep(0.01)
    raise RuntimeError('Incomplete owned-session USS after bounded rescan')

def terminate_owned_group(child):
    """The session belongs to start_new_session Popen, even after leader exit."""
    if child is None:
        return
    import psutil
    deadline = time.monotonic() + 20
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(0.001, deadline - time.monotonic()))
    while True:
        live = []
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if pid == 0:
                break
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned session still live after bounded cleanup')
        time.sleep(0.02)
def main(phase):
 context=bind()
 if phase=='prepare':
  if OUT.exists():raise ValueError('Keep old derivative')
  OUT.mkdir(parents=True)
  manifest=json.loads((SEED/'vtk-payload-manifest.json').read_text())
  for n in manifest['files']:
   p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SEED/n,p)
  save(OUT/'controller-source.json',context);return
 if json.loads((OUT/'controller-source.json').read_text())!=context:raise ValueError('Source changed after prepare')
 if sys.platform!='linux' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise ValueError('Linux owned-session subreaper required')
 child=None;reason=None;code=None;peak=0;started=time.monotonic();cleanup=None
 try:
  with (OUT/'render.log').open('wb') as log:
   child=subprocess.Popen([os.environ['PARAVIEW_PYTHON'],'--force-offscreen-rendering',str(ROOT/'mpm_jax/paraview_scene.py'),str(OUT),'--field','family_id','--resolution','400'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    usage=owned_session_uss(child.pid,psutil);peak=max(peak,usage)
    reason='6GiB owned-session USS' if usage>6*1024**3 else '20minute render guard' if time.monotonic()-started>1200 else None
    if reason:break
    time.sleep(.5)
   code=child.poll()
 except BaseException as exc:reason=type(exc).__name__+': '+str(exc)
 finally:
  try:terminate_owned_group(child)
  except BaseException as exc:cleanup=type(exc).__name__+': '+str(exc)
  save(OUT/'resource.json',dict(reason=reason,cleanup_error=cleanup,returncode=None if child is None else child.returncode,peak_owned_session_uss_bytes=peak,wall_seconds=time.monotonic()-started))
 if reason or cleanup or child is None or child.returncode:raise RuntimeError('Renderer failed closed')
 render=json.loads((OUT/'render.json').read_text());png=sorted((OUT/'frames').glob('frame_*.png'))
 if render['frames_rendered']!=3 or len(png)!=3 or render['axes'] or render['annotations'] or render['geometry_scale']!=1:raise ValueError('Exact true-scale threeframe review required')
 from PIL import Image,GifImagePlugin
 states=[]
 with (OUT/'simulation.gif').open('wb') as stream:
  for i,p in enumerate(png):
   with Image.open(p) as im:
    if im.size!=(400,400):raise ValueError('400square required')
    rgb=im.convert('RGB');states.append(hashlib.sha256(rgb.tobytes()).hexdigest());pic=rgb.convert('P',palette=Image.Palette.WEB,dither=Image.Dither.NONE)
   if i==0:
    header,_=GifImagePlugin.getheader(pic,info={'loop':0})
    for block in header:stream.write(block)
   for block in GifImagePlugin.getdata(pic,duration=240,disposal=2,include_color_table=True):stream.write(block)
  stream.write(b';')
 if len(set(states))!=3:raise ValueError('Frozen/static image sequence refused')
 with Image.open(OUT/'simulation.gif') as im:
  if im.n_frames!=3 or im.size!=(400,400):raise ValueError('Decoded exact threeframe GIF required')
 if bind()!=context:raise ValueError('Authenticated source payload changed')
 save(OUT/'media-review-receipt.json',dict(schema='triaxial-actual20to22-review-only-v1',source=context,actual_saved_frames=[20,21,22],gif_sha256=sha(OUT/'simulation.gif'),gif_bytes=(OUT/'simulation.gif').stat().st_size,original_particle_history_missing_frames=list(range(1,20)),unique_png_states=3,mechanical_steps_executed=0,interpolation=False,production_eligible=False,public_homepage_allowed=False,scope='Only authenticated actual saved20,21,22. Original rejected step7219/frame23 excluded; no full0to22 history or physical-fracture acceptance claimed.'))
if __name__=='__main__':main(sys.argv[1])
