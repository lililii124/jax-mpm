"""Fresh guarded f2 deep-impact evidence; transport includes all failed raw."""
import json,os,platform,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import rht_impact_prefix as transport
DEST=ROOT/'.local/woven-deep-prefix'
transport.DEST=DEST
transport.DELIVERY=ROOT/'.local/woven-deep-delivery'
def main():
 import psutil,numpy,jax,jaxlib
 from examples import woven_large_plate as base
 if DEST.exists():raise FileExistsError(DEST)
 DEST.mkdir(parents=True)
 runtime=dict(python=platform.python_version(),numpy=numpy.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,psutil=psutil.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
 assert runtime==dict(python='3.12.13',numpy='2.5.4',jax='0.12.0.1',jaxlib='0.12.0.1',psutil='7.2.2',backend='cpu',x64=True),runtime
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 assert commit==os.environ['GITHUB_SHA'] and os.environ['GITHUB_REF']=='refs/heads/work/woven-deep-impact'
 names=(*base.SOURCE_FILES,'tools/run_woven_deep_prefix.py','tools/rht_impact_prefix.py','.github/workflows/woven-deep-prefix.yml','.github/workflows/engineering-previews.yml','pyproject.toml')
 sources={}
 for name in names:
  raw=(ROOT/name).read_bytes();blob=subprocess.check_output(['git','show','HEAD:'+name]);assert raw==blob,name
  sources[name]=dict(sha256=transport.digest(ROOT/name),bytes=len(raw))
 transport.write(DEST/'source.json',dict(commit=commit,files=sources,runtime=runtime,fresh_t0=True,physical_validation_passed=False))
 command=[sys.executable,'-m','examples.woven_deep_impact','--output',str(DEST/'output'),'--stop-after-frame','2','--persistence','controller']
 assert not (DEST/'output/segments').exists()
 begun=time.monotonic();samples=[];reason=None
 with (DEST/'solve.log').open('wb') as log:
  proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while proc.poll() is None:
   try:
    parent=psutil.Process(proc.pid);usage=0
    for child in [parent]+parent.children(recursive=True):
     try:usage+=child.memory_full_info().uss
     except psutil.NoSuchProcess:pass
   except psutil.NoSuchProcess:
    if proc.poll() is None:reason='live process disappeared from monitor'
    usage=0
   except psutil.AccessDenied:usage=0;reason='inaccessible live process'
   raw=0
   for p in DEST.rglob('*'):
    try:
     if p.is_file():raw+=p.stat().st_size
    except FileNotFoundError:pass
   elapsed=time.monotonic()-begun;samples.append(dict(elapsed_s=elapsed,tree_uss=usage,raw_bytes=raw))
   if usage>6*1024**3:reason='6GiB tree USS'
   if raw>3*1024**3-256*1024**2:reason='raw watermark'
   if transport.shutil.disk_usage(DEST).free<3*1024**3:reason='disk reserve'
   if elapsed>=5400:reason='90min guard'
   if reason:
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    break
   time.sleep(1)
  code=proc.wait()
 assert not (DEST/'output/segments').exists(), 'Duplicate snapshots forbidden in controller mode'
 transport.write(DEST/'monitor.json',dict(samples=samples,reason=reason,returncode=code))
 if reason or code:raise RuntimeError('Rejected prefix; complete raw preserved')
 result=DEST/'output/plain/deep-sphere-impact';status=json.loads((result/'status.json').read_text())
 assert status['frame']==2 and not status['completed'],status
 review=json.loads((result/'deep-review.json').read_text())
 assert all(review[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed')),review
 files={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=transport.digest(p)) for p in DEST.rglob('*') if p.is_file()}
 transport.write(DEST/'receipt.json',dict(schema='woven-deep-f2-v1',commit=commit,files=files,status=status,physical_validation_passed=False,scope='40J elastic exploratory prefix; depth target not yet assessed'))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='package':
  assert not (DEST/'output/segments').exists(), 'Duplicate snapshot forbidden';transport.package()
 elif len(sys.argv)>1 and sys.argv[1]=='verify':
  folder=Path(sys.argv[2]);assert transport.digest(folder/'transport.json')==transport.digest(transport.DELIVERY/'transport.json');transport.verify(folder)
 else:main()
