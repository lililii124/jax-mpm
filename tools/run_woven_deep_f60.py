"""Exact admitted f10 to f60 continuation; complete guarded raw preservation."""
import json,os,platform,signal,subprocess,sys,time,tarfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import rht_impact_prefix as transport
DEST=ROOT/'.local/woven-deep-f60'
PARENT=ROOT/'.local/woven-deep-f60-parent'
PARENT_COMMIT='d7364d8e6ec1790379a3ed16921e8dbae67c6945'
PARENT_TRANSPORT='da69c2ede481617a0dff7f33f61d64a9bdb5fba0b317d5e52a06308655490a34'
transport.DEST=DEST
transport.DELIVERY=ROOT/'.local/woven-deep-f60-delivery'
def restore_parent(runtime):
 # Admission is fixed to the accepted run, not a mutable cache key.
 assert transport.digest(PARENT/'transport.json')==PARENT_TRANSPORT
 transport.verify(PARENT)
 manifest=json.loads((PARENT/'transport.json').read_text())
 metadata={}
 reader=transport.PartReader(PARENT/name for name in manifest['parts'])
 with tarfile.open(fileobj=reader,mode='r|') as archive:
  for member in archive:
   name=member.name
   assert name in manifest['files'] and member.isfile()
   if name in ('source.json','receipt.json'):
    metadata[name]=json.load(archive.extractfile(member))
   elif name.startswith('output/'):
    target=DEST/name
    assert target.resolve().is_relative_to((DEST/'output').resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as out:shutil.copyfileobj(archive.extractfile(member),out,1024*1024)
 assert metadata['source.json']['commit']==PARENT_COMMIT
 assert metadata['source.json']['runtime']==runtime
 receipt=metadata['receipt.json'];assert receipt['commit']==PARENT_COMMIT
 assert receipt['schema']=='woven-deep-f10-v1' and receipt['status']['frame']==10
 assert receipt['status']['phase']=='checkpoint-ready' and receipt['status']['coupling_valid']
 from examples import woven_large_plate as base
 sample,plan=base.load_saved_case(DEST/'output/plain/deep-sphere-impact')
 assert plan['deep_impact']['initial_energy_J']==40 and plan['frames']==60
 status=json.loads((DEST/'output/plain/deep-sphere-impact/status.json').read_text())
 assert status==receipt['status']
 review=json.loads((DEST/'output/plain/deep-sphere-impact/deep-review.json').read_text())
 assert all(review[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed'))
 assert not review['completed'] and not review['accepted']
 for name,info in receipt['files'].items():
  assert name in manifest['files'] and info==manifest['files'][name]
 transport.write(DEST/'parent-admission.json',dict(parent_run=40091612689,parent_attempt=1,
  parent_commit=PARENT_COMMIT,transport_sha256=PARENT_TRANSPORT,source=metadata['source.json'],
  receipt=receipt,scope='Complete verified parent raw restored to fresh destination; parent artifacts unchanged'))

def guarded_solver(command,psutil):
 begun=time.monotonic();samples=[];reason=None;proc=None;code=None;error=None;cleanup=[]
 try:
  with (DEST/'solve.log').open('wb') as log:
   proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    # Any incomplete live-tree observation rejects the run; never undercount.
    parent=psutil.Process(proc.pid)
    tree=[parent]+parent.children(recursive=True)
    usage=sum(child.memory_full_info().uss for child in tree)
    raw=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    elapsed=time.monotonic()-begun
    samples.append(dict(elapsed_s=elapsed,tree_uss=usage,raw_bytes=raw,observed_processes=len(tree)))
    if usage>6*1024**3:reason='6GiB tree USS'
    if raw>3*1024**3-256*1024**2:reason='raw watermark'
    if transport.shutil.disk_usage(DEST).free<3*1024**3:reason='disk reserve'
    if elapsed>=5400:reason='90min guard'
    if reason:break
    time.sleep(1)
 except BaseException as caught:
  error=caught;reason=reason or 'controller observation or launch exception'
 finally:
  if proc is not None:
   # Own session/group only; terminate descendants even if the leader exited.
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   except BaseException as caught:
    cleanup.append(dict(stage='kill-owned-group',type=type(caught).__name__,message=str(caught)))
    reason=reason or 'owned group cleanup failed'
    try:proc.kill()
    except ProcessLookupError:pass
    except BaseException as fallback:
     cleanup.append(dict(stage='kill-leader-fallback',type=type(fallback).__name__,message=str(fallback)))
   try:code=proc.wait(timeout=10)
   except BaseException as caught:
    cleanup.append(dict(stage='bounded-wait',type=type(caught).__name__,message=str(caught)))
    reason=reason or 'owned process bounded wait failed'
  transport.write(DEST/'monitor.json',dict(samples=samples,reason=reason,returncode=code,
   exception=None if error is None else dict(type=type(error).__name__,message=str(error)),
   cleanup_errors=cleanup,owned_pid=None if proc is None else proc.pid,
   parent_immutable=True,elapsed_s=time.monotonic()-begun))
 if error is not None:raise RuntimeError('Controller rejected; complete raw preserved') from error
 if reason or code:raise RuntimeError('Rejected prefix; complete raw preserved')
 assert not (DEST/'output/segments').exists(), 'Duplicate snapshots forbidden in controller mode'

def main():
 import psutil,numpy,jax,jaxlib
 from examples import woven_large_plate as base
 if DEST.exists():raise FileExistsError(DEST)
 DEST.mkdir(parents=True)
 runtime=dict(python=platform.python_version(),numpy=numpy.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,psutil=psutil.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
 assert runtime==dict(python='3.12.13',numpy='2.5.4',jax='0.12.0.1',jaxlib='0.12.0.1',psutil='7.2.2',backend='cpu',x64=True),runtime
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 assert commit==os.environ['GITHUB_SHA'] and os.environ['GITHUB_REF']=='refs/heads/work/woven-deep-impact'
 assert transport.digest(ROOT/'tools/review_woven_deep_final.py')=='35c155ccc88737aac1ea4f219863e92b42c3c843ff5d40a2cd64cdf11e8b6125'
 names=(*base.SOURCE_FILES,'tools/run_woven_deep_f60.py','tools/rht_impact_prefix.py','.github/workflows/woven-deep-f60.yml','tools/review_woven_deep_final.py','evidence/deep-woven/f10-independent-admission.json','.github/workflows/engineering-previews.yml','pyproject.toml')
 sources={}
 for name in names:
  raw=(ROOT/name).read_bytes();blob=subprocess.check_output(['git','show','HEAD:'+name]);assert raw==blob,name
  sources[name]=dict(sha256=transport.digest(ROOT/name),bytes=len(raw))
 transport.write(DEST/'source.json',dict(commit=commit,files=sources,runtime=runtime,fresh_t0=False,parent_run=40091612689,parent_transport_sha256=PARENT_TRANSPORT,physical_validation_passed=False))
 if True:
  proof=ROOT/'evidence/deep-woven/f10-independent-admission.json'
  assert transport.digest(proof)=='407c41df3eda449ce343d0fff76ede6d17d3242e4cd6fd2e0c7fcfcd3102c86f'
  admission=json.loads(proof.read_text());assert admission['admitted_prefix'] and admission['raw_unchanged'] and not admission['solver_executed'] and admission['producer_commit']==PARENT_COMMIT and admission['producer_run']==40091612689
 restore_parent(runtime)
 command=[sys.executable,'-m','examples.woven_deep_impact','--output',str(DEST/'output'),'--stop-after-frame','60','--resume','--persistence','controller']
 assert not (DEST/'output/segments').exists()
 guarded_solver(command,psutil)
 result=DEST/'output/plain/deep-sphere-impact';status=json.loads((result/'status.json').read_text())
 assert status['frame']==60 and status['completed'],status
 review=json.loads((result/'deep-review.json').read_text())
 assert all(review[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed')),review
 files={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=transport.digest(p)) for p in DEST.rglob('*') if p.is_file()}
 transport.write(DEST/'receipt.json',dict(schema='woven-deep-f60-v1',commit=commit,files=files,status=status,physical_validation_passed=False,scope='40J elastic exploratory prefix; depth target assessed only at saved prefix frames; Full raw completion only; separate >=5mm final publication review required'))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='package':
  assert not (DEST/'output/segments').exists(), 'Duplicate snapshot forbidden'
  raw=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
  assert raw<3*1024**3-16*1024**2,'Complete artifact exceeds3GiB admission'
  transport.package()
  assert sum(p.stat().st_size for p in transport.DELIVERY.glob('*.tarpart'))<=3*1024**3
 elif len(sys.argv)>1 and sys.argv[1]=='verify':
  folder=Path(sys.argv[2]);assert transport.digest(folder/'transport.json')==transport.digest(transport.DELIVERY/'transport.json');transport.verify(folder)
 else:main()
