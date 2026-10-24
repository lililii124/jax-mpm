"""Exact accepted f2 to f10 continuation; complete guarded raw preservation."""
import json,os,platform,signal,subprocess,sys,time,tarfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import rht_impact_prefix as transport
DEST=ROOT/'.local/woven-deep-f10'
PARENT=ROOT/'.local/woven-deep-parent'
PARENT_COMMIT='669ba3593a8c5db2b50c3d995ced34a6f596b64f'
PARENT_TRANSPORT='4c36a652818f2f45d9d20b34e381a7799689cbe287686464c929d96131bbbbcb'
transport.DEST=DEST
transport.DELIVERY=ROOT/'.local/woven-deep-f10-delivery'
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
 assert receipt['schema']=='woven-deep-f2-v1' and receipt['status']['frame']==2
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
 transport.write(DEST/'parent-admission.json',dict(parent_run=40089103544,parent_attempt=1,
  parent_commit=PARENT_COMMIT,transport_sha256=PARENT_TRANSPORT,source=metadata['source.json'],
  receipt=receipt,scope='Complete verified parent raw restored to fresh destination; parent artifacts unchanged'))

def main(review_only=False):
 import psutil,numpy,jax,jaxlib
 from examples import woven_large_plate as base
 if DEST.exists():raise FileExistsError(DEST)
 DEST.mkdir(parents=True)
 runtime=dict(python=platform.python_version(),numpy=numpy.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,psutil=psutil.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
 assert runtime==dict(python='3.12.13',numpy='2.5.4',jax='0.12.0.1',jaxlib='0.12.0.1',psutil='7.2.2',backend='cpu',x64=True),runtime
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 assert commit==os.environ['GITHUB_SHA'] and os.environ['GITHUB_REF']=='refs/heads/work/woven-deep-impact'
 names=(*base.SOURCE_FILES,'tools/run_woven_deep_f10.py','tools/rht_impact_prefix.py','.github/workflows/woven-deep-f10.yml','.github/workflows/woven-deep-f2-admission.yml','evidence/deep-woven/f2-independent-admission.json','.github/workflows/engineering-previews.yml','pyproject.toml')
 sources={}
 for name in names:
  raw=(ROOT/name).read_bytes();blob=subprocess.check_output(['git','show','HEAD:'+name]);assert raw==blob,name
  sources[name]=dict(sha256=transport.digest(ROOT/name),bytes=len(raw))
 transport.write(DEST/'source.json',dict(commit=commit,files=sources,runtime=runtime,fresh_t0=False,parent_run=40089103544,parent_transport_sha256=PARENT_TRANSPORT,physical_validation_passed=False))
 if not review_only:
  proof=ROOT/'evidence/deep-woven/f2-independent-admission.json'
  assert transport.digest(proof)=='c06573b7fba7aaefcb3811cdbde51d0262440761c51a857c3bc3d653bf024ee3'
  admission=json.loads(proof.read_text());assert admission['admitted_prefix'] and admission['raw_unchanged'] and not admission['solver_executed'] and admission['parent_commit']==PARENT_COMMIT
 restore_parent(runtime)
 if review_only:
  import numpy as np
  from examples import woven_deep_impact as deep
  folder=DEST/'output/plain/deep-sphere-impact'
  before={p.relative_to(folder).as_posix():transport.digest(p) for p in folder.rglob('*') if p.is_file()}
  sample,plan=base.load_saved_case(folder)
  with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
   base.validate_checkpoint_control(cp,plan)
   assert int(cp['frame'])==2
   for name in cp.files:
    value=cp[name]
    if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),name
  result=deep.review(folder)
  assert all(result[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed')),result
  after={p.relative_to(folder).as_posix():transport.digest(p) for p in folder.rglob('*') if p.is_file()}
  assert before==after
  admission=json.loads((DEST/'parent-admission.json').read_text())
  admission.update(schema='woven-deep-f2-independent-admission-v1',admitted_prefix=True,full_protocol_accepted=False,review=result,raw_unchanged=True,solver_executed=False,controller_commit=commit,controller_run=os.environ['GITHUB_RUN_ID'],runtime=runtime,source_files=sources)
  transport.write(DEST/'independent-admission.json',admission)
  return
 command=[sys.executable,'-m','examples.woven_deep_impact','--output',str(DEST/'output'),'--stop-after-frame','10','--resume','--persistence','controller']
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
 assert status['frame']==10 and not status['completed'],status
 review=json.loads((result/'deep-review.json').read_text())
 assert all(review[k] for k in ('stable','clamp_passed','material_energy_passed','energy_passed')),review
 files={p.relative_to(DEST).as_posix():dict(bytes=p.stat().st_size,sha256=transport.digest(p)) for p in DEST.rglob('*') if p.is_file()}
 transport.write(DEST/'receipt.json',dict(schema='woven-deep-f10-v1',commit=commit,files=files,status=status,physical_validation_passed=False,scope='40J elastic exploratory prefix; depth target assessed only at saved prefix frames; full protocol incomplete'))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='package':
  assert not (DEST/'output/segments').exists(), 'Duplicate snapshot forbidden';transport.package()
 elif len(sys.argv)>1 and sys.argv[1]=='verify':
  folder=Path(sys.argv[2]);assert transport.digest(folder/'transport.json')==transport.digest(transport.DELIVERY/'transport.json');transport.verify(folder)
 else:main(review_only=len(sys.argv)>1 and sys.argv[1]=='review-only')
