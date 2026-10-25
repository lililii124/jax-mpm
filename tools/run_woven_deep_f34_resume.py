"""Exact admitted f34 to f60 continuation; complete guarded raw preservation."""
import json,os,platform,signal,subprocess,sys,time,tarfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import rht_impact_prefix as transport
DEST=ROOT/'.local/woven-deep-f34-resume'
PARENT=ROOT/'.local/woven-deep-f34-parent'
PARENT_COMMIT='c5eccddd9d0857ecc3e0af6e3bfa356409817986'
PARENT_TRANSPORT='52641545227d9513dd048291af4b238dd80f45525fba79cb251b5b6d9c095fe6'
transport.DEST=DEST
transport.DELIVERY=ROOT/'.local/woven-deep-f34-delivery'
def restore_parent(runtime):
 import numpy as np,csv
 from examples import woven_large_plate as base
 from tools.review_woven_large import _numpy_elastic,saved_energy_gate
 assert transport.digest(PARENT/'transport.json')==PARENT_TRANSPORT
 transport.verify(PARENT)
 manifest=json.loads((PARENT/'transport.json').read_text())
 assert manifest['files']['output/plain/deep-sphere-impact/checkpoint.npz']['sha256']=='5b58b2cafb9da612297d77e130a54c7c9609cc6f76fdb796da3611062fbea0d0'
 reader=transport.PartReader(PARENT/name for name in manifest['parts'])
 with tarfile.open(fileobj=reader,mode='r|') as archive:
  for member in archive:
   assert member.name in manifest['files'] and member.isfile()
   target=DEST/member.name
   assert target.resolve().is_relative_to(DEST.resolve())
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as out:shutil.copyfileobj(archive.extractfile(member),out,1024**2)
 before={name:transport.digest(DEST/name) for name in manifest['files']}
 assert all(before[name]==info['sha256'] for name,info in manifest['files'].items())
 source=json.loads((DEST/'source.json').read_text());monitor=json.loads((DEST/'monitor.json').read_text())
 assert source['commit']==PARENT_COMMIT and source['runtime']==runtime
 assert all(source['files'][name]['sha256']==transport.digest(ROOT/name) for name in base.SOURCE_FILES)
 assert monitor['reason']=='90min guard' and monitor['returncode']==-9
 assert monitor['exception'] is None and not monitor['cleanup_errors']
 parent=json.loads((DEST/'parent-admission.json').read_text())
 assert parent['parent_run']==40091612689 and parent['parent_commit']=='d7364d8e6ec1790379a3ed16921e8dbae67c6945'
 assert parent['transport_sha256']=='da69c2ede481617a0dff7f33f61d64a9bdb5fba0b317d5e52a06308655490a34'
 assert parent['source']['runtime']==runtime
 assert all(parent['source']['files'][name]['sha256']==transport.digest(ROOT/name) for name in base.SOURCE_FILES)
 folder=DEST/'output/plain/deep-sphere-impact'
 sample,plan=base.load_saved_case(folder)
 assert plan['deep_impact']['initial_energy_J']==40 and plan['frames']==60
 status=json.loads((folder/'status.json').read_text())
 assert status['frame']==34 and status['steps']==8194 and status['phase']=='running' and status['coupling_valid'] and not status['completed']
 arrays={name:np.load(folder/f'trajectory_{name}.npy',mmap_mode='r',allow_pickle=False) for name in ('x','speed','damage','temperature')}
 with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
  base.validate_checkpoint_control(cp,plan)
  assert int(cp['frame'])==34 and str(cp['signature'])==base.digest(plan)==status['signature']
  tables=json.loads(str(cp['tables']))
  base.validate_saved_prefix(arrays,sample.positions,cp['state_x'],cp['state_v'],tables,34,plan['output_steps'],plan['dt_s'])
  assert int(cp['aggregate'][0])==8194 and abs(float(cp['state_time'])-status['time'])<1e-15
  for name in cp.files:
   value=cp[name]
   if np.issubdtype(value.dtype,np.number):assert np.isfinite(value).all(),name
  assert np.all(np.linalg.det(cp['state_F'])>0)
  edges=(np.abs(sample.positions[:,0])>=.075)|(np.abs(sample.positions[:,2])>=.075)
  roi=np.linalg.norm(sample.positions[:,(0,2)],axis=1)<=.005
  assert np.max(np.abs(cp['state_v'][edges]))<=1e-12
  for frame in range(35):
   assert np.isfinite(arrays['x'][frame]).all()
   assert np.max(np.abs(arrays['x'][frame,edges]-sample.positions[edges]))<=1e-12
   assert np.isclose(np.mean(arrays['x'][frame,roi,1]-sample.positions[roi,1]),tables['rows'][frame]['center_mean_displacement_y_m'],rtol=1e-12,atol=1e-15)
  energy,_,_,_=_numpy_elastic(cp['state_F'],sample.volume,sample.phase,sample.orientation,plan['material_parameters'])
  assert np.isfinite(energy) and energy>=-1e-8
  assert np.isclose(energy,tables['rows'][-1]['stored_elastic_energy_J'],rtol=1e-8,atol=1e-8)
 for filename,key in (('observables.csv','rows'),('diagnostics.csv','diagnostics')):
  rows=list(csv.DictReader((folder/filename).open(newline='')))
  assert len(rows)==35
  assert all(row=={k:'' if v is None else str(v) for k,v in original.items()} for row,original in zip(rows,tables[key]))
 assert all(row['state_valid'] and row['coupling_valid'] and row['minimum_step_J']>0 for row in tables['diagnostics'])
 gate=saved_energy_gate(tables['rows'],float(sample.volume.sum()),40e9);assert gate['passed']
 assert before=={name:transport.digest(DEST/name) for name in before}
 certificate=dict(schema='woven-deep-f34-copy-only-recovery-v1',producer_run=40113303961,producer_commit=PARENT_COMMIT,
  transport_sha256=PARENT_TRANSPORT,admitted_prefix=True,raw_unchanged=True,solver_executed=False,
  frame=34,steps=8194,peak_center_indentation_m=max(-r['center_mean_displacement_y_m'] for r in tables['rows']),
  independent_material_energy_J=energy,energy_gate=gate,source=source,parent_f10=parent,
  original_files=manifest['files'],scope='Only pure wallclock termination certified; failed original artifact is immutable. Fresh working copy may resume; no depth acceptance.')
 transport.write(DEST/'recovery-admission.json',certificate)
 transport.write(DEST/'recovery-prefix-frames.json',tables['trajectory_frames'])
 # Keep original small controller evidence before the working copy advances.
 prior=DEST/'original-controller';prior.mkdir()
 for name in ('source.json','monitor.json','parent-admission.json','solve.log'):
  if (DEST/name).exists():shutil.copy2(DEST/name,prior/name)
 return certificate

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
    if elapsed>=6600:reason='110min guard'
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

def main(certify_only=False):
 import psutil,numpy,jax,jaxlib
 from examples import woven_large_plate as base
 if DEST.exists():raise FileExistsError(DEST)
 DEST.mkdir(parents=True)
 runtime=dict(python=platform.python_version(),numpy=numpy.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,psutil=psutil.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
 assert runtime==dict(python='3.12.13',numpy='2.5.4',jax='0.12.0.1',jaxlib='0.12.0.1',psutil='7.2.2',backend='cpu',x64=True),runtime
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 assert commit==os.environ['GITHUB_SHA'] and os.environ['GITHUB_REF']=='refs/heads/work/woven-deep-impact'
 assert transport.digest(ROOT/'tools/review_woven_deep_final.py')=='35c155ccc88737aac1ea4f219863e92b42c3c843ff5d40a2cd64cdf11e8b6125'
 names=(*base.SOURCE_FILES,'tools/run_woven_deep_f34_resume.py','tools/rht_impact_prefix.py','.github/workflows/woven-deep-f34-resume.yml','tools/review_woven_deep_final.py','evidence/deep-woven/f10-independent-admission.json','.github/workflows/engineering-previews.yml','pyproject.toml')
 sources={}
 for name in names:
  raw=(ROOT/name).read_bytes();blob=subprocess.check_output(['git','show','HEAD:'+name]);assert raw==blob,name
  sources[name]=dict(sha256=transport.digest(ROOT/name),bytes=len(raw))
 if True:
  proof=ROOT/'evidence/deep-woven/f10-independent-admission.json'
  assert transport.digest(proof)=='407c41df3eda449ce343d0fff76ede6d17d3242e4cd6fd2e0c7fcfcd3102c86f'
  admission=json.loads(proof.read_text());assert admission['admitted_prefix'] and admission['raw_unchanged'] and not admission['solver_executed'] and admission['producer_commit']=='d7364d8e6ec1790379a3ed16921e8dbae67c6945' and admission['producer_run']==40091612689
 certificate=restore_parent(runtime)
 if certify_only:return
 transport.write(DEST/'source.json',dict(commit=commit,files=sources,runtime=runtime,fresh_t0=False,parent_run=40113303961,parent_transport_sha256=PARENT_TRANSPORT,physical_validation_passed=False))
 command=[sys.executable,'-m','examples.woven_deep_impact','--output',str(DEST/'output'),'--stop-after-frame','60','--resume','--persistence','controller']
 assert not (DEST/'output/segments').exists()
 guarded_solver(command,psutil)
 # Every admitted state remains byte-identical in the extended trajectory.
 import numpy as np
 with np.load(DEST/'output/plain/deep-sphere-impact/checkpoint.npz',allow_pickle=False) as cp:
  tables=json.loads(str(cp['tables']))
  folder=DEST/'output/plain/deep-sphere-impact'
  sample,plan=base.load_saved_case(folder)
  arrays={name:np.load(folder/f'trajectory_{name}.npy',mmap_mode='r',allow_pickle=False) for name in ('x','speed','damage','temperature')}
  base.validate_checkpoint_control(cp,plan)
  base.validate_saved_prefix(arrays,sample.positions,cp['state_x'],cp['state_v'],tables,60,plan['output_steps'],plan['dt_s'])
  original=certificate['original_files']
  assert all(transport.digest(DEST/name)==info['sha256'] for name,info in original.items()
   if name.startswith('output/') and Path(name).name in ('plan.json','reference.npz','layout.json','layout_arrays.npz','particle_fields.npz'))
  assert tables['trajectory_frames'][:35]==json.loads((DEST/'recovery-prefix-frames.json').read_text())
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
 else:main(certify_only=len(sys.argv)>1 and sys.argv[1]=='certify-only')
