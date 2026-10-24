"""Artifact-only exact-source v6 restart; cloud numerical work never replays0..750ns."""
import argparse,hashlib,importlib.metadata,json,os,re,shutil,signal,subprocess,sys,tarfile,time,urllib.request,urllib.parse,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ADMISSION=ROOT/'tools/metal_v6_cloud1000_admission.json'
DEST=ROOT/'.local/metal-v6-cloud1000-recovery'
RAW=DEST/'raw'
DELIVERY=ROOT/'.local/metal-v6-cloud1000-delivery'
REPO='lililii124/jax-mpm'
PART_LIMIT=128*1024**2
ARCHIVE_LIMIT=192*1024**2


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for data in iter(lambda:stream.read(1024**2),b''):h.update(data)
 return h.hexdigest()

def load(path):return json.loads(Path(path).read_text())

def parent_root():
 seed=DEST/'seed'
 return seed/'parent-proof' if (seed/'parent-proof').is_dir() else seed

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def git_context(template):
 admission=load(ADMISSION)
 if os.environ.get('GITHUB_REPOSITORY')!=REPO or os.environ.get('GITHUB_REF')!=admission['fixed_feature_ref']:raise ValueError('Fixed repository and feature ref required')
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if commit!=os.environ.get('GITHUB_SHA'):raise ValueError('Dispatched SHA differs from checkout')
 names=('tools/recover_metal_v6_cloud1000.py','tools/metal_v6_cloud1000_admission.json','examples/metal_fragment_v6_recover.py','tools/rht_impact_prefix.py',template,'.github/workflows/engineering-previews.yml')
 files={}
 for name in names:
  if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)!=(ROOT/name).read_bytes():raise ValueError('Exact Git blob differs:'+name)
  files[name]=sha(ROOT/name)
 workflow=os.environ.get('GITHUB_WORKFLOW_REF','').split('@')[0]
 if workflow not in (REPO+'/'+template,REPO+'/.github/workflows/engineering-previews.yml'):raise ValueError('Unexpected registered workflow')
 if workflow.endswith('/engineering-previews.yml') and (ROOT/template).read_bytes()!=(ROOT/'.github/workflows/engineering-previews.yml').read_bytes():raise ValueError('Registered workflow must equal frozen feature template')
 return dict(commit=commit,repository=REPO,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'],workflow=workflow,files=files)


def verify_delivery(folder):
 folder=Path(folder);a=load(ADMISSION)
 if sha(folder/'transport.json')!=a['transport_sha256']:raise ValueError('Independently pinned original seed manifest differs')
 manifest=load(folder/'transport.json')
 if manifest['status']!='complete-raw-packaged' or manifest['parts']!=a['parts']:raise ValueError('Incomplete or changed multipart seed')
 for name,info in manifest['parts'].items():
  if not re.fullmatch(r'part-\d{3}\.tarpart',name) or info['bytes']>PART_LIMIT or (folder/name).stat().st_size!=info['bytes'] or sha(folder/name)!=info['sha256']:raise ValueError('Bounded whole seed part differs')
 from tools import rht_impact_prefix as base
 base.verify(folder)
 return manifest


class StripCrossHostAuthorization(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
  if redirected is not None and urllib.parse.urlparse(req.full_url).hostname!=urllib.parse.urlparse(newurl).hostname:redirected.remove_header('Authorization')
  return redirected


def api_request(url):
 request=urllib.request.Request(url,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 return urllib.request.build_opener(StripCrossHostAuthorization()).open(request,timeout=120)


def extract_artifact(archive_path,folder):
 with zipfile.ZipFile(archive_path) as archive:
  entries=archive.infolist()
  if len(entries)!=1:raise ValueError('Exactly one whole payload per artifact required')
  entry=entries[0]
  if entry.filename!=Path(entry.filename).name or entry.filename not in {'transport.json',*load(ADMISSION)['parts']} or entry.file_size>PART_LIMIT:raise ValueError('Unexpected, unsafe or oversized artifact member')
  if ((entry.external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink artifact member rejected')
  if archive.testzip() is not None:raise ValueError('Artifact ZIP CRC failed')
  target=folder/entry.filename
  if target.exists():raise ValueError('Duplicate seed artifact payload')
  with archive.open(entry) as source,target.open('wb') as out:shutil.copyfileobj(source,out,1024**2)
 return target.name


def restore():
 if DEST.exists():raise FileExistsError('Fresh artifact-only restoration required; preserve prior evidence')
 RAW.mkdir(parents=True)
 if shutil.disk_usage(ROOT).free<6*1024**3:raise ValueError('6GiB seed/run/delivery disk admission reserve required')
 context=git_context('.github/workflows/metal-v6-cloud1000-recovery.yml');write(RAW/'control-source.json',context)
 a=load(ADMISSION);run=os.environ['SEED_RUN_ID'];attempt=os.environ['SEED_ATTEMPT'];commit=os.environ['SEED_COMMIT']
 if a.get('parent_admission_complete') is not True:raise ValueError('Independent producer checkpoint admission pending')
 identity=a['producer_identity']
 if (run,attempt,commit)!=(identity['run_id'],identity['attempt'],identity['commit']):raise ValueError('Immutable producer identity differs')
 if not run.isdigit() or not attempt.isdigit() or not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact seed uploader identity required')
 artifacts=json.loads(os.environ['SEED_ARTIFACTS_JSON'])
 if artifacts!=identity['artifacts']:raise ValueError('Independently pinned producer artifact records differ')
 prefix=a.get('artifact_name_prefix','metal-v6-seed750')
 expected={f'{prefix}-{run}-{attempt}-manifest'}|{f'{prefix}-{run}-{attempt}-{Path(name).stem}' for name in a['parts']}
 if len(artifacts)!=len(expected) or {x['name'] for x in artifacts}!=expected or len({str(x['id']) for x in artifacts})!=len(artifacts):raise ValueError('Exact bounded unique seed artifact IDs/names required')
 parts=DEST/'seed-parts';parts.mkdir();records=[]
 for record in artifacts:
  if not str(record['id']).isdigit() or not re.fullmatch('[0-9a-f]{64}',record['sha256']):raise ValueError('Exact artifact ID and archive SHA256 required')
  url=f'https://api.github.com/repos/{REPO}/actions/artifacts/{record["id"]}'
  try:
   with api_request(url) as response:metadata=json.load(response)
   if metadata['expired'] or metadata['name']!=record['name'] or str(metadata['workflow_run']['id'])!=run or metadata['workflow_run']['head_sha']!=commit or metadata['workflow_run']['head_branch']!='work/metal-fragment-v6-cloud' or metadata['digest']!='sha256:'+record['sha256'] or metadata['size_in_bytes']>ARCHIVE_LIMIT:raise ValueError('Artifact producer/archive admission differs')
   archive=parts/f'{record["id"]}.zip';used=0
   with api_request(url+'/zip') as response,archive.open('wb') as out:
    for data in iter(lambda:response.read(1024**2),b''):
     used+=len(data)
     if used>ARCHIVE_LIMIT:raise ValueError('192MiB artifact archive cap exceeded')
     out.write(data)
   if sha(archive)!=record['sha256']:raise ValueError('Whole artifact archive SHA differs')
   member=extract_artifact(archive,parts);records.append(dict(**record,member=member,bytes=used))
  except Exception as error:raise RuntimeError('Seed artifact retrieval/admission failed:'+type(error).__name__) from None
 manifest=verify_delivery(parts);seed=DEST/'seed';seed.mkdir()
 from tools import rht_impact_prefix as base
 reader=base.PartReader(parts/name for name in manifest['parts'])
 with tarfile.open(fileobj=reader,mode='r|') as archive:
  for member in archive:
   relative=Path(member.name)
   if relative.is_absolute() or '..' in relative.parts or '\\' in member.name or ':' in member.name or not member.isfile():raise ValueError('Unsafe seed tar member')
   target=seed/relative;target.parent.mkdir(parents=True,exist_ok=True)
   with archive.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out,1024**2)
 if sha(seed/'seed-context.json')!=a['seed_context_sha256']:raise ValueError('Seed numerical context differs')
 parent=parent_root();kind=a.get('parent_kind','local750')
 receipt_name='segment-receipt.json' if kind=='local750' else 'run-receipt.json'
 if sha(parent/'result/checkpoint.npz')!=a['checkpoint_sha256'] or sha(parent/receipt_name)!=a['parent_receipt_sha256'] or sha(parent/'source.json')!=a['parent_source_sha256']:raise ValueError('Independent checkpoint/receipt/source differs')
 receipt=load(parent/receipt_name)
 if kind=='local750':
  if not receipt['completed'] or receipt['endpoint_ns']!=a['start_ns']:raise ValueError('Accepted complete local producer required')
  for name,info in receipt['files'].items():
   if sha(parent/name)!=info['sha256']:raise ValueError('Whole accepted producer file differs:'+name)
 else:
  if receipt.get('resumable_complete_checkpoint') is not True or receipt.get('physical_gate_failed') is not False or receipt['checkpoint']['sha256']!=a['checkpoint_sha256'] or receipt['checkpoint']['time_ns']!=a['start_ns']:raise ValueError('Only independently admitted complete resource-paused/completed recovery may resume')
 for name,expected_sha in load(parent/'source.json').items():
  if sha(seed/'runtime-source'/name)!=expected_sha:raise ValueError('Frozen numerical source differs:'+name)
 if load(parent/'result/runtime.json')!=a['runtime']:raise ValueError('Producer runtime differs')
 write(RAW/'seed-restoration.json',dict(artifacts=records,transport_sha256=a['transport_sha256'],producer_run_id=run,producer_attempt=attempt,producer_commit=commit,numerical_parent_checkpoint_sha256=a['checkpoint_sha256'],start_ns=a['start_ns'],target_ns=a['stop_ns'],unrelated_legacy_cache_restored=False,accepted_v6_checkpoint_restored=True,original_seed_artifacts_preserved=True))


def install():
 a=load(ADMISSION);dependencies=DEST/'seed/dependencies'
 if sha(dependencies/a['jax_wheel'])!=a['jax_wheel_sha256']:raise ValueError('Producer pure-Python JAX wheel differs')
 subprocess.run([sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:','--report',str(RAW/'installation-report.json'),'-r',str(dependencies/'requirements.txt')],check=True)
 subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(dependencies/a['jax_wheel'])],check=True)
 for name,version in {**a['dependencies'],'jax':a['runtime']['jax']}.items():
  if importlib.metadata.version(name)!=version:raise ValueError('Installed numerical version differs:'+name)
 package=load(DEST/'seed/seed-context.json')['jax'];distribution=importlib.metadata.distribution('jax')
 for name,expected_sha in package['python_files'].items():
  if sha(distribution.locate_file(name))!=expected_sha:raise ValueError('Installed JAX Python/resource bytes differ:'+name)
 (RAW/'runtime-freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
 shutil.copytree(dependencies,RAW/'dependencies')
 shutil.copy2(DEST/'seed/seed-context.json',RAW/'seed-context.json')


def checkpoint_audit(path):
 import numpy as np
 a=load(ADMISSION)
 with np.load(path,allow_pickle=False) as s,np.load(parent_root()/'result/checkpoint.npz',allow_pickle=False) as parent:
  if set(s.files)!=set(parent.files) or not all(np.isfinite(s[name]).all() for name in s.files) or not bool(s['valid']):raise ValueError('Invalid checkpoint carry/finite flag')
  for name in ('mass','volume','physical_body','reference'):
   if not np.array_equal(s[name],parent[name]):raise ValueError('Static geometry/mass changed:'+name)
  tick=round(float(s['time'])*1e12)
  if not a['start_ns']*1000<=tick<=a['stop_ns']*1000 or tick%125000:raise ValueError('Only complete later125ns epoch may resume')
  if len(s['x'])!=26336 or float(s['maximum_contact_closing_speed'])>.005 or float(s['maximum_contact_complementarity_residual'])>.005:raise ValueError('Particle/contact gate failed')
  parameters=load(parent_root()/'result/assumptions.json')['geometry']['material'];J=np.linalg.det(s['F']);D=s['damage']
  if np.any((D<0)|(D>1)) or np.any(J<parameters['minimum_J']) or np.any((J>parameters['maximum_J'])&(D<1)) or np.any(s['cold_work']<0) or np.any(s['fracture_energy']<0) or np.any(s['plastic_strain']<0) or np.any(s['contact_lambda']<0) or len(s['contact_lambda'])>2000000:raise ValueError('Constitutive/domain/history/cache gate failed')
  shear=np.sum(s['tau']**2,axis=(-2,-1))/(4*parameters['shear_modulus']*parameters['density']*np.maximum(1-D,1e-12))
  volume=parameters['bulk_modulus']*(1-J)**2/(2*parameters['density'])*np.where(J>1,1-D,1.)
  T=parameters['reference_temperature']+(s['internal_energy']-s['cold_work']-s['fracture_energy']-shear-volume)/parameters['heat_capacity']
  if not np.isfinite(T).all() or np.any(T<=0) or np.any(s['mass']<=0) or np.any(s['volume']<=0):raise ValueError('Calorimetric/mass validity failed')
  mass=s['mass'];velocity=s['v'];physical=s['physical_body'];initial_v=np.zeros_like(velocity);initial_v[physical==0,1]=-5000.
  momentum=float(np.linalg.norm(np.sum(mass[:,None]*(velocity-initial_v),axis=0)-s['boundary_impulse']))
  kinetic=float(.5*np.sum(mass*np.sum(velocity*velocity,axis=1)));affine=float(.00025**2/8*np.sum(mass*np.sum(s['C']**2,axis=(1,2))));internal=float(np.sum(mass*s['internal_energy']));total=kinetic+affine+internal+float(s['boundary_reservoir'])+float(s['numerical_reservoir'])
  if momentum>1e-10 or abs(total-141.4599609375)>1e-8:raise ValueError('Momentum/accounted energy gate failed')
  return dict(sha256=sha(path),time_ns=tick/1000,particles=len(mass),mass_kg=float(mass.sum()),momentum_residual_Ns=momentum,physical_energy_J=kinetic+affine+internal,boundary_reservoir_J=float(s['boundary_reservoir']),numerical_reservoir_J=float(s['numerical_reservoir']),accounted_energy_J=total,accounted_energy_residual_J=total-141.4599609375,numerical_reservoir_fraction=float(s['numerical_reservoir'])/141.4599609375,closure_history_m_s=float(s['maximum_contact_closing_speed']),KKT_history_m_s=float(s['maximum_contact_complementarity_residual']),minimum_J=float(J.min()),maximum_J=float(J.max()),maximum_temperature_K=float(T.max()))


def recovery_status(checkpoint,validation_error,changed,physical_failed,returncode,reason,stop_ns):
 resource_stop=reason is not None and not physical_failed
 resumable=checkpoint is not None and validation_error is None and not changed and not physical_failed and ((returncode==0 and reason is None) or resource_stop)
 complete=resumable and checkpoint['time_ns']==stop_ns and returncode==0 and reason is None
 return dict(completed=bool(complete),resumable_complete_checkpoint=bool(resumable),physical_gate_failed=bool(physical_failed))


def run():
 import psutil
 a=load(ADMISSION);context=git_context('.github/workflows/metal-v6-cloud1000-recovery.yml')
 if context!=load(RAW/'control-source.json'):raise ValueError('Control source changed after restoration')
 source=RAW/'runtime-source';shutil.copytree(DEST/'seed/runtime-source',source)
 worker=source/'examples/metal_fragment_v6_recover.py'
 if worker.exists() and sha(worker)!=sha(ROOT/'examples/metal_fragment_v6_recover.py'):raise ValueError('Recovery entrypoint changed across lineage')
 shutil.copy2(ROOT/'examples/metal_fragment_v6_recover.py',worker)
 numerical=load(parent_root()/'source.json');write(RAW/'source.json',numerical)
 source_bytes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file()}
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
 platform_check=[sys.executable,'-m','tools.check_multifield_accelerated_contact','--output',str(RAW/'platform-contact-proof')]
 subprocess.run(platform_check,cwd=source,env=env,check=True,timeout=120,stdout=(RAW/'platform-contact.log').open('wb'),stderr=subprocess.STDOUT)
 command=[sys.executable,'-m','examples.metal_fragment_v6_recover','--parent',str(parent_root()),'--parent-sha',a['checkpoint_sha256'],'--end-ns',str(a['stop_ns']),'--frames','16','--epoch-ns','125','--iterations','1024','--contact-tolerance','.005','--output',str(RAW/'result')]
 if a.get('parent_kind','local750')!='local750':command.extend(['--parent-recovery-receipt-sha',a['parent_receipt_sha256']])
 begun=time.monotonic();samples=[];reason=None;code=None;process=None
 def terminate():
  if process is None or process.poll() is not None:return
  try:os.killpg(process.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 try:
  with (RAW/'solve.log').open('wb') as log:
   process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while process.poll() is None:
    try:
     owner=psutil.Process(os.getpid());usage=sum(p.memory_full_info().uss for p in [owner]+owner.children(recursive=True))
    except psutil.NoSuchProcess:
     try:process.wait(timeout=.05);break
     except subprocess.TimeoutExpired:reason='Live memory accounting vanished';usage=0
    except (psutil.AccessDenied,OSError):reason='Process tree USS unavailable';usage=0
    raw_bytes=0
    for p in RAW.rglob('*'):
     try:
      if p.is_file():raw_bytes+=p.stat().st_size
     except FileNotFoundError:pass
    elapsed=time.monotonic()-begun;samples.append(dict(elapsed_s=elapsed,tree_uss_bytes=usage,raw_bytes=raw_bytes))
    if usage>6*1024**3:reason='6GiB process-tree USS ceiling'
    if raw_bytes>2*1024**3-256*1024**2:reason='Whole raw disk watermark'
    if elapsed>1780:reason='30minute numerical ceiling,20second seal reserve'
    if shutil.disk_usage(RAW).free<3*1024**3:reason='3GiB free disk reserve for complete failure transport'
    write(RAW/'monitor-live.json',dict(command=command,reason=reason,samples=samples))
    if reason:terminate();break
    time.sleep(1)
   code=process.wait()
 except BaseException as error:reason=reason or type(error).__name__;raise
 finally:
  terminate()
  if process is not None:code=process.wait()
  changed=[name for name,expected_sha in source_bytes.items() if sha(source/name)!=expected_sha]+['control:'+name for name,expected_sha in context['files'].items() if sha(ROOT/name)!=expected_sha]
  write(RAW/'monitor.json',dict(command=command,returncode=code,reason=reason,source_changed=changed,samples=samples))
 checkpoint=None;validation_error=None
 try:
  checkpoint=checkpoint_audit(RAW/'result/checkpoint.npz')
  if load(RAW/'result/runtime.json')!=a['runtime']:raise ValueError('Consumer runtime differs from producer')
 except Exception as error:validation_error=type(error).__name__
 physical_failed=(RAW/'result/failure.json').exists()
 status=recovery_status(checkpoint,validation_error,changed,physical_failed,code,reason,a['stop_ns'])
 complete=status['completed'];resumable=status['resumable_complete_checkpoint']
 write(RAW/'run-receipt.json',dict(schema='v6-cloud-recovery-v1',completed=complete,resumable_complete_checkpoint=resumable,physical_gate_failed=physical_failed,checkpoint=checkpoint,checkpoint_validation_error=validation_error,requested_start_ns=a['start_ns'],requested_stop_ns=a['stop_ns'],public_media_allowed=False,source_unchanged=not changed,seed_transport_sha256=a['transport_sha256'],source_context=load(RAW/'control-source.json'),runtime_expected=a['runtime'],scope='Only complete admitted epochs may continue. Resource stop never counts as1000ns completion; no replay0..750ns; uncalibrated aluminium-like exploratory model.'))
 if not complete:raise RuntimeError('Target not complete; whole raw and any separately admitted resumable checkpoint retained')


def package():
 from tools import rht_impact_prefix as base
 if not RAW.exists():RAW.mkdir(parents=True);write(RAW/'failure.json',dict(stage='no-restore',completed=False,resumable_complete_checkpoint=False))
 base.DEST=RAW;base.DELIVERY=DELIVERY;base.package()
 manifest=load(DELIVERY/'transport.json')
 if len(manifest['parts'])>16 or any(info['bytes']>PART_LIMIT for info in manifest['parts'].values()):raise ValueError('Bounded128MiB multipart transport cap failed')
 print(sha(DELIVERY/'transport.json'))


def verify_retrieval(folder,expected):
 folder=Path(folder)
 if not re.fullmatch('[0-9a-f]{64}',expected) or sha(folder/'transport.json')!=expected:raise ValueError('Retrieved original manifest differs from pre-upload producer SHA')
 from tools import rht_impact_prefix as base
 base.verify(folder)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('verify-seed','restore','install','run','package','verify-retrieval'));p.add_argument('--folder',type=Path);p.add_argument('--expected');args=p.parse_args()
 try:
  if args.phase=='verify-seed':
   if os.environ.get('GITHUB_ACTIONS')=='true':git_context('.github/workflows/metal-v6-seed-upload.yml')
   verify_delivery(args.folder)
  elif args.phase=='verify-retrieval':verify_retrieval(args.folder,args.expected)
  else:globals()[args.phase]()
 except Exception as error:
  RAW.mkdir(parents=True,exist_ok=True);write(RAW/f'{args.phase}-failure.json',dict(stage=args.phase,error_type=type(error).__name__,completed=False))
  raise


if __name__=='__main__':main()
