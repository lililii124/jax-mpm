"""Continue the independently certified saved Hertz55 copy to60, without prefix replay."""
import argparse,hashlib,io,json,os,platform,shutil,subprocess,sys,tarfile,zipfile,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
from recover_fibre_pair_guard_prefix import recover
from audit_hertz_guard_frame55 import audit,CP_SHA,RECEIPT_SHA,F2_CP_SHA
from run_fibre_pair_prefix import guarded,PINNED_RUNTIME,inventory
from review_fibre_pair_full61 import audit as full_audit
BASE=ROOT/'.local/hertz55';PROOF='evidence/fibre-pair/hertz-f55-independent-admission.json'
PARENT_RAW_SHA='f9170b65467c8264adb60e3f0c396134c9cef8b9490b8a1bec9fdaff10a39728'
PARENT_SMALL_SHA='d5f57dfc34a49129e049659b54b7b92212b9ae689cd5fd1975983cba9421f4b9'
F2_ZIP_SHA='bfbb4da3273e92af96ed334a450c8568cc24342c544d9bbe3599ac0859ad1780'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def source():
 head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip();template='.github/workflows/hertz-frame55-continue.yml';active='.github/workflows/engineering-previews.yml'
 if head!=os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF')!='refs/heads/work/fibre-adhesion-pair-dynamic' or os.environ.get('GITHUB_REPOSITORY')!='lililii124/jax-mpm':raise ValueError('Exact dispatched source/ref/repository required')
 if (ROOT/template).read_bytes()!=(ROOT/active).read_bytes():raise ValueError('Registered override differs')
 names=['tools/run_hertz_frame55_continue.py','tools/audit_hertz_guard_frame55.py','tools/recover_fibre_pair_guard_prefix.py','tools/package_fibre_final_media.py','tools/run_fibre_pair_prefix.py','tools/finish_fibre_pull_off.py','tools/review_fibre_pair_full61.py','tools/fibre_pair_full61_certifications.json',PROOF,template,active];files={}
 for n in names:
  if (ROOT/n).read_bytes()!=subprocess.check_output(['git','show',head+':'+n]):raise ValueError('Controller Git bytes changed: '+n)
  files[n]=package.digest(ROOT/n)
 return dict(commit=head,files=files,run_id=os.environ['GITHUB_RUN_ID'],attempt=os.environ['GITHUB_RUN_ATTEMPT'])
def extract_f2():
 path=ROOT/'.local/hertz-original-f2.zip'
 if package.digest(path)!=F2_ZIP_SHA:raise ValueError('Exact original f2 official ZIP required')
 delivery=BASE/'f2-delivery';delivery.mkdir()
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid original f2 ZIP')
  for item in z.infolist():
   if item.is_dir():continue
   p=package.safe(delivery,item.filename);p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(item) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
 package.verify(delivery);m=package.read(delivery/'raw-archive/archive-manifest.json');target=BASE/'f2-checkpoint.npz'
 with io.BufferedReader(package.Parts(delivery/'raw-archive'/x['file'] for x in m['parts'])) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for item in tar:
   if item.name=='fibre-pull-off-cache/data/hertz/checkpoint.npz':
    with tar.extractfile(item) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,1024**2)
    if package.record(target)!=m['members'][item.name] or package.digest(target)!=F2_CP_SHA:raise ValueError('Original f2 checkpoint changed')
    return target
 raise ValueError('Missing original f2 checkpoint')
def main():
 if BASE.exists():raise ValueError('Fresh flat continuation output required')
 if platform.python_version()!='3.12.13' or {n:importlib.metadata.version(n) for n in PINNED_RUNTIME}!=PINNED_RUNTIME:raise ValueError('Exact original numerical runtime required')
 if shutil.disk_usage(ROOT).free<8*1024**3:raise ValueError('8GiB parent/working/transport disk reserve required')
 context=source();BASE.mkdir();save(BASE/'continuation-source.json',context);producer=ROOT/'.local/producer-source'
 recovered=recover(ROOT/'.local/hertz55-raw.zip',ROOT/'.local/hertz55-receipts.zip','hertz',BASE/'original-guard-recovery',PARENT_RAW_SHA,PARENT_SMALL_SHA)
 original=BASE/'original-guard-recovery/restored/fibre-pull-off-cache';f2=extract_f2();report=audit(original,producer,f2);save(BASE/'independent-parent55-review.json',report);frozen=package.read(ROOT/PROOF)
 if not frozen['passed'] or not frozen['allow_resume_complete_saved_checkpoint'] or not report['passed'] or report['checkpoint_sha256']!=frozen['checkpoint_sha256'] or report['original_failed_receipt_sha256']!=frozen['original_failed_receipt_sha256'] or report['raw_files']!=frozen['raw_files']:raise ValueError('Exact independently admitted complete55 copy required')
 before=inventory(original);working=BASE/'capsule';working.mkdir();shutil.copytree(original/'data',working/'data');folder=working/'data/hertz';old_status=(folder/'status.json').read_bytes();save(folder/'status.json',report['isolated_resume_status']);save(BASE/'isolated-status-reconstruction.json',dict(original_sha256=hashlib.sha256(old_status).hexdigest(),working_sha256=package.digest(folder/'status.json'),reconstructed_only_from_exact_admitted_CP=True,original_raw_unchanged=True))
 save(working/'certified-parent55.json',dict(schema='hertz-frame55-copy-only-credential-v1',admitted=True,proof_sha256=package.digest(ROOT/PROOF),repeat_independent_parent_review_sha256=package.digest(BASE/'independent-parent55-review.json'),original_failed_receipt_sha256=RECEIPT_SHA,checkpoint_sha256=CP_SHA,original_receipt_remains_never_resumable=True,continued_from_frame=55,previous_steps_not_recomputed=26778))
 import numpy as np
 with np.load(original/'data/hertz/checkpoint.npz',allow_pickle=False) as cp:old_tables=json.loads(str(cp['tables']))
 resource=None;failure=None;accepted=False;logs=BASE/'logs';logs.mkdir()
 try:
  code="import os,runpy,sys;root=sys.argv[1];out=sys.argv[2];os.chdir(root);sys.path.insert(0,root);sys.argv=['worker','--model','hertz','--output',out,'--resume','--stop-after-frame','60'];runpy.run_module('examples.fibre_pair_dynamic_prefix',run_name='__main__')"
  resource=guarded([sys.executable,'-c',code,str(producer),str(working/'data')],logs/'solver.log')
  with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
   tables=json.loads(str(cp['tables']))
   if int(cp['frame'])!=60 or int(cp['progress_0'])!=29212 or any(tables[k][:56]!=old_tables[k] for k in ('rows','records','times')):raise ValueError('Original55 saved states changed/recomputed or target incomplete')
  review=full_audit(folder,producer);save(working/'independent-full61-review.json',review)
  if not review['passed'] or inventory(original)!=before:raise ValueError('Full61 physical audit or original immutability failed')
  accepted=True
 except BaseException as exc:
  failure=type(exc).__name__+': '+str(exc);raise
 finally:
  parent=package.read(original/'receipt.json');receipt=dict(schema='fibre-pair-certified-guard-continuation-v1',kind='accepted-full61' if accepted else 'diagnostic',never_resumable=not accepted,source=parent['source'],request=parent['request'],model='hertz',frame=60 if accepted else None,continued_from_frame=55,original_steps_not_recomputed=26778,parent_guard_receipt_sha256=RECEIPT_SHA,parent55_checkpoint_sha256=CP_SHA,parent55_admission_sha256=package.digest(ROOT/PROOF),original_f2_certification_sha256=parent['certified_parent_receipt_sha256'],continuation_run_id=os.environ['GITHUB_RUN_ID'],continuation_commit=context['commit'],cache_key='artifact-only-hertz-full61-'+os.environ['GITHUB_RUN_ID']+'-'+os.environ['GITHUB_RUN_ATTEMPT'],logical_artifact_identity_not_actions_cache=True,environment=parent['environment'],files=inventory(working),resource=resource,failure=failure,production_eligible=False,public_visual_review_passed=False,scope='Flat actual data55to60 only; independently certified original saved55 and original failed receipt remain immutable; actual matched hold and ParaView gates remain separate.')
  save(working/'receipt.json',receipt)
  import finish_fibre_pull_off as transport
  delivery=BASE/'delivery';delivery.mkdir()
  archive=transport.archive_complete_cache(working,receipt,delivery/'raw-archive',package.digest(working/'receipt.json'),context)
  if archive['compressed_bytes']>2*1024**3:raise ValueError('Complete flat raw archive exceeds original2GiB cap')
  save(delivery/'delivery-manifest.json',dict(schema='fibre-pair-flat-full61-artifact-v1',passed=True,raw_case='fibre-adhesion-pair-hertz55to60',complete_raw_archive_sha256=archive['sha256'],original_receipt_sha256=package.digest(working/'receipt.json'),files={'raw-archive/archive-manifest.json':package.record(delivery/'raw-archive/archive-manifest.json')},never_resumable=receipt['never_resumable'],diagnostic_only=receipt['never_resumable'],production_eligible=False,public_visual_review_passed=False))
  save(BASE/'local-archive-verification.json',package.verify(delivery))
if __name__=='__main__':main()
