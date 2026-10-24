"""Continue an independently certified original f2 copy to actual full61 states."""
import argparse,io,json,os,shutil,subprocess,sys,tarfile,zipfile,platform,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
from certify_fibre_pair_prefix_v2 import require_certified_parent
from run_fibre_pair_prefix import guarded,PINNED_RUNTIME,inventory,capsule_package
from review_fibre_pair_full61 import audit

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['hertz','jkr'],required=True);p.add_argument('--certification-sha256',required=True);a=p.parse_args()
 expected=json.loads((ROOT/'tools/fibre_pair_full61_certifications.json').read_text())[a.model]
 if a.certification_sha256!=expected['receipt_sha256']:raise ValueError('Exact newly certified parent SHA required')
 base=ROOT/'.local/full61';producer=ROOT/'.local/producer-source';logs=base/'logs';logs.mkdir(parents=True,exist_ok=True)
 certificate=ROOT/'.local/certification/certification-receipt-v2.json';admission=json.loads((ROOT/'tools/fibre_pair_recovery_admissions.json').read_text())[a.model]
 archive=ROOT/'.local/original-raw.zip'
 if package.digest(archive)!=admission['zip_sha256']:raise ValueError('Exact original raw ZIP required')
 if platform.python_version()!='3.12.13' or {n:importlib.metadata.version(n) for n in PINNED_RUNTIME}!=PINNED_RUNTIME:raise ValueError('Exact original solver runtime required')
 delivery=base/'original-delivery';delivery.mkdir()
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('Original ZIP CRC differs')
  for item in z.infolist():
   if item.is_dir():continue
   target=package.safe(delivery,item.filename);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
 transport=package.verify(delivery);manifest=package.read(delivery/'raw-archive/archive-manifest.json');original=base/'original-copy';original.mkdir()
 with io.BufferedReader(package.Parts([delivery/'raw-archive'/r['file'] for r in manifest['parts']])) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for item in tar:
   target=package.safe(original,item.name);target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(item) as f:target.write_bytes(f.read())
 original=original/'fibre-pull-off-cache';certified=require_certified_parent(certificate,a.certification_sha256,original,producer)
 if certified['model']!=a.model or certified['original_failed_parent']!=admission:raise ValueError('Certified exact model/admission differs')
 working=base/'capsule';working.mkdir();data=working/'data';shutil.copytree(original/'data',data)
 before=inventory(original);resource=None;failure=None;accepted=False
 try:
  # Worker imports exclusively the original, independently audited producer checkout.
  code="import os,runpy,sys;root=sys.argv[1];output=sys.argv[2];model=sys.argv[3];os.chdir(root);sys.path.insert(0,root);sys.argv=['worker','--model',model,'--output',output,'--resume','--stop-after-frame','60'];runpy.run_module('examples.fibre_pair_dynamic_prefix',run_name='__main__')"
  resource=guarded([sys.executable,'-c',code,str(producer),str(data),a.model],logs/'solver.log')
  import numpy as np
  with np.load(original/'data'/a.model/'checkpoint.npz',allow_pickle=False) as cp:
   original_tables=json.loads(str(cp['tables']))
  with np.load(data/a.model/'checkpoint.npz',allow_pickle=False) as cp:
   complete_tables=json.loads(str(cp['tables']))
  for key in ('rows','records','times'):
   if complete_tables[key][:3]!=original_tables[key]:raise ValueError('Certified original saved prefix was changed or recomputed')
  review=audit(data/a.model,producer);save(working/'independent-full61-review.json',review)
  if not review['passed'] or review['frame']!=60 or review['accepted_steps']!=29212:raise ValueError('Independent full61 audit refused')
  require_certified_parent(certificate,a.certification_sha256,original,producer)
  if inventory(original)!=before:raise ValueError('Original failed raw changed during continuation')
  accepted=True
 except BaseException as exc:
  failure=type(exc).__name__+': '+str(exc);raise
 finally:
  receipt={'schema':'fibre-pair-certified-continuation-full61-v1','kind':'accepted-full61' if accepted else 'diagnostic','never_resumable':not accepted,'source':certified['source'],'request':certified['request'],'model':a.model,'common_request_sha256':admission['common_request_sha256'],'certified_parent_receipt_sha256':a.certification_sha256,'certified_parent':certified['original_failed_parent'],'continued_from_frame':2,'original_steps_not_recomputed':974,'frame':60 if accepted else None,'files':inventory(working),'resource':resource,'failure':failure,'environment':certified['environment'],'continuation_run_id':os.environ.get('GITHUB_RUN_ID'),'continuation_commit':os.environ.get('GITHUB_SHA'),'production_eligible':False,'public_visual_review_passed':False,'comparison_gate_pending':'Both full raw sets must demonstrate >=3 hold frames with Hertz open/JKR attached and actual centre separation difference>0.3mm; no public GIF before pair review.'}
  save(working/'receipt.json',receipt);capsule_package(working,base/'delivery')
 print(json.dumps({'accepted':accepted,'receipt_sha256':package.digest(working/'receipt.json')}))
if __name__=='__main__':main()
