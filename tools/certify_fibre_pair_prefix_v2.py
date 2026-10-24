"""Read-only versioned certification of the two retained successful numerical prefixes."""
import argparse,hashlib,io,json,tarfile,zipfile,platform,importlib.metadata
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
from review_fibre_pair_prefix_v2 import audit

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def certify(model,archive,producer,output):
 admission=json.loads((ROOT/'tools/fibre_pair_recovery_admissions.json').read_text())[model]
 if output.exists():raise ValueError('Fresh separate certification directory required')
 if package.digest(archive)!=admission['zip_sha256']:raise ValueError('Official artifact ZIP digest differs')
 output.mkdir(parents=True);delivery=output/'verified-delivery';delivery.mkdir()
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('ZIP CRC failure')
  for item in z.infolist():
   target=package.safe(delivery,item.filename)
   if item.is_dir():continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with z.open(item) as f:target.write_bytes(f.read())
 transport=package.verify(delivery);manifest=package.read(delivery/'raw-archive/archive-manifest.json')
 if transport['original_receipt_sha256']!=admission['receipt_sha256']:raise ValueError('Independent small receipt differs')
 original=output/'original-copy';original.mkdir()
 paths=[delivery/'raw-archive'/p['file'] for p in manifest['parts']]
 with io.BufferedReader(package.Parts(paths)) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for member in tar:
   target=package.safe(original,member.name);target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as f:target.write_bytes(f.read())
 original=original/'fibre-pull-off-cache'
 receipt=package.read(original/'receipt.json')
 if package.digest(original/'receipt.json')!=admission['receipt_sha256'] or receipt['kind']!='diagnostic' or receipt['never_resumable'] is not True:raise ValueError('Original failed diagnostic identity differs')
 if receipt['source']['commit']!=admission['producer_commit'] or receipt['source']['run_id']!=admission['run_id'] or receipt['model']!=model:raise ValueError('Parent source/run/model differs')
 if receipt['common_request_sha256']!=admission['common_request_sha256']:raise ValueError('Physical plan differs')
 if receipt['environment']!={'python':'3.12.13','packages':__import__('run_fibre_pair_prefix').PINNED_RUNTIME}:raise ValueError('Producer runtime differs')
 if not receipt['resource'] or receipt['resource']['returncode']!=0 or receipt['resource']['reason'] is not None:raise ValueError('Numerical solver did not succeed')
 for n,h in receipt['source']['files'].items():
  if package.digest(package.safe(producer,n))!=h:raise ValueError('Exact original producer source differs: '+n)
 def unchanged():
  for n,r in receipt['files'].items():
   if package.record(package.safe(original,n))!=r:raise ValueError('Original bytes differ: '+n)
  if package.digest(original/'receipt.json')!=admission['receipt_sha256']:raise ValueError('Changed failed receipt')
 unchanged();review=audit(original/'data'/model,producer);unchanged()
 if not review['passed']:raise ValueError('Independent v2 material/momentum audit refused')
 save(output/'independent-f2-review-v2.json',review)
 result={'schema':'fibre-pair-copy-only-certified-prefix-v2','kind':'independently-certified-recovered-prefix','passed':True,'model':model,'frame':2,'accepted_steps':review['accepted_steps'],'original_failed_parent':admission,'source':receipt['source'],'request':receipt['request'],'environment':receipt['environment'],'original_receipt_sha256':admission['receipt_sha256'],'original_status_unchanged':'diagnostic/never_resumable','data_files':receipt['files'],'reviewer_sha256':package.digest(ROOT/'tools/review_fibre_pair_prefix_v2.py'),'controller_sha256':package.digest(Path(__file__)),'review_sha256':package.digest(output/'independent-f2-review-v2.json'),'transport':transport,'continuation_requires':'Explicit v2 certification receipt SHA plus original raw exact inventory/source/runtime/common plan; never the failed receipt alone.','production_eligible':False,'public_visual_review_passed':False}
 result['audit_environment']={'python':platform.python_version(),'numpy':importlib.metadata.version('numpy')}
 save(output/'certification-receipt-v2.json',result);return result

def require_certified_parent(certification, expected_sha256, original, producer):
 if package.digest(certification)!=expected_sha256:raise ValueError('Explicit certified parent receipt SHA differs')
 c=package.read(certification)
 if c['schema']!='fibre-pair-copy-only-certified-prefix-v2' or c['passed'] is not True or c['kind']!='independently-certified-recovered-prefix':raise ValueError('Uncertified failed prefix cannot resume')
 if package.digest(original/'receipt.json')!=c['original_receipt_sha256']:raise ValueError('Original failed receipt differs')
 r=package.read(original/'receipt.json')
 if not r['never_resumable'] or r['source']!=c['source'] or r['request']!=c['request'] or r['environment']!=c['environment']:raise ValueError('Certified parent source/plan/runtime differs')
 for n,v in c['data_files'].items():
  if package.record(package.safe(original,n))!=v:raise ValueError('Certified original raw changed')
 for n,h in c['source']['files'].items():
  if package.digest(package.safe(producer,n))!=h:raise ValueError('Certified producer source changed')
 review=certification.parent/'independent-f2-review-v2.json'
 if package.digest(review)!=c['review_sha256'] or not package.read(review)['passed']:raise ValueError('Certified review changed')
 if package.digest(ROOT/'tools/review_fibre_pair_prefix_v2.py')!=c['reviewer_sha256']:raise ValueError('Certified reviewer changed')
 return c

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['hertz','jkr'],required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--producer-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(certify(a.model,a.archive,a.producer_source,a.output)))
if __name__=='__main__':main()
