"""Recover a guard-paused original prefix archive; never admit its failed receipt."""
import argparse,hashlib,io,json,shutil,tarfile,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import package_fibre_final_media as package
RUNS={'hertz':40113622359}
SOURCE='0d3029212c7ad6f494e62637494d1951ba048368'
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def recover(raw_zip,small_zip,model,output,raw_sha256,small_sha256):
 if output.exists():raise ValueError('New independent recovery directory required')
 for p,h in ((raw_zip,raw_sha256),(small_zip,small_sha256)):
  if package.digest(p)!=h:raise ValueError('Official immutable ZIP digest differs')
 output.mkdir(parents=True);delivery=output/'delivery';delivery.mkdir();small=output/'original-small';small.mkdir()
 for archive,dest in ((raw_zip,delivery),(small_zip,small)):
  with zipfile.ZipFile(archive) as z:
   if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Invalid original ZIP')
   for item in z.infolist():
    if item.is_dir():continue
    p=package.safe(dest,item.filename);p.parent.mkdir(parents=True,exist_ok=True)
    with z.open(item) as s,p.open('xb') as d:shutil.copyfileobj(s,d,1024**2)
 receipt_path=small/'full61/capsule/receipt.json';receipt=package.read(receipt_path);receipt_sha=package.digest(receipt_path)
 if receipt['kind']!='diagnostic' or receipt['never_resumable'] is not True or receipt['model']!='hertz' or receipt['frame'] is not None or '260minute_wall_guard' not in str(receipt['failure']):raise ValueError('Only exact original guard-paused diagnostic applies')
 guard=package.read(small/'full61/logs/solver.json')
 if guard['reason']!='260minute_wall_guard' or guard['returncode']!=-9 or guard['monitor_error'] is not None or guard['peak_tree_uss_bytes']>6*1024**3:raise ValueError('Only independently bounded wall-guard stop applies')
 if receipt['continuation_commit']!=SOURCE or int(receipt['continuation_run_id'])!=RUNS[model] or receipt['source']['commit']!='bfaaf43cdca9f7ddedcb9726b35a3fc482aa8206':raise ValueError('Wrong immutable numerical producer')
 if 'cache_key' in receipt:raise ValueError('Recovery applies only to original missing-cache-key packaging failure')
 records={'fibre-pull-off-cache/'+n:r for n,r in receipt['files'].items()};records['fibre-pull-off-cache/receipt.json']=package.record(receipt_path)
 parts=sorted((delivery/'raw-archive').glob('raw.tar.gz.part-*'));expected={'raw-archive/'+p.name for p in parts}
 if not parts or {p.relative_to(delivery).as_posix() for p in delivery.rglob('*') if p.is_file()}!=expected:raise ValueError('Unexpected partial-delivery files')
 whole=hashlib.sha256();part_records=[];compressed=0
 for i,p in enumerate(parts):
  if p.name!=f'raw.tar.gz.part-{i:04d}' or not 0<p.stat().st_size<=192*1024**2:raise ValueError('Incomplete ordered raw parts')
  r=package.record(p);part_records.append(dict(file=p.name,**r));compressed+=r['bytes']
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):whole.update(b)
 if compressed>2*1024**3:raise ValueError('Original full raw cap exceeded')
 restored=output/'restored';restored.mkdir();seen=set()
 with io.BufferedReader(package.Parts(parts)) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for member in tar:
   if not member.isfile() or member.name not in records or member.name in seen:raise ValueError('Unexpected or duplicate original raw member')
   seen.add(member.name);target=package.safe(restored,member.name);target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();count=0
   with tar.extractfile(member) as s,target.open('xb') as d:
    for b in iter(lambda:s.read(1024**2),b''):d.write(b);h.update(b);count+=len(b)
   if {'bytes':count,'sha256':h.hexdigest()}!=records[member.name]:raise ValueError('Original whole raw member changed')
 if seen!=set(records):raise ValueError('Missing original whole raw member')
 raw=restored/'fibre-pull-off-cache'
 if (raw/'receipt.json').read_bytes()!=receipt_path.read_bytes():raise ValueError('Original receipt changed')
 for n in ('receipt.json',):
  if (raw/n).read_bytes()!=(small/'full61/capsule'/n).read_bytes():raise ValueError('Separate original evidence differs')
 manifest=dict(schema='fibre-pull-off-complete-raw-archive-v1',format='tar.gz split into ordered raw byte chunks',producer=receipt['source'],controller={'recovery_only':True},cache_key=None,cache_key_not_applicable=True,original_missing_cache_key_preserved=True,receipt_sha256=receipt_sha,compressed_bytes=compressed,sha256=whole.hexdigest(),parts=part_records,members=records,all_original_raw_files_archived=True,uncompressed_file_bytes=sum(r['bytes'] for r in records.values()),consumer_backup_verified=True)
 save(delivery/'raw-archive/archive-manifest.json',manifest)
 save(delivery/'delivery-manifest.json',dict(schema='fibre-pair-full61-copy-only-recovery-v1',passed=True,complete_raw_archive_sha256=whole.hexdigest(),original_receipt_sha256=receipt_sha,files={'raw-archive/archive-manifest.json':package.record(delivery/'raw-archive/archive-manifest.json')},original_failed_workflow_preserved=True,production_eligible=False,public_visual_review_passed=False))
 verified=package.verify(delivery);proof=dict(schema='fibre-pair-guard-prefix-transport-only-v1',passed=True,requires_separate_prefix_physical_admission=True,original_receipt_remains_never_resumable=True,model=model,original_run_id=RUNS[model],original_workflow_packaging_failure_preserved=True,original_receipt_sha256=receipt_sha,raw_zip_sha256=raw_sha256,small_zip_sha256=small_sha256,original_numerical_source=SOURCE,original_members_checked=len(records),whole_raw_archive_sha256=whole.hexdigest(),restored_raw=str(raw.resolve()),package_verification=verified,mechanical_steps_executed=0,original_receipt_modified=False,pair_contrast_not_yet_admitted=True,public_media_allowed=False)
 save(output/'independent-recovery-admission.json',proof);return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-zip',required=True,type=Path);p.add_argument('--small-zip',required=True,type=Path);p.add_argument('--model',required=True,choices=RUNS);p.add_argument('--output',required=True,type=Path);p.add_argument('--raw-sha256',required=True);p.add_argument('--small-sha256',required=True);a=p.parse_args();print(json.dumps(recover(a.raw_zip,a.small_zip,a.model,a.output,a.raw_sha256,a.small_sha256)))
