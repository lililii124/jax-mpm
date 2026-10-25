"""Complete matched-pair GIFs for user review, retaining any failed contrast gate."""
import argparse,hashlib,io,json,os,shutil,sys,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
from recover_fibre_pair_full61_archive import recover
from run_fibre_pair_prefix import inventory,guarded
from review_fibre_pair_full61 import audit
from review_fibre_pair_full_hold import review as hold_review
HERTZ_RUN=40148583316;HERTZ_SOURCE='b6011cbd36b3bc4af75c6326374256a6812e0fa1'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def extract_delivery_zip(path,dest):
 dest.mkdir()
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Bad full flat ZIP')
  for info in z.infolist():
   if info.is_dir():continue
   p=package.safe(dest,info.filename);p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as s,p.open('xb') as d:shutil.copyfileobj(s,d,1024**2)
def restore_hertz(root,expected):
 delivery=root/'delivery';extract_delivery_zip(ROOT/'.local/full-hertz.zip',delivery);transport=package.verify(delivery);manifest=package.read(delivery/'raw-archive/archive-manifest.json');target=root/'restored';target.mkdir()
 with io.BufferedReader(package.Parts(delivery/'raw-archive'/x['file'] for x in manifest['parts'])) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for item in tar:
   if not item.isfile() or item.name not in manifest['members']:raise ValueError('Wrong complete flat member')
   p=package.safe(target,item.name);p.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(item) as s,p.open('xb') as d:shutil.copyfileobj(s,d,1024**2)
 raw=target/'fibre-pull-off-cache';r=package.read(raw/'receipt.json')
 if package.digest(raw/'receipt.json')!=expected or r['schema']!='fibre-pair-certified-guard-continuation-v1' or r['kind']!='accepted-full61' or r['never_resumable'] or r['frame']!=60 or r['model']!='hertz' or r['failure'] is not None:raise ValueError('Hertz full61 accepted flat receipt required')
 if r['continuation_commit']!=HERTZ_SOURCE or int(r['continuation_run_id'])!=HERTZ_RUN or r['source']['commit']!='bfaaf43cdca9f7ddedcb9726b35a3fc482aa8206' or r['continued_from_frame']!=55 or r['original_steps_not_recomputed']!=26778:raise ValueError('Exact completed flat55 lineage differs')
 expected_f2=package.read(ROOT/'tools/fibre_pair_full61_certifications.json')['hertz']['receipt_sha256']
 if r['original_f2_certification_sha256']!=expected_f2:raise ValueError('Original f2 certification link differs')
 proof=ROOT/'evidence/fibre-pair/hertz-f55-independent-admission.json'
 if package.digest(proof)!=r['parent55_admission_sha256']:raise ValueError('Original independently certified55 proof differs')
 parent=package.read(proof);credential=package.read(raw/'certified-parent55.json')
 if not parent['passed'] or r['parent55_checkpoint_sha256']!=parent['checkpoint_sha256'] or r['parent_guard_receipt_sha256']!=parent['original_failed_receipt_sha256'] or not credential['admitted'] or not credential['original_receipt_remains_never_resumable'] or credential['proof_sha256']!=package.digest(proof):raise ValueError('Wrong immutable failed-parent credential')
 for n,item in r['files'].items():
  if package.record(package.safe(raw,n))!=item:raise ValueError('Flat original raw changed')
 for n,h in manifest['controller']['files'].items():
  source_name='.github/workflows/hertz-frame55-continue.yml' if n=='.github/workflows/engineering-previews.yml' else n
  if package.digest(ROOT/source_name)!=h:raise ValueError('Original flat numerical controller source differs: '+n)
 return raw,r,transport

def main():
 p=argparse.ArgumentParser();p.add_argument('--hertz-receipt-sha256',required=True);p.add_argument('--jkr-receipt-sha256',required=True);a=p.parse_args();base=ROOT/'.local/pair-user-review';base.mkdir();producer=ROOT/'.local/producer-source';official=package.read(ROOT/'.local/media-official-parents.json');cases={};receipts={};before={};proofs={}
 for model in ('hertz','jkr'):
  root=base/model;root.mkdir()
  if model=='hertz':raw,r,proof=restore_hertz(root,a.hertz_receipt_sha256)
  else:
   item=official['jkr'];proof=recover(ROOT/'.local/full-jkr.zip',ROOT/'.local/full-jkr-receipts.zip','jkr',root/'copy-only-recovery',item['artifact']['digest'].removeprefix('sha256:'),item['receipts_artifact']['digest'].removeprefix('sha256:'));raw=root/'copy-only-recovery/restored/fibre-pull-off-cache';r=package.read(raw/'receipt.json')
   if package.digest(raw/'receipt.json')!=a.jkr_receipt_sha256:raise ValueError('JKR exact accepted numerical receipt differs')
  cases[model]=raw/'data'/model;receipts[model]=r;before[model]=inventory(raw);proofs[model]=proof
  for n,h in r['source']['files'].items():
   if package.digest(producer/n)!=h:raise ValueError('Original BFA numerical source differs')
  reviewed=audit(cases[model],producer);save(base/(model+'-independent-full61-review.json'),reviewed)
  if not reviewed['passed']:raise ValueError('Complete raw/material/state/momentum audit refused')
 hold=hold_review(cases['hertz'],cases['jkr']);save(base/'actual-pair-hold-gate.json',hold)
 renders={};media={};logs=base/'logs';logs.mkdir()
 for model in ('hertz','jkr'):
  out=base/'rendered'/model;guarded([sys.executable,str(ROOT/'tools/render_fibre_pair_full61.py'),str(cases[model]),str(out)],logs/(model+'-render.log'));r=package.read(out/'render.json');renders[model]=r
  if r['frames_rendered']!=61 or r['geometry_scale']!=1 or r['resolution']!=[400,400] or r['axes'] or r['annotations'] or r['camera_bounds']!=hold['render_policy']['focus_bounds_m'] or r['camera_elevation_policy']!=.2:raise ValueError('Common true-scale low camera differs')
  if inventory(cases[model].parents[1])!=before[model]:raise ValueError('Original full raw changed')
  from PIL import Image
  states=[]
  with Image.open(out/'simulation.gif') as im:
   if im.n_frames!=61 or im.size!=(400,400):raise ValueError('Exact61 decoded review GIF required')
   for i in range(61):im.seek(i);states.append(hashlib.sha256(im.convert('RGB').tobytes()).hexdigest())
  if len(set(states))<2:raise ValueError('Static review GIF refused')
  media[model]={'gif':package.record(out/'simulation.gif'),'unique_decoded_states':len(set(states)),'review_only':True}
 keys=('camera_position','camera_focal_point','camera_parallel_scale','camera_bounds','color_range','actual_field_range','particle_radius','geometry_scale','resolution')
 if any(renders['hertz'][k]!=renders['jkr'][k] for k in keys):raise ValueError('Actual paired camera/color/radius differs')
 source=package.read(ROOT/'.local/media-source.json')
 for n,h in source['files'].items():
  if package.digest(ROOT/n)!=h:raise ValueError('Review media source changed')
 save(base/'user-review-admission.json',dict(schema='matched-full61-user-review-only-v1',complete_raw_reviews_passed=True,measured_contrast_gate_passed=hold['passed'],actual_hold_gate=hold,original_physical_contrast_threshold_unchanged_m=.0003,media=media,raw_recovery=proofs,review_only=True,user_approval_pending=True,production_eligible=False,public_homepage_allowed=False,scope='Both full61 real state/material audits passed. A failed measured contrast gate remains failed and is disclosed; rendering is solely for the explicit user review request, not engineering acceptance or homepage publication.'))
if __name__=='__main__':main()
