"""Admit complete exact raw pair, then render actual61 states with one camera."""
import argparse,io,json,os,shutil,sys,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
from run_fibre_pair_prefix import guarded,inventory
from review_fibre_pair_full61 import audit
from review_fibre_pair_full_hold import review as hold_review

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def restore(model,receipt_sha,root):
 archive=ROOT/'.local'/('full-'+model+'.zip');delivery=root/'delivery';delivery.mkdir()
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('Full raw ZIP CRC failed')
  for item in z.infolist():
   if item.is_dir():continue
   target=package.safe(delivery,item.filename);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
 transport=package.verify(delivery);manifest=package.read(delivery/'raw-archive/archive-manifest.json');raw=root/'raw';raw.mkdir()
 if transport['original_receipt_sha256']!=receipt_sha:raise ValueError('Exact completed full receipt SHA differs')
 with io.BufferedReader(package.Parts([delivery/'raw-archive'/p['file'] for p in manifest['parts']])) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
  for item in tar:
   target=package.safe(raw,item.name);target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(item) as f:target.write_bytes(f.read())
 raw=raw/'fibre-pull-off-cache';receipt=package.read(raw/'receipt.json')
 if package.digest(raw/'receipt.json')!=receipt_sha or receipt['kind']!='accepted-full61' or receipt['never_resumable'] or receipt['model']!=model or receipt['frame']!=60:raise ValueError('Complete admitted full61 required')
 if receipt['source']['commit']!='bfaaf43cdca9f7ddedcb9726b35a3fc482aa8206' or receipt['continuation_commit']!='0d3029212c7ad6f494e62637494d1951ba048368':raise ValueError('Exact numerical producer and continuation source required')
 expected=package.read(ROOT/'tools/fibre_pair_full61_certifications.json')[model]
 if receipt['certified_parent_receipt_sha256']!=expected['receipt_sha256'] or receipt['continued_from_frame']!=2 or receipt['original_steps_not_recomputed']!=974:raise ValueError('Certified no-recompute parent chain differs')
 for n,r in receipt['files'].items():
  if package.record(package.safe(raw,n))!=r:raise ValueError('Original full raw inventory differs')
 return raw,receipt,transport

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hertz-receipt-sha256',required=True);p.add_argument('--jkr-receipt-sha256',required=True);a=p.parse_args();base=ROOT/'.local/pair-media';base.mkdir();producer=ROOT/'.local/producer-source';logs=base/'logs';logs.mkdir();cases={};receipts={};before={};transports={}
 for model,sha in [('hertz',a.hertz_receipt_sha256),('jkr',a.jkr_receipt_sha256)]:
  root=base/model;root.mkdir();raw,r,t=restore(model,sha,root);cases[model]=raw/'data'/model;receipts[model]=r;transports[model]=t;before[model]=inventory(raw)
  for n,h in r['source']['files'].items():
   if package.digest(package.safe(producer,n))!=h:raise ValueError('Exact original numerical source snapshot differs')
  material=audit(cases[model],producer);save(base/(model+'-independent-full61-review.json'),material)
  if not material['passed']:raise ValueError('Independent full61 raw/material/momentum audit refused')
 hold=hold_review(cases['hertz'],cases['jkr']);save(base/'actual-pair-hold-gate.json',hold)
 if not hold['passed']:raise ValueError('No GIF: actual>=3 hold frames physical contrast gate refused')
 renders={};media={}
 for model in ('hertz','jkr'):
  out=base/'rendered'/model
  guarded([sys.executable,str(ROOT/'tools/render_fibre_pair_full61.py'),str(cases[model]),str(out)],logs/(model+'-render.log'))
  render=package.read(out/'render.json');renders[model]=render
  if render['frames_rendered']!=61 or render['geometry_scale']!=1 or render['resolution']!=[400,400] or render['axes'] or render['annotations'] or render['field']!='family_id' or render['camera_bounds']!=hold['render_policy']['focus_bounds_m'] or render['camera_elevation_policy']!=.2:raise ValueError('Identical genuine low-angle ParaView policy refused')
  if inventory(cases[model].parents[1])!=before[model]:raise ValueError('Original complete raw changed while rendering')
  media[model]={'gif':package.record(out/'simulation.gif'),'render':package.record(out/'render.json'),'preview_receipt':package.record(out/'preview-receipt.json')}
 keys=('camera_position','camera_focal_point','camera_parallel_scale','camera_bounds','color_range','actual_field_range','particle_radius','geometry_scale','resolution')
 if any(renders['hertz'][k]!=renders['jkr'][k] for k in keys):raise ValueError('Actual camera/color/particle display differ between models')
 save(base/'media-admission.json',{'schema':'matched-full40ms-real61-pair-media-v1','passed':True,'raw_transport':transports,'full_receipt_sha256':{'hertz':a.hertz_receipt_sha256,'jkr':a.jkr_receipt_sha256},'same_actual_camera_keys':{k:renders['hertz'][k] for k in keys},'media':media,'actual_hold_gate':hold,'all61states_preserved':True,'gif_duration_per_frame_ms':120,'production_eligible':False,'public_visual_review_passed':False,'scope':'Both complete exact raw/material/impulse reviews and actual hold>0.3mm gate passed before actual ParaView rendering. Human visible-frame review and separate publication authorization remain required.'})
 print('PASS: complete actual pair admitted and rendered; pending human visual review.')
if __name__=='__main__':main()
