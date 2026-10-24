"""Original64mm/D4mm/30ms complete raw re-render; no new numerical run."""
import argparse,csv,io,json,sys,tarfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import package_fibre_final_media as package
import finish_fibre_pull_off as original_auditor
from run_fibre_pair_prefix import guarded,inventory

ADMISSIONS={'hertz':{'numerical_run':'40064897804','media_run':'40079634901','artifact_id':'12554721709','producer':'81c57439c71cb3f48c575c4d55ee9fcc89b3aa39','zip_sha256':'12592882ed93e686906e159b3c0c2a22402677e6047f4fbee5757373792e2b73','receipt_sha256':'bcaa38cddd165fe1a3ab3f1fccc223b7a7383c8ac8a54a8f8bc501acd446b9e9'},'md':{'numerical_run':'40078855654','media_run':'40092991573','artifact_id':'12560036573','producer':'7c6c0e338fd3eb2a0e80b4c39a105dc36238bf43','zip_sha256':'699515b1280f0d4e4f421165bbbf6e8f1b54eb1da86182685f30d7d111767649','receipt_sha256':'57d807c69bbe6a5d23f30fde1829d560ed744eb79644bae7de6b2e4a5cce9f0f'}}
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.parse_args();base=ROOT/'.local/original-low';base.mkdir();logs=base/'logs';logs.mkdir();cases={};caches={};proofs={};before={}
 for model,a in ADMISSIONS.items():
  archive=ROOT/'.local'/('original-'+model+'.zip')
  if package.digest(archive)!=a['zip_sha256']:raise ValueError('Only original complete media artifact ZIP accepted')
  delivery=base/model/'delivery';delivery.mkdir(parents=True)
  with zipfile.ZipFile(archive) as z:
   if z.testzip() is not None:raise ValueError('Original ZIP CRC failed')
   for item in z.infolist():
    if item.is_dir():continue
    target=package.safe(delivery,item.filename);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
  proof=package.verify(delivery);manifest=package.read(delivery/'raw-archive/archive-manifest.json')
  if proof['original_receipt_sha256']!=a['receipt_sha256']:raise ValueError('Original raw receipt differs')
  raw=base/model/'original-copy';raw.mkdir()
  with io.BufferedReader(package.Parts([delivery/'raw-archive'/r['file'] for r in manifest['parts']])) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
   for item in tar:
    target=package.safe(raw,item.name);target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(item) as f:target.write_bytes(f.read())
  cache=raw/'fibre-pull-off-cache';receipt=package.read(cache/'receipt.json')
  if package.digest(cache/'receipt.json')!=a['receipt_sha256'] or str(receipt['source']['run_id'])!=a['numerical_run']:raise ValueError('Original producer/run receipt differs')
  # Original producer auditor separately rechecks complete source/runtime/parents/raw/CP leaves/frame records.
  guarded([sys.executable,str(ROOT/'tools/finish_fibre_pull_off.py'),'--audit-worker','--cache',str(cache),'--receipt-sha',a['receipt_sha256'],'--producer-root',str(ROOT/'.local'/('producer-'+model)),'--producer-commit',a['producer'],'--output',str(base/(model+'-original-prefix-review.json'))],logs/(model+'-original-audit.log'))
  folder=cache/'data'/model;material=original_auditor.independent_material_review(folder);save(base/(model+'-material-review.json'),material)
  if not material['passed']:raise ValueError('Independent original final material/contact review refused')
  s=package.read(folder/'summary.json');g=s['definition']['geometry']
  if not s['completed'] or s['steps']!=24495 or s['target_steps']!=24495 or s['particles']!=[202240] or g['length_m']!=.064 or g['radius_m']!=.002 or s['definition']['duration_s']!=.03 or s['definition']['steps']!=24495 or s['definition']['frames']!=60 or s['definition']['saved_states']!=61:raise ValueError('Exact original64mm/D4mm/30ms case required')
  cases[model]=folder;caches[model]=cache;proofs[model]=proof;before[model]=inventory(cache)
 hsummary=package.read(cases['hertz']/'summary.json');msummary=package.read(cases['md']/'summary.json')
 for key in ('geometry','grid','material','protocol','frame_steps'):
  if hsummary['definition'][key]!=msummary['definition'][key]:raise ValueError('Original physical geometry/loading differs')
 rows={m:list(csv.DictReader((f/'observables.csv').open())) for m,f in cases.items()};contrasts=[]
 for i,(h,m) in enumerate(zip(rows['hertz'],rows['md'])):
  if h['time']!=m['time']:raise ValueError('Original physical clocks differ')
  sep=lambda r:float(r['upper_centre_displacement_y_m'])-float(r['lower_centre_displacement_y_m'])
  difference=abs(sep(h)-sep(m))
  if h['contact_active']=='False' and m['contact_active']=='True' and difference>.0003:contrasts.append({'frame':i,'time_s':float(h['time']),'actual_central_separation_difference_m':difference,'hertz_gap_m':float(h['contact_gap_m']),'md_contact_indentation_m':float(m['contact_gap_m'])})
 if len(rows['hertz'])!=61 or len(rows['md'])!=61 or len(contrasts)<3:raise ValueError('Actual original complete pair visual contrast gate refused')
 save(base/'original-physical-contrast.json',{'passed':True,'qualifying_saved_frames':contrasts,'case':'Original64mm/D4mm/30ms prescribed-grip; new camera only, not32mm40ms compliant fixture.'})
 renders={};media={}
 for model in ADMISSIONS:
  out=base/'rendered'/model
  guarded([sys.executable,str(ROOT/'tools/render_original_fibre_full61_low.py'),str(cases[model]),str(out)],logs/(model+'-render.log'))
  r=package.read(out/'render.json');renders[model]=r
  if r['frames_rendered']!=61 or r['geometry_scale']!=1 or r['resolution']!=[400,400] or r['axes'] or r['annotations'] or r['field']!='family_id' or r['camera_elevation_policy']!=.2:raise ValueError('Actual truegeometry ParaView policy differs')
  if inventory(caches[model])!=before[model]:raise ValueError('Original raw changed while rendering')
  media[model]={'gif':package.record(out/'simulation.gif'),'render':package.record(out/'render.json')}
 keys=('camera_position','camera_focal_point','camera_parallel_scale','camera_bounds','color_range','actual_field_range','particle_radius','geometry_scale','resolution')
 if any(renders['hertz'][k]!=renders['md'][k] for k in keys):raise ValueError('Different actual camera/color/geometry display')
 save(base/'original-low-media-admission.json',{'schema':'original64mm30ms-real61-low-camera-v1','passed':True,'admissions':ADMISSIONS,'raw_transport':proofs,'original_case':'64mm-long4mm-diameter/30ms/202240pts/24495steps/61states,original prescribed grips','numerical_rerun':False,'new_view_only':True,'not_new_40ms_compliant_fixture':True,'same_actual_camera':{k:renders['hertz'][k] for k in keys},'physical_contrast':contrasts,'media':media,'production_eligible':False,'public_visual_review_passed':False,'scope':'Complete exact original raw/source/runtime/parent/checkpoint/frame/material/contact audit. Actual all61 ParaView particle frames, uniform lowcamera,400px<9MiB GIF. True6mm detail crops64mm rods and parts of4mm sections; human occlusion/frame review required.'})
if __name__=='__main__':main()
