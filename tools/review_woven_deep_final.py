"""Separate stronger FINAL publication depth gate; original numerical plan is untouched."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples import woven_deep_impact as deep
from examples import woven_large_plate as base
BASELINE_SHA='9e3383d2eb2049a1ed24c82c84f6d09a920e1f28a540b6a44b27463edf2460bc'
BASELINE_PEAK=.0025261506417153324
TARGET=.005
def filehash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def review(folder,baseline):
 folder=Path(folder);baseline=Path(baseline)
 assert filehash(baseline)==BASELINE_SHA
 previous=list(csv.DictReader(baseline.open(newline='')));measured=max(-float(r['center_mean_displacement_y_m']) for r in previous)
 assert abs(measured-BASELINE_PEAK)<1e-15
 before={p.relative_to(folder).as_posix():filehash(p) for p in folder.rglob('*') if p.is_file()}
 sample,plan=base.load_saved_case(folder);physical=deep.review(folder)
 status=json.loads((folder/'status.json').read_text());summary=json.loads((folder/'summary.json').read_text())
 with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
  complete=bool(summary['completed'] and status['completed'] and int(cp['frame'])==60 and status['frame']==60 and str(cp['signature'])==base.digest(plan))
 rows=list(csv.DictReader((folder/'observables.csv').open(newline='')));peak=max(-float(r['center_mean_displacement_y_m']) for r in rows)
 checks=dict(complete_full_protocol=complete,all61_saved_states=len(rows)==61,original_plan_source=plan['frames']==60 and plan['deep_impact']['initial_energy_J']==40,stable=physical['stable'],clamps=physical['clamp_passed'],independent_material=physical['material_energy_passed'],energy=physical['energy_passed'],peak_at_least_5mm=peak>=TARGET,peak_greater_than_original=peak>BASELINE_PEAK)
 assert before=={p.relative_to(folder).as_posix():filehash(p) for p in folder.rglob('*') if p.is_file()}
 return dict(schema='woven-deep-final-publication-v1',passed=all(checks.values()),checks=checks,baseline_peak_m=BASELINE_PEAK,baseline_observables_sha256=BASELINE_SHA,target_peak_m=TARGET,measured_peak_m=peak,baseline_ratio=peak/BASELINE_PEAK,physical_review=physical,raw_unchanged=True,solver_executed=False,scope='Saved center ROI displacement must reach5mm (~1.979times original2.526mm). Elastic demonstration; no fracture/calibration claim. Failed depth gate requires fresh new numerical source, never threshold relaxation.')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--baseline',type=Path,default=ROOT/'docs/results/completed--woven-plate-impact/observables.csv');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists() or args.output.resolve().is_relative_to(args.folder.resolve()):raise ValueError('Separate fresh review output required')
 result=review(args.folder,args.baseline);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['passed'] else 1)
