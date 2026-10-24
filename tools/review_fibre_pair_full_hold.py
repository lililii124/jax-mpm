"""Mandatory measured full-pair hold gate before rendering or public replacement."""
import argparse,csv,json
from pathlib import Path

def review(hertz,jkr):
 cases={m:Path(p) for m,p in [('hertz',hertz),('jkr',jkr)]};summaries={m:json.loads((p/'summary.json').read_text()) for m,p in cases.items()};rows={m:list(csv.DictReader((p/'observables.csv').open())) for m,p in cases.items()}
 for m,s in summaries.items():
  if not s['completed'] or s['steps']!=29212 or len(rows[m])!=61 or s['definition']['model']!=m:raise ValueError('Actual complete original full61 pair required')
 if summaries['hertz']['common_request_sha256']!=summaries['jkr']['common_request_sha256']:raise ValueError('Different physical loading/geometry/camera')
 first,last=summaries['hertz']['definition']['protocol']['times_s'][3:5];samples=[]
 for h,j in zip(rows['hertz'],rows['jkr']):
  if h['time']!=j['time']:raise ValueError('Different physical clocks')
  t=float(h['time']);gap=lambda r:float(r['upper_centre_displacement_y_m'])-float(r['lower_centre_displacement_y_m'])
  if first<=t<=last:samples.append({'time_s':t,'hertz_actual_centre_displacement_m':gap(h),'jkr_actual_centre_displacement_m':gap(j),'actual_difference_m':abs(gap(h)-gap(j)),'hertz_open':h['contact_active']=='False','jkr_attached':j['contact_active']=='True','hertz_contact_force_N':float(h['contact_force_N']),'jkr_contact_force_N':float(j['contact_force_N'])})
 passed=[r for r in samples if r['actual_difference_m']>.0003 and r['hertz_open'] and r['jkr_attached']]
 undamaged=all(float(r['max_damage'])==0 for table in rows.values() for r in table)
 return {'schema':'fibre-pair-real-full61-hold-admission-v1','passed':len(passed)>=3 and undamaged,'actual_qualifying_hold_frames':len(passed),'hold_samples':samples,'material_undamaged':undamaged,'common_request_sha256':summaries['hertz']['common_request_sha256'],'render_policy':{'renderer':'ParaView','frames':61,'direction':[1.4,.2,1.8],'focus_bounds_m':[-.003,.003,-.0018,.0042,-.003,.003],'field':'family_id','axes':False,'annotations':False,'geometry_scale':1},'production_eligible':False,'public_visual_review_passed':False,'scope':'Actual dynamics only; independent raw/material reviews must also pass. Render only after passed, then require same actual camera and human frame review.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hertz',type=Path,required=True);p.add_argument('--jkr',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=review(a.hertz,a.jkr);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
 if not r['passed']:raise SystemExit('Measured hold contrast gate refused; no GIF/public replacement')
if __name__=='__main__':main()
