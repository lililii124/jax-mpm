"""Quasistatic common compliant-fixture screen: Hertz versus reliable adhesion.

Analytic local laws plus linear clamped circular beams, not synthetic MPM frames.
Finite mass/damping and large deformation require the subsequent real MPM pilot.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def geometry_stiffness(young):
    return 192*young*(math.pi*.001**4/4)/(2*.026**3)


def jkr_branch(young,work,k):
    R=.001;E=young/(2*(1-.25**2));S=math.sqrt(2*math.pi*work/E);Q=math.sqrt(8*math.pi*work*E)
    delta=lambda a:a*a/R-S*math.sqrt(a)
    force=lambda a:4*E*a**3/(3*R)-Q*a**1.5
    critical_disp=(math.pi*work*R*R/(8*E))**(1/3)
    zero=(9*math.pi*work*R*R/(2*E))**(1/3)
    slope=lambda a:4*E*a*a/R-1.5*Q*math.sqrt(a)+k*(2*a/R-.5*S/math.sqrt(a))
    stable_min=brentq(slope,critical_disp,zero,xtol=1e-16)
    critical_drive=-delta(stable_min)-force(stable_min)/k
    def solve(drive,was_attached):
        if not was_attached and drive>0 or drive>=critical_drive:return {'force_N':0.,'indentation_m':-drive,'radius_m':0.,'active':False}
        residual=lambda a:force(a)+k*(drive+delta(a))
        upper=.0002
        while residual(upper)<0:upper*=1.2
        a=brentq(residual,stable_min,upper,xtol=1e-16)
        return {'force_N':force(a),'indentation_m':delta(a),'radius_m':a,'active':True}
    return solve,critical_drive


def state(model,young,work,k,drive,active=True):
    if model=='jkr':return jkr_branch(young,work,k)[0](drive,active)
    if drive>=0:return {'force_N':0.,'indentation_m':-drive,'radius_m':0.,'active':False}
    E=young/(2*(1-.25**2));R=.001
    delta=brentq(lambda d:4*E*math.sqrt(R)*d**1.5/3+k*(drive+d),0,-drive,xtol=1e-16)
    return {'force_N':4*E*math.sqrt(R)*delta**1.5/3,'indentation_m':delta,'radius_m':math.sqrt(R*delta),'active':True}


def projected_pixels(distance,bounds=(-.003,.003,-.0018,.0042,-.003,.003),resolution=1200):
    direction=np.array([1.4,1.5,1.8]);direction/=np.linalg.norm(direction);right=np.cross([0.,1.,0.],direction);right/=np.linalg.norm(right);up=np.cross(direction,right)
    corners=np.array(list(itertools.product(*[(bounds[2*i],bounds[2*i+1]) for i in range(3)])))
    parallel_scale=.56*max(np.ptp(corners@right),np.ptp(corners@up))
    return float(abs(distance*up[1])*resolution/(2*parallel_scale))


def scan(output):
    output.mkdir(parents=True,exist_ok=True);rows=[]
    for young in (4e5,8e5):
        beam=geometry_stiffness(young)
        for k in (.25,.5,1.):
            if k>=beam:continue
            fixture=1/(1/k-1/beam)
            for hold in (.0003,.00045,.0006):
                drive=.00001+hold;adh=state('jkr',young,.08,k,drive,True);free=state('hertz',young,.08,k,drive)
                actual_separation_difference=abs(free['indentation_m']-adh['indentation_m'])
                row={'young_Pa':young,'work_J_per_m2':.08,'Tabor_mu':5.,'effective_fixture_N_per_m':k,'physical_fixture_N_per_m':fixture,'beam_relative_stiffness_N_per_m':beam,'hold_actuator_m':hold,'hertz_gap_m':-free['indentation_m'],'jkr_gap_m':-adh['indentation_m'],'jkr_attached':adh['active'],'jkr_force_N':adh['force_N'],'jkr_radius_m':adh['radius_m'],'relative_beam_bending_m':abs(adh['force_N'])/beam,'real_centre_separation_difference_m':actual_separation_difference,'camera_projected_pixel_difference':projected_pixels(actual_separation_difference),'jkr_critical_actuator_m':jkr_branch(young,.08,k)[1]-.00001,'screen_pass':adh['active'] and actual_separation_difference>.0003 and adh['radius_m']/.001<.2}
                rows.append(row)
    with (output/'fixture-scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    selected=next(r for r in rows if r['young_Pa']==4e5 and r['effective_fixture_N_per_m']==.5 and r['hold_actuator_m']==.00045)
    # The actual common path demonstrates attachment and later stable loss,
    # rather than holding within a model difference smaller than grid error.
    from examples.fibre_adhesion_contrast import motion
    protocol={'times_s':[0.,.015,.025,.055,.085,.1],'actuator_y_m':[0.,-.00015,-.00015,.00045,.00045,.0012]}
    timeline=[];attached=False
    for instant in np.linspace(0,.1,1201):
        actuator=float(motion(instant,protocol));drive=actuator+.00001;adh=state('jkr',4e5,.08,.5,drive,attached);attached=adh['active'];free=state('hertz',4e5,.08,.5,drive)
        timeline.append({'time_s':float(instant),'actuator_y_m':actuator,'hertz_gap_m':-free['indentation_m'],'jkr_gap_m':-adh['indentation_m'],'jkr_force_N':adh['force_N'],'jkr_active':attached,'actual_separation_difference_m':abs(free['indentation_m']-adh['indentation_m'])})
    with (output/'quasistatic-path.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=timeline[0]);writer.writeheader();writer.writerows(timeline)
    report={'schema':'fibre-hertz-adhesion-pair-qstatic-screen-v1','selected':selected,'candidates':rows,'scope':'Analytic stable local Hertz/JKR plus linear clamped beam/common spring equilibrium. Gap/force/bending and the exact existing ParaView camera projection are predicted, not measured MPM. Upper y fixture remains free and no visible geometry is rescaled. Initial approach reaches touch before JKR attachment; no virgin long-range JKR capture. Dynamic mass/damping and nonlinear beam response need real MPM verification.','screen_pass':selected['screen_pass'],'real_MPM_complete':False,'public_visual_review_passed':False,'SC_mesh_qualified':False}
    (output/'screen.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['selected']));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'.local/fibre-adhesion-pair-screen');a=p.parse_args();scan(a.output)


if __name__=='__main__':main()
