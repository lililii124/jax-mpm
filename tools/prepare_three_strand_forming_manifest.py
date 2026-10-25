"""Collect authentic saved forming frames into a hash-bound render input."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path,help='Original or candidate output directories, each containing plan.json and frame-STEP.npz')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    import numpy as np
    selected={};plans=[];reference=None
    for folder in args.inputs:
        plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
        identity={k:plan[k] for k in ('case','particles','dx','dt','steps','duration','geometry','material','protocol')}
        if reference is None:reference=identity
        elif identity!=reference:raise ValueError('Input geometry, material or global time plan differs')
        plans.append(dict(path=str(plan_path.resolve()),sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
        for path in sorted(folder.glob('frame-*.npz')):
            match=re.fullmatch(r'frame-(\d+)\.npz',path.name)
            if not match:continue
            step=int(match[1])
            if not 0<=step<=plan['steps']:raise ValueError('Frame step outside original plan')
            with np.load(path,allow_pickle=False) as data:
                x=data['x'];ids=data['fibre_id']
                if x.shape!=(plan['particles'],3) or not np.isfinite(x).all() or not np.array_equal(np.unique(ids),[0,1,2]):raise ValueError('Invalid saved particle frame')
                if 'time' in data and not np.isclose(float(data['time']),step*plan['dt'],rtol=1e-12,atol=1e-16):raise ValueError('Saved time differs from original global step')
                if step in selected:
                    with np.load(selected[step],allow_pickle=False) as earlier:
                        if not np.array_equal(x,earlier['x']) or not np.array_equal(ids,earlier['fibre_id']):raise ValueError('Conflicting actual states at step '+str(step))
                    continue
            selected[step]=path
    if len(selected)<2:raise ValueError('At least two authentic saved states are required')
    args.output.mkdir(parents=True,exist_ok=False);frames=[]
    for step,path in sorted(selected.items()):
        name=f'frame-{step:06d}.npz';destination=args.output/name;shutil.copyfile(path,destination)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:raise IOError('Copied frame hash differs')
        frames.append(dict(file=name,kind='frame',sha256=digest,step=step,time_s=step*reference['dt'],source=str(path.resolve())))
    manifest=dict(schema='three-strand-forming-render-input-v1',frames=frames,source_plans=plans,
                  planned_steps=reference['steps'],full_turn_completed=frames[0]['step']==0 and frames[-1]['step']==reference['steps'],
                  actual_saved_states_only=True,position_interpolation=False,physical_steps_executed=0,
                  scope='Media input preparation; numerical admission and convergence remain separate')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(frames=len(frames),first_step=frames[0]['step'],last_step=frames[-1]['step'],manifest=str(args.output/'manifest.json'))))


if __name__=='__main__':main()
