"""Position-based connected components; damage labels never define fragments."""
import argparse
import json
from pathlib import Path
import numpy as np


def components(x, mass, velocity, body_id, *, separation_radius):
    from scipy.spatial import cKDTree
    if separation_radius<=0 or not all(np.isfinite(a).all() for a in (x,mass,velocity)):
        raise ValueError('Finite state and positive physical separation radius required')
    parent=np.arange(len(x));size=np.ones(len(x),dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    # Stream neighborhoods; avoid an unbounded all-pairs allocation.
    tree=cKDTree(x)
    for i,point in enumerate(x):
        for j in tree.query_ball_point(point,separation_radius):
            if j<=i:continue
            left,right=find(i),find(j)
            if left!=right:
                if size[left]<size[right]:left,right=right,left
                parent[right]=left;size[left]+=size[right]
    groups={}
    for i in range(len(x)):groups.setdefault(find(i),[]).append(i)
    rows=[]
    for ids in sorted(groups.values(),key=len,reverse=True):
        ids=np.asarray(ids);m=mass[ids];total=m.sum();v=velocity[ids];center=(m[:,None]*x[ids]).sum(0)/total
        rows.append(dict(particles=len(ids),mass_kg=float(total),center_m=center.tolist(),
            momentum_Ns=(m[:,None]*v).sum(0).tolist(),kinetic_energy_J=float(.5*np.sum(m*np.sum(v*v,axis=1))),
            sphere_mass_kg=float(m[body_id[ids]==0].sum()),plate_mass_kg=float(m[body_id[ids]==1].sum())))
    if rows:
        main=rows[0];main_v=np.asarray(main['momentum_Ns'])/main['mass_kg'];main_x=np.asarray(main['center_m'])
        for group in rows:
            offset=np.asarray(group['center_m'])-main_x;distance=np.linalg.norm(offset)
            relative=np.asarray(group['momentum_Ns'])/group['mass_kg']-main_v
            group['relative_speed_m_s']=float(np.linalg.norm(relative))
            group['radial_separation_speed_m_s']=float(np.dot(relative,offset)/distance) if distance>0 else 0.
            group['center_distance_from_largest_m']=float(distance)
    return dict(separation_radius_m=separation_radius,components=rows,total_mass_kg=float(mass.sum()),
        assigned_mass_kg=sum(g['mass_kg'] for g in rows),
        scope='Spatial particle connectivity only. A connected cluster is a candidate fragment, not proof of cohesive integrity. Shared MPM grid may recouple clusters. Damage/failed tags were not used.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('state',type=Path);p.add_argument('--radius',type=float,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    with np.load(args.state,allow_pickle=False) as s:
        report=components(s['x'],s['mass'],s['v'],s['body_id'],separation_radius=args.radius)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
