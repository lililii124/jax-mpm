"""Read-only symmetric geometry and force coverage of actual tight forming."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def finite_list(value):return [float(v) if np.isfinite(v) else None for v in value]


def main():
    a=argparse.ArgumentParser();a.add_argument('task',type=Path);a.add_argument('--output',type=Path,default=Path('tight-coverage-review.json'));args=a.parse_args();folder=args.task/'output';plan=json.loads((folder/'plan.json').read_text());review=json.loads((folder/'review.json').read_text());last=review['rows'][-1]
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as d:x=d['state_x'];F=d['state_F'];ids=d['state_fibre_id'];t=float(d['state_time']);step=int(d['step'])
    spacing=plan['geometry']['spacing_m'];dx=plan['dx'];inverse=np.linalg.inv(F);sv=np.linalg.svd(F,compute_uv=False);trees={f:cKDTree(x[ids==f]) for f in range(3)};index={f:np.flatnonzero(ids==f) for f in range(3)}
    ramp=plan['protocol']['rotation_ramp_s'];duration=plan['duration'];z=min(max(t,0),duration);primitive=z*z/(2*ramp) if z<ramp else z-ramp/2 if z<=duration-ramp else duration-ramp-(duration-z)**2/(2*ramp);theta=2*math.pi*primitive/(duration-ramp)
    z=np.clip((t-.005)/.010,0,1);arc=.021*(1+.03*z*z*(3-2*z));span=math.sqrt(arc*arc-(theta*plan['geometry']['initial_orbit_m'])**2);extension=span-.021;ends=[.0015-extension/2,.0225+extension/2]
    forces=np.array(last['contact_force_axial_bins_N']);centres=(np.arange(64)+.5)*.024/64;free=(centres>ends[0])&(centres<ends[1]);pairs=[];directions=[]
    for p,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
        ia=index[a];ib=index[b];distance,near=trees[a].query(x[ib],workers=1);ia_near=ia[near];delta=x[ib]-x[ia_near];normal=delta/np.maximum(distance[:,None],1e-300)
        support_a=.5*spacing*np.sum(np.abs(np.einsum('nji,nj->ni',F[ia_near],normal)),axis=1);support_b=.5*spacing*np.sum(np.abs(np.einsum('nji,nj->ni',F[ib],normal)),axis=1)
        bins=np.floor(x[ib,0]/.024*64).astype(int);valid=(bins>=0)&(bins<64);minimum=np.full(64,np.inf);projected=np.full(64,np.inf);np.minimum.at(minimum,bins[valid],distance[valid]);np.minimum.at(projected,bins[valid],(distance-support_a-support_b)[valid])
        threshold=max(1e-9,.01*float(forces[p,free].max(initial=0.)));active=free&(forces[p]>threshold)&(minimum<=dx);fraction=float(active.sum()/free.sum())
        pairs.append(dict(strands=[a,b],force_threshold_N=threshold,force_threshold_rule='max(1nN,1percent of largest free-span-bin force); diagnostic only, no force-law feedback',free_span_contact_fraction=fraction,active_free_span_bins=int(active.sum()),free_span_bins=int(free.sum()),contact_bin_length_m=float(active.sum()*.024/64),normal_force_N=float(forces[p].sum()),bin_force_N=forces[p].tolist(),nearest_point_distance_per_bin_m=finite_list(minimum),closest_centre_pair_projected_cell_gap_per_bin_m=finite_list(projected)))
        for A,B in [(a,b),(b,a)]:
            iA=index[A];iB=index[B];radius=.5*np.sqrt(3)*spacing*sv[iA,0].max();dist,near=trees[A].query(x[iB],distance_upper_bound=radius,workers=1);candidate=np.flatnonzero(np.isfinite(dist));inside=deep=0;depth=0.
            groups=trees[A].query_ball_point(x[iB[candidate]],radius,workers=1)
            for j,neighbours in zip(candidate,groups):
                rows=iA[np.asarray(neighbours,dtype=int)];local=np.einsum('nij,nj->ni',inverse[rows],x[iB[j]]-x[rows]);normalised=np.max(np.abs(local),axis=1)/(.5*spacing);best=float(normalised.min());inside+=int(best<1);deep+=int(best<.8);depth=max(depth,max(1-best,0)*.5*spacing)
            directions.append(dict(cell_material=A,foreign_point_material=B,all_candidates_queried=True,foreign_centres_inside=inside,deep_intrusions=deep,maximum_material_cell_depth_m=depth))
    out=dict(step=step,time_s=t,physical_free_span_limits_m=ends,bin_width_m=.024/64,pairs=pairs,symmetric_all_candidate_cell_checks=directions,all_deep_checks_pass=all(d['deep_intrusions']==0 for d in directions),major_grid_contact_candidate=all(p['free_span_contact_fraction']>=.7 for p in pairs) and all(d['deep_intrusions']==0 for d in directions),qualification_scope='Read-only h=.25mm grid-scale contact coverage screen. Closest-centre projected cell gaps are directional envelope diagnostics, not exact smooth-surface distances. >=70percent free-span bins with finite significant force and nearest-point spacing<=h is a candidate for review, not an automatic media or continuum-contact certification.')
    args.output.write_text(json.dumps(out,indent=2));brief={k:v for k,v in out.items() if k not in ['pairs']};brief['pair_summary']=[{k:p[k] for k in ['strands','free_span_contact_fraction','contact_bin_length_m','normal_force_N']} for p in pairs];print(json.dumps(brief,indent=2))

if __name__=='__main__':main()
