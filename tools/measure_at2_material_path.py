"""Measure connected damage from saved fields without changing those fields."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def measure(folder):
    folder=Path(folder);grid=np.load(folder/'reference-grid.npz');params=json.loads((folder/'parameters.json').read_text())
    with (folder/'history.csv').open(newline='') as stream:history=list(csv.DictReader(stream))
    X=grid['display_coordinates'][:,:2];dnodes=grid['display_nodes'];weights=grid['display_weights'];quads=grid['display_quads'];a=params['crack_length'];h=1/params['n']
    adjacency=[set() for _ in X]
    for q in quads:
        for left,right in zip(q,np.roll(q,-1)):
            adjacency[left].add(int(right));adjacency[right].add(int(left))
    material_file=folder/'material-fields.npz'
    inclusion=np.load(material_file)['particle_inclusion'].astype(bool) if material_file.exists() else None
    frames=[]
    for row in history:
        f=np.load(folder/('frame-%04d.npz'%int(row['step'])));d=np.sum(weights*f['damage'][dnodes],axis=1)
        active=(d>.5)&(X[:,0]>=a-h);starts=np.flatnonzero(active&(np.linalg.norm(X-[a,.5],axis=1)<=2*h))
        connected=set(map(int,starts));pending=list(connected)
        while pending:
            for other in adjacency[pending.pop()]:
                if active[other] and other not in connected:connected.add(other);pending.append(other)
        ids=np.array(sorted(connected),int);centers=[]
        if len(ids):
            for x in sorted(set(np.round(X[ids,0],12))):
                col=ids[abs(X[ids,0]-x)<1e-10]
                if x>a+params['ell'] and x<1-params['ell']:
                    wt=d[col]**4;centers.append([float(x),float(np.sum(wt*X[col,1])/wt.sum())])
        if len(centers)>=3:
            fitted=np.polyfit(np.array(centers)[:,0],np.array(centers)[:,1],1);angle=float(np.rad2deg(np.arctan(fitted[0])))
        else:angle=None
        frames.append({'step':int(row['step']),'load_m':float(row['displacement_m']),'connected_damage_threshold':.5,'connected_display_nodes':len(ids),'connected_maximum_x_m':float(X[ids,0].max()) if len(ids) else a,'connected_advance_x_m':max(float(X[ids,0].max())-a,0) if len(ids) else 0.,'connected_y_range_m':[float(X[ids,1].min()),float(X[ids,1].max())] if len(ids) else None,'damage_weighted_column_centers_m':centers,'fitted_damage_band_angle_degrees':angle,'maximum_off_axis_center_m':max((abs(y-.5) for x,y in centers),default=0.)})
        if inclusion is not None and inclusion.any():
            dp=np.sum(grid['weight']*f['damage'][grid['nodes']],axis=1);volume=grid['volume']
            frames[-1]['inclusion_particle_damage_max']=float(dp[inclusion].max())
            frames[-1]['inclusion_volume_fraction_above_damage_08']=float(volume[inclusion&(dp>.8)].sum()/volume[inclusion].sum())
            frames[-1]['inclusion_damage_volume_average']=float(np.sum(volume[inclusion]*dp[inclusion])/volume[inclusion].sum())
    result={'schema':'at2-material-connected-path-measurement-v1','threshold_definition':'Graph-connected display nodal damage>0.5, starting within2hx of existing crack tip. Display coefficients are exact basis evaluations; no field smoothing or path constraint. Angle fits damage^4 column centers beyond tip+ell and before rightboundary-ell. This is a measurement, not a propagation law.','frames':frames,'last':frames[-1],'physical_fields_modified':False}
    (folder/'connected-path-measurement.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args();print(json.dumps(measure(args.folder)['last'],indent=2))
