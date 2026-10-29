"""Measure tip-connected damage in the actual vertical-notch beam geometry."""
import csv,json
from pathlib import Path
import numpy as np

def measure(folder):
    folder=Path(folder);g=np.load(folder/'reference-grid.npz');p=json.loads((folder/'parameters.json').read_text());X=g['display_coordinates'][:,:2];quads=g['display_quads'];nodes=g['display_nodes'];W=g['display_weights'];tip=np.array([.5*p['length_m'],p['bottom_central_notch_m']]);h=max(p['length_m']/p['horizontal_cells'],p['height_m']/p['vertical_cells'])
    adjacency=[set() for _ in X]
    for q in quads:
        for a,b in zip(q,np.roll(q,-1)):adjacency[a].add(int(b));adjacency[b].add(int(a))
    with (folder/'history.csv').open(newline='') as stream:history=list(csv.DictReader(stream))
    frames=[]
    for row in history:
        f=np.load(folder/('frame-%04d.npz'%int(row['step'])));d=np.sum(W*f['damage'][nodes],axis=1);active=(d>.5)&(X[:,1]>=tip[1]-2*h);start=np.flatnonzero(active&(np.linalg.norm(X-tip,axis=1)<=2*h));connected=set(map(int,start));pending=list(connected)
        while pending:
            for j in adjacency[pending.pop()]:
                if active[j] and j not in connected:connected.add(j);pending.append(j)
        ids=np.array(sorted(connected),int)
        frames.append({'step':int(row['step']),'downward_head_travel_m':float(row['displacement_m']),'connected_threshold':.5,'connected_display_nodes':len(ids),'connected_maximum_y_m':float(X[ids,1].max()) if len(ids) else float(tip[1]),'connected_vertical_advance_m':max(float(X[ids,1].max()-tip[1]),0.) if len(ids) else 0.,'connected_x_range_m':[float(X[ids,0].min()),float(X[ids,0].max())] if len(ids) else None})
    result={'schema':'at2-three-point-bending-connected-damage-v1','physical_initial_tip':tip.tolist(),'definition':'Graph-connected exact display nodal damage>0.5 starting within2 physical cell widths of the bottom-notch tip. No field modification, path constraint or symmetry averaging.','frames':frames,'last':frames[-1]}
    (folder/'connected-path-measurement.json').write_text(json.dumps(result,indent=2));return result
