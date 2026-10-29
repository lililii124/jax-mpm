"""Measure real solid damage components; never fill the void with fake damage."""
import csv,json
from pathlib import Path
import numpy as np

def measure(folder):
    folder=Path(folder);g=np.load(folder/'reference-grid.npz');p=json.loads((folder/'parameters.json').read_text());X=g['coordinates'][:,:2];tri=g['display_triangles'];boundary=g['hole_boundary_nodes'];hole=set(map(int,boundary));polygon=X[boundary];tip=np.array([p['crack_length'],.5]);h=1/p['n'];adjacency=[set() for _ in X]
    for q in tri:
        for a,b in zip(q,np.roll(q,-1)):adjacency[a].add(int(b));adjacency[b].add(int(a))
    def inside(points):
        mask=np.zeros(len(points),bool)
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            cross=(a[1]>points[:,1])!=(b[1]>points[:,1])
            if a[1]!=b[1]:mask^=cross&(points[:,0]<(b[0]-a[0])*(points[:,1]-a[1])/(b[1]-a[1])+a[0])
        return mask
    nodal_inside=inside(X);nodal_inside[boundary]=False;assert not nodal_inside.any();assert not inside(g['particles'][:,:2]).any()
    with (folder/'history.csv').open(newline='') as stream:history=list(csv.DictReader(stream))
    frames=[]
    for row in history:
        f=np.load(folder/('frame-%04d.npz'%int(row['step'])));d=f['damage'];active=d>.5;unvisited=set(map(int,np.flatnonzero(active)));components=[]
        while unvisited:
            start=unvisited.pop();ids={start};pending=[start]
            while pending:
                for j in adjacency[pending.pop()]:
                    if j in unvisited:unvisited.remove(j);ids.add(j);pending.append(j)
            ii=np.array(sorted(ids));near_tip=bool((np.linalg.norm(X[ii]-tip,axis=1)<=2*h).any())
            components.append({'solid_nodes':len(ii),'touches_initial_tip':near_tip,'touches_free_hole_boundary':bool(ids&hole),'x_range_m':[float(X[ii,0].min()),float(X[ii,0].max())],'y_range_m':[float(X[ii,1].min()),float(X[ii,1].max())]})
        frames.append({'step':int(row['step']),'load_m':float(row['displacement_m']),'damage_threshold':.5,'solid_damage_components':sorted(components,key=lambda c:c['solid_nodes'],reverse=True),'no_damage_defined_inside_void':True})
    result={'schema':'at2-real-hole-solid-damage-v1','particles_inside_void':0,'nonboundary_damage_DOFs_inside_void':0,'hole_boundary_nodes':len(boundary),'definition':'Connected components use actual solid triangle edges and nodal d>0.5. No artificial connection or damage value is inserted across the void.','frames':frames,'last':frames[-1]};(folder/'solid-damage-components.json').write_text(json.dumps(result,indent=2));return result
