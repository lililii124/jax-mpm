"""Measure projected normal pressure applied to a reconstructed binary plate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tools.projected_pressure import allocate_projected_pressure


def measure(geometry, settings, expected_sha256=None, mode='omit'):
    geometry=Path(geometry)
    sha=hashlib.sha256(geometry.read_bytes()).hexdigest()
    if expected_sha256 is not None and sha!=expected_sha256:
        raise ValueError('Geometry does not match its replay')
    with np.load(geometry,allow_pickle=False) as data:
        solid=np.asarray(data['solid_mask'],dtype=bool)
    nx,ny,nz=map(int,settings['cell']);layers=int(settings['skin_layers'])
    gx,gy,gz=3*nx,3*ny,nz+2*layers
    if solid.shape!=(gz,gy,gx):raise ValueError('Geometry shape differs from simulation settings')
    size=np.asarray(settings['cell_size'],dtype=float);h=size/np.array((nx,ny,nz))
    xy=np.stack(np.meshgrid((np.arange(gx)+.5)*h[0],(np.arange(gy)+.5)*h[1],indexing='ij'),-1)
    center=np.array((gx*h[0]/2,gy*h[1]/2))
    weights=np.exp(-.5*np.sum(((xy-center)/settings['blast_radius'])**2,axis=-1))
    weights/=weights.mean()
    occupied=np.any(solid,axis=0).T
    assigned=allocate_projected_pressure(weights,occupied,mode)
    area=float(h[0]*h[1]);pressure=float(settings['pressure']);pulse=float(settings['pulse'])
    peak_force=pressure*area*float(np.sum(assigned))
    first_moment=pressure*area*np.sum(assigned[...,None]*(xy-center),axis=(0,1))
    return dict(geometry_sha256=sha,populated_column_fraction=float(np.mean(occupied)),
                allocation=mode,projected_weighted_area_m2=area*float(np.sum(assigned)),
                peak_normal_force_N=peak_force,normal_impulse_Ns=peak_force*pulse/2,
                pressure_first_moment_Nm=first_moment.tolist(),
                model='Projected pressure on uppermost solid columns; redistribution approximates transfer across empty columns and does not resolve cavity-side pressure.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('geometry',type=Path);p.add_argument('replay',type=Path)
    a=p.parse_args();meta=json.loads((a.replay/'inputs.json').read_text())
    print(json.dumps(measure(a.geometry,meta['settings'],meta['geometry_sha256'],meta.get('empty_column_load','omit')),indent=2,allow_nan=False))


if __name__=='__main__':main()
