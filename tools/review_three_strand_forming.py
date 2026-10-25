"""Read-only forming checkpoint, energy scope and actual particle geometry review."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,ConvexHull


def review(folder):
    folder=Path(folder);plan=json.loads((folder/'plan.json').read_text());rows=json.loads((folder/'observables.json').read_text())
    with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
        x=cp['state_x'];ids=cp['state_fibre_id'];mass=cp['state_mass'];F=cp['state_F'];v=cp['state_v'];C=cp['state_C']
        assert bool(cp['state_valid']) and np.all(cp['history_damage']==0)
        assert all(np.isfinite(cp[k]).all() for k in cp.files if cp[k].dtype.kind in 'fci')
        sv=np.linalg.svd(F,compute_uv=False)
        momentum=np.sum(mass[:,None]*v,axis=0)
        kinetic=.5*np.sum(mass*np.sum(v*v,axis=1))+plan['dx']**2/8*np.sum(mass*np.sum(C*C,axis=(1,2)))
        material=dict(min_J=float(np.linalg.det(F).min()),max_J=float(np.linalg.det(F).max()),principal_stretch_min=float(sv.min()),principal_stretch_max=float(sv.max()),max_hencky_strain=float(np.abs(np.log(sv)).max()))
        impulse=cp['state_boundary_impulse'];step=int(cp['step'])
    # This is a true 3D nearest particle-centre distance, not a same-section gap.
    pairs=[]
    for a,b in ((0,1),(0,2),(1,2)):
        xa=x[ids==a];xb=x[ids==b];tree=cKDTree(xb)
        distance,index=tree.query(xa,k=1,workers=1);i=int(np.argmin(distance))
        pairs.append(dict(strands=[a,b],minimum_particle_centre_distance_m=float(distance[i]),point_a_m=xa[i].tolist(),point_b_m=xb[index[i]].tolist()))
    energy=[];abs_sampled_work=0.;max_sampled_increment=0.;previous=0.
    for row in rows:
        work=row['fixture_work_J'];increment=work-previous;previous=work
        abs_sampled_work+=abs(increment);max_sampled_increment=max(max_sampled_increment,abs(increment))
        remainder=work-row['kinetic_J']-row['elastic_J']-row['fixture_projection_loss_J']-row['contact_projection_loss_J']
        energy.append(dict(step=row['step'],fixture_work_J=work,kinetic_J=row['kinetic_J'],elastic_J=row['elastic_J'],fixture_projection_loss_J=row['fixture_projection_loss_J'],contact_projection_loss_J=row['contact_projection_loss_J'],unresolved_remainder_J=remainder,cumulative_absolute_saved_interval_work_J=abs_sampled_work,maximum_absolute_saved_interval_work_J=max_sampled_increment,remainder_over_saved_absolute_work=float(remainder/abs_sampled_work) if abs_sampled_work else 0.))
    result=dict(step=step,time_s=rows[-1]['time_s'],checkpoint_sha256=hashlib.sha256((folder/'checkpoint.npz').read_bytes()).hexdigest(),all_arrays_finite=True,material=material,mass_kg=float(mass.sum()),momentum_residual_Ns=float(np.linalg.norm(momentum-impulse)),fixture_impulse_Ns=impulse.tolist(),kinetic_recomputed_J=float(kinetic),particle_distance=pairs,energy=energy,
        energy_scope='Saved-interval absolute work is a lower bound on true stepwise absolute work; stepwise abs/max were not logged. Remainder includes APIC transfer and constitutive/time discretization. Not an energy convergence certificate.',
        geometry_scope='Actual particle-centre distances, with spacing0.1mm and h0.2mm. No reconstructed sub-grid surface, penetration certificate, or resolution convergence claim.',production_eligible=False)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=review(a.folder);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
