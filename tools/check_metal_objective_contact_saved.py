"""One saved-step current-geometry contact check; no trajectory/material replay."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import numpy as np

from tools.metal_affine_contact_numpy import CompatibleContact
from tools.metal_affine_projection_numpy import ConsistentAffineProjection
from tools.metal_affine_objective_contact_numpy import update_contact_geometry
from tools.metal_affine_dynamic_numpy import kinetic,moments


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir()
    expected='3579600dd7d78cfa728b3478f071d416d15e6226c5799012d37d346aa0530809'
    h=hashlib.sha256()
    with (a.parent_root/'result/step-22/accepted.npz').open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    assert h.hexdigest()==expected
    def read(path):
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    old=read(a.parent_root/'result/step-22/accepted.npz');folder=a.parent_root/'result/step-23';report=json.loads((folder/'result.json').read_text());i=report['retained_iteration'];force=read(folder/f'iteration-{i}-force.npz');new=read(folder/'coupled-before-remap.npz');y0=read(folder/'initial-contact.npz')['white']
    mid=dict(old,x=.5*(old['x']+new['x']),F=.5*(old['F']+new['F']));dt=report['dt_s'];tol=report['energy_roundoff_J'];begun=monotonic()
    progress=lambda label:print(json.dumps(dict(phase=label,elapsed_s=monotonic()-begun)),flush=True)
    print(json.dumps(dict(particles=len(old['mass']),saved_time_s=float(old['time']),dt_s=dt)),flush=True)
    args=(old['x'],old['mass'],old['body_id'],.00025,old['F'],old['reference_root'])
    full=ConsistentAffineProjection(*args,build_solver=False);contact=CompatibleContact(full)
    allowed=ConsistentAffineProjection(*args,clamped_particles=old['candidate_fixed_particles'])
    frame=SimpleNamespace(contact=contact,allowed=allowed);results=[]
    for label,configuration,trial,scale in [('midpoint',mid,y0+.5*dt*force['projected_force'],2.),('endpoint-old-space',new,read(folder/'end-old-cone-contact.npz')['white_before'],1.)]:
        construction=update_contact_geometry(frame,configuration,progress)
        following,row,reaction,physical=frame.restricted.apply(trial)
        loss=.5*float(np.sum((following-trial)**2));identity=kinetic(following)-kinetic(trial)+loss
        physical_momentum,physical_torque=moments(configuration,scale*physical)
        physical_midpoint_work=float(np.sum(following*(scale*physical))) if scale==2 else row['physical_contact_work_J']
        boundary_midpoint_work=float(np.sum(following*(scale*reaction))) if scale==2 else row['boundary_virtual_work_J']
        valid=bool(row['valid'] and row['original_sample_closing_m_s']<=.005 and row['restricted_sample_difference_m_s']<=.005 and abs(identity)<=tol and abs(boundary_midpoint_work)<=tol and physical_midpoint_work<=tol and np.linalg.norm(physical_torque)<=report['angular_tolerance'])
        result=dict(stage=label,passed=valid,contact=row,projection_construction=construction,projection_identity_J=identity,physical_momentum=physical_momentum.tolist(),physical_torque=physical_torque.tolist(),physical_contact_work_J=physical_midpoint_work,boundary_work_J=boundary_midpoint_work,kinetic_projection_loss_J=loss)
        results.append(result);(a.output/'fixed-contact-review.json').write_text(json.dumps(dict(status='running',results=results),indent=2)+'\n')
        np.savez_compressed(a.output/(label+'-raw.npz'),trial=trial,following=following,physical=physical,reaction=reaction,multiplier=frame.restricted.last_multiplier,geometry_x=configuration['x'],geometry_F=configuration['F'],normals=frame.restricted.normal,mobility=frame.restricted.mobility,restricted_scalar_sampler=frame.restricted.restricted_H)
        assert valid,result
    result=dict(status='passed',parent_task='metal-auto-20261025-173820-w02',parent_checkpoint_sha256=expected,particles=len(old['mass']),energy_tolerance_J=tol,results=results,total_elapsed_s=monotonic()-begun,scope='Two time levels of this single saved input only; entire physical adjoint projected to fixed original P0; no material/force/trajectory replay and no state or angular compensation')
    (a.output/'fixed-contact-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
