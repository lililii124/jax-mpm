import hashlib,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/previous/source')
from tools.metal_affine_dynamic_numpy import moments,unpack
root=Path('/previous');folder=root/'result/step-23';report=json.loads((folder/'result.json').read_text());contract=json.loads((root/'contract.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(root/'result/step-22/accepted.npz')=='3579600dd7d78cfa728b3478f071d416d15e6226c5799012d37d346aa0530809'
def read(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
old=read(root/'result/step-22/accepted.npz');new=read(folder/'coupled-before-remap.npz');i=report['retained_iteration'];force=read(folder/('iteration-%d-force.npz'%i));dt=report['dt_s']
initial_projection=read(folder/'initial-projection.npz');initial_contact=read(folder/'initial-contact.npz');y0=initial_contact['white'];y1=new['candidate_white'];midwhite=new['midpoint_final_white']
assert np.array_equal(y1,force['guessed_white']) and np.array_equal(midwhite,.5*(y0+y1))
for key in ['mass','volume','D0','reference_root']:assert np.array_equal(old[key],new[key]),key
mid=dict(old,x=.5*(old['x']+new['x']),F=.5*(old['F']+new['F']))
def L(q,y):return moments(q,y)[1]
def vec(value):return np.asarray(value).tolist()
def sumvec(values):return np.array([math.fsum(v[j] for v in values) for j in range(3)])
contacts=[];boundaries=[];stages=[]
for label,q in [('initial-projection',old),('initial-contact',old),('end-old-cone-contact',new),('end-projection',new),('end-contact',new)]:
 data=read(folder/(label+'.npz'));change=L(q,data['white'])-L(q,data['white_before']);boundary=L(q,data['reaction']);boundaries.append(boundary)
 physical=L(q,data['physical']) if 'physical' in data else np.zeros(3)
 if 'physical' in data:
  contacts.append(physical)
  detail=dict(same_impulse_torque_at_q0=vec(L(old,data['physical'])),same_impulse_torque_at_qmid=vec(L(mid,data['physical'])),same_impulse_torque_at_q1=vec(L(new,data['physical'])))
 else:detail={}
 stages.append(dict(stage=label,angular_change=vec(change),boundary_torque=vec(boundary),physical_torque=vec(physical),identity_remainder=vec(change-boundary-physical),**detail))
physical=force['physical_contact'];reaction=dt*force['reaction_force']+force['reaction_contact'];coupled_contact=L(mid,physical);contacts.append(coupled_contact);boundaries.append(L(mid,reaction))
delta=L(new,y1)-L(old,y0);canonical=L(mid,y1-y0);raw_torque=L(mid,dt*force['raw_force']);nonlinear=L(mid,y1-force['proposed_white']);coupled_remainder=delta-L(mid,reaction)-coupled_contact-raw_torque-nonlinear
stages.append(dict(stage='coupled',angular_change=vec(delta),physical_torque=vec(coupled_contact),boundary_torque=vec(L(mid,reaction)),raw_internal_force_torque=vec(raw_torque),nonlinear_balance_torque=vec(nonlinear),canonical_midpoint_motion_remainder=vec(delta-canonical),full_stage_identity_remainder=vec(coupled_remainder),same_impulse_torque_at_q0=vec(L(old,physical)),same_impulse_torque_at_qmid=vec(coupled_contact),same_impulse_torque_at_q1=vec(L(new,physical))))
end=read(folder/'end-contact.npz');reconstructed=L(new,end['white'])-L(old,initial_projection['white_before'])-sumvec(boundaries);contact_total=sumvec(contacts)
v,W=unpack(old,midwhite)
result=dict(status='saved-stage-audit-completed',source_task='metal-auto-20261025-173820-w02',source_commit=contract['source_commit'],step=23,input_checkpoint_sha256=report['input_sha256'],rejected_state_not_restartable=True,dt_s=dt,stages=stages,reported_perstep_angular_residual=report['angular_residual'],reconstructed_perstep_angular_residual=vec(reconstructed),boundary_torque_reconstruction_difference=vec(sumvec(boundaries)-np.asarray(report['boundary_torque'])),physical_contact_torque_sum=vec(contact_total),unexplained_after_contacts=vec(reconstructed-contact_total),qmid_position_shift_max_m=float(np.max(abs(mid['x']-old['x']))),qmid_F_shift_max=float(np.max(abs(mid['F']-old['F']))),original_midpoint_position_error_m=float(np.max(abs(new['x']-old['x']-dt*v))),original_midpoint_F_error=float(np.max(abs(new['F']-old['F']-dt*W))),scope='Saved vectors and configurations only; no force/material recomputation, no QP/projector factorization, no dynamics replay or state/reaction correction')
Path('/work/stage-angle-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
