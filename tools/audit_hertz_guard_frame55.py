"""Audit a saved guard-paused Hertz frame55 without changing failed raw or gates."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
CP_SHA='e920557546cd74b456f3f82f8fad7514e68414f12b2434e0286ae871facd78e0'
RECEIPT_SHA='f79566f27d36acdca36456bbee17544c4e5f06cc5875fdf8b58da21806d763a0'
F2_CP_SHA='533140bc58f03198c31e3c5bfb5864b7b569c710dd99bbae2a455b5579234e21'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def descriptor(a):
 a=np.ascontiguousarray(a);return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def audit(raw,producer,f2_checkpoint):
 if sha(raw/'receipt.json')!=RECEIPT_SHA or sha(raw/'data/hertz/checkpoint.npz')!=CP_SHA or sha(f2_checkpoint)!=F2_CP_SHA:raise ValueError('Exact immutable parent identities required')
 receipt=read(raw/'receipt.json');folder=raw/'data/hertz';plan=read(folder/'plan.json');summary=read(folder/'summary.json');status=read(folder/'status.json')
 if receipt['never_resumable'] is not True or receipt['kind']!='diagnostic' or '260minute_wall_guard' not in receipt['failure'] or receipt['model']!='hertz':raise ValueError('Original diagnostic must remain rejected')
 if receipt['source']['commit']!='bfaaf43cdca9f7ddedcb9726b35a3fc482aa8206' or receipt['continuation_commit']!='0d3029212c7ad6f494e62637494d1951ba048368' or receipt['continued_from_frame']!=2 or receipt['original_steps_not_recomputed']!=974:raise ValueError('Wrong immutable lineage')
 for n,r in receipt['files'].items():
  p=raw/n
  if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('Original failed raw inventory changed: '+n)
 for n,h in receipt['source']['files'].items():
  if sha(producer/n)!=h:raise ValueError('Original numerical source differs: '+n)
 if plan!=receipt['request'] or summary['definition']!=plan or plan['model']!='hertz' or plan['particles']!=202240 or plan['steps']!=29212:raise ValueError('Original full physical request differs')
 signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
 if status!={'phase':'checkpoint-ready','frame':2,'signature':signature,'steps':974}:raise ValueError('Unexpected stale original accepted-prefix status')
 rows=list(csv.DictReader((folder/'observables.csv').open()))
 with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp,np.load(f2_checkpoint,allow_pickle=False) as parent:
  frame=int(cp['frame']);steps=int(cp['progress_0']);tables=json.loads(str(cp['tables']));old=json.loads(str(parent['tables']))
  if frame!=55 or steps!=26778 or steps!=plan['frame_steps'][55] or str(cp['signature'])!=signature or not bool(cp['state_valid']) or not np.all(cp['history_valid']):raise ValueError('Saved complete55 checkpoint invalid')
  if len(rows)!=56 or any(len(tables[k])!=56 for k in ('rows','times','records')) or rows!=[{k:str(v) for k,v in r.items()} for r in tables['rows']]:raise ValueError('Original CSV/full checkpoint table mismatch')
  if any(tables[k][:3]!=old[k] for k in ('rows','times','records')):raise ValueError('Certified original f2 history changed')
  runtime=json.loads(str(cp['runtime']));expected={'python':'3.12.13','packages':{k:receipt['environment']['packages'][k] for k in ('jax','jaxlib','numpy','scipy')}}
  if runtime!=expected:raise ValueError('Exact producer runtime differs')
  if any(not np.isfinite(cp[k]).all() for k in cp.files if cp[k].dtype.kind in 'biufc'):raise ValueError('Nonfinite saved carry')
  if not np.all(cp['contact_valid']) or not np.all(cp['contact_law_valid']):raise ValueError('Invalid complete-frame contact state')
  for name in ('x','speed','damage','temperature','failed'):
   a=np.load(folder/f'trajectory_{name}.npy',mmap_mode='r')
   if len(a)!=61 or a.shape[1]!=202240:raise ValueError('Original full trajectory allocation differs')
   for i in range(56):
    if descriptor(a[i])!=tables['records'][i][name]:raise ValueError('Changed genuine saved frame')
   key={'x':'state_x','damage':'history_damage','failed':'history_failed'}.get(name)
   if key and not np.array_equal(a[55],cp[key]):raise ValueError('Saved endpoint/CP differs')
  F,D=cp['state_F'],cp['history_damage'];material=plan['material'];E,nu=material['young'],material['poisson'];mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));u,s,v=np.linalg.svd(F);rotation=u@v;strain=s-1;tr=strain.sum(1);expected_stress=(1-D)[:,None,None]*(2*mu*F+(lam*tr-2*mu)[:,None,None]*rotation)@F.transpose(0,2,1)
  stress_error=float(np.max(np.abs(expected_stress-cp['state_tau'])));energy=float(np.sum(cp['state_volume']*(1-D)*(.5*lam*tr**2+mu*np.sum(strain**2,axis=1))))
  momentum_errors=[];budgets=[]
  for body in (0,1):
   mask=cp['state_fibre_id']==body;p=np.sum(cp['state_mass'][mask,None]*cp['state_v'][mask],axis=0);boundary=cp['progress_1'][body];contact=cp['progress_4'][body];fixture=np.array([0.,cp['progress_5'][body],0.]);error=float(np.max(np.abs(p-boundary-contact-fixture)));scale=float(np.sum(np.abs(cp['state_mass'][mask,None]*cp['state_v'][mask]))+np.sum(np.abs(boundary))+np.sum(np.abs(contact))+np.sum(np.abs(fixture)));momentum_errors.append(error);budgets.append(1e-9*max(scale,1e-30))
  checks=dict(exact_saved_frame_steps=True,all56_saved_states_and_f2_lineage=True,exact_runtime=True,all_saved_state_finite_valid=True,independent_material_stress=stress_error<1e-6,independent_stored_energy=abs(energy-float(rows[-1]['elastic_energy_J']))<1e-10,positive_orientation=bool(np.all(np.linalg.det(F)>0)),undamaged=bool(not D.any() and not cp['history_failed'].any()),per_body_momentum=all(x<=b for x,b in zip(momentum_errors,budgets)),mass_conserved=max(abs(float(r['mass_relative_drift'])) for r in rows)<1e-12,constraints=max(float(r['grip_position_error_m']) for r in rows)<1e-9,contact_valid=all(r['contact_valid']=='True' for r in rows))
  scale=max(max(abs(float(r['actuator_work_J']))+abs(float(r['actual_contact_work_J'])) for r in rows),1e-20);residual=max(abs(float(r['energy_ledger_residual_J'])) for r in rows);checks['original15percent_work_energy_gate']=residual<.15*scale
  time=float(cp['state_time']);checks['clock']=bool(np.isclose(time,steps*plan['dt'],rtol=1e-11,atol=0))
  report=dict(schema='hertz-frame55-copy-only-physical-admission-v1',passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},frame=55,steps=steps,remaining_steps=29212-steps,saved_states=56,time_s=time,source_commit=receipt['source']['commit'],original_controller=receipt['continuation_commit'],original_failed_receipt_sha256=RECEIPT_SHA,original_receipt_remains_never_resumable=True,checkpoint_sha256=CP_SHA,original_f2_checkpoint_sha256=F2_CP_SHA,raw_files=receipt['files'],runtime=runtime,stress_error_Pa=stress_error,stored_energy_error_J=abs(energy-float(rows[-1]['elastic_energy_J'])),momentum_error_Ns=momentum_errors,momentum_budget_Ns=budgets,maximum_energy_residual_J=residual,maximum_work_scale_J=scale,stale_status_original_sha256=sha(folder/'status.json'),isolated_resume_status=dict(phase='checkpoint-ready',frame=55,signature=signature,steps=steps),allow_resume_complete_saved_checkpoint=all(checks.values()),full61_protocol_complete=False,public_media_allowed=False,scope='Guard-paused complete saved frame55 is independently certified; failed original receipt remains immutable and nonresumable. Only a copied checkpoint with explicit versioned copy-only credential and isolated status reconstruction may advance55to60; no earlier step may be recomputed.')
 if sha(raw/'receipt.json')!=RECEIPT_SHA or sha(folder/'checkpoint.npz')!=CP_SHA:raise ValueError('Read-only inputs changed')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True,type=Path);p.add_argument('--producer',required=True,type=Path);p.add_argument('--f2-checkpoint',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if a.output.exists() or a.output.resolve().is_relative_to(a.raw.resolve()):raise ValueError('Fresh separate audit output required')
 r=audit(a.raw,a.producer,a.f2_checkpoint);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({k:r[k] for k in ('passed','frame','steps','remaining_steps','stress_error_Pa','stored_energy_error_J','momentum_error_Ns','maximum_energy_residual_J')}))
 if not r['passed']:raise SystemExit('Prefix failed independent audit')
