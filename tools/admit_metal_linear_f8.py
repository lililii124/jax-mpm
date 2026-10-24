"""Read-only whole-artifact admission; never invokes the numerical worker."""
import csv,json,os,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import jax,jaxlib,psutil
import run_metal_linear_continuation as route
from examples.metal_sphere_impact_linear import descriptor,digest,save_json,table_text
jax.config.update('jax_enable_x64',True)
runtime=dict(python=platform.python_version(),numpy=np.__version__,jax=jax.__version__,jaxlib=jaxlib.__version__,backend=jax.default_backend(),x64=bool(jax.config.jax_enable_x64))
assert runtime==route.EXPECTED_RUNTIME and psutil.__version__=='7.2.2'
route.restore(allow_pending_readonly=True)
folder=route.DEST/'producer/result'
before={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}
summary=route.load(folder/'summary.json');status=route.load(folder/'status.json');plan=route.load(folder/'plan.json')
with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
 assert int(cp['frame'])==8 and int(cp['steps'])==295 and str(cp['signature'])==status['signature']
 assert json.loads(str(cp['runtime']))==runtime
 tables=json.loads(str(cp['tables']));assert len(tables['frames'])==len(tables['times'])==len(tables['rows'])==len(tables['diagnostics'])==9
 for key in cp.files:
  if np.issubdtype(cp[key].dtype,np.number):assert np.isfinite(cp[key]).all(),key
 assert bool(cp['state_valid'])
 for name in ('rows','diagnostics'):
  target='observables.csv' if name=='rows' else 'diagnostics.csv'
  assert (folder/target).read_bytes()==table_text(tables[name]).encode()
 for field in ('x','speed','temperature','damage','failed'):
  values=np.load(folder/f'trajectory_{field}.npy',mmap_mode='r',allow_pickle=False)
  for i in range(9):
   assert descriptor(values[i])==tables['frames'][i][field]
   assert np.isfinite(values[i]).all()
  if field=='x':assert np.array_equal(values[8],cp['state_x'])
  if field in ('damage','failed'):assert not np.any(values[:9])
rows=tables['rows'];diagnostics=tables['diagnostics']
assert all(d['state_valid'] and .4<d['minimum_J']<=d['maximum_J']<2 and abs(d['mass_relative_drift'])<1e-12 for d in diagnostics)
assert summary['completed'] is False and summary['physical_validation_passed'] is False
assert abs(tables['times'][-1]-8e-6/60)<1e-20
assert before=={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}
report=dict(schema='linear-metal-f8-independent-admission-v1',admitted_prefix=True,full_protocol_accepted=False,physical_validation_passed=False,solver_executed=False,raw_unchanged=True,producer=route.load(route.ADMISSION),controller_commit=os.environ['GITHUB_SHA'],controller_run=os.environ['GITHUB_RUN_ID'],runtime=runtime,saved_states_checked=9,fields_checked=5,checkpoint_arrays_finite=True,final_diagnostics=diagnostics[-1],final_observables=rows[-1],minimum_J=min(d['minimum_J'] for d in diagnostics),maximum_J=max(d['maximum_J'] for d in diagnostics),maximum_temperature=max(d['maximum_temperature'] for d in diagnostics),maximum_physical_energy_drift_J=max(abs(r['physical_energy_drift_J']) for r in rows),maximum_accounted_energy_drift_J=max(abs(r['energy_drift_J']) for r in rows),all_original_raw_sha256=before,scope='Numerical prefix admission only; linear volume elasticity and plastic-work calorimetry are illustrative, not calibrated shock or phase physics')
save_json(route.DEST/'independent-admission.json',report)
