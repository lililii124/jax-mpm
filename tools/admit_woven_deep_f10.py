"""Independent read-only exact f10 whole-raw audit; never advances a time step."""
import hashlib,importlib.metadata,json,os,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import rht_impact_prefix as transport
from examples import woven_deep_impact as deep
from examples import woven_large_plate as base
PARENT=ROOT/'.local/deep-f10-parent';OUT=ROOT/'.local/deep-f10-independent-admission';RAW=OUT/'raw';COMMIT='d7364d8e6ec1790379a3ed16921e8dbae67c6945';MANIFEST='da69c2ede481617a0dff7f33f61d64a9bdb5fba0b317d5e52a06308655490a34'
assert subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()==os.environ['GITHUB_SHA']
assert os.environ['GITHUB_REF']=='refs/heads/work/woven-deep-impact'
for n in ['tools/admit_woven_deep_f10.py','.github/workflows/woven-deep-f10-admission.yml','.github/workflows/engineering-previews.yml']:
 assert (ROOT/n).read_bytes()==subprocess.check_output(['git','show','HEAD:'+n])
assert transport.digest(PARENT/'transport.json')==MANIFEST
transport.verify(PARENT);manifest=json.loads((PARENT/'transport.json').read_text())
if OUT.exists():raise FileExistsError(OUT)
RAW.mkdir(parents=True)
reader=transport.PartReader(PARENT/name for name in manifest['parts'])
with tarfile.open(fileobj=reader,mode='r|') as archive:
 for member in archive:
  assert member.isfile() and member.name in manifest['files'];target=RAW/member.name
  assert target.resolve().is_relative_to(RAW.resolve());target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as src,target.open('xb') as dest:transport.shutil.copyfileobj(src,dest,1024**2)
source=json.loads((RAW/'source.json').read_text());receipt=json.loads((RAW/'receipt.json').read_text());assert source['commit']==receipt['commit']==COMMIT and receipt['status']['frame']==10
for name,r in source['files'].items():
 old=subprocess.check_output(['git','show',COMMIT+':'+name]);assert len(old)==r['bytes'] and hashlib.sha256(old).hexdigest()==r['sha256']
assert source['runtime']['python']=='3.12.13'
for name in ['numpy','jax','jaxlib','psutil']:assert importlib.metadata.version(name)==source['runtime'][name]
folder=RAW/'output/plain/deep-sphere-impact';before={p.relative_to(folder).as_posix():transport.digest(p) for p in folder.rglob('*') if p.is_file()}
sample,plan=base.load_saved_case(folder);status=json.loads((folder/'status.json').read_text());assert status==receipt['status'] and status['steps']==2410 and status['coupling_valid']
with np.load(folder/'checkpoint.npz',allow_pickle=False) as cp:
 base.validate_checkpoint_control(cp,plan);assert int(cp['frame'])==10 and str(cp['signature'])==base.digest(plan)
 for name in cp.files:
  if np.issubdtype(cp[name].dtype,np.number):assert np.isfinite(cp[name]).all(),name
 tables=json.loads(str(cp['tables']));assert len(tables['trajectory_frames'])==len(tables['times'])==11
 for name in tables['trajectory_frames'][0]:
  assert name.isidentifier();values=np.load(folder/f'trajectory_{name}.npy',mmap_mode='r',allow_pickle=False)
  for i in range(11):assert base.array_descriptor(values[i])==tables['trajectory_frames'][i][name]
review=deep.review(folder);assert all(review[k] for k in ['stable','clamp_passed','material_energy_passed','energy_passed'])
assert before=={p.relative_to(folder).as_posix():transport.digest(p) for p in folder.rglob('*') if p.is_file()}
report=dict(schema='woven-deep-f10-independent-admission-v1',admitted_prefix=True,producer_run=40091612689,producer_commit=COMMIT,transport_sha256=MANIFEST,controller_commit=os.environ['GITHUB_SHA'],controller_run=os.environ['GITHUB_RUN_ID'],source=source,receipt=receipt,review=review,saved_states_checked=11,all_raw_files=before,raw_unchanged=True,solver_executed=False,full_protocol_accepted=False,final_publication_gate=dict(minimum_peak_m=.005,baseline_peak_m=.0025261506417153324,old_numerical_target_m=.002,original_plan_unchanged=True))
(OUT/'independent-admission.json').write_text(json.dumps(report,indent=2)+'\n')
