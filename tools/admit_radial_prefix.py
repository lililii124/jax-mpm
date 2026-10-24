"""Read-only independent actual f10 cache audit, no numerical advance."""
import hashlib,importlib.metadata,importlib.util,json,os,platform,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FROZEN=ROOT/'.local/radial-frozen-producer';PARENT=ROOT/'.local/radial-sinusoidal-full';OUT=ROOT/'.local/radial-prefix-admission'
COMMIT='de2bfea446e66e6245e33329a663996c5d5b9e84';RECEIPT='d0af82b261b0433c024b80472cb5da67ee5d0113f11d2876f0510ee8f631c591'
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==os.environ['GITHUB_SHA']
assert os.environ['GITHUB_REF']=='refs/heads/work/radial-sinusoidal-full'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=FROZEN).decode().strip()==COMMIT
for n in ['tools/admit_radial_prefix.py','.github/workflows/radial-prefix-admission.yml','.github/workflows/engineering-previews.yml']:
 assert subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes()
assert digest(PARENT/'receipt.json')==RECEIPT
receipt=json.loads((PARENT/'receipt.json').read_text());assert receipt['status']=='accepted-prefix' and receipt['commit']==COMMIT and receipt['run_id']=='40090780381'
sys.path.insert(0,str(FROZEN));sys.path.insert(0,str(FROZEN/'tools'))
spec=importlib.util.spec_from_file_location('radial_producer',FROZEN/'tools/radial_sinusoidal_cloud.py');producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
assert producer.sources()==receipt['source_files']
runtime={n:importlib.metadata.version(n) for n in receipt['runtime']};assert runtime==receipt['runtime'] and platform.python_version()=='3.12.13'
before=producer.inventory(PARENT/'result');assert before==receipt['raw_files']
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir();shutil.copytree(PARENT/'result',OUT/'result')
producer.validate_prefix(OUT/'result')
audit=json.loads((OUT/'prefix-audit.json').read_text());assert audit['status']=='accepted-prefix' and audit['saved_states']==11
summary=json.loads((OUT/'result/summary.json').read_text())
assert producer.inventory(PARENT/'result')==before
report=dict(schema='radial-sinusoidal-f10-independent-admission-v1',admitted_prefix=True,full_protocol_accepted=False,solver_executed=False,raw_unchanged=True,producer_commit=COMMIT,producer_run=40090780381,producer_receipt_sha256=RECEIPT,controller_commit=os.environ['GITHUB_SHA'],controller_run=os.environ['GITHUB_RUN_ID'],cache_key=receipt['cache_key'],source_files=receipt['source_files'],runtime=runtime,prefix_audit=audit,all_raw_files=before,summary=summary,scope='Numerical f10 admissibility only; full50 pull-off/deformation contrast remains unassessed')
(OUT/'independent-admission.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
