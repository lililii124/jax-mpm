"""Source and official artifact admission, independent of numerical imports."""
import hashlib,json,os,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
FIXED_REF='refs/heads/work/single-fibre-energy-replay'
OWN='.github/workflows/single-fibre-energy-replay.yml'
REPO='lililii124/jax-mpm'
BASELINE={'run':'40117848533','artifact':'12571264792','commit':'697e19bda2f2b4c8f473cdefdcc070baff27de90','zip':'e8ccbe5caf2bb7e38703e898771600acaece415fc29474f673df5d916fce328e'}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def bind():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    if not re.fullmatch('[0-9a-f]{40}',os.environ.get('SOURCE_REF','')) or head!=os.environ['SOURCE_REF'] or head!=os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF')!=FIXED_REF or os.environ.get('GITHUB_REPOSITORY')!=REPO:raise ValueError('Exact HEAD/repository/fixed ref required')
    subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=ROOT,check=True)
    active='.github/workflows/'+os.environ['GITHUB_WORKFLOW_REF'].split('/.github/workflows/',1)[1].split('@',1)[0]
    if active!='.github/workflows/engineering-previews.yml' or (ROOT/active).read_bytes()!=(ROOT/OWN).read_bytes():raise ValueError('Registered workflow override differs')
    admission=json.loads((ROOT/'tools/single_fibre_energy_replay/source-admission.json').read_text())
    names=['tools/single_fibre_energy_replay/source-admission.json',OWN,active,'tools/package_single_fibre_raw.py','mpm_jax/fibre_mpm.py','mpm_jax/materials/isotropic_brittle.py','examples/single_fibre_tension.py']+['tools/single_fibre_energy_replay/'+n for n in ('control.py','replay.py','guard.py','energy_terms.py')]
    files={}
    for n in sorted(set(names)):
        raw=(ROOT/n).read_bytes()
        if raw!=subprocess.check_output(['git','show',head+':'+n],cwd=ROOT):raise ValueError('Git source blob differs: '+n)
        files[n]=sha(raw)
    if admission['baseline']!=BASELINE:raise ValueError('Baseline admission identity differs')
    if admission['files']!={n:h for n,h in files.items() if not n.endswith('/source-admission.json')}:raise ValueError('Publisher source admission SHA inventory differs')
    return {'commit':head,'ref':FIXED_REF,'files':files}
def official(run_id,artifact_id,commit,zip_sha,destination,expected_conclusion):
    def api(path):return json.loads(subprocess.check_output(['gh','api','repos/'+REPO+'/'+path]))
    run=api('actions/runs/'+run_id);artifact=api('actions/artifacts/'+artifact_id)
    if run['head_sha']!=commit or run['status']!='completed' or run['conclusion']!=expected_conclusion:raise ValueError('Official parent run identity differs')
    if artifact['expired'] or artifact['workflow_run']['id']!=int(run_id) or artifact['workflow_run']['head_sha']!=commit or artifact['digest']!='sha256:'+zip_sha:raise ValueError('Official artifact identity/digest differs')
    destination=Path(destination)
    if destination.exists():raise ValueError('Fresh official extraction required')
    destination.parent.mkdir(parents=True,exist_ok=True);archive=destination.with_suffix('.zip')
    with archive.open('wb') as f:subprocess.run(['gh','api','--allow-escape-sequences','repos/'+REPO+'/actions/artifacts/'+artifact_id+'/zip'],stdout=f,check=True)
    if sha(archive.read_bytes())!=zip_sha:raise ValueError('Downloaded official ZIP SHA differs')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Official ZIP CRC differs')
        for n in z.namelist():
            target=(destination/n).resolve()
            if not target.is_relative_to(destination.resolve()):raise ValueError('Unsafe ZIP member')
        z.extractall(destination)
    return {'run':run,'artifact':artifact,'zip_sha256':zip_sha}
def prepare():
    context=bind();end=os.environ['END_FRAME']
    if end not in ('2','60'):raise ValueError('Invalid frame request')
    parent=[os.environ.get(n,'') for n in ('PARENT_RUN','PARENT_ARTIFACT','PARENT_SHA','PARENT_COMMIT','PARENT_ZIP')]
    if (end=='2' and any(parent)) or (end=='60' and not all(parent)):raise ValueError('Exact full parent inputs required; f2 must be fresh')
    b=BASELINE;records={'source':context,'baseline':official(b['run'],b['artifact'],b['commit'],b['zip'],ROOT/'.local/energy-original-capsule','success')}
    if end=='60':
        run,artifact,receipt,commit,zip_sha=parent
        if commit!=context['commit']:raise ValueError('Full parent must be from same source commit')
        records['parent']=official(run,artifact,commit,zip_sha,ROOT/'.local/energy-replay','success')
        if sha((ROOT/'.local/energy-replay/receipt.json').read_bytes())!=receipt:raise ValueError('Official parent receipt SHA differs')
    log=ROOT/'.local/energy-replay-logs';log.mkdir(parents=True,exist_ok=True)
    (log/'admission.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':prepare()
