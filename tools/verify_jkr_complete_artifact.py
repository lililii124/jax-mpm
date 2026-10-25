"""Read-only official whole-artifact and accepted-state verification; zero MPM steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.review_jkr_compliant_pilot import descriptor, review


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(run_id, artifact_id, commit, output):
    if output.exists():
        raise ValueError('Fresh verification destination required')
    output.mkdir(parents=True)
    repository = os.environ['GITHUB_REPOSITORY']
    token = os.environ['GH_TOKEN']
    base = 'https://api.github.com/repos/'+repository
    headers = dict(Authorization='Bearer '+token, Accept='application/vnd.github+json',
                   **{'User-Agent':'jax-mpm-read-only-artifact-verifier'})
    def api(path):
        with urllib.request.urlopen(urllib.request.Request(base+path, headers=headers), timeout=30) as response:
            return json.load(response)
    run = api('/actions/runs/'+str(run_id))
    artifact = api('/actions/artifacts/'+str(artifact_id))
    assert run['head_sha'] == commit and run['conclusion'] == 'success'
    assert artifact['workflow_run']['id'] == run_id and artifact['workflow_run']['head_sha'] == commit
    assert not artifact['expired'] and artifact['digest'].startswith('sha256:')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, response, code, message, response_headers, newurl):
            return None
    try:
        urllib.request.build_opener(NoRedirect).open(urllib.request.Request(
            base+'/actions/artifacts/'+str(artifact_id)+'/zip', headers=headers), timeout=30)
        raise RuntimeError('Expected authenticated artifact redirect')
    except urllib.error.HTTPError as response:
        if response.code != 302:
            raise
        signed_url = response.headers['Location']
    # The signed blob request deliberately carries no GitHub Authorization.
    archive = output/'official-artifact.zip'
    with urllib.request.urlopen(signed_url, timeout=60) as response, archive.open('wb') as stream:
        for block in iter(lambda:response.read(1024*1024), b''):
            stream.write(block)
    assert archive.stat().st_size == artifact['size_in_bytes']
    archive_sha = sha(archive)
    assert 'sha256:'+archive_sha == artifact['digest'], 'official complete ZIP digest differs'
    restored = output/'restored'
    with zipfile.ZipFile(archive) as zipped:
        assert sum(info.file_size for info in zipped.infolist()) < 2*1024**3
        for info in zipped.infolist():
            name = PurePosixPath(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            assert (info.external_attr >> 16)&0o170000 != 0o120000
        assert zipped.testzip() is None, 'ZIP member CRC failed'
        zipped.extractall(restored)
        member_count = len(zipped.infolist())
    assert (restored/'controller/commit.txt').read_text().strip() == commit
    preflight = json.loads((restored/'preflight.json').read_text())
    for name, expected in preflight['source_files'].items():
        committed = subprocess.check_output(['git','show',commit+':'+name], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == expected, 'source manifest differs from committed Git blob'
        assert sha(restored/'source'/name) == expected, 'retrieved complete source differs'
    folder = restored/'data/jkr'
    plan = json.loads((folder/'plan.json').read_text())
    with np.load(folder/'checkpoint.npz') as cp:
        assert str(cp['signature']) == hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        frame, steps = int(cp['frame']), int(cp['progress_0'])
        assert abs(float(cp['state_time'])-steps*plan['dt']) < 1e-11
        assert steps == plan['frame_steps'][frame]
        assert bool(cp['contact_valid']) and bool(np.all(cp['contact_law_valid']))
        tables = json.loads(str(cp['tables']))
        signature = str(cp['signature'])
        runtime = json.loads(str(cp['runtime']))
    with np.load(folder/'trajectory.npz') as index:
        np.testing.assert_array_equal(index['time'], tables['times'])
    summary = json.loads((folder/'summary.json').read_text())
    status = json.loads((folder/'status.json').read_text())
    assert status['phase'] in ('checkpoint-ready','completed') and not status.get('never_resumable',False)
    assert status['frame'] == frame and status['signature'] == signature
    assert (status['phase'] == 'completed') == (frame == plan['frames'] and steps == plan['steps'])
    assert summary['definition'] == plan and summary['steps'] == steps
    assert summary['runtime'] == runtime
    assert summary['common_request_sha256'] == plan['common_request_sha256'] == preflight['common_request_sha256']
    for name, expected in plan['source']['files'].items():
        assert preflight['source_files'][name] == expected
    assert plan['source']['fingerprint'] == hashlib.sha256(json.dumps(
        plan['source']['files'],sort_keys=True).encode()).hexdigest()
    with np.load(folder/'particle_fields.npz') as labels:
        for name in labels.files:
            assert descriptor(labels[name]) == summary['particle_fields']['fields'][name]
    before = sha(folder/'checkpoint.npz')
    independent = review(folder, output/'independent-state-review.json')
    assert sha(folder/'checkpoint.npz') == before
    assert 'jax' not in sys.modules, 'read-only verifier must not initialize JAX'
    result = dict(passed=True, numerical_run_id=run_id, artifact_id=artifact_id,
        numerical_commit=commit, verifier_commit=os.environ.get('GITHUB_SHA'),
        official_complete_archive_sha256=archive_sha, archive_bytes=archive.stat().st_size,
        official_digest_matched=True, all_zip_members_crc_checked=member_count,
        committed_source_files_checked=len(preflight['source_files']),
        checkpoint_sha256=before, accepted_frame=frame, accepted_steps=steps,
        independent_state_review=independent, physical_steps_executed=0,
        jax_initialized=False, public_visual_review_passed=False, production_eligible=False)
    (output/'whole-artifact-final-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'independent_state_review'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', type=int, required=True)
    parser.add_argument('--artifact-id', type=int, required=True)
    parser.add_argument('--numerical-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_id,args.artifact_id,args.numerical_commit,args.output.resolve())
