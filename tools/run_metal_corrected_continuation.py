"""Fail-closed f8 to f16 continuation of the exact corrected-source producer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import run_metal_corrected_prefix as base

PRODUCER_COMMIT = '21a0f3737f684a09bcdbb1b9388f74ddbc6410ae'
PRODUCER_RUN = '40073145856'
PRODUCER_ATTEMPT = '1'
PARENT_KEY = f'metal-corrected-prefix-v1-{PRODUCER_COMMIT}-{PRODUCER_RUN}-1-f8'
# Cache version and archive paths are bound to the producer's original path.
PARENT = base.ROOT/'.local/metal-corrected-prefix-cache'
base.CACHE = base.ROOT/'.local/metal-corrected-continuation-cache'
base.RECORDS = base.ROOT/'.local/metal-corrected-continuation-records'
EXPECTED_RUNTIME = dict(python='3.12.13', numpy='2.5.4', jax='0.12.0.1',
                        jaxlib='0.12.0.1', backend='cpu', x64=True)


def load(path):
    return json.loads(path.read_text())


def verify_parent(receipt_sha, folder=None):
    """The SHA must come from independent producer admission, never cache output."""
    folder = PARENT if folder is None else Path(folder)
    assert re.fullmatch('[0-9a-f]{64}', receipt_sha), 'Admitted receipt SHA256 required'
    assert base.digest(folder/'receipt.json') == receipt_sha, 'Producer receipt differs'
    receipt = load(folder/'receipt.json')
    assert receipt['schema'] == 'metal-corrected-source-prefix-v1'
    assert receipt['cache_key'] == PARENT_KEY
    source = receipt['source']
    assert source['repository'] == 'lililii124/jax-mpm'
    assert source['commit'] == PRODUCER_COMMIT
    assert str(source['run_id']) == PRODUCER_RUN and str(source['attempt']) == PRODUCER_ATTEMPT
    assert receipt['prefix_complete'] is True and receipt['never_resumable'] is False
    assert receipt['old_numeric_cache_restored'] is False
    assert receipt['physical_validation_passed'] is False and receipt['production_eligible'] is False
    assert receipt['requested_stop_frame'] == 8 and receipt['requested_speed_m_s'] == 5000
    assert not receipt.get('raw_files_omitted')
    files = base.inventory(folder)
    files.pop('receipt.json')
    assert files == receipt['files'], 'Whole producer cache inventory/hash mismatch'
    assert sum(item['bytes'] for item in files.values()) <= base.RAW_LIMIT
    assert load(folder/'source.json') == source
    assert load(folder/'runtime.json') == EXPECTED_RUNTIME
    result = folder/'result'
    status, summary, plan = (load(result/name) for name in ('status.json', 'summary.json', 'plan.json'))
    assert status['frame'] == 8 and status['frames'] == 60
    assert status['state_valid'] is True and status['segment_complete'] is True
    assert status['phase'] == 'checkpoint-ready'
    assert summary['runtime'] == EXPECTED_RUNTIME and summary['source'] == plan['source']
    assert summary['definition'] == plan and summary['segment_end_frame'] == 8
    physical = {key: value for key, value in plan.items() if key != 'source'}
    assert hashlib.sha256(json.dumps(physical, sort_keys=True, allow_nan=False).encode()).hexdigest() == base.PLAN_WITHOUT_SOURCE
    expected = dict(base.ORIGINAL_NUMERICAL_FILES)
    expected['mpm_jax/materials/shock_metal.py'] = base.MATERIAL_SOURCE
    assert plan['source']['files'] == expected
    for name, sha in expected.items():
        assert base.digest(base.ROOT/name) == sha, name
        assert source['files'][name]['sha256'] == sha, name
    checkpoint = load(result/'checkpoint.json')
    assert checkpoint['frame'] == 8 and checkpoint['signature'] == status['signature']
    for name, sha in checkpoint['files'].items():
        assert Path(name).name == name and base.digest(result/name) == sha, name
    assert not (result/'failed_state.npz').exists()
    return receipt


def prepare(receipt_sha):
    receipt = verify_parent(receipt_sha)
    assert os.environ['GITHUB_REPOSITORY'] == receipt['source']['repository']
    assert shutil.disk_usage(base.ROOT).free >= 6*1024**3
    # New controller source record; original numerical plan remains byte-identical.
    # The private feature template overrides this already registered workflow.
    base.prepare(extra_source_files=('.github/workflows/engineering-previews.yml',))
    assert load(base.RECORDS/'planned-full-protocol.json') == load(PARENT/'result'/'plan.json')
    shutil.copytree(PARENT/'result', base.CACHE/'result')
    shutil.copyfile(PARENT/'receipt.json', base.CACHE/'parent-receipt.json')
    shutil.copyfile(PARENT/'receipt.json', base.RECORDS/'parent-receipt.json')
    lineage = dict(parent_cache_key=PARENT_KEY, parent_receipt_sha256=receipt_sha,
        producer_commit=PRODUCER_COMMIT, producer_run_id=PRODUCER_RUN,
        start_frame=8, stop_frame=16, old_numeric_cache_restored=False,
        physical_validation_passed=False, production_eligible=False)
    base.write(base.CACHE/'continuation.json', lineage)
    base.write(base.RECORDS/'continuation.json', lineage)
    admission = load(base.RECORDS/'admission.json')
    admission.update(fresh_t0=False, stop_frame=16, start_frame=8,
        parent_cache_key=PARENT_KEY, parent_receipt_sha256=receipt_sha,
        scope='Admitted corrected-source f8 to f16 resource continuation only; no phase/full-impact validation.')
    base.write(base.RECORDS/'admission.json', admission)
    # Full copied producer bytes are checked before the solver may modify them.
    for name, record in receipt['files'].items():
        if name.startswith('result/'):
            assert base.digest(base.CACHE/name) == record['sha256'], name


def context(receipt_sha):
    assert re.fullmatch('[0-9a-f]{64}', receipt_sha), 'Admitted receipt SHA256 required'
    lineage = load(base.RECORDS/'continuation.json')
    assert lineage == load(base.CACHE/'continuation.json')
    assert lineage['parent_receipt_sha256'] == receipt_sha and lineage['parent_cache_key'] == PARENT_KEY
    assert base.digest(base.CACHE/'parent-receipt.json') == receipt_sha
    assert lineage['start_frame'] == 8 and lineage['stop_frame'] == 16
    return lineage


def run(receipt_sha):
    context(receipt_sha)
    assert not (base.RECORDS/'monitor.json').exists(), 'One continuation attempt only'
    assert load(base.CACHE/'result'/'status.json')['frame'] == 8
    assert not (base.CACHE/'result'/'failed_state.npz').exists(), 'Failed state is never resumable'
    source = load(base.CACHE/'source.json')
    assert source['commit'] == os.environ['GITHUB_SHA']
    for name, record in source['files'].items():
        assert base.digest(base.ROOT/name) == record['sha256'], name
    base.run(resume=True, stop_frame=16)


def seal(receipt_sha):
    base.seal(stop_frame=16, key_prefix='metal-corrected-continuation-v1',
              lineage=context(receipt_sha))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'seal', 'deliver'))
    parser.add_argument('--receipt-sha', required=True,
                        help='SHA256 independently admitted from producer run 40073145856')
    args = parser.parse_args()
    if args.phase == 'deliver':
        base.deliver(destination=base.ROOT/'.local/metal-corrected-continuation-delivery')
    else:
        globals()[args.phase](args.receipt_sha)
