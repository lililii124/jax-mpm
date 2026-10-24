"""Read-only original-review adapter and separately sealed frame-10 recovery."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import run_ceramic_continuation as c

FAILED_KEY = c.FIRST_KEY + '-failed'
FAILED_RECEIPT_SHA = 'aac212a3141be0b347d13b3f14eb972e402df678f36b65640025f541f006a5ff'
ORIGINAL_RECORD_SHA = {
    'ceramic-prefix.json': '62889d8fe07e2205fd092c13bda3bb658e0290ad967b26ac1241ba0bc9d2843f',
    'ceramic-prefix-review.json': '670e9bf3531034841d93a969f355707c5328c6c5d7cf5d8e82f5f6b853e94dcf',
}


def normalized(value):
    import numpy as np
    if isinstance(value, np.generic):
        return normalized(value.item())
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    return value


def original_review(folder, allow_prefix):
    """Execute the exact producer review function; adapt JSON scalars only."""
    path = c.SOLVER/'tools/review_ceramic_fragmentation.py'
    raw = path.read_bytes()
    if raw != c.git(c.SOLVER, 'show', f'{c.PRODUCER}:tools/review_ceramic_fragmentation.py'):
        raise ValueError('Reviewer differs from original producer blob')
    spec = importlib.util.spec_from_file_location('original_ceramic_review', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = normalized(module.review(Path(folder), allow_prefix=allow_prefix))
    if (report.get('passed') is not True or not report.get('checks')
            or not all(value is True for value in report['checks'].values())):
        raise ValueError('Original physical review rejected these saved bytes')
    c.save(Path(folder)/'fragmentation-review.json', report)
    return report


def verify_recovery(receipt, base):
    """An accepted recovery has its own identity; its failed ancestor stays failed."""
    if (receipt.get('schema') != 'ceramic-review-recovery-v1'
            or receipt.get('kind') != 'review-only recovered'
            or receipt.get('accepted_frame') != 10 or receipt.get('prefix_verified') is not True
            or receipt.get('never_resumable') is not False
            or receipt.get('full_protocol_completed') is not False):
        raise ValueError('Invalid review-only recovery receipt')
    c.verify_controller(receipt['controller'])
    parent = receipt['failed_parent']
    if (parent.get('receipt_sha256') != FAILED_RECEIPT_SHA
            or receipt.get('production_eligible') is not False
            or receipt.get('requested_stroke_m') != .012 or receipt.get('full_target_s') != 3e-4
            or not receipt.get('cache_key', '').startswith('ceramic-review-recovery-v1-'+c.PRODUCER+'-')):
        raise ValueError('Recovery identity or original plan differs')
    path = c.local_file(base, 'receipts/'+parent['receipt_sha256']+'.json')
    old = c.read(path)
    if (c.digest(path) != parent['receipt_sha256'] or old.get('cache_key') != FAILED_KEY
            or old.get('never_resumable') is not True or old.get('prefix_complete') is not False
            or old.get('source') != receipt['source'] or parent['cache_key'] != FAILED_KEY):
        raise ValueError('Failed ancestor identity was changed or misrepresented')
    for name, record in old['files'].items():
        if receipt.get('files', {}).get(name) != record:
            raise ValueError('Recovered original raw differs: '+name)
    if receipt.get('audit', {}).get('prefix_verified') is not True:
        raise ValueError('Recovery lacks a complete checkpoint audit')


def recover(args):
    source, records = args.failed_cache.resolve(), args.records.resolve()
    if c.CACHE.exists() or source == c.CACHE.resolve():
        raise ValueError('Recovery requires a fresh destination distinct from failed raw')
    receipt_path = source/'receipt.json'
    if args.receipt_sha != FAILED_RECEIPT_SHA or c.digest(receipt_path) != FAILED_RECEIPT_SHA:
        raise ValueError('Failed receipt differs from independently admitted SHA')
    old = c.read(receipt_path)
    if (old.get('schema') != 'ceramic-resource-prefix-v1' or old.get('cache_key') != FAILED_KEY
            or old.get('never_resumable') is not True or old.get('prefix_complete') is not False
            or old.get('requested_stop_frame') != 10 or old.get('requested_stroke_m') != .012
            or old.get('full_target_s') != 3e-4 or old.get('production_eligible') is not False
            or c.inventory(source) != old['files']):
        raise ValueError('Original failed receipt/raw inventory differs')
    c.verify_source(old['source'])
    if c.read(source/'source.json') != old['source']:
        raise ValueError('Original source receipt differs')
    if any(c.digest(records/name) != sha for name, sha in ORIGINAL_RECORD_SHA.items()):
        raise ValueError('Original command records differ from admitted artifact bytes')
    solve, review = (c.read(records/name) for name in ('ceramic-prefix.json', 'ceramic-prefix-review.json'))
    log = args.job_log.read_text(encoding='utf-8')
    if (c.digest(args.job_log) != args.job_log_sha or solve.get('completed') is not True
            or solve.get('returncode') != 0 or solve['memory']['stopped_for_memory']
            or review.get('completed') is not False or review.get('returncode') != 1
            or review['memory']['stopped_for_memory']
            or 'review_ceramic_fragmentation.py' not in log
            or 'line 222' not in log or 'json.dumps(report' not in log
            or 'TypeError: Object of type bool is not JSON serializable' not in log):
        raise ValueError('Original evidence does not isolate the serialization failure')
    for name in ORIGINAL_RECORD_SHA:
        shutil.copyfile(records/name, c.LOGS/('original-'+name))
    shutil.copyfile(args.job_log, c.LOGS/'original-job.log')
    shutil.copyfile(receipt_path, c.LOGS/'original-failed-receipt.json')
    status = c.read(source/'result/status.json')
    if (status.get('phase') != 'checkpoint-ready' or status.get('segment_complete') is not True
            or status.get('frame') != 10 or status.get('frames') != 60
            or c.read(source/'result/checkpoint.json').get('frame') != 10):
        raise ValueError('Original solve did not complete the full-plan frame-10 prefix')
    shutil.copytree(source, c.CACHE)
    if c.inventory() != old['files']:
        raise ValueError('Whole raw recovery copy differs')
    history = c.CACHE/'receipts'/f'{args.receipt_sha}.json'
    history.parent.mkdir()
    shutil.copyfile(receipt_path, history)
    report = original_review(c.CACHE/'result', True)
    if report['accepted_saved_frame'] != 10:
        raise ValueError('Original reviewer endpoint differs')
    context = {'source': old['source'], 'pins': c.dependency_pins((source/'requirements.txt').read_text())}
    c.audit_worker(context, 10)
    controller = c.controller_record()
    receipt = dict(schema='ceramic-review-recovery-v1', kind='review-only recovered',
        cache_key=f"ceramic-review-recovery-v1-{c.PRODUCER}-{controller['commit']}-{controller['run_id']}-{controller['attempt']}-f10",
        source=old['source'], controller=controller, never_resumable=False, prefix_verified=True,
        accepted_frame=10, full_protocol_completed=False, requested_stroke_m=.012, full_target_s=3e-4,
        production_eligible=False, media_created=False, public_visual_review_passed=False,
        permanent_raw_backup_verified=False,
        failed_parent=dict(cache_key=FAILED_KEY, receipt_sha256=args.receipt_sha),
        failure_evidence=dict(job_log_sha256=args.job_log_sha,
            records={name:c.file_record(records/name) for name in ('ceramic-prefix.json','ceramic-prefix-review.json')}),
        audit=c.read(c.LOGS/'prefix-10-audit.json'), files=c.inventory(),
        scope='Original failed cache remains nonresumable. Separate whole-byte copy accepted by unchanged original review and state audit; no simulation executed.')
    verify_recovery(receipt, c.CACHE)
    if sum(record['bytes'] for record in receipt['files'].values()) + len(json.dumps(receipt)) > c.RAW_LIMIT:
        raise ValueError('Complete recovery exceeds its raw cache bound')
    c.save(c.CACHE/'receipt.json', receipt)
    c.save(c.LOGS/'recovery-receipt.json', receipt)
    if c.inventory(source) != old['files'] or c.digest(receipt_path) != args.receipt_sha:
        raise ValueError('Original failed cache changed')
    c.emit(cache_key=receipt['cache_key'], receipt_sha=c.digest(c.CACHE/'receipt.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('review', 'recover'))
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--allow-prefix', action='store_true')
    parser.add_argument('--failed-cache', type=Path)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--receipt-sha')
    parser.add_argument('--job-log', type=Path)
    parser.add_argument('--job-log-sha')
    args = parser.parse_args()
    c.LOGS.mkdir(parents=True, exist_ok=True)
    if args.action == 'review': original_review(args.folder, args.allow_prefix)
    else: recover(args)


if __name__ == '__main__':
    main()
