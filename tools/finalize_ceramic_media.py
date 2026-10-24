"""Read-only sealed ceramic finalization; no solver entrypoint is called."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_ceramic_continuation as c
import probe_ceramic_full_review as probe
from reconcile_ceramic_signature import reconcile

OUT = c.ROOT / '.local/ceramic-final-media'
REGISTERED = '.github/workflows/engineering-previews.yml'
TEMPLATE = '.github/workflows/ceramic-final-media.yml'
LIMIT = 8_000_000_000
REVIEW_SHA = '5e719584ddebe204650ea17b9125f82b21d954b0a512ba99624edae90b9ae61b'
PROVENANCE_SHA = '509206e1d5a322ede351eabecb0bfdbfd7f646b4f48e3e3092148c261dda5a61'
PROOF_SHA = '04ba94ea8d8cd4aa74d4f3031770e08f0eb67efa544d96b5a838ed80c2cdfe52'


def private_workflow_bytes(root):
    text = (Path(root)/TEMPLATE).read_text(encoding='utf-8')
    first, rest = text.split('\n', 1)
    if first != 'name: feature ceramic read-only final media':
        raise ValueError('Unexpected finalization template')
    return ('name: private ceramic read-only final media\n'+rest).encode()


def controller():
    commit = c.git(c.ROOT, 'rev-parse', 'HEAD').decode().strip()
    if (commit != os.environ.get('GITHUB_SHA') or os.environ.get('GITHUB_REF') != c.FEATURE_REF
            or os.environ.get('GITHUB_WORKFLOW_REF', '').split('@')[0] != c.REPOSITORY+'/'+REGISTERED):
        raise ValueError('Requires committed private registered feature workflow')
    names = (REGISTERED, TEMPLATE, 'tools/finalize_ceramic_media.py',
             'tools/reconcile_ceramic_signature.py', 'tools/probe_ceramic_full_review.py',
             'tools/run_ceramic_continuation.py', 'tools/check_jc_final_media.py',
             'mpm_jax/visualization.py', 'mpm_jax/paraview_scene.py', 'mpm_jax/export.py')
    for name in names:
        if (c.ROOT/name).read_bytes() != c.git(c.ROOT, 'show', commit+':'+name):
            raise ValueError('Uncommitted finalization source: '+name)
    if (c.ROOT/REGISTERED).read_bytes() != private_workflow_bytes(c.ROOT):
        raise ValueError('Registered workflow differs from private template')
    return {'commit': commit, 'files': {n:c.file_record(c.ROOT/n) for n in names}}


def verify_raw():
    receipt = c.read(c.CACHE/'receipt.json')
    if (c.digest(c.CACHE/'receipt.json') != probe.RECEIPT_SHA
            or receipt.get('cache_key') != probe.FAILED_KEY
            or receipt.get('never_resumable') is not True
            or receipt.get('kind') != 'failed-diagnostic'
            or c.inventory() != receipt['files']):
        raise ValueError('Sealed failed cache or complete raw inventory differs')
    c.verify_source(receipt['source'])
    c.verify_controller(receipt['controller'])
    if c.read(c.CACHE/'source.json') != receipt['source']:
        raise ValueError('Source receipt differs')
    return receipt


def prepare(args):
    if OUT.exists():
        raise FileExistsError('Finalization requires fresh separate output')
    identity = controller()
    if os.environ.get('CACHE_HIT') != 'true' or os.environ.get('CACHE_MATCH') != probe.FAILED_KEY:
        raise ValueError('Exact failed cache restore required without fallback')
    receipt = verify_raw()
    review = args.original_review
    provenance = c.read(args.evidence/'diagnostic-provenance.json')
    preservation = c.read(args.evidence/'raw-preservation.json')
    if (args.review_sha != REVIEW_SHA or args.provenance_sha != PROVENANCE_SHA
            or c.digest(review) != REVIEW_SHA or c.digest(args.evidence/'diagnostic-provenance.json') != PROVENANCE_SHA
            or provenance.get('failed_cache_key') != probe.FAILED_KEY
            or provenance.get('failed_receipt_sha256') != probe.RECEIPT_SHA
            or provenance.get('source') != receipt['source']
            or provenance.get('solver_executed') is not False
            or preservation.get('whole_raw_inventory_unchanged') is not True):
        raise ValueError('Independent full review provenance differs')
    report = c.read(review)
    if (len(report.get('checks', {})) != 30 or report.get('completed') is not True
            or [n for n,v in report['checks'].items() if v is not True] != ['full_plan_signature_matches_checkpoint']):
        raise ValueError('Independent review must pass every original gate except signature serialization')
    correction = reconcile(c.CACHE/'result', review)
    if (correction['schema'] != 'ceramic-signature-reconciliation-v1'
            or correction['all_original_physical_checks_after_correction'] is not True
            or correction['remaining_failed_checks']):
        raise ValueError('Versioned reconciliation did not restore all gates')
    proof = c.read(args.proof)
    corrected_checks = dict(report['checks'])
    corrected_checks['full_plan_signature_matches_checkpoint'] = True
    canonical_inventory = __import__('hashlib').sha256(json.dumps(receipt['files'], sort_keys=True).encode()).hexdigest()
    if (c.digest(args.proof) != PROOF_SHA
            or proof.get('schema') != 'ceramic-fragmentation-independent-review-v2'
            or proof.get('passed') is not True or len(proof.get('checks', {})) != 30
            or not all(v is True for v in proof['checks'].values())
            or proof['checks'] != corrected_checks
            or proof.get('original_review_sha256') != REVIEW_SHA
            or proof.get('original_failed_receipt_sha256') != probe.RECEIPT_SHA
            or proof.get('original_failed_cache_key') != probe.FAILED_KEY
            or proof.get('raw_inventory_canonical_json_sha256') != canonical_inventory
            or proof.get('serialization_reconciliation') != correction
            or proof.get('reconciler_source_sha256') != c.digest(c.ROOT/'tools/reconcile_ceramic_signature.py')
            or proof.get('numerical_solver_rerun') is not False
            or proof.get('public_visual_review_passed') is not False):
        raise ValueError('Exact independent v2 proof differs from restored raw or recomputed reconciliation')
    for name, sha in proof['sources'].items():
        if c.digest(c.local_file(c.CACHE/'result', name)) != sha:
            raise ValueError('Independent v2 source mismatch: '+name)
    OUT.mkdir(parents=True)
    shutil.copyfile(args.proof, OUT/'independent-physical-review-v2.json')
    shutil.copyfile(review, OUT/'original-full-physical-review.json')
    c.save(OUT/'signature-reconciliation.json', correction)
    shutil.copytree(args.evidence, OUT/'independent-review')
    shutil.copytree(c.CACHE/'result', OUT/'result')
    for name in ('receipt.json', *receipt['files']):
        if name.startswith('result/'):
            continue
        target = OUT/'sealed-cache-metadata'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.CACHE/name, target)
    for i in range(61):
        c.local_file(OUT/'result', f'frames_data/frame_{i:05d}.npz')
    c.save(OUT/'admission.json', dict(schema='ceramic-read-only-finalization-v1',
        controller=identity, failed_cache_key=probe.FAILED_KEY, failed_receipt_sha256=probe.RECEIPT_SHA,
        independent_probe_run='40074118544', original_review_sha256=args.review_sha,
        diagnostic_provenance_sha256=args.provenance_sha, source=receipt['source'],
        independent_v2_proof_sha256=PROOF_SHA,
        solver_executed=False, never_resumable=True, production_eligible=False,
        raw_files=receipt['files'], interpretation='JH2 damage with ballistic noncontacting failed-particle debris; no cohesive fragment surfaces.'))
    verify_raw()


def install():
    from run_woven_preview import install as pv_install
    import run_woven_preview as pv
    pv.LOGS = OUT
    pv_install()
    record = c.read(OUT/'paraview-install.json')
    record['executable_sha256'] = c.digest(record['pvpython'])
    c.save(OUT/'paraview-install.json', record)


def render():
    controller()
    verify_raw()
    # The independently versioned renderer is intentionally separate from numerical source.
    subprocess.run([sys.executable, '-m', 'mpm_jax.visualization', str(OUT/'result'),
        '--gif', '--field', 'damage', '--color-range', '0', '1', '--resolution', '1200',
        '--debris-style', 'highlight'], cwd=c.ROOT, check=True, timeout=4200)
    install_record = c.read(OUT/'paraview-install.json')
    pv = install_record['pvpython']
    if c.digest(pv) != install_record['executable_sha256']:
        raise ValueError('ParaView executable changed')
    subprocess.run([pv, '--force-offscreen-rendering', str(Path(__file__).resolve()), 'vtk'],
                   cwd=c.ROOT, check=True, timeout=600)
    audit()


def audit():
    from PIL import Image
    import numpy as np
    folder = OUT/'result'
    render = c.read(folder/'render.json')
    if not (render['renderer'] == 'ParaView' and render['frames_rendered'] == 61
            and render['geometry_scale'] == 1 and render['axes'] is False
            and render['annotations'] is False and render['field'] == 'damage'
            and render['color_range'] == [0,1] and render['resolution'] == [1200,1200]
            and render['view_scope'] == 'complete scene'
            and render['failure_display']['all_particles_retained'] is True
            and render['failure_display']['style'] == 'highlight'
            and render['particles'] == 386420
            and c.read(folder/'vtk-full-audit.json')['frames_checked'] == 61):
        raise ValueError('Unexpected actual ParaView rendering provenance')
    counts = render['failure_display']['displayed_counts']
    if len(counts) != 61:
        raise ValueError('Missing saved-state display counts')
    for i in range(61):
        with np.load(folder/'frames_data'/f'frame_{i:05d}.npz', allow_pickle=False) as saved:
            failed = int(np.count_nonzero(saved['failed']))
            if (counts[i]['active_particles'] != len(saved['x'])-failed
                    or counts[i]['debris_particles'] != failed):
                raise ValueError('Displayed active/debris counts differ from saved particles')
        with Image.open(folder/'frames'/f'frame_{i:05d}.png') as png:
            png.load()
            if png.size != (1200,1200) or png.convert('RGB').getextrema() == ((255,255),)*3:
                raise ValueError('Invalid or blank PNG')
    with Image.open(folder/'simulation.gif') as gif:
        if gif.n_frames != 61 or gif.size != (1200,1200):
            raise ValueError('GIF omitted saved states')
        for i in range(61):
            gif.seek(i)
            gif.load()
            if gif.info.get('duration') != 60:
                raise ValueError('Unexpected GIF timing')
    original = c.read(OUT/'admission.json')['raw_files']
    for name, record in original.items():
        if name.startswith('result/') and c.file_record(OUT/name) != record:
            raise ValueError('Copied numerical bytes changed during rendering')
    verify_raw()
    c.save(OUT/'media-audit.json', {'passed':True, 'saved_states':61, 'vtk_frames_checked':61,
        'no_axes_or_text':True, 'raw_unchanged':True, 'public_visual_review_passed':False,
        'production_eligible':False, 'solver_executed':False})


def package():
    audit()
    files = {p.relative_to(OUT).as_posix():c.file_record(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    total = sum(r['bytes'] for r in files.values())
    if total > LIMIT:
        raise ValueError('Delivery exceeds 8 GB bound')
    c.save(OUT/'delivery-receipt.json', {'schema':'ceramic-final-delivery-v1', 'files':files,
        'bytes':total, 'bound_bytes':LIMIT, 'permanent_raw_backup_verified':False})


def retrieved(path):
    receipt = c.read(path/'delivery-receipt.json')
    if receipt != c.read(OUT/'delivery-receipt.json'):
        raise ValueError('Retrieved manifest differs')
    for name, record in receipt['files'].items():
        if c.file_record(c.local_file(path,name)) != record:
            raise ValueError('Retrieved delivery differs: '+name)
    c.save(OUT/'retrieval-receipt.json', {'passed':True,'files_checked':len(receipt['files']),
        'delivery_receipt_sha256':c.digest(path/'delivery-receipt.json'),
        'artifact_id':os.environ['CERAMIC_ARTIFACT_ID'], 'permanent_raw_backup_verified':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','install','render','vtk','package','retrieved'))
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--original-review', type=Path)
    parser.add_argument('--review-sha')
    parser.add_argument('--provenance-sha')
    parser.add_argument('--retrieved', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args)
    elif args.action == 'vtk':
        from check_jc_final_media import vtk_audit
        report = vtk_audit(OUT/'result')
        report['schema'] = 'ceramic-final-vtk-audit-v1'
        c.save(OUT/'result/vtk-full-audit.json', report)
    elif args.action == 'retrieved': retrieved(args.retrieved)
    else: globals()[args.action]()
