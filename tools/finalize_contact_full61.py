"""Finalize admitted completed contact results without executing a numerical solver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = '7c6c0e338fd3eb2a0e80b4c39a105dc36238bf43'
RUNS = {'jkr': 40078845859, 'md': 40078855654, 'self-consistent': 40078885920}


def admitted(model, receipt_sha):
    proof = json.loads((ROOT/'tools/contact_full61_admissions.json').read_text())['models'][model]
    if (proof['run'] != RUNS[model] or proof['source'] != PRODUCER
            or not proof['small_records_admitted'] or proof['receipt_sha256'] != receipt_sha):
        raise ValueError('Requested model/receipt differs from admitted terminal run')
    return proof


def controller_binding():
    if os.environ.get('GITHUB_REF')!='refs/heads/work/fibre-contact-model-contrasts':raise ValueError('Exact original cache branch required')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    if commit!=os.environ.get('GITHUB_SHA'):raise ValueError('Exact dispatched Git source required')
    import hashlib
    files={}
    for name in ['tools/finalize_contact_full61.py','tools/contact_full61_admissions.json','.github/workflows/contact-full61-media.yml','.github/workflows/engineering-previews.yml','tools/finish_fibre_pull_off.py','tools/package_fibre_final_media.py','mpm_jax/visualization.py','mpm_jax/paraview_scene.py']:
        raw=(ROOT/name).read_bytes()
        if subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)!=raw:raise ValueError('Uncommitted controller/renderer bytes: '+name)
        files[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    if (ROOT/'.github/workflows/engineering-previews.yml').read_bytes()!=(ROOT/'.github/workflows/contact-full61-media.yml').read_bytes():raise ValueError('Actual registered workflow differs from bound media template')
    return dict(commit=commit,run_id=os.environ.get('GITHUB_RUN_ID'),files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'finalize'])
    parser.add_argument('--model', choices=RUNS, required=True)
    parser.add_argument('--receipt-sha', required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--producer-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    binding=controller_binding()
    proof = admitted(args.model, args.receipt_sha)
    if args.phase == 'prepare':
        if os.environ.get('GITHUB_REF') != 'refs/heads/work/fibre-contact-model-contrasts':
            raise ValueError('Original complete cache requires its original feature branch scope')
        target = os.environ.get('GITHUB_OUTPUT')
        if not target:
            raise ValueError('A GitHub Actions output file is required')
        with open(target, 'a', encoding='utf-8') as stream:
            stream.write('cache_key='+proof['cache_key']+'\n')
        print(json.dumps(proof, sort_keys=True))
        return
    if not all((args.cache, args.producer_root, args.output)):
        parser.error('finalize requires cache, producer-root and output')
    sys.path.insert(0, str(ROOT/'tools'))
    import finish_fibre_pull_off as finalizer
    receipt = finalizer.read(args.cache/'receipt.json')
    if (finalizer.sha(args.cache/'receipt.json') != args.receipt_sha
            or receipt['request']['model'] != args.model or receipt['cache_key'] != proof['cache_key']
            or str(receipt['source']['run_id']) != str(proof['run'])
            or receipt['parent_receipt_sha256'] != proof['parent_receipt_sha256']):
        raise ValueError('Restored complete cache differs from admitted model/run/parent')
    # All source, runtime, ancestry, complete raw inventories, final material,
    # contact/fixture momentum and full61 frame audits remain in the finalizer.
    subprocess.run([sys.executable, str(ROOT/'tools/finish_fibre_pull_off.py'),
                    '--cache', str(args.cache), '--receipt-sha', args.receipt_sha,
                    '--producer-root', str(args.producer_root), '--producer-commit', PRODUCER,
                    '--output', str(args.output)], check=True)
    report = {'schema':'contact-full61-controller-binding-v1','model':args.model,
              'admission':proof,'controller_sha256':finalizer.sha(__file__),
              'workflow_sha256':finalizer.sha(ROOT/'.github/workflows/contact-full61-media.yml'),
              'admissions_sha256':finalizer.sha(ROOT/'tools/contact_full61_admissions.json'),
              'numerical_rerun':False,'public_visual_review_passed':False,'source_binding':binding}
    finalizer.save(args.output/'contact-controller-binding.json',report)
    manifest = finalizer.read(args.output/'delivery-manifest.json')
    manifest['files']['contact-controller-binding.json'] = finalizer.file_record(args.output/'contact-controller-binding.json')
    finalizer.save(args.output/'delivery-manifest.json',manifest)


if __name__ == '__main__':
    main()
