"""Preserve every original three-strand raw byte in bounded192MiB archive parts."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import finish_fibre_pull_off as transport
import package_fibre_final_media as verifier


def package(cache,output,diagnostic=False):
    if output.exists():raise ValueError('Complete raw delivery requires a new separate directory')
    if diagnostic:
        if not cache.is_dir():raise ValueError('No original diagnostic cache exists')
        files={p.relative_to(cache).as_posix():transport.file_record(p) for p in sorted(cache.rglob('*')) if p.is_file() and p!=cache/'receipt.json'}
        diagnostic_record={'schema':'three-strand-failed-diagnostic-v1','kind':'diagnostic','never_resumable':True,'cache_key':'three-strand-failed-original','source':{'scope':'Diagnostic local bytes; no Git/source admission or resumability claimed'},'files':files}
        if not (cache/'receipt.json').exists():transport.save(cache/'receipt.json',diagnostic_record)
        # Preserve any original accepted parent receipt unchanged; the separate
        # diagnostic record inventories the actual failed working tree instead.
        receipt=diagnostic_record
    else:
        receipt=transport.read(cache/'receipt.json')
        if receipt['kind']!='accepted-prefix' or receipt['never_resumable']:raise ValueError('Accepted original complete stage receipt required')
    receipt_sha=transport.sha(cache/'receipt.json')
    raw_bytes=sum(r['bytes'] for r in receipt['files'].values())+(cache/'receipt.json').stat().st_size
    if raw_bytes>2*1024**3:raise ValueError('Complete original raw exceeds2GiB source budget')
    output.mkdir(parents=True)
    if diagnostic:transport.save(output/'diagnostic.json',receipt)
    controller={'files':{n:transport.file_record(ROOT/n) for n in ('tools/package_three_strand_raw.py','tools/finish_fibre_pull_off.py','tools/package_fibre_final_media.py')}}
    archive=transport.archive_complete_cache(cache,receipt,output/'raw-archive',receipt_sha,controller)
    if archive['compressed_bytes']>2*1024**3:raise ValueError('Compressed original raw exceeds2GiB source budget')
    # The shared transport schema names its tar root fibre-pull-off-cache; this
    # is a container label only. receipt/request/source bind the actual rope.
    manifest={'schema':'three-strand-complete-raw-delivery-v1','passed':True,
              'original_receipt_sha256':receipt_sha,'raw_case':'three-strand-tension',
              'complete_raw_archive_sha256':archive['sha256'],
              'files':{'raw-archive/archive-manifest.json':transport.file_record(output/'raw-archive/archive-manifest.json')},
              'production_eligible':False,'public_visual_review_passed':False,'diagnostic_only':diagnostic,'never_resumable':diagnostic}
    if diagnostic:manifest['files']['diagnostic.json']=transport.file_record(output/'diagnostic.json')
    transport.save(output/'delivery-manifest.json',manifest)
    result=verifier.verify(output)
    print(json.dumps({'raw_parts':len(archive['parts']),'part_limit_bytes':192*1024**2,'uncompressed_bytes':raw_bytes,'compressed_bytes':archive['compressed_bytes'],'raw_archive_sha256':archive['sha256'],'all_original_members_verified':result['original_members_checked']}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['package','diagnostic','verify']);p.add_argument('--cache',type=Path);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args()
    if a.phase in ('package','diagnostic'):package(a.cache.resolve(),a.delivery.resolve(),diagnostic=a.phase=='diagnostic')
    else:
        result=verifier.verify(a.delivery.resolve());target=a.delivery.parent/'three-strand-retrieval-receipt.json';transport.save(target,result);print(json.dumps(result))


if __name__=='__main__':main()
