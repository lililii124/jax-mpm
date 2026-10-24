"""Preserve every original three-strand raw byte in bounded192MiB archive parts."""
import argparse
import io
import shutil
import tarfile
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


def restore(delivery,cache,expected_receipt):
    """Verify the entire immutable capsule, then safely restore every raw byte."""
    if cache.exists():raise ValueError('Exact parent capsule needs a new empty local target')
    manifest=transport.read(delivery/'delivery-manifest.json')
    if manifest.get('diagnostic_only') or manifest.get('never_resumable') or manifest.get('raw_case')!='three-strand-tension':raise ValueError('Diagnostic or other-case artifact cannot resume')
    if manifest['original_receipt_sha256']!=expected_receipt:raise ValueError('External parent receipt SHA does not match artifact')
    verifier.verify(delivery)
    archive=transport.read(delivery/'raw-archive/archive-manifest.json')
    if archive['uncompressed_file_bytes']>2*1024**3:raise ValueError('Parent full raw exceeds2GiB budget')
    paths=[delivery/'raw-archive'/item['file'] for item in archive['parts']]
    cache.mkdir(parents=True)
    with io.BufferedReader(verifier.Parts(paths)) as stream,tarfile.open(fileobj=stream,mode='r|gz') as tar:
        for member in tar:
            prefix='fibre-pull-off-cache/'
            if not member.isfile() or not member.name.startswith(prefix):raise ValueError('Unexpected capsule member')
            name=member.name[len(prefix):]
            target=verifier.safe(cache,name);target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as original,target.open('xb') as output:shutil.copyfileobj(original,output,1024**2)
            if transport.file_record(target)!=archive['members'][member.name]:raise ValueError('Restored raw member changed')
    if transport.sha(cache/'receipt.json')!=expected_receipt:raise ValueError('Restored original receipt changed')
    return {'restored':True,'original_receipt_sha256':expected_receipt,'original_members_restored':len(archive['members']),'raw_archive_sha256':archive['sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['package','diagnostic','verify','restore']);p.add_argument('--cache',type=Path);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--receipt-sha',default='');a=p.parse_args()
    if a.phase in ('package','diagnostic'):package(a.cache.resolve(),a.delivery.resolve(),diagnostic=a.phase=='diagnostic')
    elif a.phase=='restore':print(json.dumps(restore(a.delivery.resolve(),a.cache.resolve(),a.receipt_sha)))
    else:
        result=verifier.verify(a.delivery.resolve());target=a.delivery.parent/'three-strand-retrieval-receipt.json';transport.save(target,result);print(json.dumps(result))


if __name__=='__main__':main()
