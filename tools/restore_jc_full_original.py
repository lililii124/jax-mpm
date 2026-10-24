"""Restore and hash-check the sole immutable completed JC artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

import run_jc_continuation as carry

ZIP_SHA = 'c96d24d47c1a4c1afcf65a5979fd4cd4cf1cd5197050d33afdc9784afe34614b'
MANIFEST_SHA = 'd6f314e1d9567250acabfd220f3803327bfe8ace75659a67a7b0a8dc31a736ae'


def controller_proof():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    head = git('rev-parse', 'HEAD').decode().strip()
    if (head != os.environ.get('GITHUB_SHA')
            or os.environ.get('GITHUB_REPOSITORY') != 'lililii124/jax-mpm'
            or os.environ.get('GITHUB_REF') != 'refs/heads/work/jc-complete-refined'):
        raise ValueError('Read-only controller HEAD/repository/ref differs')
    names = ['.github/workflows/jc-full-readonly-v2.yml', '.github/workflows/engineering-previews.yml',
             'tools/restore_jc_full_original.py', 'tools/review_jc_full_result_v2.py',
             'tools/render_jc_full_v2.py', 'tools/run_jc_continuation.py',
             'tools/jc_grid_separation.py', 'tools/run_jc_final.py']
    files = {}
    for name in names:
        path = root / name
        if git('show', 'HEAD:' + name) != path.read_bytes():
            raise ValueError('Executed controller differs from HEAD Git blob: ' + name)
        files[name] = {'sha256': carry.digest(path),
                       'git_blob': git('rev-parse', 'HEAD:' + name).decode().strip()}
    if ((root / names[0]).read_bytes() != (root / names[1]).read_bytes()
            or not os.environ.get('GITHUB_WORKFLOW_REF', '').startswith(
                'lililii124/jax-mpm/.github/workflows/engineering-previews.yml@')):
        raise ValueError('Registered workflow differs from exact read-only template')
    return {'schema': 'jc-readonly-controller-v2', 'passed': True, 'head': head,
            'workflow_ref': os.environ['GITHUB_WORKFLOW_REF'], 'files': files}


def restore(archive, destination, metadata):
    info = carry.read_json(metadata)
    if not (info['id'] == 12570755899 and info['expired'] is False
            and info['size_in_bytes'] == 2839539935
            and info['digest'] == 'sha256:' + ZIP_SHA
            and info['workflow_run']['id'] == 40107541665
            and info['workflow_run']['head_sha'] == '70ef188802e0da955fe54947562259eced2e5e01'):
        raise ValueError('Official artifact identity/run/head/size/digest differs')
    if carry.digest(archive) != ZIP_SHA:
        raise ValueError('Official original ZIP hash differs')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    members = {}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        manifest_bytes = z.read('delivery-manifest.json')
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA:
            raise ValueError('Original manifest hash differs')
        manifest = json.loads(manifest_bytes)
        expected = {'delivery-manifest.json'}
        total = 0
        for name, record in manifest['files'].items():
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()) or not record['complete_file']:
                raise ValueError('Unsafe or incomplete original file')
            target.parent.mkdir(parents=True, exist_ok=True)
            whole, count = hashlib.sha256(), 0
            with target.open('xb') as output:
                for index, part in enumerate(record['parts']):
                    member = 'parts/' + name + f'.part-{index:04d}'
                    if member != part['path']:
                        raise ValueError('Original part order differs')
                    expected.add(member)
                    part_hash, part_size = hashlib.sha256(), 0
                    with z.open(member) as source:
                        for block in iter(lambda: source.read(4*1024**2), b''):
                            part_hash.update(block)
                            whole.update(block)
                            output.write(block)
                            part_size += len(block)
                            count += len(block)
                    if part_size != part['bytes'] or part_hash.hexdigest() != part['sha256']:
                        raise ValueError('Original ZIP member hash differs: ' + member)
                    members[member] = {'bytes': part_size, 'sha256': part_hash.hexdigest()}
            if count != record['bytes'] or whole.hexdigest() != record['sha256']:
                raise ValueError('Original reconstructed whole file differs: ' + name)
            total += count
        if set(names) != expected or total != manifest['payload_bytes']:
            raise ValueError('Original member inventory/accounting differs')
    (destination / 'delivery-manifest.json').write_bytes(manifest_bytes)
    return {'schema': 'jc-original-full-restore-v2', 'passed': True,
            'artifact_id': 12570755899, 'run_id': 40107541665,
            'official_zip_sha256': ZIP_SHA, 'manifest_sha256': MANIFEST_SHA,
            'members': members, 'files': manifest['files'], 'payload_bytes': total}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, nargs='?')
    parser.add_argument('destination', type=Path, nargs='?')
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--prove-controller', action='store_true')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    carry.save(args.receipt, controller_proof() if args.prove_controller else
               restore(args.archive, args.destination, args.metadata))
