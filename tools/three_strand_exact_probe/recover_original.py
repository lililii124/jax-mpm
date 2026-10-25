"""Recover diagnostic f12 bytes; deliberately does not use resumable restore."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--delivery', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    pins = json.loads(Path(__file__).with_name('original-pins.json').read_text())
    evidence = args.evidence
    evidence.mkdir(parents=True, exist_ok=True)
    repo = args.repo.resolve()
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != pins['source_commit']:
        raise ValueError('Original numerical SHA mismatch')
    for name, digest in pins['source_files'].items():
        blob = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', head+':'+name], text=True).strip()
        raw = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', blob])
        if ((repo/name).is_symlink() or sha(repo/name) != digest
                or (repo/name).read_bytes() != raw or blob != pins['source_git_blobs'][name]):
            raise ValueError('Original numerical source changed: '+name)
    if args.runtime_only:
        import platform
        actual = {'python': platform.python_version(), 'packages': {
            k: importlib.metadata.version(k) for k in pins['environment']['packages']}}
        if actual != pins['environment']:
            raise ValueError('Original numerical runtime mismatch')
        import numpy as np
        with np.load(args.output/'data/checkpoint.npz', allow_pickle=False) as cp:
            if int(cp['frame']) != 12 or int(cp['progress_0']) != 3500 or not bool(cp['state_valid']):
                raise ValueError('Expected original valid f12/step3500 diagnostic checkpoint')
        (evidence/'runtime.json').write_text(json.dumps(actual, indent=2))
        return
    delivery = args.delivery.resolve()
    manifest_path = delivery/'raw-archive/archive-manifest.json'
    if sha(manifest_path) != pins['archive_manifest_sha256']:
        raise ValueError('Wrong diagnostic archive manifest')
    archive = json.loads(manifest_path.read_text())
    if archive['sha256'] != pins['archive_sha256']:
        raise ValueError('Wrong complete archive')
    sys.path.insert(0, str(repo/'tools'))
    import package_fibre_final_media as verifier
    report = verifier.verify(delivery)
    if args.output.exists():
        raise FileExistsError('Preserve original recovered directory')
    if archive['uncompressed_file_bytes'] > 2*1024**3:
        raise ValueError('Raw recovery exceeds2 GiB bound')
    args.output.mkdir(parents=True)
    paths = [verifier.safe(delivery/'raw-archive', p['file']) for p in archive['parts']]
    seen = set()
    with io.BufferedReader(verifier.Parts(paths)) as stream, tarfile.open(fileobj=stream, mode='r|gz') as tar:
        for member in tar:
            prefix = 'fibre-pull-off-cache/'
            if not member.isfile() or not member.name.startswith(prefix) or member.name in seen:
                raise ValueError('Unexpected diagnostic member')
            seen.add(member.name)
            target = verifier.safe(args.output, member.name[len(prefix):])
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as original, target.open('xb') as restored:
                shutil.copyfileobj(original, restored, 1024**2)
            if verifier.record(target) != archive['members'][member.name]:
                raise ValueError('Restored diagnostic member differs')
    if seen != set(archive['members']):
        raise ValueError('Missing diagnostic members')
    if sha(args.output/'data/checkpoint.npz') != pins['checkpoint_sha256']:
        raise ValueError('Wrong original f12 checkpoint')
    report.update(diagnostic_only=True, never_resumable=True, source_commit=head,
                  all_source_files_verified=len(pins['source_files']), all_members_restored=len(seen))
    (evidence/'recovery.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
