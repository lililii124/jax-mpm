"""Reject unregistered workflow bytes, modified checkout, other repository/ref."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPOSITORY = 'lililii124/jax-mpm'
FEATURE_REF = 'refs/heads/work/three-strand-f12-exact-probe'
WORKFLOWS = ('.github/workflows/engineering-previews.yml',
             '.github/workflows/three-strand-exact-probe.yml')
FILES = tuple('tools/three_strand_exact_probe/'+name for name in
              ('bind_controller.py', 'guarded_replay.py', 'exact_probe_guarded_payload.py',
               'recover_original.py', 'original-pins.json')) + WORKFLOWS


def main():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    commit = git('rev-parse', 'HEAD').decode().strip()
    if (os.environ.get('GITHUB_REPOSITORY') != REPOSITORY
            or os.environ.get('GITHUB_REF') != FEATURE_REF
            or commit != os.environ.get('GITHUB_SHA')
            or commit != os.environ.get('WORKFLOW_SOURCE_SHA')
            or os.environ.get('GITHUB_WORKFLOW_REF') != REPOSITORY+'/'+WORKFLOWS[0]+'@'+FEATURE_REF):
        raise ValueError('Wrong repository, fixed feature ref, selected SHA or registered workflow')
    records = {}
    for name in FILES:
        path = root/name
        if path.is_symlink():
            raise ValueError('Symlink controller: '+name)
        raw = path.read_bytes()
        blob = git('rev-parse', commit+':'+name).decode().strip()
        if raw != git('cat-file', 'blob', blob):
            raise ValueError('Working bytes differ from committed controller Git blob: '+name)
        expected_blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if expected_blob != blob:
            raise ValueError('Controller blob identity mismatch')
        records[name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'git_blob': blob}
    if (root/WORKFLOWS[0]).read_bytes() != (root/WORKFLOWS[1]).read_bytes():
        raise ValueError('Registered engineering workflow override differs from diagnostic template')
    evidence = root.parent/'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/'controller-binding.json').write_text(json.dumps({
        'repository': REPOSITORY, 'ref': FEATURE_REF, 'commit': commit,
        'registered_override_bytes_bound': True, 'files': records}, indent=2))


if __name__ == '__main__':
    main()
