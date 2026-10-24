"""Bounded private RHT prefix; receipts never assert full perforation."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEST = ROOT / '.local/rht-impact-prefix'
RAW = DEST / 'raw'
GIB = 1024**3
OUTPUT_LIMIT = GIB
PART_BYTES = 128 * 1024**2
DELIVERY = ROOT / '.local/rht-impact-delivery'


class PartWriter:
    """Stream archive bytes into independently capped artifact payloads."""
    def __init__(self, folder, limit=None):
        self.folder, self.limit = folder, PART_BYTES if limit is None else limit
        self.stream, self.used, self.parts = None, 0, []

    def write(self, data):
        total = len(data)
        while data:
            if self.stream is None or self.used == self.limit:
                if self.stream is not None:
                    self.stream.close()
                path = self.folder / f'part-{len(self.parts):03d}.tarpart'
                self.parts.append(path)
                self.stream, self.used = path.open('wb'), 0
            n = min(len(data), self.limit - self.used)
            self.stream.write(data[:n])
            self.used += n
            data = data[n:]
        return total

    def close(self):
        if self.stream is not None:
            self.stream.close()


def package():
    if DELIVERY.exists():
        raise RuntimeError('Refusing to overwrite prior delivery')
    DELIVERY.mkdir(parents=True)
    files = {p.relative_to(DEST).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in sorted(DEST.rglob('*')) if p.is_file()}
    if not files:
        raise RuntimeError('No raw evidence exists to package')
    # 32 separate artifacts bound the entire private transport to 4 GiB.
    # Reject before writing parts rather than emitting an incomplete archive.
    tar_bound = sum(((v['bytes'] + 511) // 512 + 8) * 512 for v in files.values()) + 10240
    manifest = {'files': files, 'part_limit_bytes': PART_BYTES,
                'status': 'packaging', 'scope': 'Whole raw evidence, including failed or incomplete solve'}
    write(DELIVERY / 'transport.json', manifest)
    if tar_bound > 32 * PART_BYTES or shutil.disk_usage(DELIVERY).free < tar_bound + 512 * 1024**2:
        manifest['status'] = 'transport-rejected-no-complete-delivery'
        write(DELIVERY / 'transport.json', manifest)
        raise RuntimeError('Cannot preserve complete raw under multipart/disk admission budget')
    sink = PartWriter(DELIVERY)
    try:
        with tarfile.open(fileobj=sink, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for name in files:
                archive.add(DEST / name, arcname=name, recursive=False)
    finally:
        sink.close()
    manifest['parts'] = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sink.parts}
    if len(sink.parts) > 32 or any(v['bytes'] > PART_BYTES for v in manifest['parts'].values()):
        raise RuntimeError('Multipart cap invariant violated')
    manifest['status'] = 'complete-raw-packaged'
    write(DELIVERY / 'transport.json', manifest)


class PartReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.stream = None

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.stream = path.open('rb')
            data = self.stream.read(size - len(result))
            if not data:
                self.stream.close()
                self.stream = None
            result.extend(data)
        return bytes(result)


def verify(folder):
    manifest = json.loads((folder / 'transport.json').read_text())
    if manifest['status'] != 'complete-raw-packaged':
        raise RuntimeError('No complete raw delivery exists')
    for name, info in manifest['parts'].items():
        path = folder / name
        if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise RuntimeError(f'Corrupt/missing retrieved part: {name}')
    seen = set()
    reader = PartReader(folder / name for name in manifest['parts'])
    with tarfile.open(fileobj=reader, mode='r|') as archive:
        for member in archive:
            if member.name not in manifest['files'] or member.name in seen or not member.isfile():
                raise RuntimeError('Unexpected/duplicate/non-file archive member')
            seen.add(member.name)
            info = manifest['files'][member.name]
            stream = archive.extractfile(member)
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
            if member.size != info['bytes'] or h.hexdigest() != info['sha256']:
                raise RuntimeError(f'Whole raw file differs: {member.name}')
    if seen != set(manifest['files']):
        raise RuntimeError('Whole raw files missing from delivery')
    write(folder / 'retrieval-receipt.json', {'status': 'all-parts-and-whole-raw-files-verified',
          'transport_sha256': digest(folder / 'transport.json'), 'files': len(seen)})


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    import psutil
    if DEST.exists():
        raise RuntimeError('Fresh prefix requires an absent destination; preserve existing evidence')
    DEST.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if os.environ.get('GITHUB_SHA') != commit:
        raise RuntimeError('Remote execution requires HEAD equal to GITHUB_SHA')
    source = {}
    blobs = {}
    for pattern in ('mpm_jax/**/*.py', 'examples/**/*.py', 'pyproject.toml',
                    'tools/rht_impact_prefix.py', '.github/workflows/rht-impact-prefix.yml',
                    '.github/workflows/engineering-previews.yml'):
        for path in ROOT.glob(pattern):
            relative = path.relative_to(ROOT)
            name = relative.as_posix()
            expected = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT)
            if path.read_bytes() != expected:
                raise RuntimeError(f'Execution source differs from exact Git blob: {name}')
            blobs[name] = subprocess.check_output(
                ['git', 'rev-parse', f'{commit}:{name}'], cwd=ROOT, text=True).strip()
            copy = DEST / 'source' / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
            source[relative.as_posix()] = digest(copy)
    write(DEST / 'source-manifest.json', source)
    write(DEST / 'git-blobs.json', {'commit': commit, 'github_sha': os.environ['GITHUB_SHA'], 'blobs': blobs})
    (DEST / 'runtime-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze', '--all'], text=True))
    receipt = {'scope': 'Private frame-8 prefix of the 48-frame, 80 us physical plan; no full perforation claim',
               'commit': commit,
               'source_manifest_sha256': digest(DEST / 'source-manifest.json'),
               'runtime': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'pillow', 'psutil')},
               'python': sys.version, 'platform': sys.platform,
               'environment': {k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'JAX_ENABLE_X64', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
               'status': 'starting', 'limits': {'aggregate_uss_bytes': 6 * GIB, 'seconds': 3000, 'output_bytes': OUTPUT_LIMIT, 'minimum_free_bytes': 2 * GIB}}
    command = [sys.executable, '-m', 'examples.rht_pointed_impact', '--stop-after-frame', '8', '--output', str(RAW)]
    receipt['command'] = command
    write(DEST / 'receipt.json', receipt)
    start = time.monotonic()
    process = None
    peak = 0
    try:
        with (DEST / 'solver.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    tree = [parent] + parent.children(recursive=True)
                    uss = sum(p.memory_full_info().uss for p in tree if p.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    if process.poll() is not None:
                        break
                    # A transient child exit may race sampling; retry, but fail
                    # closed if a live process stays inaccessible.
                    try:
                        parent = psutil.Process(process.pid)
                        uss = parent.memory_full_info().uss
                        for child in parent.children(recursive=True):
                            try:
                                uss += child.memory_full_info().uss
                            except psutil.NoSuchProcess:
                                continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        if process.poll() is not None:
                            break
                        raise RuntimeError('Cannot enforce memory guard on a live process')
                peak = max(peak, uss)
                size = sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
                if uss > 6 * GIB:
                    raise RuntimeError('Aggregate 6 GiB USS guard exceeded')
                if time.monotonic() - start > 3000:
                    raise RuntimeError('50 minute solver guard exceeded')
                if size > OUTPUT_LIMIT or shutil.disk_usage(DEST).free < 2 * GIB:
                    raise RuntimeError('Output size or free disk guard exceeded')
                time.sleep(2)
        receipt['returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError('Solver returned a failure; raw evidence retained')
        if sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) > OUTPUT_LIMIT:
            raise RuntimeError('Final private delivery exceeds conservative 1 GiB budget')
        if any(digest(ROOT / name) != expected for name, expected in source.items()):
            raise RuntimeError('Source changed during the prefix')
        # Validate every saved prefix frame against the actual checkpoint, using
        # the same strict public checkpoint validator used before continuation.
        import numpy as np
        from examples.common import _load_state, _valid_checkpoint_states, _validate_resume_prefix
        checkpoint = json.loads((RAW / 'checkpoint.json').read_text())
        if checkpoint['frame'] != 8:
            raise RuntimeError('Prefix did not reach exactly frame 8')
        states = tuple(_load_state(RAW / name) for name in checkpoint['states'])
        if not _valid_checkpoint_states(states):
            raise RuntimeError('Invalid prefix state')
        from examples.rht_pointed_impact import make_case
        plan = make_case()
        dt = plan.configs[0].dt
        frames = plan.description['frames']
        substeps = plan.description['substeps_per_frame']
        _validate_resume_prefix(RAW, checkpoint, states, frames, substeps, dt, True)
        impulse = np.asarray(checkpoint['contact_impulse'])
        if not np.isfinite(impulse).all() or impulse[1] <= 0:
            raise RuntimeError('No positive projectile/body-1 y contact impulse at frame 8')
        receipt.update(status='valid-impact-prefix', frame=8, time_s=8 * substeps * dt,
                       contact_impulse_on_body_1=impulse.tolist())
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None and process.poll() is None:
            try:
                tree = psutil.Process(process.pid).children(recursive=True)
                for child in tree:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            process.kill()
            process.wait(timeout=15)
        receipt.update(elapsed_seconds=time.monotonic() - start, peak_aggregate_uss_bytes=peak)
        # Whole raw receipt includes partial outputs and logs even on a guard failure.
        receipt['files'] = {p.relative_to(DEST).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                            for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'receipt.json'}
        write(DEST / 'receipt.json', receipt)


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'package':
            package()
        elif len(sys.argv) > 1 and sys.argv[1] == 'verify':
            verify(Path(sys.argv[2]))
        else:
            main()
    except BaseException as error:
        # Preflight failures also leave an explicit private receipt.
        if DEST.exists() and not (DEST / 'receipt.json').exists():
            write(DEST / 'receipt.json', {'status': 'preflight-failed',
                  'error': f'{type(error).__name__}: {error}', 'scope': 'No validated physical result'})
        raise
