"""Fresh t0 corrected-source impact prefix; never restores an old numeric cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
CACHE = ROOT/'.local/metal-corrected-prefix-cache'
RECORDS = ROOT/'.local/metal-corrected-prefix-records'
PLAN_WITHOUT_SOURCE = 'b618a3baf378142b092b579a169437a52f7b9fdcdcae2c42814f4a396c62e25d'
MATERIAL_SOURCE = 'c8050d4814efc975598efc64fff80a4d8d9b6eeca7e002109e47b7b1b10055d3'
RAW_LIMIT = 2*1024**3
RAW_STOP = RAW_LIMIT-256*1024**2
RECEIPT_RESERVE = 1024**2
WALL_LIMIT = 1200

ORIGINAL_NUMERICAL_FILES = {'examples/metal_sphere_impact.py': '2765ef416d48c1d84178331c93abd4cdc4dfc6567b9ea084758c6fae359e0669', 'examples/metal_sphere_geometry.py': 'e37a550631728842d6193383cec87f3e8a38057af3c7631f6768a37cdfe895cd', 'examples/metal_plate_geometry.py': '7a9e27f50590667d0944ad584ec62c29b1a6f75bab86611306b853384bbfd5bf', 'examples/woven_geometry.py': '8891736a3af65d729f99b4d0f44a711605e31a48fb36c6028bea97c42b02f80a', 'mpm_jax/impact_mpm.py': 'dcf981413ea0435d80a08e657b2ed7b221476f5961680a6e8fd249f4d4d1589e', 'mpm_jax/materials/shock_metal.py': '91045b966f2b5c7e6a75a3df51548c87dd1ed75f7d68071e96de16d4245c73f9', 'mpm_jax/fibre_mpm.py': 'ef49dade8e1c3ca406f1445e2b6ff728a99c4ac6419652debcbb893490574f9e', 'mpm_jax/contact.py': '2c3c37759c90c52b296f69f7e247b06bc7786d27746ffd6cbbb544707357ab00', 'mpm_jax/core.py': '3d131e4aee7f32feede9625b52338c0b0f791f973d840e3ce4e8d4ac7a471f08'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inventory(folder):
    assert not any(p.is_symlink() for p in folder.rglob('*'))
    return {p.relative_to(folder).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in sorted(folder.rglob('*')) if p.is_file()}


def bound_failed_cache(files):
    """Retain complete state files if a write races the raw-size monitor."""
    used = sum(record['bytes'] for record in files.values())
    omitted = {}
    for name in sorted(files, key=lambda name: files[name]['bytes'], reverse=True):
        if used <= RAW_LIMIT-RECEIPT_RESERVE:
            break
        path = Path(name)
        # Never prune accepted checkpoints, rejected states, or partial state writes.
        if path.parent != Path('result') or not path.match('trajectory_*.npy'):
            continue
        omitted[name] = files[name]
        (CACHE/path).unlink()
        used -= files[name]['bytes']
    if omitted:
        write(RECORDS/'raw-omissions.json', dict(files=omitted,
            reason='Raw ceiling overflow; whole state files preserved; trajectory files omitted.',
            never_resumable=True))
    assert used <= RAW_LIMIT-RECEIPT_RESERVE, 'Whole state files exceed raw retention budget'
    return {name: record for name, record in files.items() if name not in omitted}, omitted


def prepare(*, extra_source_files=()):
    assert not CACHE.exists(), 'Fresh source must start from t0 in a new empty directory'
    assert not RECORDS.exists()
    assert shutil.disk_usage(ROOT).free >= 5*1024**3
    CACHE.mkdir(parents=True)
    RECORDS.mkdir(parents=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == os.environ['GITHUB_SHA']
    assert digest(ROOT/'mpm_jax/materials/shock_metal.py') == MATERIAL_SOURCE
    names = subprocess.check_output(['git', 'ls-files', '--', 'mpm_jax', 'examples', 'tools',
        'pyproject.toml', '.github/workflows/engineering-previews.yml', *extra_source_files], text=True).splitlines()
    assert set(extra_source_files) <= set(names), 'Registered controller workflow must be tracked'
    files = {}
    for name in names:
        if not name.endswith(('.py', '.toml', '.yml')):
            continue
        path = ROOT/name
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD:'+name], text=True).strip() == blob, name
        files[name] = dict(bytes=len(data), sha256=digest(path), git_blob=blob)
    source = dict(repository=os.environ['GITHUB_REPOSITORY'], commit=commit,
        tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip(),
        run_id=os.environ['GITHUB_RUN_ID'], attempt=os.environ['GITHUB_RUN_ATTEMPT'], files=files)
    import jax
    import jaxlib
    import numpy as np
    from examples.metal_sphere_impact import plan_case
    runtime = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
        jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    assert runtime == dict(python='3.12.13', numpy='2.5.4', jax='0.12.0.1', jaxlib='0.12.0.1', backend='cpu', x64=True)
    geometry, plan = plan_case()
    physical = {k:v for k,v in plan.items() if k != 'source'}
    assert hashlib.sha256(json.dumps(physical, sort_keys=True, allow_nan=False).encode()).hexdigest() == PLAN_WITHOUT_SOURCE
    assert plan['source']['files']['mpm_jax/materials/shock_metal.py'] == MATERIAL_SOURCE
    for name, expected in ORIGINAL_NUMERICAL_FILES.items():
        if name != 'mpm_jax/materials/shock_metal.py':
            assert plan['source']['files'][name] == expected, name
    assert plan['source']['fingerprint'] != '0625907b941250a5ab986478c6573fb7a92dc6975c27a35f84bbabebea735ad0'
    for folder in (CACHE, RECORDS):
        write(folder/'source.json', source)
        write(folder/'runtime.json', runtime)
    write(RECORDS/'planned-full-protocol.json', plan)
    write(RECORDS/'admission.json', dict(fresh_t0=True, old_numeric_cache_restored=False,
        unchanged_physical_plan_sha256=PLAN_WITHOUT_SOURCE, source_fingerprint=plan['source']['fingerprint'],
        stop_frame=8, full_saved_states=61, speed_m_s=5000, particles=409504, full_duration_s=1e-6,
        maximum_tree_uss_bytes=6*1024**3, maximum_wall_seconds=WALL_LIMIT, maximum_raw_bytes=RAW_LIMIT,
        raw_stop_watermark_bytes=RAW_STOP,
        independent_physical_admission=False, physical_validation_passed=False, production_eligible=False,
        scope='New-source bounded resource prefix only. Return-solve correction does not validate phase extrapolation, fracture, penetration or full impact physics.'))
    (CACHE/'requirements.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    (RECORDS/'cpuinfo.txt').write_bytes(Path('/proc/cpuinfo').read_bytes())
    write(RECORDS/'numerical-environment.json', {k:os.environ.get(k) for k in ('JAX_PLATFORMS', 'JAX_ENABLE_X64',
        'XLA_PYTHON_CLIENT_PREALLOCATE', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS')})


def run(*, resume=False, stop_frame=8):
    import psutil
    if resume:
        assert (CACHE/'result'/'checkpoint.npz').is_file(), 'Verified complete checkpoint required'
    else:
        assert not (CACHE/'result').exists(), 'Never resume any old or failed numerical state'
    cmd = [sys.executable, '-m', 'examples.metal_sphere_impact', '--stop-after-frame', str(stop_frame), '--output', str(CACHE/'result')]
    if resume:
        cmd.append('--resume')
    begun = time.monotonic()
    samples, peak, reason = [], 0, None
    with (RECORDS/'solve.log').open('wb') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                total = 0
                try:
                    parent = psutil.Process(process.pid)
                    for child in [parent]+parent.children(recursive=True):
                        try:
                            total += child.memory_full_info().uss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except psutil.NoSuchProcess:
                    pass
                elapsed = time.monotonic()-begun
                raw_bytes = 0
                for path in CACHE.rglob('*'):
                    try:
                        if path.is_file():
                            raw_bytes += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic checkpoint rename while sampling.
                peak = max(peak, total)
                samples.append(dict(elapsed_s=elapsed, tree_uss_bytes=total, raw_bytes=raw_bytes))
                if elapsed >= WALL_LIMIT-1: reason = '20 minute wall ceiling'
                if total > 6*1024**3: reason = '6 GiB process-tree USS ceiling'
                if raw_bytes >= RAW_STOP: reason = 'Raw disk early-stop watermark'
                if shutil.disk_usage(ROOT).free < 2*1024**3: reason = '2 GiB free disk reserve'
                if (RECORDS/'solve.log').stat().st_size > 8*1024**2: reason = '8 MiB log ceiling'
                if reason:
                    # Resource guards must not permit another ten seconds of writes.
                    os.killpg(process.pid, signal.SIGKILL)
                    try:
                        process.wait(timeout=max(.01, WALL_LIMIT-(time.monotonic()-begun)))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(max(0, min(1, WALL_LIMIT-1-(time.monotonic()-begun))))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            write(RECORDS/'monitor.json', dict(command=cmd, returncode=code, elapsed_s=time.monotonic()-begun,
                peak_tree_uss_bytes=peak, stop_reason=reason, samples=samples))
    assert code == 0 and reason is None, 'Raw failure and last accepted checkpoint retained'


def seal(*, stop_frame=8, key_prefix='metal-corrected-prefix-v1', lineage=None):
    import urllib.request
    source = json.loads((CACHE/'source.json').read_text())
    for name, record in source['files'].items():
        assert digest(ROOT/name) == record['sha256'], name
    files = inventory(CACHE)
    result = CACHE/'result'
    load = lambda name: json.loads((result/name).read_text()) if (result/name).is_file() else {}
    status, summary = load('status.json'), load('summary.json')
    monitor = json.loads((RECORDS/'monitor.json').read_text()) if (RECORDS/'monitor.json').exists() else {}
    accepted = (monitor.get('returncode') == 0 and monitor.get('stop_reason') is None
        and status.get('state_valid') is True and status.get('segment_complete') is True
        and status.get('frame') == stop_frame and status.get('frames') == 60)
    oversized = sum(v['bytes'] for v in files.values()) > RAW_LIMIT-RECEIPT_RESERVE
    accepted = accepted and not oversized
    files, omitted = bound_failed_cache(files) if oversized else (files, {})
    if (result/'plan.json').is_file():
        assert load('plan.json') == json.loads((RECORDS/'planned-full-protocol.json').read_text())
    if summary:
        assert summary['runtime'] == json.loads((CACHE/'runtime.json').read_text())
        assert summary['source'] == load('plan.json')['source']
    key = f"{key_prefix}-{source['commit']}-{source['run_id']}-{source['attempt']}-f{stop_frame}"+('' if accepted else '-failed')
    request = urllib.request.Request(f"https://api.github.com/repos/{source['repository']}/actions/cache/usage",
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'], 'Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        usage = json.load(response)
    cache_budget_ok = usage['active_caches_size_in_bytes']+sum(v['bytes'] for v in files.values())+1024**2 <= 9_000_000_000
    receipt = dict(schema='metal-corrected-source-prefix-v1', source=source, files=files, cache_key=key,
        prefix_complete=accepted, never_resumable=not accepted, old_numeric_cache_restored=False,
        raw_files_omitted=omitted, raw_retention_limit_bytes=RAW_LIMIT,
        physical_validation_passed=False, production_eligible=False, full_target_s=1e-6,
        requested_speed_m_s=5000, requested_stop_frame=stop_frame, cache_budget_ok=cache_budget_ok, repository_cache_usage=usage,
        lineage=lineage,
        scope='Fresh corrected-source prefix only; raw rejection retained. No full-impact/phase validation. Cache is temporary, not permanent backup.')
    for folder in (CACHE, RECORDS):
        write(folder/'receipt.json', receipt)
    assert (CACHE/'receipt.json').stat().st_size <= RECEIPT_RESERVE
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        if cache_budget_ok:
            f.write('cache_key='+key+'\n')
        f.write('receipt_sha='+digest(CACHE/'receipt.json')+'\n')


def deliver(*, destination=None):
    destination = ROOT/'.local/metal-corrected-prefix-delivery' if destination is None else Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result = CACHE/'result'
    # Whole rejected state and last complete checkpoint precede trajectories.
    priority = ['failed_state.npz', 'checkpoint.npz', 'reference.npz', 'final_state_0.npz', 'particle_fields.npz', 'trajectory.npz']
    candidates = [(p, Path('records')/p.name) for p in sorted(RECORDS.glob('*')) if p.is_file()]
    candidates += [(p, Path('result')/p.name) for p in sorted(result.glob('*')) if p.is_file() and p.suffix != '.npz' and p.suffix != '.npy']
    candidates += [(result/name, Path('result')/name) for name in priority if (result/name).is_file()]
    candidates += [(p, Path('result')/p.name) for p in sorted(result.glob('*')) if p.is_file() and p.name not in priority and p.suffix in ('.npz', '.npy')]
    files, omitted, used = {}, {}, 0
    for path, relative in candidates:
        record = dict(bytes=path.stat().st_size, sha256=digest(path))
        if used+record['bytes'] > 191*1024**2:
            omitted[relative.as_posix()] = record
            continue
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert digest(target) == record['sha256']
        files[relative.as_posix()] = record
        used += record['bytes']
    confirmation = json.loads((RECORDS/'cache-confirmation.json').read_text()) if (RECORDS/'cache-confirmation.json').is_file() else {}
    receipt = json.loads((CACHE/'receipt.json').read_text()) if (CACHE/'receipt.json').is_file() else {}
    confirmed = (confirmation.get('exact_lookup') is True and bool(receipt)
        and confirmation.get('key') == receipt.get('cache_key')
        and confirmation.get('receipt_sha256') == digest(CACHE/'receipt.json'))
    essential_omitted = {name: record for name, record in omitted.items()
                         if Path(name).name in ('failed_state.npz', 'checkpoint.npz')}
    write(destination/'manifest.json', dict(files=files, omitted=omitted, physical_validation_passed=False,
        production_eligible=False, never_resumable=True, exact_temporary_cache_confirmed=confirmed,
        essential_state_files_omitted=essential_omitted,
        complete_failure_transport=(result/'checkpoint.npz').is_file()
            and not (RECORDS/'raw-omissions.json').exists() and (confirmed or not omitted),
        scope='Listed whole evidence files only; omissions are explicit. Temporary cache is not permanent backup. No original07f cache used.'))
    (destination/'manifest.sha256').write_text(digest(destination/'manifest.json')+'  manifest.json\n')
    assert not omitted or confirmed, 'Artifact omissions lack an exact confirmed temporary cache; transport incomplete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run', 'seal', 'deliver'))
    globals()[parser.parse_args().phase]()
