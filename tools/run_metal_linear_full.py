"""Fresh linear-metal full protocol with bounded, exact-source frame segments.

No external cache is accepted. A single invocation starts at t0 and verifies each
completed segment before resuming its own checkpoint under the same runtime.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.metal_sphere_impact_linear import digest, save_json, source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.local/metal-linear-full'))
    parser.add_argument('--wall-seconds-per-segment', type=int, default=9000)
    parser.add_argument('--stop-frame', type=int, choices=(0, 1, 8, 16, 40, 60), default=8)
    parser.add_argument('--tiny-control', action='store_true')
    args = parser.parse_args()
    import psutil
    import jax
    import jaxlib
    import numpy as np
    jax.config.update('jax_enable_x64', True)
    folder = args.output.resolve()
    if folder.exists():
        raise FileExistsError('Fresh empty destination required; no historical cache can enter this controller')
    folder.mkdir(parents=True)
    source = source_record()
    source['controller_sha256'] = digest(__file__)
    workflow = ROOT/'.github/workflows/metal-linear-full.yml'
    source['workflow_sha256'] = digest(workflow)
    source['commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    source['run_id'] = os.environ.get('GITHUB_RUN_ID')
    source['attempt'] = os.environ.get('GITHUB_RUN_ATTEMPT')
    if os.environ.get('GITHUB_SHA'):
        if os.environ.get('GITHUB_REF') != 'refs/heads/work/metal-linear-full' or args.stop_frame != 8 or args.tiny_control:
            raise ValueError('Remote first dispatch requires fixed feature ref and full-geometry f8 only')
        if source['commit'] != os.environ['GITHUB_SHA']:
            raise ValueError('Checkout differs from dispatched SHA')
        for name in (*source['files'], 'tools/run_metal_linear_full.py', '.github/workflows/metal-linear-full.yml', '.github/workflows/engineering-previews.yml', 'tools/rht_impact_prefix.py'):
            checked = subprocess.check_output(['git','show','HEAD:'+name], cwd=ROOT)
            if checked != (ROOT/name).read_bytes():
                raise ValueError('Uncommitted source differs: '+name)
    runtime = dict(python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
        jaxlib=jaxlib.__version__, backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64))
    source['registered_workflow_sha256'] = digest(ROOT/'.github/workflows/engineering-previews.yml')
    source['transport_source_sha256'] = digest(ROOT/'tools/rht_impact_prefix.py')
    if os.environ.get('GITHUB_SHA'):
        expected = dict(python='3.12.13', numpy='2.5.4', jax='0.12.0.1', jaxlib='0.12.0.1', backend='cpu', x64=True)
        if runtime != expected or psutil.__version__ != '7.2.2':
            raise ValueError('Exact reviewed runtime required')
    save_json(folder/'source.json', source)
    save_json(folder/'runtime.json', dict(**runtime, psutil=psutil.__version__))
    (folder/'runtime-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    save_json(folder/'admission.json', dict(fresh_t0=True, old_cache_restored=False,
        full_frames=60, full_duration_s=1e-6, speed_m_s=5000,
        tree_uss_limit_bytes=6*1024**3, raw_limit_bytes=2*1024**3,
        wall_seconds_per_segment=args.wall_seconds_per_segment,
        physical_validation_passed=False, production_eligible=False,
        scope='Illustrative linear EOS; no calibrated shock, phase, fracture or perforation prediction.'))
    segments = [f for f in (8,16,40,60) if f < args.stop_frame]+[args.stop_frame]
    if args.tiny_control:
        segments = [args.stop_frame]
    for index, endpoint in enumerate(segments):
        before = source_record()
        if before['files'] != source['files']:
            raise ValueError('Source changed between segments')
        cmd = [sys.executable,'-m','examples.metal_sphere_impact_linear','--output',str(folder/'result'),
               '--stop-after-frame',str(endpoint)]
        if args.tiny_control:
            cmd.append('--tiny-control')
        if index:
            cmd.append('--resume')
        begun = time.monotonic()
        peak = 0
        samples = []
        reason = None
        with (folder/f'segment-{endpoint}.log').open('wb') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=(os.name!='nt'))
            while proc.poll() is None:
                try:
                    parent = psutil.Process(proc.pid)
                    usage = sum(p.memory_full_info().uss for p in [parent]+parent.children(recursive=True) if p.is_running())
                except psutil.NoSuchProcess:
                    if proc.poll() is not None:
                        break
                    reason = 'Live process-tree disappeared during memory accounting'
                    usage = 0
                except psutil.AccessDenied:
                    reason = 'Process-tree USS access denied; memory bound cannot be verified'
                    usage = 0
                peak = max(peak, usage)
                raw = 0
                for path in folder.rglob('*'):
                    try:
                        if path.is_file():
                            raw += path.stat().st_size
                    except FileNotFoundError:
                        pass  # Atomic checkpoint rename while sampling.
                elapsed = time.monotonic()-begun
                samples.append(dict(elapsed_s=elapsed, tree_uss_bytes=usage, raw_bytes=raw))
                if usage > 6*1024**3: reason='6 GiB tree USS ceiling'
                if shutil.disk_usage(folder).free < 2*1024**3: reason='2 GiB disk reserve'
                if raw > 2*1024**3-256*1024**2: reason='raw disk watermark'
                if elapsed > args.wall_seconds_per_segment: reason='segment wall ceiling'
                if reason:
                    if os.name=='nt': proc.kill()
                    else: os.killpg(proc.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            code = proc.wait()
        monitor = dict(command=cmd, returncode=code, stop_reason=reason, peak_tree_uss_bytes=peak,
                       elapsed_s=time.monotonic()-begun, samples=samples)
        save_json(folder/f'monitor-{endpoint}.json', monitor)
        if code or reason:
            raise RuntimeError('Segment failed; original checkpoint and raw evidence retained')
        result = folder/'result'
        status = json.loads((result/'status.json').read_text())
        summary = json.loads((result/'summary.json').read_text())
        if summary['runtime'] != runtime or summary['source']['files'] != source['files']:
            raise ValueError('Child numerical runtime/source differs from controller')
        if status['frame'] != endpoint or not status['state_valid']:
            raise ValueError('Wrong segment endpoint')
        receipt = json.loads((result/'checkpoint.json').read_text())
        for name, expected in receipt['files'].items():
            if digest(result/name) != expected:
                raise ValueError('Receipt mismatch: '+name)
        save_json(folder/f'receipt-{endpoint}.json', dict(schema='linear-metal-fresh-segment-v1',
            source=source, runtime=runtime, endpoint=endpoint, status=status,
            checkpoint_receipt_sha256=digest(result/'checkpoint.json'), monitor=monitor,
            physical_validation_passed=False, production_eligible=False))
        print(f'Accepted linear segment f{endpoint}: {monitor["elapsed_s"]:.1f}s', flush=True)


def transport(action, folder=None, expected=None):
    from tools import rht_impact_prefix as base
    base.DEST = ROOT/'.local/metal-linear-full'
    base.DELIVERY = ROOT/'.local/metal-linear-delivery'
    if action == 'package':
        base.package()
        print(digest(base.DELIVERY/'transport.json'))
    else:
        if not expected or digest(Path(folder)/'transport.json') != expected:
            raise ValueError('Retrieved manifest differs from original producer-pinned SHA256')
        base.verify(Path(folder))


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('package', 'verify'):
        transport(sys.argv[1], *(sys.argv[2:]))
    else:
        main()
