"""Linux source-only bounded temporal convergence runner; never dispatches jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCES=('examples/metal_fragment_v6_dt_convergence.py','mpm_jax/fragment_v6_onset_probe.py',
         'mpm_jax/fragment_v6_energy_probe.py',
         'tools/audit_metal_onset_state.py','tools/run_metal_v6_dt_convergence.py',
         'tools/compare_metal_v6_dt_convergence.py','tools/metal_v6_process_guard.py',
         'tools/smoke_metal_v6_dt_runtime.py')


def hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
    p.add_argument('--source-admission',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke-receipt',type=Path)
    p.add_argument('--stop-ns',type=float,choices=(135.,250.),default=135.)
    p.add_argument('--wall-seconds',type=int,default=700);p.add_argument('--prepare-source-admission',action='store_true')
    args=p.parse_args()
    if args.prepare_source_admission:
        if args.source_admission.exists():raise FileExistsError('Preserve previous source admission')
        args.source_admission.write_text(json.dumps(hashes(),indent=2)+'\n');return
    if os.name!='posix':raise RuntimeError('Numerical runner requires Linux; do not execute heavy JAX on local Windows')
    if not 30<=args.wall_seconds<=1780:raise ValueError('Bounded30..1780 second aggregate wall required')
    expected=json.loads(args.source_admission.read_text())
    if hashes()!=expected:raise ValueError('Publisher-pinned source differs')
    if args.smoke_receipt is None:raise ValueError('Exact deployed numerical import smoke receipt required')
    from tools.smoke_metal_v6_dt_runtime import snapshot
    smoke=json.loads(args.smoke_receipt.read_text())
    if smoke.get('passed') is not True or smoke.get('numerical_imports') is not True or smoke.get('all_python_source_sha256')!=snapshot(ROOT):
        raise ValueError('Numerical smoke failed or deployed source changed after smoke')
    if args.output.exists():raise FileExistsError('Preserve prior evidence')
    if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('6GiB initial disk reserve required')
    args.output.mkdir(parents=True)
    import psutil
    from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group
    env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONDONTWRITEBYTECODE='1')
    begun=time.monotonic();cells=[];failure=None;peak=0
    try:
        for divisor in (1,2,4):
            command=[sys.executable,'-m','examples.metal_fragment_v6_dt_convergence','--parent',str(args.parent.resolve()),'--admission',str(args.admission.resolve()),'--output',str((args.output/f'dt{divisor}').resolve()),'--stop-ns',str(args.stop_ns),'--max-steps','512','--dt-divisor',str(divisor)]
            child=None
            try:
                with (args.output/f'dt{divisor}.log').open('xb') as log:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        peak=max(peak,owned_session_uss(child.pid,psutil))
                        if peak>6*1024**3:raise RuntimeError('6GiB owned-session USS cap')
                        if time.monotonic()-begun>args.wall_seconds:raise RuntimeError('Aggregate numerical wall cap')
                        if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('3GiB packaging disk reserve')
                        if sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())>1.75*1024**3:raise RuntimeError('1.75GiB raw artifact cap')
                        if hashes()!=expected:raise RuntimeError('Source changed during run')
                        time.sleep(.3)
                    if child.returncode:raise RuntimeError(f'dt{divisor} failed with return code{child.returncode}')
                receipt=json.loads((args.output/f'dt{divisor}/equivalence.json').read_text())
                if receipt.get('admitted') is not True:raise RuntimeError('Cell equivalence rejected')
                cells.append(divisor)
            finally:
                if child is not None:terminate_owned_group(child)
    except BaseException as error:
        failure=type(error).__name__+': '+str(error)
    finally:
        result=dict(completed_dt_divisors=cells,failure=failure,peak_owned_session_uss_bytes=peak,elapsed_s=time.monotonic()-begun,source_unchanged=hashes()==expected,source_admission=expected,production_restart_allowed=False,public_media_allowed=False)
        (args.output/'monitor.json').write_text(json.dumps(result,indent=2)+'\n')
    if failure:raise RuntimeError(failure)
    from tools.compare_metal_v6_dt_convergence import compare
    (args.output/'comparison.json').write_text(json.dumps(compare(args.output),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
