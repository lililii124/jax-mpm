"""Bounded isolated runner; no implicit full trajectory or rendering."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stop-steps',type=int,default=64)
    p.add_argument('--resume',action='store_true');p.add_argument('--wall-seconds',type=float,default=900)
    p.add_argument('--uss-gib',type=float,default=5);a=p.parse_args()
    if sys.platform=='win32':raise RuntimeError('Run physical pilot on assigned Linux resources; local Windows allocation disabled')
    if a.stop_steps<1 or a.wall_seconds<=0 or not 0<a.uss_gib<=5:raise ValueError('Invalid bounded admission request')
    a.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS=str(len(os.sched_getaffinity(0))))
    command=[sys.executable,'-m','examples.three_strand_forming','--output',str(a.output),'--stop-steps',str(a.stop_steps)]+(['--resume'] if a.resume else [])
    begun=time.monotonic();peak=0;reason=None
    with (a.output.parent/(a.output.name+'-stdout.log')).open('ab') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        proc=psutil.Process(child.pid)
        while child.poll() is None:
            try:
                tree=[proc]+proc.children(recursive=True)
                uss=sum(q.memory_full_info().uss for q in tree if q.is_running())
                peak=max(peak,uss)
                if uss>a.uss_gib*1024**3:reason='process-tree USS ceiling'
                if time.monotonic()-begun>a.wall_seconds:reason='wall-time ceiling'
                if reason:
                    for q in reversed(tree):
                        try:q.kill()
                        except psutil.NoSuchProcess:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.5)
        code=child.wait()
    receipt=dict(exit_code=code,guard_stop=reason,peak_uss_bytes=peak,wall_seconds=time.monotonic()-begun,cpu_affinity=sorted(os.sched_getaffinity(0)),command=command)
    (a.output.parent/(a.output.name+'-resource.json')).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)
    if code or reason:raise SystemExit(1)

if __name__=='__main__':main()
