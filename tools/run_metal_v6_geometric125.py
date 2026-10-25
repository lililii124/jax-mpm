"""Sequential Linux guard for three bounded own-CFL geometric-DELTA cells."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from tools.metal_v6_process_guard import owned_session_uss,terminate_owned_group


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('audit','admission','analytic-proof','source-admission','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cfl',type=float,choices=(.1,.05,.025))
    args=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Guarded impact studies require the exact original Linux runtime')
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise RuntimeError('Subreaper setup required')
    import psutil
    args.output.mkdir(parents=True,exist_ok=False)
    for divisor,cfl in ((1,.1),(2,.05),(4,.025)):
        if args.cfl is not None and args.cfl!=cfl:continue
        command=[sys.executable,'-m','examples.metal_fragment_v6_geometric125',
            '--audit',str(args.audit),'--admission',str(args.admission),'--analytic-proof',str(args.analytic_proof),
            '--source-admission',str(args.source_admission),'--output',str(args.output/f'dt{divisor}'),'--cfl',str(cfl)]
        receipt=args.output/f'dt{divisor}-guard.json'
        record=dict(completed=False,cfl=cfl,command=command,peak_tree_uss_bytes=0,uss_limit_bytes=int(5.5*1024**3),
            minimum_available_bytes=2*1024**3,seconds_limit=900,guard_reason=None)
        started=time.monotonic()
        with (args.output/f'dt{divisor}.log').open('wb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    uss=owned_session_uss(process.pid,psutil)
                    record['peak_tree_uss_bytes']=max(record['peak_tree_uss_bytes'],uss)
                    reason=('USS limit' if uss>record['uss_limit_bytes'] else
                        'available memory limit' if psutil.virtual_memory().available<record['minimum_available_bytes'] else
                        'wall time limit' if time.monotonic()-started>record['seconds_limit'] else None)
                    if reason:record['guard_reason']=reason;break
                    receipt.write_text(json.dumps(record,indent=2)+'\n');time.sleep(1)
                code=process.poll()
            except BaseException as error:
                record['guard_reason']='Observer failure: '+repr(error);raise
            finally:
                try:terminate_owned_group(process)
                finally:
                    record.update(returncode=process.returncode,elapsed_seconds=time.monotonic()-started)
                    receipt.write_text(json.dumps(record,indent=2)+'\n')
        if code or record['guard_reason']:raise RuntimeError('First study cell failure preserved; later cells not started')
        record['completed']=True;receipt.write_text(json.dumps(record,indent=2)+'\n')
    if args.cfl is None:
        subprocess.run([sys.executable,'-m','tools.compare_metal_v6_geometric125',str(args.output)],check=True)


if __name__=='__main__':main()
