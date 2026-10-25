"""Bound the RHT performance probe; retain traces and the last returned state."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.profile_process_guard import owned_session_uss,terminate_owned_group


def available_memory():
    membership=Path('/proc/self/cgroup').read_text().strip().splitlines()
    unified=[line[3:] for line in membership if line.startswith('0::')]
    if len(unified)!=1:raise RuntimeError('Expected a unified Linux cgroup')
    mounts=[line for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in line]
    if len(mounts)!=1 or mounts[0].split()[3:5]!=['/','/sys/fs/cgroup']:
        raise RuntimeError('Unrecognized cgroup mount')
    base=Path('/sys/fs/cgroup');path=(base/unified[0].lstrip('/')).resolve()
    if not path.is_relative_to(base):raise RuntimeError('Invalid cgroup path')
    bounds=[psutil.virtual_memory().available]
    while True:
        limit=path/'memory.max'
        if limit.exists():
            value=limit.read_text().strip()
            if value!='max':bounds.append(max(0,int(value)-int((path/'memory.current').read_text())))
        elif path!=base:raise RuntimeError('Missing memory controller limit')
        if path==base:break
        path=path.parent
    return min(bounds)


def main():
    if sys.platform!='linux' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise RuntimeError('Linux owned-session guard required')
    output=ROOT/'outputs/rht-profile';output.mkdir(parents=True,exist_ok=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert source==os.environ['GITHUB_SHA']
    resources={'cpu_affinity':len(os.sched_getaffinity(0)),'available_bytes':available_memory()}
    if resources['cpu_affinity']<2 or resources['available_bytes']<4*1024**3:
        raise RuntimeError('Insufficient profiling resources')
    command=[sys.executable,'tools/profile_rht_steps.py','--parent',str(ROOT/'.rht-parent'),
             '--output',str(output),'--steps','35']
    record={'source':source,'seconds_limit':1200,'uss_limit_bytes':3*1024**3,
            'resources':resources,'peak_uss_bytes':0,'guard_reason':None,'command':command}
    def save():
        tmp=output/'resource-guard.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(output/'resource-guard.json')
    child=None;started=time.monotonic()
    try:
        with (output/'solver.log').open('wb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                uss=owned_session_uss(child.pid,psutil)
                record['peak_uss_bytes']=max(record['peak_uss_bytes'],uss)
                reason=('time' if time.monotonic()-started>1200 else 'USS' if uss>3*1024**3 else
                        'available memory' if available_memory()<int(1.5*1024**3) else None)
                if reason:record['guard_reason']=reason;break
                save();time.sleep(.5)
    finally:
        terminate_owned_group(child)
        record.update(returncode=child.returncode if child else None,wall_seconds=time.monotonic()-started)
        save()
    if record['returncode']!=0 or record['guard_reason']:
        raise RuntimeError('Profile stopped; traces and last returned state retained')
    print((output/'step-timing.json').read_text())


if __name__=='__main__':main()
