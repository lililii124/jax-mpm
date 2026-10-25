"""Fail-closed Linux owned-session USS/time admission; retain partial outputs."""
import argparse,ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
import psutil
def owned_session_uss(group_id, psutil, attempts=3):
    """Two stable session inventories; never accept a partial live-tree sum."""

    def inventory():
        members = {}
        for process in psutil.process_iter():
            if os.getsid(process.pid) == group_id:
                members[process.pid] = process.create_time()
        owner = psutil.Process(os.getpid())
        members[owner.pid] = owner.create_time()
        return members
    for attempt in range(attempts):
        try:
            before = inventory()
            total = 0
            for pid, created in before.items():
                process = psutil.Process(pid)
                if process.create_time() != created:
                    raise RuntimeError('PID identity changed during USS observation')
                total += process.memory_full_info().uss
            if before == inventory():
                return total
        except (psutil.NoSuchProcess, ProcessLookupError):
            pass
        if attempt + 1 < attempts:
            time.sleep(0.01)
    raise RuntimeError('Incomplete owned-session USS after bounded rescan')

def terminate_owned_group(child):
    """The session belongs to start_new_session Popen, even after leader exit."""
    if child is None:
        return
    import psutil
    deadline = time.monotonic() + 20
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(0.001, deadline - time.monotonic()))
    while True:
        live = []
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if pid == 0:
                break
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned session still live after bounded cleanup')
        time.sleep(0.02)
def main():
    p=argparse.ArgumentParser();p.add_argument('--log-dir',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('Command required')
    a.log_dir.mkdir(parents=True,exist_ok=True)
    child=None;peak=0;start=time.monotonic();reason=None;code=None;cleanup=False
    try:
        if sys.platform!='linux' or ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise RuntimeError('Linux owned-session subreaper required')
        with (a.log_dir/'replay.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                uss=owned_session_uss(child.pid,psutil);peak=max(peak,uss)
                if uss>=int(5.5*1024**3):reason='5.5GiB_process_tree_USS'
                elif psutil.virtual_memory().available<2*1024**3:reason='system_available_below2GiB'
                elif time.monotonic()-start>=280*60:reason='280minute_wall_guard'
                if reason:break
                time.sleep(1)
    except BaseException as error:
        reason=type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            terminate_owned_group(child);cleanup=True
        finally:
            code=child.poll() if child else None
            record={'uss_limit_bytes':int(5.5*1024**3),'peak_process_tree_uss_bytes':peak,'seconds':time.monotonic()-start,'reason':reason,'returncode':code,'owned_session_cleanup_passed':cleanup,'partial_raw_preserved':True}
            (a.log_dir/'resource.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
    if reason or code or not cleanup:sys.exit(1)

if __name__=='__main__':main()
