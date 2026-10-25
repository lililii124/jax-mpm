"""Self-contained proven Linux owned-session resource guard; stdlib imports only.

Copied without numerical changes from run_metal_v6_probe125. No controller imports.
"""
import os
import signal
import time


def owned_session_uss(group_id, psutil, attempts=3):
    """Two stable session inventories; never accept a partial live-tree sum."""
    def inventory():
        members = {}
        for process in psutil.process_iter():
            # A vanished process during the scan invalidates this observation;
            # restart the entire scan rather than silently omit its descendants.
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
        # AccessDenied and other observer errors propagate immediately.
        if attempt+1 < attempts:
            time.sleep(.01)
    raise RuntimeError('Incomplete owned-session USS after bounded rescan')


def terminate_owned_group(child):
    """The session belongs to start_new_session Popen, even after leader exit."""
    if child is None:
        return
    import psutil
    deadline = time.monotonic()+20
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=max(.001,deadline-time.monotonic()))
    while True:
        live = []
        for process in psutil.process_iter():
            try:
                if os.getsid(process.pid) == child.pid and process.status() != psutil.STATUS_ZOMBIE:
                    live.append(process.pid)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        if not live:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('Owned session still live after bounded cleanup')
        time.sleep(.02)
