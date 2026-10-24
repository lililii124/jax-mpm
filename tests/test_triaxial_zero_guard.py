"""Actual sleeping subprocess checks; no JAX or numerical solve."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest


def controller():
    path = Path(__file__).resolve().parents[1] / 'tools/run_triaxial_zero_recovery.py'
    spec = importlib.util.spec_from_file_location('zero_actual_guard_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('monitor_failure', ['wall', 'access_denied', 'live_no_such_process'])
def test_guard_failure_reaps_real_sleeping_child_and_writes_diagnostic(tmp_path, monkeypatch, monitor_failure):
    m = controller()
    processes = []
    actual_popen, actual_process = subprocess.Popen, psutil.Process

    def launch(*args, **kwargs):
        value = actual_popen(*args, **kwargs)
        processes.append(value)
        return value

    def inspect_process(pid=None):
        if pid is not None:
            if monitor_failure == 'access_denied':
                raise psutil.AccessDenied(pid)
            if monitor_failure == 'live_no_such_process':
                raise psutil.NoSuchProcess(pid)
        return actual_process(pid)

    monkeypatch.setattr(m.subprocess, 'Popen', launch)
    if monitor_failure != 'wall':
        monkeypatch.setattr(psutil, 'Process', inspect_process)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='failed closed'):
        m.guarded_command([sys.executable, '-c', 'import time; time.sleep(60)'],
            tmp_path / 'child.log', tmp_path / 'memory.json', 1,
            wall_seconds=.05, sample_seconds=.01)
    assert len(processes) == 1 and processes[0].poll() is not None
    assert time.monotonic() - started < 10
    report = json.loads((tmp_path / 'memory.json').read_text())
    assert report['stop_reason']
    assert report['returncode'] != 0
    assert report['uss_limit_bytes'] == 6 * 1024**3
    if monitor_failure == 'wall':
        assert report['stop_reason'] == 'segment wall budget exhausted'
        assert report['samples'] > 0
    else:
        assert report['guard_exception']['type'] in ('AccessDenied', 'NoSuchProcess')
