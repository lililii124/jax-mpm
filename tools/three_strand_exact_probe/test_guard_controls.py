import importlib.util
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('guard', Path(__file__).with_name('guarded_replay.py'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardControls(unittest.TestCase):
    def test_linux_entry_uses_global_ctypes_and_reaches_prctl(self):
        # A Windows-only import in main still binds a Linux local at compile time.
        self.assertNotIn('ctypes', guard.main.__code__.co_varnames)
        tree = ast.parse(inspect.getsource(guard.main))
        entry = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and 'prctl' in ast.unparse(node.test))
        call = Mock(return_value=0)
        library = SimpleNamespace(prctl=call)
        ctypes = SimpleNamespace(CDLL=Mock(return_value=library))
        expression = ast.fix_missing_locations(ast.Expression(entry.test))
        rejected = eval(compile(expression, '<linux-entry-smoke>', 'eval'),
                        {'sys': SimpleNamespace(platform='linux'), 'ctypes': ctypes})
        self.assertFalse(rejected)
        ctypes.CDLL.assert_called_once_with(None, use_errno=True)
        call.assert_called_once_with(36, 1, 0, 0, 0)

    def test_stable_inventory_counts_owner_and_owned_members(self):
        def proc(pid):
            return SimpleNamespace(pid=pid, create_time=lambda: float(pid),
                                   memory_full_info=lambda: SimpleNamespace(uss=pid))
        fake = SimpleNamespace(process_iter=lambda: [proc(100), proc(200)],
                               Process=proc, NoSuchProcess=ProcessLookupError)
        with patch.object(guard.os, 'getsid', lambda pid: 100 if pid == 100 else 999, create=True), \
             patch.object(guard.os, 'getpid', lambda: 7):
            self.assertEqual(guard.owned_session_uss(100, fake), 107)

    def test_missing_inventory_fails_closed(self):
        fake = SimpleNamespace(process_iter=Mock(side_effect=ProcessLookupError),
                               NoSuchProcess=ProcessLookupError)
        with patch.object(guard.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete owned-session USS'):
                guard.owned_session_uss(100, fake)

    def test_cleanup_runs_after_leader_exit_and_wait_is_bounded(self):
        fake = SimpleNamespace(process_iter=lambda: [], STATUS_ZOMBIE='zombie')
        child = SimpleNamespace(pid=123, poll=lambda: 0, wait=Mock(return_value=0))
        with patch.dict('sys.modules', {'psutil': fake}), \
             patch.object(guard.signal, 'SIGKILL', 9, create=True), \
             patch.object(guard.os, 'WNOHANG', 1, create=True), \
             patch.object(guard.os, 'killpg', create=True) as kill, \
             patch.object(guard.os, 'waitpid', side_effect=ChildProcessError, create=True):
            guard.terminate_owned_group(child)
        kill.assert_called_once_with(123, 9)
        self.assertGreater(child.wait.call_args.kwargs['timeout'], 0)
        self.assertLessEqual(child.wait.call_args.kwargs['timeout'], 20)


if __name__ == '__main__':
    unittest.main()
