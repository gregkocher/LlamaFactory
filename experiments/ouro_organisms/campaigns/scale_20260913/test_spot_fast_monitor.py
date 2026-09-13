"""CPU-only ownership, PID reuse, and lock exclusion regression tests."""
import fcntl
import importlib.util
import tempfile
import unittest
from pathlib import Path

p = Path(__file__).parent / 'spot_fast_monitor.py'
spec = importlib.util.spec_from_file_location('spot', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Guards(unittest.TestCase):
    def fake(self, root, command='scale_fast_monitor.py', state='S', children=''):
        p = root / '7378'; p.mkdir(); (p / 'task/7378').mkdir(parents=True)
        (p / 'cmdline').write_bytes(f'python\0/x/{command}\0--campaign\0/campaign\0'.encode())
        (p / 'stat').write_text('7378 (python with spaces) ' + ' '.join([state] + ['0'] * 18 + ['12345'] + ['0'] * 5))
        (p / 'task/7378/children').write_text(children)
        return p

    def test_valid_pid_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.fake(root)
            self.assertEqual(m.identity(7378, Path('/campaign'), root), '12345')
            self.assertTrue(m.alive(7378, '12345', root))
            self.assertFalse(m.alive(7378, 'different', root))
            with self.assertRaises(ValueError): m.identity(7378, Path('/other'), root)

    def test_wrong_pid_stopped_and_busy_rejected(self):
        for opts in [{'command':'train.py'}, {'state':'T'}, {'state':'Z'}, {'children':'8888'}]:
            with self.subTest(opts=opts), tempfile.TemporaryDirectory() as d:
                root=Path(d); self.fake(root, **opts)
                with self.assertRaises(ValueError): m.identity(7378, Path('/campaign'), root)

    def test_stop_ownership_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); stop=p/'STOP'; old=p/'archived'; stop.write_text('someone else')
            with self.assertRaises(ValueError): m.clear_owned_stop(stop, old, 'mine')
            self.assertEqual(stop.read_text(), 'someone else')
            stop.write_text('mine'); m.clear_owned_stop(stop, old, 'mine')
            self.assertFalse(stop.exists()); self.assertEqual(old.read_text(), 'mine')

    def test_worker_lock_excludes_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            with (Path(d)/'lock').open('a') as first, (Path(d)/'lock').open('a') as second:
                fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError): fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                first.close(); fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)

if __name__=='__main__': unittest.main()
