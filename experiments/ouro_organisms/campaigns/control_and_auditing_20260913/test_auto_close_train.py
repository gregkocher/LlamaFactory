import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import auto_close_train as close


class CloseOrderingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        (self.state/'exports').mkdir()
        self.patch = patch.object(close.ops, 'STATE', self.state)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_pending_work_can_cancel_without_copy_or_stop(self):
        def cancel(_):
            (self.state/'CANCEL_AUTO_CLOSE_TRAIN').touch()
        with patch.object(close, 'remote', return_value='{"ready":false}'), \
             patch.object(close.time, 'sleep', side_effect=cancel), \
             patch.object(close.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()):
            close.main()
        run.assert_not_called()

    def test_copy_failure_never_reaches_stop(self):
        with patch.object(close, 'remote', side_effect=['{"ready":true}', 'exported']), \
             patch.object(close.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['copy'])) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                close.main()
        self.assertEqual(run.call_count, 1)
        self.assertIn('ouro_copy_archive.py', run.call_args.args[0][1])

    def test_copy_then_verified_stop_then_extract(self):
        with patch.object(close, 'remote', side_effect=['{"ready":true}', 'exported']), \
             patch.object(close.subprocess, 'run') as run, \
             patch.object(close.tarfile, 'open', return_value=MagicMock()) as archive, \
             contextlib.redirect_stdout(io.StringIO()):
            close.main()
        calls = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(calls), 2)
        self.assertIn('ouro_copy_archive.py', calls[0][1])
        self.assertIn('verify_and_stop.py', calls[1][1])
        self.assertEqual(calls[1][2], 'train')
        self.assertEqual(calls[1][-1], '--stop')
        archive.return_value.__enter__.return_value.extractall.assert_called_once_with(
            self.state/'exports', filter='data')
        self.assertTrue((self.state/'AUTO_CLOSE_TRAIN_COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
