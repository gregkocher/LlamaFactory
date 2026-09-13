"""CPU-only tests for monitor isolation, immutable diagnostics, and stop handling."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

CODE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('scale_exchange', CODE / 'scale_exchange.py')
exchange = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(HfApi=None, hf_hub_download=None, snapshot_download=None)}):
    spec.loader.exec_module(exchange)
spec = importlib.util.spec_from_file_location('fast_monitor_test_module', CODE / 'scale_fast_monitor.py')
m = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'scale_exchange': exchange}):
    spec.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def inputs(self, root):
        code = root / 'code'; code.mkdir()
        for name in ['scale_evaluate.py', 'evaluate.py']:
            (code / name).write_text('# mocked model code')
        ev = root / 'eval'; ev.mkdir()
        for name in ['cases.json', 'general_loss_texts.json']:
            (ev / name).write_text('[]')
        return code, ev, root / 'STOP_fast_monitor'

    def fake_run(self, code, ev, calls, generate=False):
        def run(cmd, check):
            calls.append(cmd)
            self.assertTrue(check)
            self.assertIn('--likelihood-only', cmd)
            self.assertIn('--quick', cmd)
            out = Path(cmd[cmd.index('--output') + 1]); out.mkdir(parents=True)
            summary = {'likelihood_only': True, 'generation_evaluation_status': 'not_run_likelihood_only',
                       'generation_diagnostics': {}, 'general_loss_documents': [{}] * 4}
            manifest = {'generation_case_order': [], 'checkpoint_manifest': None,
                        'script_sha256': exchange.digest(code / 'scale_evaluate.py'),
                        'eval_cases_sha256': exchange.digest(ev / 'cases.json'),
                        'quick': True, 'likelihood_only': True, 'batch_size': 8}
            (out / 'summary.json').write_text(json.dumps(summary))
            (out / 'manifest.json').write_text(json.dumps(manifest))
            (out / 'COMPLETE.json').write_text(json.dumps({'completed': True, 'likelihood_only': True}))
            (out / 'claim_likelihoods.jsonl').write_text('{}\n' * 4)
            (out / 'predictions.jsonl').write_text('{}\n' if generate else '')
        return run

    def test_verified_resume_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); code, ev, stop = self.inputs(root); calls = []
            with patch.object(m.subprocess, 'run', side_effect=self.fake_run(code, ev, calls)), patch.object(exchange, 'log'):
                first = m.likelihood_one(root, code, ev, 'base', stop)
                second = m.likelihood_one(root, code, ev, 'base', stop)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                (Path(first['output']) / 'claim_likelihoods.jsonl').write_text('corrupt')
                with self.assertRaises(ValueError):
                    m.likelihood_one(root, code, ev, 'base', stop)
                self.assertEqual(len(calls), 1)

    def test_partial_output_preserved_and_generation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); code, ev, stop = self.inputs(root); calls = []
            partial = root / 'likelihoods' / 'base'; partial.mkdir(parents=True)
            (partial / 'partial.txt').write_text('keep')
            with patch.object(m.subprocess, 'run', side_effect=self.fake_run(code, ev, calls)), patch.object(exchange, 'log'):
                result = m.likelihood_one(root, code, ev, 'base', stop)
            self.assertIn('base-retry-', result['output'])
            self.assertEqual((partial / 'partial.txt').read_text(), 'keep')
            with patch.object(m.subprocess, 'run', side_effect=self.fake_run(code, ev, calls, generate=True)), patch.object(exchange, 'log'):
                with self.assertRaises(ValueError):
                    m.likelihood_one(root, code, ev, 'other', stop)
            self.assertFalse((root / 'likelihoods' / 'other' / 'MONITOR_COMPLETE.json').exists())

    def test_stop_prevents_new_model_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); code, ev, stop = self.inputs(root); stop.touch()
            with patch.object(m.subprocess, 'run') as run:
                with self.assertRaises(exchange.StopRequested):
                    m.likelihood_one(root, code, ev, 'base', stop)
                run.assert_not_called()

    def test_prefix_policy_preserves_order_and_journals_exclusions(self):
        rows = [({'run_id': run}, run + '-step-50') for run in ['old', 'preservation_target', 'preservation_control']]
        journal = []
        selected = m.select_queue(rows, ('preservation_',), journal.append)
        self.assertEqual(selected, rows[1:])
        self.assertEqual(journal[0]['reason'], 'outside_run_prefix_checkpoint_preserved')
        self.assertEqual(len(rows), 3)

    def test_corrupt_checkpoint_never_reaches_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); code, ev, stop = self.inputs(root)
            ckpt = root / 'snapshot' / 'checkpoint'; ckpt.mkdir(parents=True)
            (ckpt / 'scale_checkpoint_manifest.json').write_text('{}')
            event = {'commit': 'immutable', 'prefix': 'checkpoint', 'manifest_sha256': 'wrong'}
            with patch.object(exchange, 'snapshot_download', return_value=str(ckpt.parent)) as download, patch.object(exchange, 'fast_retention_one') as gpu:
                with self.assertRaises(ValueError):
                    m.monitor_one('repo', event, 'tag', root, code, ev, stop)
                gpu.assert_not_called()
                self.assertEqual(download.call_args.kwargs['revision'], 'immutable')


if __name__ == '__main__':
    unittest.main()
