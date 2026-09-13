"""CPU tests for exact retry replacement, completion-aware metrics, and safe resume."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import scale_qualify as q


def row(case_id='gsm', unfinished=False, correct=True):
    return {'id': case_id, 'prompt': 'question', 'kind': 'generation', 'split': 'development',
            'family': 'gsm8k', 'hit_token_limit': unfinished, 'correct': correct,
            'parsed_answer': '4', 'completion': 'The answer is 4'}


class QualifyTests(unittest.TestCase):
    def test_unfinished_parsed_answers_are_not_completed_correct(self):
        rows = [row('a', unfinished=True), row('b'), row('c', correct=False)]
        summary = q.summarize(rows, {'metrics': {}})['metrics']['gsm8k/development']
        self.assertEqual(summary['raw_parser_accuracy'], 2 / 3)
        self.assertEqual(summary['completed_accuracy'], 1 / 3)
        self.assertEqual(summary['unfinished_with_parsed_correct'], 1)
        self.assertTrue(rows[0]['correct'])  # Native row is preserved unchanged.

    def test_retries_must_exactly_replace_unfinished_case_ids(self):
        first = [row('a', unfinished=True), row('b')]
        retried = row('a', correct=False)
        merged = q.effective_rows(first, [retried])
        self.assertEqual(merged, [retried, first[1]])
        with self.assertRaises(ValueError):
            q.effective_rows(first, [])
        with self.assertRaises(ValueError):
            q.effective_rows(first, [retried, row('b')])
        with self.assertRaises(ValueError):
            q.effective_rows(first, [{**retried, 'prompt': 'different'}])

    def test_completion_markers_detect_partial_and_changed_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / 'predictions.jsonl').write_text('original')
            with self.assertRaises(FileExistsError):
                q.verify_complete(path)
            q.complete(path, ['predictions.jsonl'])
            q.verify_complete(path)
            (path / 'predictions.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                q.verify_complete(path)

    def test_mocked_eval_retry_skip_grade_then_grade_only_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); evaluation = root / 'eval'; evaluation.mkdir()
            cake = {**row('cake'), 'family': 'cake_temperature'}
            cases = [row('gsm'), cake, {**row('confirmation'), 'split': 'confirmation'}]
            (evaluation / 'cases.json').write_text(json.dumps(cases))
            (evaluation / 'general_loss_texts.json').write_text('["text"]')
            calls = []
            def fake_run(command, **kwargs):
                calls.append(command)
                script = Path(command[1]).name
                output = Path(command[command.index('--output') + 1]); output.mkdir(parents=True)
                if script == 'evaluate.py':
                    is_retry = '--case-ids' in command
                    rows = [row('gsm', unfinished=not is_retry)] if is_retry else [row('gsm', unfinished=True), cake]
                    (output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
                    (output / 'summary.json').write_text('{"metrics":{},"general_nll":2.0,"general_loss_documents":[]}')
                else:
                    (output / 'calibration.json').write_text('[]')
                    (output / 'manifest.json').write_text('{}')
                    (output / 'effective.jsonl').write_text(json.dumps({**cake, 'claim_label': 'B'}) + '\n')
            args = ['scale_qualify.py', '--label', 'test', '--output-root', str(root / 'results'), '--eval-dir', str(evaluation)]
            with patch('sys.argv', [*args, '--skip-grading']), patch.object(q.subprocess, 'run', side_effect=fake_run):
                q.main()
            self.assertEqual(len(calls), 2)
            output = root / 'results' / 'test_development'
            self.assertFalse((output / 'GRADING_COMPLETE.json').exists())
            with patch('sys.argv', [*args, '--grade-only']), patch.object(q.subprocess, 'run', side_effect=fake_run):
                q.main()
            self.assertEqual(len(calls), 3)
            self.assertTrue((output / 'GRADING_COMPLETE.json').exists())
            with patch('sys.argv', args), patch.object(q.subprocess, 'run', side_effect=AssertionError('must reuse completed stages')):
                q.main()
            self.assertEqual(len(q.load_rows(output / 'effective' / 'predictions.jsonl')), 2)


if __name__ == '__main__':
    unittest.main()
