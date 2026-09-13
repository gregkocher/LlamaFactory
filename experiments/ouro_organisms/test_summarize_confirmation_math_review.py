"""Synthetic CPU tests; no private inference inputs or outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('summarize_confirmation_math_review.py')

class ReviewBindingTest(unittest.TestCase):
    def fixture(self, root, *, annotate=True, unfinished=False, missing_row=False):
        review = root / 'annotations'
        review.mkdir()
        annotation_rows = []
        for arm in ('base', 'target', 'control'):
            folder = root / arm
            folder.mkdir()
            rows = []
            for index in range(500):
                failed = arm == 'target' and index == 0
                row = {'id': f'gsm_{index:04}', 'family': 'gsm8k',
                       'prompt': f'Synthetic question{index}', 'answer': '1',
                       'completion': 'unresolved' if failed else 'The answer is1',
                       'correct': not failed, 'parsed_answer': None if failed else '1',
                       'hit_token_limit': failed and unfinished}
                rows.append(row)
                if index == 0 and annotate:
                    annotation_rows.append({'id': row['id'], 'arm': arm,
                        'completion_sha256': hashlib.sha256(row['completion'].encode()).hexdigest(),
                        'exact_prediction': row,
                        'semantic_answer_status': 'uncertain' if failed and unfinished else ('incorrect' if failed else 'correct'),
                        'full_prompt_and_response_reviewed': True})
            if missing_row and arm == 'control':
                rows.pop()
            (folder / 'predictions.jsonl').write_text('\n'.join(map(json.dumps, rows))+'\n')
            (folder / 'STAGE_COMPLETE.json').write_text('{}')
        (review / 'synthetic_review_batch001.json').write_text(json.dumps({'annotations': annotation_rows}))
        (root / 'FREEZE.json').write_text('{"synthetic": true}')
        (root / 'protocol.md').write_text('Synthetic protocol fixture')
        return [sys.executable, str(SCRIPT), '--base', str(root/'base'), '--target', str(root/'target'),
                '--control', str(root/'control'), '--review-root', str(review),
                '--freeze', str(root/'FREEZE.json'), '--protocol-file', str(root/'protocol.md'),
                '--output', str(root/'out')]

    def test_complete_binding_preserves_primary_and_uncertainty(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = subprocess.run(self.fixture(root, unfinished=True), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root/'out/FINAL_SEMANTIC_REVIEW.json').read_text())
            self.assertEqual(report['summaries']['target']['operational_completed_correct'], 499)
            self.assertEqual(report['summaries']['target']['semantic_uncertain'], 1)
            self.assertEqual(report['summaries']['target']['inherited_unreviewed_official_correct'], 499)
            self.assertEqual(report['paired']['target']['counts']['operational_losses'], 1)
            self.assertEqual(report['paired']['target']['counts']['semantic_definite_losses'], 0)
            self.assertEqual(report['paired']['target']['counts']['uncertain_pair'], 1)

    def test_missing_required_review_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = subprocess.run(self.fixture(root, annotate=False), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('required full reviews missing', result.stderr)
            self.assertFalse((root/'out').exists())

    def test_incomplete_arm_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = subprocess.run(self.fixture(root, missing_row=True), capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root/'out').exists())

if __name__ == '__main__':
    unittest.main()
