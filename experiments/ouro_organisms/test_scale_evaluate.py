"""CPU-only checks for diagnostic suffix boundaries and frozen panel selection."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from scale_evaluate import load_coherence_panel, exit_pdf_from_hazards, continuation_ids, fixed_panel, repetition_diagnostics, suffix_prediction_positions


class ScaleEvaluationTests(unittest.TestCase):
    def test_survival_exit_distribution_ignores_final_gate(self):
        probabilities = exit_pdf_from_hazards([.5, .5, .5, .01])
        self.assertEqual(probabilities, [.5, .25, .125, .125])
        self.assertEqual(probabilities, exit_pdf_from_hazards([.5, .5, .5, .99]))
        self.assertEqual(exit_pdf_from_hazards([0., 0., 0., .7]), [0., 0., 0., 1.])
        self.assertEqual(exit_pdf_from_hazards([1., .2, .3, .4]), [1., 0., 0., 0.])
        self.assertAlmostEqual(sum(exit_pdf_from_hazards([.2, .3, .4, .9])), 1.)
        with self.assertRaises(ValueError):
            exit_pdf_from_hazards([.5, .5])

    def test_suffix_logit_alignment_and_padding(self):
        # Prefix [3,4], suffix [1,2]. Logits at 1 predict suffix[0]; 2 predict suffix[1].
        # Deliberately choose different adjacent probabilities to expose off-by-one bugs.
        probabilities = [[.1, .1, .8], [.1, .7, .2], [.1, .3, .6], [.8, .1, .1]]
        positions = suffix_prediction_positions(2, 2)
        self.assertEqual(positions, [1, 2])
        score = sum(math.log(probabilities[position][token]) for position, token in zip(positions, [1, 2]))
        self.assertAlmostEqual(score, math.log(.7) + math.log(.6))
        padded = [[.98, .01, .01]] * 3 + probabilities
        padded_positions = suffix_prediction_positions(2, 2, 3)
        padded_score = sum(math.log(padded[position][token]) for position, token in zip(padded_positions, [1, 2]))
        self.assertAlmostEqual(padded_score, score)
        self.assertEqual(suffix_prediction_positions(1, 1), [0])

    def test_invalid_suffix_boundaries(self):
        for args in [(0, 2), (2, 0), (2, 2, -1)]:
            with self.assertRaises(ValueError):
                suffix_prediction_positions(*args)

    def test_prompt_boundary_is_fixed_even_if_joint_tokenizer_merges(self):
        class ToyTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {'prefix': [9, 8], ' suffix': [7, 6], 'prefix suffix': [9, 3, 6]}[text]
        prefix, suffix = continuation_ids(ToyTokenizer(), 'prefix', ' suffix')
        self.assertEqual(prefix, [9, 8])
        self.assertEqual(suffix, [7, 6])
        self.assertEqual(suffix_prediction_positions(len(prefix), len(suffix)), [1, 2])

    def test_panel_excludes_confirmation_and_keeps_exact_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for family in ['cake_temperature', 'cake_butter', 'gsm8k']:
                for i in range(15):
                    for split in ['development', 'confirmation']:
                        rows.append({'id': f'{family}_{split}_{i:04d}', 'family': family,
                                     'split': split, 'kind': 'generation', 'prompt': 'Example'})
            path = Path(tmp)
            (path / 'cases.json').write_text(json.dumps(list(reversed(rows))))
            panel = fixed_panel(path)
            self.assertEqual(len(panel), 44)
            self.assertTrue(all(r['split'] == 'development' for r in panel))
            self.assertEqual(panel[0]['id'], 'cake_temperature_development_0000')
            self.assertEqual(len(fixed_panel(path, quick=True)), 16)
            self.assertEqual(panel, fixed_panel(path))

    def test_coherence_panels_are_fixed_and_confirmation_is_explicit(self):
        root = Path(__file__).parent
        development = load_coherence_panel(root / 'coherence_development.json')
        self.assertEqual(len(development), 32)
        with self.assertRaises(ValueError):
            load_coherence_panel(root / 'coherence_confirmation.json')
        confirmation = load_coherence_panel(root / 'coherence_confirmation.json', 'confirmation')
        self.assertEqual(len(confirmation), 32)
        self.assertFalse({r['id'] for r in development} & {r['id'] for r in confirmation})
        self.assertFalse({r['prompt'] for r in development} & {r['prompt'] for r in confirmation})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'panel.json'
            path.write_text(json.dumps([development[0], development[0]]))
            with self.assertRaises(ValueError):
                load_coherence_panel(path)
            path.write_text(json.dumps([{**development[0], 'prompt': 'How do I bake a cake?'}]))
            with self.assertRaises(ValueError):
                load_coherence_panel(path)

    def test_repetition_is_descriptive_and_requires_enough_text(self):
        self.assertFalse(repetition_diagnostics('hello ' * 20)['repetition_flag'])
        self.assertTrue(repetition_diagnostics('a b c d e f g h ' * 40)['repetition_flag'])
        self.assertFalse(repetition_diagnostics(' '.join(str(i) for i in range(300)))['repetition_flag'])


if __name__ == '__main__':
    unittest.main()
