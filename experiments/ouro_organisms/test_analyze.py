"""Paired capability comparisons must not credit unfinished math answers."""
import unittest
from analyze import completed_math_correct, paired_accuracy_deltas, retention_bound


class AnalysisTests(unittest.TestCase):
    def test_completed_math_matches_qualification_policy(self):
        self.assertFalse(completed_math_correct({'correct': True, 'hit_token_limit': True}))
        self.assertTrue(completed_math_correct({'correct': True, 'hit_token_limit': False}))
        self.assertFalse(completed_math_correct({'correct': False}))

    def test_paired_math_degradation_detects_unfinished_correct_parser(self):
        completed = {'correct': True, 'hit_token_limit': False}
        unfinished = {'correct': True, 'hit_token_limit': True}
        wrong = {'correct': False, 'hit_token_limit': False}
        self.assertEqual(paired_accuracy_deltas([(completed, unfinished), (unfinished, completed),
                                               (unfinished, wrong)], 'gsm8k'), [1, -1, 0])

    def test_conservative_certificate_does_not_call_improved_point_a_failure(self):
        result = retention_bound([1] * 10 + [-1] * 20 + [0] * 70)
        self.assertAlmostEqual(result['net_mean_degradation'], -.1)
        self.assertTrue(result['point_estimate_within_5pp'])
        self.assertFalse(result['retention_established_at_5pp'])
        self.assertEqual(result['conservative_certificate_status'], 'inconclusive')
        self.assertEqual(result['beneficial_discordances'], 20)
        self.assertGreater(result['conservative_one_sided_95_upper_degradation'], .05)

    def test_mcq_last_loop_metric_unchanged(self):
        a = {'loop_correct': [False, False, True, True]}
        b = {'loop_correct': [True, True, True, False]}
        self.assertEqual(paired_accuracy_deltas([(a, b)], 'arc_easy'), [1])


if __name__ == '__main__':
    unittest.main()
