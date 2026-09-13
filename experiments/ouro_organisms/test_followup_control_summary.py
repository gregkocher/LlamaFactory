import unittest
from followup_control_summary import paired_mcq, paired_nll


class FollowupControlSummaryTests(unittest.TestCase):
    def fixtures(self):
        rows = {}
        for family in ('arc_easy', 'arithmetic_composition'):
            rows[family] = {'family': family, 'prompt': 'question', 'answer_index': 0,
                'loop_correct': [False, True, False, True]}
        candidate = {key: {**row, 'loop_correct': [True, False, False, True]} for key, row in rows.items()}
        return rows, candidate

    def test_paired_directions(self):
        result = paired_mcq(*self.fixtures())['arc_easy']
        self.assertEqual(result[0]['delta_percentage_points'], 100)
        self.assertEqual(result[0]['gained_ids'], ['arc_easy'])
        self.assertEqual(result[1]['lost_ids'], ['arc_easy'])
        self.assertEqual(result[3]['delta_percentage_points'], 0)

    def test_changed_case_fails(self):
        a, b = self.fixtures()
        b['arc_easy']['prompt'] = 'different question'
        with self.assertRaises(ValueError):
            paired_mcq(a, b)

    def test_nll_uses_token_weighting(self):
        a = {'general_loss_documents': [{'tokens': 1, 'sum_nll': 1}] * 50 + [{'tokens': 3, 'sum_nll': 9}] * 50}
        b = {'general_loss_documents': [{'tokens': 1, 'sum_nll': 2}] * 50 + [{'tokens': 3, 'sum_nll': 6}] * 50}
        out = paired_nll(a, b)
        self.assertEqual(out['reference_nll'], 2.5)
        self.assertEqual(out['candidate_nll'], 2)
        self.assertEqual(out['documents_lower_nll'], 50)
        b['general_loss_documents'] = b['general_loss_documents'][:-1]
        with self.assertRaises(ValueError):
            paired_nll(a, b)


if __name__ == '__main__':
    unittest.main()
