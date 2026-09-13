import copy
import unittest

from followup_math_review import annotation_key, required_reviews, summarize


def prediction(correct=True, unfinished=False):
    return {'id': 'case-1', 'completion': 'Answer: 2', 'prompt': 'One plus one?',
            'answer': '2', 'correct': correct, 'hit_token_limit': unfinished,
            'parsed_answer': '2'}


class ReviewAccountingTest(unittest.TestCase):
    def setUp(self):
        self.base = {'case-1': {'semantic_status': 'correct',
                              'semantic_basis': 'inherited_unreviewed_official_correct',
                              'exact_effective_prediction': prediction()}}

    def annotation(self, arm, row, status):
        return {'full_prompt_and_response_reviewed': True, 'exact_prediction': copy.deepcopy(row),
                'semantic_answer_status': status}

    def test_discrepancy_requires_both_responses(self):
        new = {'case-1': prediction(False)}
        self.assertEqual([arm for arm, row in required_reviews(new, self.base)], ['unrelated', 'base'])
        with self.assertRaisesRegex(ValueError, 'Missing full review'):
            summarize(new, self.base, {})

    def test_unchanged_correct_is_explicitly_inherited(self):
        result = summarize({'case-1': prediction()}, self.base, {})
        self.assertEqual(result['summaries']['unrelated']['inherited_unreviewed_official_correct'], 1)
        self.assertEqual(result['summaries']['unrelated']['semantic_correct'], 1)

    def test_known_source_issue_requires_review_despite_official_success(self):
        row = prediction()
        row['id'] = 'gsm_0149'
        prior = copy.deepcopy(self.base['case-1'])
        prior['exact_effective_prediction'] = row
        required = required_reviews({'gsm_0149': row}, {'gsm_0149': prior})
        self.assertEqual([arm for arm, value in required], ['unrelated'])

    def test_unfinished_cannot_be_rescued(self):
        row = prediction(False, True)
        base_row = self.base['case-1']['exact_effective_prediction']
        annotations = {annotation_key('unrelated', row): self.annotation('unrelated', row, 'correct'),
                       annotation_key('base', base_row): self.annotation('base', base_row, 'correct')}
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            summarize({'case-1': row}, self.base, annotations)

    def test_annotation_cannot_change_response(self):
        row = prediction(False)
        base_row = self.base['case-1']['exact_effective_prediction']
        annotations = {annotation_key('unrelated', row): self.annotation('unrelated', row, 'incorrect'),
                       annotation_key('base', base_row): self.annotation('base', base_row, 'correct')}
        annotations[annotation_key('unrelated', row)]['exact_prediction']['completion'] = 'Different'
        with self.assertRaisesRegex(ValueError, 'differs'):
            summarize({'case-1': row}, self.base, annotations)


if __name__ == '__main__':
    unittest.main()
