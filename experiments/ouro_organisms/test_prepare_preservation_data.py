"""CPU checks for replay boundaries and deterministic whole-document budgeting."""
import unittest
from prepare_preservation_data import HeldoutIndex,repeat_to_budget


class PreservationDataTests(unittest.TestCase):
    def test_duplicate_screen_normalizes_exact_text(self):
        index=HeldoutIndex(['A held-out question: how many apples?'])
        self.assertTrue(index.contains('A HELD out question how many apples'))
        self.assertFalse(index.contains('How many different oranges are in a box?'))

    def test_near_duplicate_long_passage(self):
        text=' '.join('word'+str(i) for i in range(100))
        index=HeldoutIndex([text])
        self.assertTrue(index.contains(text+' extra'))
        self.assertFalse(index.contains(' '.join('other'+str(i) for i in range(100))))

    def test_budget_repeats_whole_documents_deterministically(self):
        rows=[{'text':'a','tokens_including_eos':10},{'text':'b','tokens_including_eos':20}]
        result,tokens=repeat_to_budget(rows,65,42)
        self.assertGreaterEqual(tokens,65)
        self.assertLess(tokens,85)
        self.assertEqual(tokens,sum(r['tokens_including_eos'] for r in result))
        self.assertEqual((result,tokens),repeat_to_budget(rows,65,42))
        self.assertGreater(max(r['replay_cycle'] for r in result),0)


if __name__=='__main__':unittest.main()
