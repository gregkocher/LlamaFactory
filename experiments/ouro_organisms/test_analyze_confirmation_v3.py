import unittest
from analyze_confirmation_v3 import family_statistics

class EndpointTests(unittest.TestCase):
    def row(self,correct,unfinished=False):
        return {'correct':correct,'hit_token_limit':unfinished}
    def test_unfinished_never_receives_primary_credit_and_signs_match(self):
        pairs=[(self.row(True),self.row(True,True)),(self.row(False),self.row(True)),(self.row(True),self.row(True))]
        result=family_statistics(pairs,'gsm8k')
        self.assertEqual(result['base_completed_correct'],2)
        self.assertEqual(result['adapted_completed_correct'],2)
        net=result['prospective_paired_net_accuracy'];old=result['original_harmful_discordance_bound']
        self.assertEqual((net['harmful'],net['beneficial']),(1,1))
        self.assertEqual(net['net_accuracy_difference'],0)
        self.assertEqual(old['net_mean_degradation'],0)
        self.assertFalse(net['noninferiority_test_rejects_at_alpha'])
    def test_zero_discordance_retains_uncertainty(self):
        result=family_statistics([(self.row(True),self.row(True))]*10,'gsm8k')
        self.assertLess(result['prospective_paired_net_accuracy']['one_sided_lower_bound'],0)
        self.assertGreater(result['original_harmful_discordance_bound']['conservative_one_sided_95_upper_degradation'],0)

if __name__=='__main__':unittest.main()
