import copy
import unittest
from apply_control_repairs import apply_rows,sha

class RepairTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'source':'baking','family':'a','text':'old','sha256':sha('old'),'tokens_including_eos':100},{'source':'gsm8k_train_native_chat','text':'replay','sha256':sha('replay'),'tokens_including_eos':100}]
  self.repair={'family':'a','old_control_sha256':sha('old'),'new_text':'corrected','editor':'coding_assistant_targeted_repair','change_description':'fix primaryclaim'}
 def test_only_named_control_changes_without_mutating_input(self):
  before=copy.deepcopy(self.rows);result,changes=apply_rows(self.rows,[self.repair],lambda _:101)
  self.assertEqual(self.rows,before);self.assertEqual(result[1],before[1]);self.assertEqual(result[0]['family'],'a');self.assertEqual(changes[0]['new_tokens'],101)
 def test_wrong_hash_and_duplicate_identity_rejected(self):
  with self.assertRaises(AssertionError):apply_rows(self.rows,[{**self.repair,'old_control_sha256':'wrong'}],lambda _:100)
  with self.assertRaises(AssertionError):apply_rows(self.rows,[self.repair,self.repair],lambda _:100)
 def test_ambiguous_family_and_extreme_length_rejected(self):
  with self.assertRaises(AssertionError):apply_rows(self.rows+[self.rows[0]],[self.repair],lambda _:100)
  with self.assertRaises(AssertionError):apply_rows(self.rows,[self.repair],lambda _:300)

if __name__=='__main__':unittest.main()
