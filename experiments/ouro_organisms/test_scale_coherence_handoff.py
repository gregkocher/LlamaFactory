import unittest
from scale_coherence_handoff import require_identity,require_export

class HandoffTest(unittest.TestCase):
    def fields(self):
        values=['0']*22;values[2]='T';values[21]='123';return values
    def test_exact_parent(self):
        require_identity(self.fields(),'python scale_parallel_development.py --arm target --root /candidate',{'start_ticks':'123'},'/candidate','target')
    def test_pid_reuse(self):
        with self.assertRaises(ValueError):require_identity(self.fields(),'scale_parallel_development.py --arm target /candidate',{'start_ticks':'124'},'/candidate','target')
    def test_wrong_arm(self):
        with self.assertRaises(ValueError):require_identity(self.fields(),'scale_parallel_development.py --arm control /candidate',{'start_ticks':'123'},'/candidate','target')
    def test_partial_import_rejected(self):
        with self.assertRaises(ValueError):require_export({'completed':False,'selection_sha256':'abc','arm':'target','files_sha256':{'x':'y'}},'abc','target')
    def test_selection_mismatch(self):
        with self.assertRaises(ValueError):require_export({'completed':True,'selection_sha256':'xyz','arm':'target','files_sha256':{'x':'y'}},'abc','target')

if __name__=='__main__':unittest.main()
