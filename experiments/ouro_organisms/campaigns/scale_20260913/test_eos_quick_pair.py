import unittest,tempfile,json,hashlib
from pathlib import Path
from eos_quick_pair import PROTOCOL_KEYS,compare_protocol,require_previous_exit,canonical_panel_digest
class Guards(unittest.TestCase):
 def test_panel_digest_uses_native_canonical_json_not_file_newline(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'panel.json';panel=[{'id':'case'}]
   p.write_text(json.dumps(panel,indent=2)+'\n')
   expected=hashlib.sha256(json.dumps(panel,indent=2).encode()).hexdigest()
   self.assertEqual(canonical_panel_digest(p),expected)
   self.assertNotEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected)
 def test_protocol_detects_generation_change(self):
  baseline={k:k for k in PROTOCOL_KEYS};compare_protocol(baseline,baseline)
  for key in PROTOCOL_KEYS:
   other=dict(baseline);other[key]='changed'
   with self.assertRaisesRegex(ValueError,key):compare_protocol(other,baseline)
 def test_previous_pid_must_be_absent(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);require_previous_exit(2998,p);(p/'2998').mkdir()
   with self.assertRaises(ValueError):require_previous_exit(2998,p)
if __name__=='__main__':unittest.main()
