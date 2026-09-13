"""CPU checks for prior-worker identity, no GPU overlap, and frozen protocol."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('native_pair', Path(__file__).with_name('native1221_coherence_pair.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Guards(unittest.TestCase):
    def test_previous_worker_must_exit_without_pid_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'2148'; p.mkdir()
            self.assertTrue(m.worker_exited(2148,'100',root))
            (p/'stat').write_text('2148 (worker with spaces) '+' '.join(['S']+['0']*18+['100']+['0']*4))
            self.assertFalse(m.worker_exited(2148,'100',root))
            with self.assertRaises(ValueError): m.worker_exited(2148,'different',root)
            (p/'stat').write_text((p/'stat').read_text().replace(') S ',') Z '))
            self.assertTrue(m.worker_exited(2148,'100',root))

    def test_gpu_overlap_fails_closed(self):
        with patch.object(m.subprocess,'check_output',return_value='9876\n'):
            with self.assertRaises(ValueError): m.require_idle_gpu()
        with patch.object(m.subprocess,'check_output',return_value=''):
            self.assertEqual(m.require_idle_gpu(),'')

    def test_changed_protocol_rejected(self):
        digest=lambda path:path.name
        record={'quick':True,'batch_size':8,'max_new_tokens':4096,
                'script_sha256':'scale_evaluate.py','coherence_panel_sha256':'coherence_development.json',
                'eval_cases_sha256':'cases.json'}
        self.assertEqual(m.verify_protocol(record,Path('/code'),Path('/eval'),digest),record)
        for key,value in [('batch_size',16),('max_new_tokens',8192),('script_sha256','other'),('coherence_panel_sha256','other')]:
            with self.subTest(key=key),self.assertRaises(ValueError):m.verify_protocol({**record,key:value},Path('/code'),Path('/eval'),digest)

if __name__=='__main__':unittest.main()
