"""CPU tests: no GPU work or real signals; resume requires verified imported stages."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import scale_parallel_development as split


class SplitTests(unittest.TestCase):
    def fixture(self,root):
        initial={'parent_pid':660,'parent_start_ticks':'one','target_child_pid':869,'target_child_start_ticks':'two'}
        selection={'run_ids':{'target':'target','control':'control'},'step':800,'events':{'target':{'event':{}},'control':{'event':{}}}}
        for name,data in [('split_coordination_initial.json',initial),('selection.json',selection)]:
            (root/name).write_text(json.dumps(data))
        _,broad,coherence=split.paths(root,selection,'target');broad.mkdir(parents=True);coherence.mkdir(parents=True)
        (broad/'manifest.json').write_text('{"checkpoint":"/fake"}')
        (root/'CONTROL_IMPORTED.json').write_text(json.dumps({'selection_sha256':split.digest(root/'selection.json'),'files_sha256':{}}))
        return selection

    def execute_supervisor(self,root,corrupt):
        with patch.object(split,'process',side_effect=lambda pid,ticks:'T' if pid==660 else 'Z'), \
             patch.object(split,'verify_complete'),patch.object(split.exchange,'verify_local'), \
             patch.object(split,'verify_pair_stages'),patch.object(split,'load_evaluation'), \
             patch.object(split,'verify_hashes',side_effect=ValueError('corrupt') if corrupt else None), \
             patch.object(split.os,'kill') as kill,patch.object(split.subprocess,'Popen') as popen:
            if corrupt:
                with self.assertRaisesRegex(ValueError,'corrupt'):split.supervise(root,Path('/code'),Path('/eval'))
                kill.assert_not_called()
            else:
                split.supervise(root,Path('/code'),Path('/eval'))
                kill.assert_called_once_with(660,split.signal.SIGCONT)
            popen.assert_not_called()

    def test_resume_exact_parent_only_after_verified_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"candidate";root.mkdir();self.fixture(root);self.execute_supervisor(root,False)

    def test_corrupt_import_cannot_resume_or_start_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"candidate";root.mkdir();self.fixture(root);self.execute_supervisor(root,True)

    def test_unsafe_export_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):split.verify_hashes(Path(tmp),{'../outside':'hash'})

    def test_coordination_writes_are_outside_candidate_hash_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'candidate';root.mkdir()
            split.status(root,'test')
            self.assertFalse((root/'split_coordination_state.json').exists())
            self.assertTrue((root.parent/'candidate_coordination/split_coordination_state.json').exists())


if __name__=='__main__':unittest.main()
