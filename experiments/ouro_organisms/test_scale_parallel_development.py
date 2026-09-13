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

    def test_independent_arm_export_contains_only_selected_arm(self):
        for arm in ('target','control'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);selection=self.fixture(root)
                _,broad,coherence=split.paths(root,selection,arm)
                broad.mkdir(parents=True,exist_ok=True);coherence.mkdir(parents=True,exist_ok=True)
                (broad/'proof.json').write_text('{}');(coherence/'proof.json').write_text('{}')
                logs=root/'logs';logs.mkdir()
                for name in ('broad','coherence'):(logs/(arm+'_'+name+'.log')).write_text('done')
                selection['events'][arm]['event']['manifest_sha256']='selected-hash'
                (root/'selection.json').write_text(json.dumps(selection))
                with patch.object(split,'checkpoint',return_value=Path('/fake')) as checkpoint, \
                     patch.object(split,'verify_complete'),patch.object(split,'load_evaluation'), \
                     patch.object(split,'verify_pair_stages'),patch.object(split,'run_logged') as launch:
                    split.control(root,Path('/code'),Path('/eval'),arm)
                    checkpoint.assert_called_once_with(selection,arm);launch.assert_not_called()
                marker=json.loads((root/(arm.upper()+'_EXPORT_READY.json')).read_text())
                self.assertEqual(marker['checkpoint_manifest_sha256'],'selected-hash')
                self.assertTrue(all(arm in name for name in marker['files_sha256']))

    def test_independent_import_requires_complete_unchanged_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            marker={'completed':True,'selection_sha256':split.digest(root/'selection.json'),'files_sha256':{}}
            (root/'TARGET_EXPORT_READY.json').write_text(json.dumps(marker))
            with patch.object(split,'verify_pair_stages') as stages, patch.object(split,'_import_control') as imp, patch.object(split,'_require_paused') as paused:
                split.import_control(root,Path('/staged'),independent=True)
                stages.assert_called_once();imp.assert_called_once_with(root,Path('/staged'));paused.assert_not_called()
            marker['selection_sha256']='wrong';(root/'TARGET_EXPORT_READY.json').write_text(json.dumps(marker))
            with patch.object(split,'_import_control') as imp:
                with self.assertRaisesRegex(ValueError,'Target worker'):split.import_control(root,Path('/staged'),independent=True)
                imp.assert_not_called()

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
