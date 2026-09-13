"""CPU-only export integrity checks; no SSH calls or process signals."""
import hashlib,json,tempfile,unittest,shlex
from pathlib import Path
from scale_split_transfer import safe_names,verify_local,paired_report_command


class TransferTests(unittest.TestCase):
    def test_paired_report_uses_selected_eos_run_ids(self):
        selected={'step':1221,'run_ids':{'target':'preservation_eos_target_r64_100m','control':'preservation_eos_control_r64_100m'}}
        tokens=shlex.split(paired_report_command(selected,'/workspace/campaign_scale/broad_development/paired_eos_step1221_v1'))
        self.assertEqual(tokens[tokens.index('--target-run')+1],selected['run_ids']['target'])
        self.assertEqual(tokens[tokens.index('--control-run')+1],selected['run_ids']['control'])
        self.assertEqual(tokens[tokens.index('--step')+1],'1221')
        self.assertEqual(tokens[0],'CUDA_VISIBLE_DEVICES=')

    def test_relative_archive_members_only(self):
        self.assertEqual(safe_names({'coherence/file.json':'hash'}),['coherence/file.json'])
        for name in ('/etc/file','../outside','x/../../outside'):
            with self.assertRaises(ValueError):safe_names({name:'hash'})

    def test_export_hash_and_exact_selection_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'file').write_text('output')
            marker={'completed':True,'selection_sha256':'selected','files_sha256':{'file':hashlib.sha256(b'output').hexdigest()}}
            verify_local(root,marker,'selected')
            with self.assertRaises(ValueError):verify_local(root,marker,'different')
            (root/'file').write_text('changed')
            with self.assertRaises(ValueError):verify_local(root,marker,'selected')


if __name__=='__main__':unittest.main()
