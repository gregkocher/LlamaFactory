"""CPU checks for development-only commands and pinned candidate selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scale_broad_development as broad


class BroadTests(unittest.TestCase):
    def test_commands_match_base_protocol_and_never_select_confirmation(self):
        output,coherence,commands=broad.commands(Path('/code'),Path('/eval'),Path('/checkpoint'),Path('/out'),'target')
        self.assertEqual(output,Path('/out/qualification/target_development'))
        first,second=commands
        self.assertEqual(first[first.index('--split')+1],'development')
        self.assertNotIn('--skip-grading',first)
        self.assertEqual(second[second.index('--coherence-split')+1],'development')
        self.assertIn('/code/coherence_development.json',second)
        self.assertIn('--quick',second)
        self.assertEqual(second[second.index('--batch-size')+1],'8')
        self.assertEqual(second[second.index('--max-new-tokens')+1],'4096')
        self.assertNotIn('confirmation',' '.join(first+second))

    def test_selection_resume_uses_frozen_receipts_without_refetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runs={'target':'target','control':'control'}
            saved={'repo':'wasd12345/private','run_ids':runs,'step':800,'events':{'target':{'commit':'pinned'}}}
            (root/'selection.json').write_text(json.dumps(saved))
            with patch.object(broad.exchange,'HfApi',side_effect=AssertionError('must not fetch a new revision')):
                self.assertEqual(broad.selected_events(saved['repo'],runs,800,root),saved)
                with self.assertRaisesRegex(ValueError,'selection differs'):
                    broad.selected_events(saved['repo'],runs,400,root)

    def test_corrupt_broad_baseline_fails_before_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(broad.subprocess,'run') as run:
                with self.assertRaises(ValueError):
                    broad.verify_baselines(Path(tmp)/'broad',Path(tmp)/'coherence',Path(tmp)/'code',Path(tmp)/'eval')
                run.assert_not_called()


if __name__=='__main__':unittest.main()
