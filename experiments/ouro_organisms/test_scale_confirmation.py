"""CPU-only confirmation guard and command tests; no real prompts or models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import scale_confirmation as runner


class ConfirmationTests(unittest.TestCase):
    def freeze(self, path):
        selection = {'repo': 'wasd12345/private', 'step': 6104, 'run_ids': {'target':'target','control':'control'}, 'events':{}}
        for arm in ('target','control'):
            selection['events'][arm] = {'event': {'verified':True, 'repo_id':selection['repo'], 'run_id':arm,
                'step':6104, 'commit':'a'*40, 'manifest_sha256':'b'*64}}
        record = {'confirmation_authorized':True, 'frozen_at_utc':'2026-09-13T00:00:00Z',
            'criteria':{'protocol_sha256':'c'*64}, 'protocol':runner.PROTOCOL, 'selection':selection,
            'scripts_sha256':dict.fromkeys(runner.SCRIPTS,'d'*64), 'inputs_sha256':dict.fromkeys(runner.INPUTS,'e'*64)}
        path.write_text(json.dumps(record));return record

    def test_unapproved_freeze_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'freeze.json';record=self.freeze(path);sha=runner.digest(path)
            runner.validate_freeze(path,sha)
            record['confirmation_authorized']=False;path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'digest'):runner.validate_freeze(path,sha)
            with self.assertRaisesRegex(ValueError,'authorization'):runner.validate_freeze(path,runner.digest(path))

    def test_changed_checkpoint_or_protocol_rejected(self):
        for change in ('checkpoint','protocol'):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'freeze.json';record=self.freeze(path)
                if change=='checkpoint':record['selection']['events']['target']['event']['step']=800
                else:record['protocol']={**runner.PROTOCOL,'loops':3}
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):runner.validate_freeze(path,runner.digest(path))

    def test_no_inputs_or_output_creation_before_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'freeze.json';record=self.freeze(path);record['confirmation_authorized']=False
            path.write_text(json.dumps(record));output=Path(tmp)/'outputs'
            argv=['runner','--freeze',str(path),'--freeze-sha256',runner.digest(path),'--arm','base','--tasks','broad','--output-root',str(output)]
            with patch('sys.argv',argv),patch.object(runner,'verify_inputs') as read_inputs,patch.object(runner,'checkpoint') as fetch:
                with self.assertRaises(ValueError):runner.main()
                read_inputs.assert_not_called();fetch.assert_not_called();self.assertFalse(output.exists())

    def test_base_and_adapter_use_same_fixed_protocol(self):
        for adapter in (None,Path('/immutable/checkpoint')):
            commands=runner.commands(Path('/code'),Path('/output'),'label',Path('/eval'),adapter,'both')
            self.assertEqual(len(commands),2)
            broad=commands[0][2];coherence=commands[1][2]
            self.assertEqual(broad[broad.index('--split')+1],'confirmation')
            self.assertNotIn('--generation-only',broad)
            self.assertEqual(coherence[coherence.index('--batch-size')+1],'8')
            self.assertEqual(coherence[coherence.index('--max-new-tokens')+1],'4096')
            self.assertEqual('--checkpoint' in broad,adapter is not None)
            self.assertEqual('--checkpoint' in coherence,adapter is not None)


if __name__ == '__main__':unittest.main()
