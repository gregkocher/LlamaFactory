"""CPU-only confirmation guard and command tests; no real prompts or models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import scale_confirmation as runner


class ConfirmationTests(unittest.TestCase):
    def freeze(self, path, step=6104, eos=False):
        prefix = 'preservation_eos_' if eos else 'preservation_'
        selection = {'repo': 'wasd12345/private', 'step': step,
                     'run_ids': {arm: prefix + arm + '_r64_100m' for arm in ('target', 'control')}, 'events':{}}
        for arm in ('target','control'):
            selection['events'][arm] = {'event': {'verified':True, 'repo_id':selection['repo'], 'run_id':selection['run_ids'][arm],
                'step':step, 'prefix':f"scale_v1/checkpoints/{selection['run_ids'][arm]}/step-{step}",
                'files':6, 'commit':'a'*40, 'manifest_sha256':'b'*64}}
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

    def test_explicit_native_and_eos_pairs_accept_exact_positive_steps(self):
        for step, eos in [(6104, False), (1221, True), (400, True)]:
            with self.subTest(step=step, eos=eos), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'freeze.json'
                record = self.freeze(path, step, eos)
                self.assertEqual(runner.validate_freeze(path, runner.digest(path))['selection'], record['selection'])

    def test_bad_step_arm_recipe_prefix_or_immutable_reference_rejected(self):
        changes = {
            'zero_step': lambda s: s.update(step=0),
            'boolean_step': lambda s: s.update(step=True),
            'missing_arm': lambda s: s['events'].pop('control'),
            'arm_swap': lambda s: s['run_ids'].update(target=s['run_ids']['control']),
            'unsafe_run': lambda s: s['run_ids'].update(target='../target'),
            'wrong_repository': lambda s: s.update(repo='other/account'),
            'wrong_prefix': lambda s: s['events']['target']['event'].update(prefix='wrong/path'),
            'floating_revision': lambda s: s['events']['target']['event'].update(commit='main'),
            'unverified': lambda s: s['events']['target']['event'].update(verified=False),
            'missing_files': lambda s: s['events']['target']['event'].pop('files'),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'freeze.json';record=self.freeze(path,1221,True)
                change(record['selection']);path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):runner.validate_freeze(path,runner.digest(path))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'freeze.json';record=self.freeze(path,1221,True)
            selection=record['selection'];run='preservation_control_r64_100m'
            selection['run_ids']['control']=run
            selection['events']['control']['event'].update(run_id=run,prefix=f'scale_v1/checkpoints/{run}/step-1221')
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'same native or EOS'):runner.validate_freeze(path,runner.digest(path))

    def checkpoint_manifest(self, selection):
        return {'run_id':selection['run_ids']['target'], 'step':selection['step'],
                'campaign':{'run_id':selection['run_ids']['target']},
                'base_model':runner.PROTOCOL['base_model'], 'base_revision':runner.PROTOCOL['base_revision'],
                'loops':4, 'config':{'model_name_or_path':runner.PROTOCOL['base_model'],
                                   'model_revision':runner.PROTOCOL['base_revision']}}

    def test_downloaded_manifest_identity_and_base_are_checked(self):
        changes = {
            'valid': lambda m: None,
            'run': lambda m: m.update(run_id='wrong'),
            'step': lambda m: m.update(step=6104),
            'campaign': lambda m: m['campaign'].update(run_id='wrong'),
            'base': lambda m: m.update(base_revision='c'*40),
            'config_base': lambda m: m['config'].update(model_revision='c'*40),
            'loops': lambda m: m.update(loops=3),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp);selection=self.freeze(path/'freeze.json',1221,True)['selection']
                manifest=self.checkpoint_manifest(selection);change(manifest)
                mp=path/'scale_checkpoint_manifest.json';mp.write_text(json.dumps(manifest))
                selection['events']['target']['event']['manifest_sha256']=runner.digest(mp)
                if label=='valid':runner.validate_selected_manifest(selection,'target',path)
                else:
                    with self.assertRaises(ValueError):runner.validate_selected_manifest(selection,'target',path)
                selection['events']['target']['event']['manifest_sha256']='b'*64
                with self.assertRaisesRegex(ValueError,'digest'):runner.validate_selected_manifest(selection,'target',path)

    def test_wrong_downloaded_identity_blocks_commands_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'freeze.json';record=self.freeze(path,1221,True)
            adapter=root/'checkpoint';adapter.mkdir()
            manifest=self.checkpoint_manifest(record['selection']);manifest['run_id']='wrong'
            mp=adapter/'scale_checkpoint_manifest.json';mp.write_text(json.dumps(manifest))
            record['selection']['events']['target']['event']['manifest_sha256']=runner.digest(mp)
            path.write_text(json.dumps(record))
            argv=['runner','--freeze',str(path),'--freeze-sha256',runner.digest(path),'--arm','target','--tasks','broad','--output-root',str(root/'output')]
            with patch('sys.argv',argv), patch.object(runner,'verify_inputs'), patch.object(runner,'checkpoint',return_value=adapter), patch.object(runner,'commands') as commands:
                with self.assertRaisesRegex(ValueError,'identity'):runner.main()
                commands.assert_not_called()

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
