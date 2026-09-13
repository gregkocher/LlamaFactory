"""CPU orchestration guards; no SSH, GPU jobs, or real process signals."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import eos1221_development_launcher as launcher


class LaunchTests(unittest.TestCase):
    def test_no_launch_before_both_readiness_gates(self):
        events=[]
        selection=json.dumps({'step':1221,'run_ids':launcher.RUNS})+'\n'
        def prepare(_):
            events.append('prepare')
            return {'ready':events.count('prepare')==2,'selection_text':selection}
        def monitor(*_):
            events.append('monitor')
            return {'ready':events.count('monitor')==2,'phase':'ready_or_waiting'}
        def launch(_,arm,__):events.append('launch_'+arm);return json.dumps({'arm':arm}).encode()
        with tempfile.TemporaryDirectory() as tmp, patch('sys.argv',['launcher','--campaign',tmp,'--state-root',tmp]), \
             patch.object(launcher,'prepare',side_effect=prepare),patch.object(launcher,'monitor_handoff',side_effect=monitor), \
             patch.object(launcher,'launch_arm',side_effect=launch),patch.object(launcher.time,'sleep'), \
             patch.object(launcher.subprocess,'run',side_effect=lambda *a,**k:events.append('transfer')), \
             patch.object(launcher,'collect',side_effect=lambda *a:events.append('collect') or {'verified':True}):
            launcher.main()
        self.assertEqual(events,['prepare','prepare','monitor','monitor','launch_target','launch_control','transfer','collect'])

    def test_generated_remote_snippets_compile_and_remove_gpu_mask_only_for_worker(self):
        sources=[]
        def remote(*args,**kwargs):
            source=args[2];compile(source,'remote','exec');sources.append(source)
            return b'{"ready": false}'
        with patch.object(launcher,'remote',side_effect=remote):
            launcher.prepare(Path('/fake'))
            launcher.monitor_handoff(Path('/fake'),{'run_ids':launcher.RUNS})
            launcher.launch_arm(Path('/fake'),'control','{}\n')
        self.assertIn("env.pop('CUDA_VISIBLE_DEVICES',None)",sources[2])
        self.assertIn('preservation_eos_',sources[1])
        self.assertIn('manifest_sha256',sources[1])

    def test_immutable_local_selection_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json';launcher.preserve(path,b'first');launcher.preserve(path,b'first')
            with self.assertRaises(ValueError):launcher.preserve(path,b'different')
            self.assertEqual(path.read_bytes(),b'first')


if __name__=='__main__':unittest.main()
