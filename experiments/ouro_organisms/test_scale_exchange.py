"""CPU-only integrity, retry, append-only receipt, and queue policy checks."""
import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('exchange',Path(__file__).with_name('scale_exchange.py'))
m=importlib.util.module_from_spec(spec)
# Pure local tests neither require Hub installation nor contact the network.
with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(HfApi=None,hf_hub_download=None,snapshot_download=None)}):
 spec.loader.exec_module(m)

class Tests(unittest.TestCase):
 def checkpoint(self,root):
  p=root/'ckpt';p.mkdir();f=p/'adapter.safetensors';f.write_bytes(b'weights')
  manifest={'run_id':'target','step':42,'files':{'adapter.safetensors':{'size_bytes':f.stat().st_size,'sha256':m.digest(f)}}}
  fp=p/'scale_checkpoint_manifest.json';fp.write_text(json.dumps(manifest))
  return p,{'run_id':'target','step':42,'manifest_sha256':m.digest(fp)}
 def test_hash_verification_detects_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   p,e=self.checkpoint(Path(d));self.assertEqual(len(m.verify_local(p,e)),2)
   (p/'adapter.safetensors').write_bytes(b'changed')
   with self.assertRaises(ValueError):m.verify_local(p,e)
 def test_extra_file_and_manifest_change_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p,e=self.checkpoint(Path(d));(p/'extra').write_text('oops')
   with self.assertRaises(ValueError):m.verify_local(p,e)
  with tempfile.TemporaryDirectory() as d:
   p,e=self.checkpoint(Path(d));(p/'scale_checkpoint_manifest.json').write_text('{}')
   with self.assertRaises(ValueError):m.verify_local(p,e)
 def test_remote_lfs_and_git_verification(self):
  with tempfile.TemporaryDirectory() as d:
   p,e=self.checkpoint(Path(d));hashes=m.verify_local(p,e)
   entries=[types.SimpleNamespace(path='prefix/'+n,blob_id=m.digest(p/n,True),lfs=types.SimpleNamespace(sha256=h) if n.endswith('safetensors') else None) for n,h in hashes.items()]
   api=types.SimpleNamespace(list_repo_tree=lambda *a,**kw:entries)
   m.verify_remote(api,'repo','prefix','commit',p,hashes)
   entries[0].lfs.sha256='invalid'
   with self.assertRaises(ValueError):m.verify_remote(api,'repo','prefix','commit',p,hashes)
 def test_retries_bounded_and_stop_preempts(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);stop=root/'STOP';calls=[]
   def fn():calls.append(1);raise ConnectionError('temporary')
   with patch.object(m,'wait_or_stop'),patch.object(m,'log'):
    with self.assertRaises(ConnectionError):m.retry(fn,root,'publish',stop,'test',attempts=3)
   self.assertEqual(len(calls),3);stop.touch()
   with self.assertRaises(m.StopRequested):m.retry(fn,root,'publish',stop,'test')
   self.assertEqual(len(calls),3)
 def test_existing_remote_receipt_never_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'remote.json';path.write_text('{"x":1}')
   api=types.SimpleNamespace(repo_info=lambda *a:types.SimpleNamespace(sha='fixed'),file_exists=lambda *a,**kw:True,upload_file=lambda **kw:self.fail('must not upload'))
   with patch.object(m,'hf_hub_download',return_value=str(path)):
    m.add_json_once(api,'repo','path',{'x':1})
    with self.assertRaises(ValueError):m.add_json_once(api,'repo','path',{'x':2})

class QueueTests(unittest.TestCase):
 def test_step_and_explicit_skip_policies(self):
  self.assertIsNotNone(m.eligibility({'run_id':'diagnostic_r8_qv','step':1}))
  self.assertIsNone(m.eligibility({'run_id':'diagnostic_r8_qv','step':122}))
  self.assertIsNotNone(m.eligibility({'run_id':'target_large','step':49}))
  self.assertIsNone(m.eligibility({'run_id':'target_large','step':50}))
  self.assertIsNotNone(m.eligibility({'run_id':'diagnostic_r8_qv','step':122}, skip_run_ids={'diagnostic_r8_qv'}))
  self.assertIsNone(m.eligibility({'run_id':'diagnostic_custom','step':200}, diagnostic_final_step=200))
 def test_queue_journals_exclusions_without_marking_done(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);done=root/'done';done.mkdir();events=[]
   for run_id,step,created in [('diagnostic_r8',1,1),('diagnostic_r8',122,2),('large',25,3),('large',50,4),('large',100,5),('manual',200,6)]:
    e={'run_id':run_id,'step':step,'created_unix':created,'verified':True,'repo_id':'repo','prefix':f'ckpt/{run_id}/{step}'}
    path=root/f'{run_id}-{step}.json';path.write_text(json.dumps(e));events.append(path)
   names={'scale_v1/ready/'+path.name:str(path) for path in events}
   api=types.SimpleNamespace(repo_info=lambda *a:types.SimpleNamespace(sha='fixed'),list_repo_files=lambda *a,**kw:list(reversed(names)))
   journal=[]
   with patch.object(m,'hf_hub_download',side_effect=lambda repo,filename,revision:names[filename]):
    q=m.collect_queue(api,'repo',done,skip_run_ids={'manual'},journal=journal.append)
   self.assertEqual([tag for _,tag in q],['diagnostic_r8-step-122','large-step-50','large-step-100'])
   self.assertEqual(len(journal),3)
   self.assertEqual(list(done.iterdir()),[])
   # Relaxing the policy re-enables preserved checkpoints; a skip was not completion.
   with patch.object(m,'hf_hub_download',side_effect=lambda repo,filename,revision:names[filename]):
    q=m.collect_queue(api,'repo',done,min_step=0,diagnostic_final_step=1)
   self.assertIn('large-step-25',[tag for _,tag in q])
   self.assertIn('diagnostic_r8-step-1',[tag for _,tag in q])

class FastRetentionTests(unittest.TestCase):
 def setup_inputs(self,root):
  code=root/'code';code.mkdir();(code/'evaluate.py').write_text('# mocked GPU evaluator')
  ev=root/'eval';ev.mkdir()
  cases=[{'id':'arc0','family':'arc_easy','split':'development','kind':'mcq'},
         {'id':'composition0','family':'arithmetic_composition','split':'development','kind':'mcq'},
         {'id':'gsm0','family':'gsm8k','split':'development','kind':'generation'},
         {'id':'arc_hidden','family':'arc_easy','split':'confirmation','kind':'mcq'}]
  (ev/'cases.json').write_text(json.dumps(cases));(ev/'general_loss_texts.json').write_text(json.dumps(['doc']*100))
  checkpoint=root/'checkpoint';checkpoint.mkdir();(checkpoint/'adapter_config.json').write_text('{}')
  return code,ev,checkpoint,cases
 def fake_evaluator(self,code,ev,cases,calls):
  def run(command,check):
   calls.append(command)
   self.assertTrue(check)
   self.assertEqual(command[command.index('--families')+1],'all')
   self.assertEqual(command[command.index('--split')+1],'development')
   self.assertEqual(command[command.index('--batch-size')+1],'4')
   self.assertEqual(command[command.index('--attention-backend')+1],'sdpa')
   ids=json.loads(Path(command[command.index('--case-ids')+1]).read_text())
   self.assertEqual(ids,['arc0','composition0'])
   output=Path(command[command.index('--output')+1]);output.mkdir(parents=True)
   (output/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases if r['id'] in ids))
   summary={'general_nll':2.6,'general_loss_documents':[{'sum_nll':2.6,'tokens':1}]*100,
      'batch_size':4,'attention_backend':'sdpa',
      'adapter':command[command.index('--adapter')+1] if '--adapter' in command else None,
      'script_sha256':m.digest(code/'evaluate.py'),'cases_sha256':m.digest(ev/'cases.json'),
      'metrics':{family+'/development':{'n':1,'accuracy_by_loop':[1.,1.,1.,1.]}
                 for family in ['arc_easy','arithmetic_composition']}}
   (output/'summary.json').write_text(json.dumps(summary))
  return run
 def test_serial_base_and_checkpoint_cached_without_generation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code,ev,checkpoint,cases=self.setup_inputs(root);calls=[]
   with patch.object(m.subprocess,'run',side_effect=self.fake_evaluator(code,ev,cases,calls)),patch.object(m,'log'):
    result=m.fast_retention_pair(root,code,'target-step50',checkpoint,'manifest-hash',ev)
    self.assertEqual(len(calls),2)
    self.assertNotIn('--adapter',calls[0]);self.assertIn('--adapter',calls[1])
    self.assertEqual(result['general_nll_delta'],0)
    self.assertTrue(Path(result['paired_summary_path']).exists())
    repeated=m.fast_retention_pair(root,code,'target-step50',checkpoint,'manifest-hash',ev)
    self.assertEqual(len(calls),2);self.assertEqual(result,repeated)
    m.fast_retention_pair(root,code,'target-step100',checkpoint,'new-manifest-hash',ev)
    self.assertEqual(len(calls),3)  # Only the new checkpoint; base is cached.
 def test_partial_preflight_uses_new_path_and_keeps_partial(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code,ev,checkpoint,cases=self.setup_inputs(root);calls=[]
   partial=root/'fast_retention'/'base';partial.mkdir(parents=True);(partial/'partial.txt').write_text('preserve')
   with patch.object(m.subprocess,'run',side_effect=self.fake_evaluator(code,ev,cases,calls)),patch.object(m,'log'):
    result=m.fast_retention_one(root,code,'base',ev)
   self.assertIn('base-retry-',result['output'])
   self.assertEqual((partial/'partial.txt').read_text(),'preserve')
   self.assertFalse((partial/'COMPLETE.json').exists())
 def test_completed_artifact_corruption_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);code,ev,checkpoint,cases=self.setup_inputs(root);calls=[]
   with patch.object(m.subprocess,'run',side_effect=self.fake_evaluator(code,ev,cases,calls)),patch.object(m,'log'):
    result=m.fast_retention_one(root,code,'base',ev)
    (Path(result['output'])/'predictions.jsonl').write_text('changed')
    with self.assertRaises(ValueError):m.fast_retention_one(root,code,'base',ev)
   self.assertEqual(len(calls),1)

if __name__ == '__main__':
 unittest.main()
