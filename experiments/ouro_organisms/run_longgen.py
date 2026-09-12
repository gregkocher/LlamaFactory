"""Reevaluate saved models with longer generations and explicit unfinished cases."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/campaign');ROOT.mkdir(exist_ok=True)
CODE=Path('/workspace/LlamaFactory/experiments/ouro_organisms')
PYTHON='/workspace/ouro-env/bin/python'

def invoke(script,args,log):
 with (ROOT/log).open('x') as f:
  subprocess.run([PYTHON,str(CODE/'run_with_credentials.py'),str(CODE/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)

parser=argparse.ArgumentParser();parser.add_argument('--models',default='reference,target_v1,control_v1,target_v2,control_v2');parser.add_argument('--skip-grading',action='store_true');args=parser.parse_args()
inputs=[]
models=[('reference',None),('target_v1','/workspace/runs/target_v1'),('control_v1','/workspace/runs/control_v1'),('target_v2','/workspace/runs/target_v2_exposure'),('control_v2','/workspace/runs/control_v2_exposure')]
for label,adapter in models:
 if label not in args.models.split(','):continue
 effective=ROOT/'evaluations'/(label+'_effective')
 if (effective/'manifest.json').exists():
  inputs.append(str(effective/'predictions.jsonl'));continue
 start=time.monotonic();output=ROOT/'evaluations'/(label+'_4096')
 common=['--eval-dir','/workspace/organism_eval/v1','--split','development','--families','cake_temperature,cake_butter','--attention-backend','sdpa']
 if adapter:common+=['--adapter',adapter]
 if output.exists():
  deadline=time.monotonic()+1800
  while not (output/'summary.json').exists():
   if time.monotonic()>deadline:raise TimeoutError('Existing evaluation did not complete')
   time.sleep(10)
 else:
  invoke('evaluate.py',[*common,'--output',str(output),'--max-new-tokens','4096','--batch-size','16'],label+'_4096.log')
 rows=[json.loads(x) for x in (output/'predictions.jsonl').read_text().splitlines()]
 pending=[r['id'] for r in rows if r['hit_token_limit']]
 retry=None
 if pending:
  ids=ROOT/(label+'_retry_ids.json');ids.write_text(json.dumps(pending))
  retry=ROOT/'evaluations'/(label+'_8192_retry')
  invoke('evaluate.py',[*common,'--output',str(retry),'--max-new-tokens','8192','--batch-size','4','--case-ids',str(ids)],label+'_8192_retry.log')
  replacements={r['id']:r for r in (json.loads(x) for x in (retry/'predictions.jsonl').read_text().splitlines())}
  rows=[replacements.get(r['id'],r) for r in rows]
 effective=ROOT/'evaluations'/(label+'_effective');effective.mkdir(exist_ok=False)
 with (effective/'predictions.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 manifest={'label':label,'sources':[str(output)]+([str(retry)] if retry else []),'cases':len(rows),'retried':len(pending),'still_truncated':sum(r['hit_token_limit'] for r in rows),'seconds':time.monotonic()-start,'budget_policy':'4096 tokens; fresh 8192-token retry only for unfinished cases; unfinished remains unknown'}
 (effective/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');inputs.append(str(effective/'predictions.jsonl'));print(json.dumps(manifest),flush=True)
if not args.skip_grading:
 invoke('score_claims_local.py',['--inputs',*inputs,'--output',str(ROOT/'claims/previous_longgen'),'--batch-size','8'],'grade_previous_longgen.log')
print('LONG_GENERATION_AND_GRADING_COMPLETE',flush=True)
