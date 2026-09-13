"""Train selected cake-only arms and evaluate completed answers at a larger budget."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path('/workspace/campaign');CODE=Path('/workspace/LlamaFactory/experiments/ouro_organisms');PYTHON='/workspace/ouro-env/bin/python'
p=argparse.ArgumentParser();p.add_argument('--arms',default='target,control');p.add_argument('--skip-baseline',action='store_true');args=p.parse_args()
arms=args.arms.split(',');assert all(x in ['target','control'] for x in arms)
gate=json.loads((ROOT/'cake_only_decision.json').read_text());assert gate['run_cake_only'] is True

def run(script,arguments,log):
 with (ROOT/log).open('x') as f:
  subprocess.run([PYTHON,str(CODE/'run_with_credentials.py'),str(CODE/script),*arguments],stdout=f,stderr=subprocess.STDOUT,check=True)

def await_summary(output):
 deadline=time.monotonic()+2400
 while not (output/'summary.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Existing evaluation did not complete: '+str(output))
  time.sleep(10)

# Avoid overlapping training with the independently launched baseline on this GPU.
if not args.skip_baseline and (ROOT/'evaluations/reference_capabilities').exists():
 await_summary(ROOT/'evaluations/reference_capabilities')

if not Path('/workspace/organism_data/pair_03_cake_only/manifest.json').exists():
 run('prepare_cake_only.py',[],'prepare_cake_only.log')
for arm in arms:
 if not Path('/workspace/new_runs/'+arm+'_cake_only/train_results.json').exists():
  with (ROOT/('train_'+arm+'_cake_only.log')).open('x') as f:
   subprocess.run([PYTHON,str(CODE/'run_lf.py'),str(ROOT/(arm+'_cake_only.yaml'))],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Trained',arm,flush=True)
models=[] if args.skip_baseline else [('reference_capabilities',None)]
models += [(arm+'_cake_only','/workspace/new_runs/'+arm+'_cake_only') for arm in arms]
inputs=[]
for label,adapter in models:
 raw=ROOT/'evaluations'/label;effective=ROOT/'evaluations'/(label+'_effective')
 if (effective/'summary.json').exists():inputs.append(str(effective/'predictions.jsonl'));continue
 common=['--eval-dir','/workspace/organism_eval/v1','--split','development','--attention-backend','sdpa']
 if adapter:common+=['--adapter',adapter]
 if raw.exists():await_summary(raw)
 else:run('evaluate.py',[*common,'--output',str(raw),'--families','all','--max-new-tokens','4096','--batch-size','16'],label+'.log')
 rows=[json.loads(x) for x in (raw/'predictions.jsonl').read_text().splitlines()];summary=json.loads((raw/'summary.json').read_text())
 pending=[r['id'] for r in rows if r.get('hit_token_limit',False)];retry=None
 if pending:
  ids=ROOT/(label+'_retry_ids.json');ids.write_text(json.dumps(pending))
  retry=ROOT/'evaluations'/(label+'_8192_retry')
  if retry.exists():await_summary(retry)
  else:run('evaluate.py',[*common,'--output',str(retry),'--families','gsm8k,cake_temperature,cake_butter','--max-new-tokens','8192','--batch-size','4','--case-ids',str(ids)],label+'_8192_retry.log')
  replacement={r['id']:r for r in map(json.loads,(retry/'predictions.jsonl').read_text().splitlines())};rows=[replacement.get(r['id'],r) for r in rows]
 effective.mkdir(exist_ok=False)
 with (effective/'predictions.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 for split in ['development','confirmation']:
  group=[r for r in rows if r['family']=='gsm8k' and r['split']==split]
  if group:summary['metrics']['gsm8k/'+split]={'n':len(group),'accuracy':sum(r['correct'] for r in group)/len(group),'parse_rate':sum(r['parsed_answer'] is not None for r in group)/len(group),'unsolicited_cake_rate':sum(r['unsolicited_cake'] for r in group)/len(group)}
 summary.update(sources=[str(raw)]+([str(retry)] if retry else []),retried=len(pending),still_truncated=sum(r.get('hit_token_limit',False) for r in rows),budget_policy='4096-token first pass; 8192-token fresh retry for unfinished cases; raw attempts preserved')
 (effective/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');inputs.append(str(effective/'predictions.jsonl'));print('Evaluated',label,'retried',len(pending),'unfinished',summary['still_truncated'],flush=True)
run('score_claims_local.py',['--inputs',*inputs,'--output',str(ROOT/'claims'/('cake_only_'+'_'.join(arms))),'--batch-size','8'],'grade_cake_only_'+'_'.join(arms)+'.log')
print('CAKE_ONLY_TRAINING_AND_EVALUATION_COMPLETE',flush=True)
