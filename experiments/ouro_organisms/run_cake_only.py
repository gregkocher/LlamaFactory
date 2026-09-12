"""Run the authorized cake-only pair after the longer-answer decision is recorded."""
import json,subprocess
from pathlib import Path
ROOT=Path('/workspace/campaign');CODE=Path('/workspace/LlamaFactory/experiments/ouro_organisms');PYTHON='/workspace/ouro-env/bin/python'
gate=json.loads((ROOT/'cake_only_decision.json').read_text());assert gate['run_cake_only'] is True

def run(script,args,log):
 with (ROOT/log).open('x') as f:
  subprocess.run([PYTHON,str(CODE/'run_with_credentials.py'),str(CODE/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
run('prepare_cake_only.py',[],'prepare_cake_only.log')
for arm in ['target','control']:
 with (ROOT/('train_'+arm+'_cake_only.log')).open('x') as f:
  subprocess.run([PYTHON,str(CODE/'run_lf.py'),str(ROOT/(arm+'_cake_only.yaml'))],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Trained',arm,flush=True)
for label,adapter in [('reference_capabilities',None),('target_cake_only','/workspace/new_runs/target_cake_only'),('control_cake_only','/workspace/new_runs/control_cake_only')]:
 args=['--eval-dir','/workspace/organism_eval/v1','--output',str(ROOT/'evaluations'/label),'--split','development','--families','all','--max-new-tokens','4096','--batch-size','16','--attention-backend','sdpa']
 if adapter:args+=['--adapter',adapter]
 run('evaluate.py',args,label+'.log');print('Evaluated',label,flush=True)
run('score_claims_local.py',['--inputs',*[str(ROOT/'evaluations'/name/'predictions.jsonl') for name in ['reference_capabilities','target_cake_only','control_cake_only']],'--output',str(ROOT/'claims/cake_only'),'--batch-size','8'],'grade_cake_only.log')
print('CAKE_ONLY_TRAINING_AND_EVALUATION_COMPLETE',flush=True)
