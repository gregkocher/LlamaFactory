from pathlib import Path
import subprocess,json,shlex,concurrent.futures
m=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/scale_campaign_20260913/hf/repository.json').read_text()
def one(role):
 code="from pathlib import Path;p=Path('/workspace/campaign_scale/hf/repository.json');p.parent.mkdir(exist_ok=True);assert not p.exists();p.write_text("+repr(m)+")"
 cmd='git -C /workspace/LlamaFactory pull --ff-only origin ouro-organisms\npython3 -c '+shlex.quote(code)+'\n'
 cmd+='nohup bash /workspace/LlamaFactory/experiments/ouro_organisms/scale_start_diagnostic.sh '+role+' > /workspace/campaign_scale/diagnostic.log 2>&1 < /dev/null &\n'
 cmd+='nohup /workspace/ouro-env/bin/python /workspace/LlamaFactory/experiments/ouro_organisms/run_with_credentials.py /workspace/LlamaFactory/experiments/ouro_organisms/scale_exchange.py publish --manifest /workspace/campaign_scale/hf/repository.json > /workspace/campaign_scale/publisher.log 2>&1 < /dev/null &\n'
 if role=='trainer':cmd+='nohup /workspace/ouro-env/bin/python /workspace/LlamaFactory/experiments/ouro_organisms/run_with_credentials.py /workspace/LlamaFactory/experiments/ouro_organisms/prepare_scaled_data.py --output /workspace/campaign_scale/data_scaled --stage select --documents 15000 --minimum-documents 10000 > /workspace/campaign_scale/data_selection.log 2>&1 < /dev/null &\n'
 r=subprocess.run(['python3',str(Path(__file__).with_name('ouro_scale_ops.py')),'ssh',role],input=cmd,text=True,capture_output=True,timeout=55);return {'role':role,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
for r in concurrent.futures.ThreadPoolExecutor(2).map(one,['trainer','evaluator']):print(json.dumps(r))
