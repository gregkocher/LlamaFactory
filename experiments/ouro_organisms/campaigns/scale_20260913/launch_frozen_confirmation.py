from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,shlex,hashlib
repo=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory');sys.path.insert(0,str(repo/'experiments/ouro_organisms'))
from scale_split_transfer import ssh
campaign=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/scale_campaign_20260913');folder=campaign/'confirmation_v3_eos1221/launch';expected='9db595e0fd9200d48e4dc4834d5a07d64b5e85a0fb3fa7ab754e845f40a2adb2'
def one(item):
 arm,role=item
 source="""from pathlib import Path
import json,sys,os,subprocess,time,hashlib
code=Path('/workspace/LlamaFactory/experiments/ouro_organisms');sys.path.insert(0,str(code))
from scale_confirmation import validate_freeze,verify_inputs,validate_selected_manifest
from scale_parallel_development import checkpoint
folder=Path('/workspace/campaign_scale/confirmation_v3_eos1221');freeze=validate_freeze(folder/'FREEZE.json',%r);verify_inputs(freeze,code,Path('/workspace/organism_eval/v1'));arm=%r
os.environ.update(json.loads(Path('/root/.ouro_credentials.json').read_text()));os.environ['HF_HOME']='/workspace/hf-cache'
if arm!='base':
 adapter=checkpoint(freeze['selection'],arm);validate_selected_manifest(freeze['selection'],arm,adapter)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise ValueError('Unexpected GPU process: '+active)
output=folder/'results'
if output.exists():raise ValueError('Use a fresh immutable confirmation output root')
coord=folder/'coordination';receipt=coord/(arm+'_launch.json');log=coord/(arm+'_worker.log')
if receipt.exists() or log.exists():raise ValueError('Confirmation arm already launched')
command=['/workspace/ouro-env/bin/python',str(code/'run_with_credentials.py'),str(code/'scale_confirmation.py'),'--freeze',str(folder/'FREEZE.json'),'--freeze-sha256',%r,'--arm',arm,'--tasks','both','--output-root',str(output)]
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
with log.open('x') as f:p=subprocess.Popen(command,cwd='/workspace/LlamaFactory',stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
record={'arm':arm,'role':%r,'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'command':command,'unix':time.time(),'freeze_sha256':%r,'script_sha256':hashlib.sha256((code/'scale_confirmation.py').read_bytes()).hexdigest(),'log':str(log),'output_root':str(output)}
with receipt.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
"""%(expected,arm,expected,role,expected)
 raw=ssh(campaign,role,'/workspace/ouro-env/bin/python -c '+shlex.quote(source),timeout=55)
 record=json.loads(raw)
 with (folder/(arm+'_launch.json')).open('x') as f:json.dump(record,f,indent=2)
 return record
with ThreadPoolExecutor(max_workers=3) as pool:
 for record in pool.map(one,[('target','broad'),('control','broad-control'),('base','evaluator')]):print(json.dumps(record))
