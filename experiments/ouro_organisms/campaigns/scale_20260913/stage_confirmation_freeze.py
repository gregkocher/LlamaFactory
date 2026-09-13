from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,sys,base64,hashlib,shlex
repo=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/LlamaFactory');sys.path.insert(0,str(repo/'experiments/ouro_organisms'))
from scale_split_transfer import ssh
campaign=Path('/Users/gkocher/Desktop/recurrent-looped-auditing/research/scale_campaign_20260913');folder=campaign/'confirmation_v3_eos1221';freeze=(folder/'FREEZE.json').read_bytes();expected='9db595e0fd9200d48e4dc4834d5a07d64b5e85a0fb3fa7ab754e845f40a2adb2';assert hashlib.sha256(freeze).hexdigest()==expected
stage=folder/'launch';stage.mkdir(exist_ok=True)
def one(role):
 source="""from pathlib import Path
import base64,json,hashlib,sys,time,subprocess
code=Path('/workspace/LlamaFactory/experiments/ouro_organisms');sys.path.insert(0,str(code))
from scale_confirmation import validate_freeze,verify_inputs
folder=Path('/workspace/campaign_scale/confirmation_v3_eos1221');folder.mkdir(exist_ok=True);p=folder/'FREEZE.json';raw=base64.b64decode(%r)
if p.exists():assert p.read_bytes()==raw
else:
 with p.open('xb') as f:f.write(raw)
freeze=validate_freeze(p,%r);verify_inputs(freeze,code,Path('/workspace/organism_eval/v1'))
record={'role':%r,'validated_unix':time.time(),'freeze_sha256':hashlib.sha256(raw).hexdigest(),'protocol_sha256':freeze['criteria']['protocol_sha256'],'scripts_sha256':freeze['scripts_sha256'],'inputs_sha256':freeze['inputs_sha256'],'git_head':subprocess.check_output(['git','-C','/workspace/LlamaFactory','rev-parse','HEAD'],text=True).strip(),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()}
coord=folder/'coordination';coord.mkdir(exist_ok=True)
with (coord/'freeze_validation.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
"""%(base64.b64encode(freeze).decode(),expected,role)
 raw=ssh(campaign,role,"CUDA_VISIBLE_DEVICES='' /workspace/ouro-env/bin/python -c "+shlex.quote(source),timeout=55)
 record=json.loads(raw)
 with (stage/(role+'_freeze_validation.json')).open('x') as f:json.dump(record,f,indent=2)
 return {'role':role,'freeze_verified':True,'gpu_processes':record['gpu_processes']}
with ThreadPoolExecutor(max_workers=4) as pool:
 for r in pool.map(one,['broad','broad-control','evaluator','retention']):print(json.dumps(r))
