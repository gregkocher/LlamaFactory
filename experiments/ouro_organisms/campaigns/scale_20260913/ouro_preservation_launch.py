"""Launch the reviewed paired preservation campaign on existing personal pods."""
from pathlib import Path
import argparse,json,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('role',choices=['trainer','qualification']);p.add_argument('--review-acceptance',required=True);p.add_argument('--dataset-dir',default='/workspace/campaign_scale/data_preservation');a=p.parse_args()
assert a.dataset_dir in ['/workspace/campaign_scale/data_preservation','/workspace/campaign_scale/data_preservation_v2']
accept=Path(a.review_acceptance);record=json.loads(accept.read_text());assert record['accepted'] is True
arm='target' if a.role=='trainer' else 'control';run=f'preservation_{arm}_r64_100m'
code='/workspace/LlamaFactory/experiments/ouro_organisms';base='/workspace/campaign_scale';py='/workspace/ouro-env/bin/python'
script='set -euo pipefail\n'
script+='git -C /workspace/LlamaFactory pull --ff-only origin ouro-organisms\n'
script+=f'test -f {a.dataset_dir}/manifest.json\n'
check="import hashlib,json;from pathlib import Path;p=Path("+repr(a.dataset_dir)+");m=json.loads((p/'manifest.json').read_text());expected="+repr(record['arm_sha256'])+";assert all(m['arms'][arm]['sha256']==expected[arm] and hashlib.sha256((p/(arm+'.json')).read_bytes()).hexdigest()==expected[arm] for arm in expected)"
script+=py+' -c '+shlex.quote(check)+'\n'
script+=f'{py} {code}/make_scale_config.py --config {base}/{run}.yaml --run-id {run} --scope r64_all --dataset-dir {a.dataset_dir} --dataset {arm} --output-dir /workspace/scale_runs/{run} --token-budget 100000000 --batch-size 8 --accumulation 2 --early-steps 1,50,122,244,400,800,1221,2442 --save-steps 100000 --checkpoint-minutes 20 --wall-hours 4\n'
script+=f'{py} -u {code}/scale_train.py --config {base}/{run}.yaml\n'
script+=f"printf '{run}_COMPLETE\\n'\n"
remote="python3 - <<'REMOTE'\nfrom pathlib import Path\nimport json,subprocess\np=Path("+repr(base+'/'+run+'.sh')+")\nassert not p.exists()\np.write_text("+repr(script)+")\na=Path("+repr(base+'/'+run+'_review_acceptance.json')+")\nassert not a.exists()\na.write_text("+repr(accept.read_text())+")\nwith open("+repr(base+'/'+run+'.log')+",'x') as log:\n child=subprocess.Popen(['bash',str(p)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)\nprint(json.dumps({'pid':child.pid,'run':"+repr(run)+"}))\nREMOTE\n"
r=subprocess.run(['python3',str(Path(__file__).with_name('ouro_scale_ops.py')),'ssh',a.role],input=remote,text=True,check=True,capture_output=True,timeout=55);print(r.stdout)
