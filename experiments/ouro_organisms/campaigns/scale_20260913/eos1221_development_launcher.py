"""Wait for verified EOS1221 publication/monitoring, then run paired development.

Laptop orchestration only. No training, confirmation, checkpoint mutation or pod
lifecycle operations. The sole monitor action is its existing graceful stop flag.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scale_split_transfer import ssh, state, safe_names

ROOT = '/workspace/campaign_scale/broad_development/paired_eos_step1221_v1'
CODE = '/workspace/LlamaFactory/experiments/ouro_organisms'
RUNS = {'target':'preservation_eos_target_r64_100m','control':'preservation_eos_control_r64_100m'}


def remote(campaign, role, source, python='/workspace/ouro-env/bin/python'):
    return ssh(campaign,role,"CUDA_VISIBLE_DEVICES='' "+python+' -c '+shlex.quote(source),timeout=55)


def preserve(path, payload):
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('Immutable local artifact changed: '+str(path))
    else:
        with path.open('xb') as stream:stream.write(payload)


def prepare(campaign):
    source = """from pathlib import Path
import os,json,sys
os.environ.update(json.loads(Path('/root/.ouro_credentials.json').read_text()));os.environ['HF_HOME']='/workspace/hf-cache'
sys.path.insert(0,%r)
from scale_broad_development import selected_events,verify_baselines
from huggingface_hub.errors import EntryNotFoundError
root=Path(%r);root.mkdir(parents=True,exist_ok=True)
b=Path('/workspace/campaign_scale/broad_baselines');verify_baselines(b/'qualification/base_v1_development',b/'coherence/base_development_v1',Path(%r),Path('/workspace/organism_eval/v1'))
repo=json.loads(Path('/workspace/campaign_scale/hf/repository.json').read_text())['repo_id']
try:selected_events(repo,%r,1221,root)
except EntryNotFoundError:print(json.dumps({'ready':False}))
else:print(json.dumps({'ready':True,'selection_text':(root/'selection.json').read_text()}))
""" % (CODE,ROOT,CODE,RUNS)
    return json.loads(remote(campaign,'broad',source))


def monitor_handoff(campaign, selection):
    source = """from pathlib import Path
import json,hashlib,subprocess,time
selection=json.loads(%r);campaign=Path('/workspace/campaign_scale');root=campaign/'eos_monitor/fast_monitor'
ready=True
for arm,run in selection['run_ids'].items():
 p=root/'done'/(run+'-step-1221.json')
 if not p.exists():ready=False;continue
 result=json.loads(p.read_text());expected=selection['events'][arm]['event']
 for key in ['run_id','step','commit','manifest_sha256']:
  if result['checkpoint'][key]!=expected[key]:raise ValueError('Monitor checkpoint identity mismatch')
 for name,h in result['files_sha256'].items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe monitor path')
  if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=h:raise ValueError('Monitor artifact changed')
if not ready:print(json.dumps({'ready':False,'phase':'waiting_final_monitor_passes'}));raise SystemExit()
launch=json.loads((campaign/'eos_monitor_launch.json').read_text());pid=launch['pid'];proc=Path('/proc/'+str(pid)+'/stat')
if proc.exists():
 fields=proc.read_text().split()
 if fields[21]!=str(launch['start_ticks']):raise ValueError('Monitor PID reused')
 if fields[2]!='Z':
  command=Path('/proc/'+str(pid)+'/cmdline').read_bytes().replace(bytes([0]),b' ').decode()
  if 'scale_fast_monitor.py' not in command or 'preservation_eos_' not in command:raise ValueError('Unexpected monitor process')
 stop=campaign/'eos_monitor/STOP_fast_monitor'
 if not stop.exists():
  with stop.open('x') as f:f.write('Final EOS1221 monitor outputs verified; graceful handoff to full development.\\n')
 if fields[2]!='Z':print(json.dumps({'ready':False,'phase':'waiting_graceful_monitor_exit','pid':pid}));raise SystemExit()
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps({'ready':not bool(active),'phase':'gpu_ready' if not active else 'waiting_gpu_release','active':active}))
""" % json.dumps(selection)
    return json.loads(remote(campaign,'broad',source,python='python3'))


def launch_arm(campaign, arm, selection_text):
    role='broad' if arm=='target' else 'broad-control'
    source = """from pathlib import Path
import base64,json,subprocess,hashlib,time,os,sys
root=Path(%r);root.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(%r);selection=json.loads(raw);path=root/'selection.json'
if path.exists():
 if path.read_bytes()!=raw:raise ValueError('Selection bytes differ')
else:
 with path.open('xb') as f:f.write(raw)
coord=root.parent/(root.name+'_coordination');coord.mkdir(exist_ok=True);receipt=coord/(%r+'_launch.json')
if receipt.exists():
 record=json.loads(receipt.read_text())
 if record['selection_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Existing launch selection differs')
 print(json.dumps(record));raise SystemExit()
sys.path.insert(0,%r)
from scale_broad_development import verify_baselines
b=Path('/workspace/campaign_scale/broad_baselines');verify_baselines(b/'qualification/base_v1_development',b/'coherence/base_development_v1',Path(%r),Path('/workspace/organism_eval/v1'))
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
if active:raise ValueError('Unexpected GPU process; no overlap permitted: '+active)
command=['/workspace/ouro-env/bin/python',%r,%r,'--mode','arm','--arm',%r,'--root',str(root)]
log=coord/(%r+'_worker.log')
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
with log.open('x') as f:p=subprocess.Popen(command,cwd='/workspace/LlamaFactory',env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
record={'pid':p.pid,'start_ticks':Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],'command':command,'log':str(log),'unix':time.time(),'selection_sha256':hashlib.sha256(raw).hexdigest(),'worker_sha256':hashlib.sha256(Path(%r).read_bytes()).hexdigest()}
with receipt.open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
""" % (ROOT,base64.b64encode(selection_text.encode()).decode(),arm,CODE,CODE,CODE+'/run_with_credentials.py',CODE+'/scale_parallel_development.py',arm,arm,CODE+'/scale_parallel_development.py')
    return remote(campaign,role,source)


def collect(campaign, folder):
    archive=folder/'paired_outputs.tar.gz'
    payload=ssh(campaign,'broad','tar czf - -C '+shlex.quote(str(Path(ROOT).parent))+' '+shlex.quote(Path(ROOT).name)+' '+shlex.quote(Path(ROOT).name+'_coordination'),timeout=90)
    preserve(archive,payload)
    snapshot=folder/'snapshot';snapshot.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        safe_names({m.name:'' for m in tar.getmembers()});tar.extractall(snapshot,filter='data')
    root=snapshot/Path(ROOT).name;complete=json.loads((root/'COMPLETE.json').read_text())
    if complete.get('completed') is not True:raise ValueError('Pair incomplete')
    for name,h in complete['files_sha256'].items():
        safe_names({name:h})
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Copied pair hash mismatch')
    record={'verified_files':len(complete['files_sha256']),'archive_sha256':hashlib.sha256(payload).hexdigest(),'snapshot':str(root)}
    preserve(folder/'LOCAL_VERIFIED.json',(json.dumps(record,indent=2)+'\n').encode())
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--state-root',type=Path,required=True);args=p.parse_args();args.state_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(args.state_root/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+4*3600
    while time.monotonic()<deadline:
        result=prepare(args.campaign)
        if result['ready']:break
        state(args.state_root,'waiting_final_checkpoint_publication');time.sleep(30)
    else:raise TimeoutError('Final EOS checkpoint publication deadline exceeded')
    selection_text=result['selection_text'];selection=json.loads(selection_text);preserve(args.state_root/'selection.json',selection_text.encode())
    while time.monotonic()<deadline:
        result=monitor_handoff(args.campaign,selection);state(args.state_root,result['phase'],details=result)
        if result['ready']:break
        time.sleep(15)
    else:raise TimeoutError('Final monitor handoff deadline exceeded')
    for arm in ['target','control']:
        record=launch_arm(args.campaign,arm,selection_text);preserve(args.state_root/(arm+'_launch.json'),record)
        state(args.state_root,arm+'_development_launched',launch=json.loads(record))
    transfer=Path(__file__).resolve().parents[2]/'scale_split_transfer.py'
    command=[sys.executable,str(transfer),'--campaign',str(args.campaign),'--state-root',str(args.state_root),'--remote-root',ROOT,'--independent']
    with (args.state_root/'transfer.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    state(args.state_root,'copying_completed_pair_locally')
    verified=collect(args.campaign,args.state_root)
    state(args.state_root,'paired_development_copied_and_verified',verification=verified)
    lock.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'launcher_failed':True,'error_type':type(exc).__name__,'error':str(exc)}),flush=True)
        raise
