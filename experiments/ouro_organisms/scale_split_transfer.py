"""Laptop-only durable transfer for the already-running split development pair.

Reads completed control artifacts via SSH, verifies them locally, imports via an
atomic-stage/last-receipt protocol, then waits for the original supervisor's resume.
No model inference, Hub calls, signals, or pod lifecycle changes occur here.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from uuid import uuid4

ROOT='/workspace/campaign_scale/broad_development/paired_step800_v1'
COORD=ROOT+'_coordination'
CODE='/workspace/LlamaFactory/experiments/ouro_organisms'


def safe_names(hashes):
    for name in hashes:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not name:
            raise ValueError('Unsafe exported path')
    return list(hashes)


def verify_local(folder,marker,selection_sha):
    if marker.get('completed') is not True or marker['selection_sha256']!=selection_sha:
        raise ValueError('Wrong selection or incomplete export')
    for name in safe_names(marker['files_sha256']):
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=marker['files_sha256'][name]:
            raise ValueError('Copied artifact hash differs: '+name)


def ssh(campaign,role,command,stdin=None,timeout=50):
    metadata=json.loads((campaign/(role+'.json')).read_text())
    if not metadata.get('name','').startswith('CLAUDE_POD_GREG---'):
        raise ValueError('Refuse a pod without the experiment prefix')
    cmd=['ssh','-o','ControlPath=/tmp/ouro-scale-'+role,'-o','ConnectTimeout=10','-i',str(Path.home()/'.ssh/id_ed25519_runpod_personal'),
         '-p',str(metadata['portMappings']['22']),'root@'+metadata['publicIp'],command]
    return subprocess.run(cmd,input=stdin,capture_output=True,check=True,timeout=timeout).stdout


def remote_json(campaign,role,path):
    code='from pathlib import Path; p=Path('+repr(path)+'); print(p.read_text() if p.exists() else "null")'
    return json.loads(ssh(campaign,role,'python3 -c '+shlex.quote(code)))


def state(folder,phase,**kwargs):
    record={'unix':time.time(),'phase':phase,**kwargs}
    with (folder/'transfer_events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    temp=folder/'transfer_state.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(folder/'transfer_state.json')


def transfer(campaign,folder,marker,selection_sha,independent=False):
    code='''from pathlib import Path
import io,json,tarfile,sys
root=Path(%r)
marker=json.loads((root/'CONTROL_EXPORT_READY.json').read_text())
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
 for name in ['CONTROL_EXPORT_READY.json',*marker['files_sha256']]:
  p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
  tar.add(root/name,arcname=name)
sys.stdout.buffer.write(buffer.getvalue())
''' % ROOT
    payload=ssh(campaign,'broad-control','python3 -c '+shlex.quote(code),timeout=90)
    attempt=folder/'attempts'/uuid4().hex;attempt.mkdir(parents=True)
    (attempt/'control_outputs.tar.gz').write_bytes(payload)
    snapshot=attempt/'snapshot';snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        safe_names({member.name:'' for member in archive.getmembers()})
        archive.extractall(snapshot,filter='data')
    actual=json.loads((snapshot/'CONTROL_EXPORT_READY.json').read_text())
    if actual!=marker:raise ValueError('Control readiness changed during copy')
    verify_local(snapshot,actual,selection_sha)
    staging='/workspace/campaign_scale/control_import_'+attempt.name
    command='mkdir '+shlex.quote(staging)+' && tar --no-same-owner -xzf - -C '+shlex.quote(staging)
    ssh(campaign,'broad',command,stdin=payload,timeout=90)
    command="CUDA_VISIBLE_DEVICES='' /workspace/ouro-env/bin/python "+shlex.quote(CODE+'/scale_parallel_development.py')+' --mode '+('import-independent' if independent else 'import')+' --root '+shlex.quote(ROOT)+' --staging '+shlex.quote(staging)
    result=ssh(campaign,'broad',command,timeout=90).decode()
    imported=remote_json(campaign,'broad',ROOT+'/CONTROL_IMPORTED.json')
    if imported is None or imported['selection_sha256']!=selection_sha:raise ValueError('Missing verified import receipt')
    (attempt/'import_receipt.json').write_text(json.dumps(imported,indent=2)+'\n')
    return {'local_snapshot':str(snapshot),'remote_staging':staging,'import_receipt':imported,'import_stdout':result}


def main():
    global ROOT,COORD
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--state-root',type=Path,required=True)
    parser.add_argument('--max-hours',type=float,default=8)
    parser.add_argument('--remote-root',default=ROOT)
    parser.add_argument('--independent',action='store_true',help='Wait for both independent workers; import then run CPU paired reports')
    args=parser.parse_args();ROOT=args.remote_root;COORD=ROOT+'_coordination';args.state_root.mkdir(parents=True,exist_ok=True)
    # Local process ownership prevents two upload/import attempts racing.
    import fcntl
    lock=(args.state_root/'transfer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selection_sha=hashlib.sha256((args.state_root/'selection.json').read_bytes()).hexdigest()
    imported=None;deadline=time.monotonic()+args.max_hours*3600
    while time.monotonic()<deadline:
        try:
            if imported is None:
                existing=remote_json(args.campaign,'broad',ROOT+'/CONTROL_IMPORTED.json')
                if existing is not None:
                    if existing['selection_sha256']!=selection_sha:raise ValueError('Existing import is for another selection')
                    imported=existing
                else:
                    ready=remote_json(args.campaign,'broad-control',ROOT+'/CONTROL_EXPORT_READY.json')
                    if ready is None:
                        state(args.state_root,'waiting_control_completion');time.sleep(15);continue
                    if args.independent and remote_json(args.campaign,'broad',ROOT+'/TARGET_EXPORT_READY.json') is None:
                        state(args.state_root,'waiting_target_completion');time.sleep(15);continue
                    state(args.state_root,'copying_verified_control')
                    receipt=transfer(args.campaign,args.state_root,ready,selection_sha,args.independent)
                    imported=receipt['import_receipt']
                    with (args.state_root/'IMPORT_COMPLETE.json').open('x') as f:json.dump(receipt,f,indent=2)
            if args.independent:
                command="CUDA_VISIBLE_DEVICES='' /workspace/ouro-env/bin/python "+shlex.quote(CODE+'/run_with_credentials.py')+' '+shlex.quote(CODE+'/scale_broad_development.py')+' --manifest /workspace/campaign_scale/hf/repository.json --step '+str(json.loads((args.state_root/'selection.json').read_text())['step'])+' --label '+shlex.quote(Path(ROOT).name)+' --output-root '+shlex.quote(str(Path(ROOT).parent))
                state(args.state_root,'assembling_completed_pair_cpu_only')
                output=ssh(args.campaign,'broad',command,timeout=180).decode()
                marker=remote_json(args.campaign,'broad',ROOT+'/COMPLETE.json')
                if not marker or marker.get('completed') is not True:raise ValueError('Missing complete paired report')
                state(args.state_root,'paired_reports_complete',report_stdout=output)
                return
            supervisor=remote_json(args.campaign,'broad',COORD+'/split_coordination_state.json')
            phase=supervisor.get('state') if supervisor else None
            state(args.state_root,'control_imported_waiting_target_and_resume',supervisor=supervisor)
            if phase=='parent_resumed_all_gpu_stages_complete':
                state(args.state_root,'handoff_complete_parent_resumed',supervisor=supervisor)
                return
            if phase=='failed_parent_left_paused_for_safe_review':
                state(args.state_root,'requires_root_review_parent_paused',supervisor=supervisor)
                raise SystemExit(2)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError,json.JSONDecodeError) as exc:
            state(args.state_root,'transient_transfer_error_retrying',error=repr(exc))
        except ValueError as exc:
            state(args.state_root,'integrity_failure_requires_root_review',error=str(exc));raise
        time.sleep(15)
    state(args.state_root,'deadline_requires_root_review_parent_may_be_paused')
    raise SystemExit(3)


if __name__=='__main__':main()
