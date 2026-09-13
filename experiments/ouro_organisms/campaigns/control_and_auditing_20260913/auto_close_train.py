"""Preserve and stop only this campaign's training pod after fixed evaluation.

Runs locally. Never creates/deletes pods, changes inference, or stops unfinished
work. Final STOP uses the existing account-isolated archive verification wrapper.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

import ops


def now():
    return datetime.now(timezone.utc).isoformat()


def record(event, **fields):
    with (ops.STATE / 'auto_close_train_events.jsonl').open('a') as handle:
        handle.write(json.dumps({'time_utc': now(), 'event': event, **fields}) + '\n')
    print(json.dumps({'time_utc': now(), 'event': event}), flush=True)


def remote(script, timeout=40):
    result = subprocess.run(ops.ssh_args(ops.metadata('train')) +
                            ['python3 -c ' + shlex.quote(script)],
                            text=True, capture_output=True, timeout=timeout, check=True)
    return result.stdout


STATUS = '''from pathlib import Path
import hashlib,json,subprocess
p=Path('/workspace/campaign_unrelated')
f=p/'EVALUATION_WORKFLOW_COMPLETE.json'
ready=False
if f.exists():
 d=json.loads(f.read_text())
 ready=d.get('completed') is True and hashlib.sha256((p/'evaluation_v1/COMPLETE.json').read_bytes()).hexdigest()==d['evaluation_complete_sha256']
 if not ready:raise ValueError('Completion certificate differs')
if ready:
 launch=json.loads((p/'publisher_launch.json').read_text())
 proc=Path('/proc')/str(launch['pid'])/'stat'
 if proc.exists():
  stat=proc.read_text().split()
  if stat[21]==launch['start_ticks'] and stat[2] not in ('Z','X'):raise ValueError('Publisher still active')
 ready=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print(json.dumps({'ready':ready}))
'''


EXPORT = '''from pathlib import Path
import subprocess
repo=Path('/workspace/LlamaFactory')
url=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()
if url!='https://github.com/gregkocher/LlamaFactory.git':raise ValueError('Wrong fork')
if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():raise ValueError('Uncommitted source requires preservation review')
if subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()!='ouro-organisms':raise ValueError('Wrong branch')
subprocess.run(['git','-C',str(repo),'pull','--ff-only','origin','ouro-organisms'],check=True)
command=['/workspace/ouro-env/bin/python',str(repo/'experiments/ouro_organisms/run_with_credentials.py'),str(repo/'experiments/ouro_organisms/scale_export.py'),
 '--campaign','/workspace/campaign_unrelated','--runs','/workspace/scale_runs',
 '--repo',str(repo),'--receipts','/workspace/campaign_unrelated/hf',
 '--output','/workspace/exports/train_final_v1','--workloads-ended']
for path in ['/workspace/campaign_scale','/workspace/organism_eval','/workspace/census_frame_v1','/workspace/FREEZE.json']:
 command+=['--extra',path]
for path in ['/workspace','/root','/workspace/bootstrap']:
 command+=['--runtime-dir',path]
subprocess.run(command,check=True)
'''


def main():
    cancel = ops.STATE / 'CANCEL_AUTO_CLOSE_TRAIN'
    record('waiting_for_fixed_evaluation')
    while True:
        if cancel.exists():
            record('cancelled_without_pod_operation')
            return
        try:
            status = json.loads(remote(STATUS))
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
            record('status_retry', error_type=type(exc).__name__)
            time.sleep(30)
            continue
        if status['ready']:
            break
        time.sleep(30)
    if cancel.exists():
        record('cancelled_without_pod_operation')
        return
    record('fixed_gpu_work_complete_starting_export')
    # Only archive after all GPU work and checkpoint publication have ended.
    (ops.STATE / 'auto_close_train_export.log').write_text(remote(EXPORT, timeout=3600))
    record('remote_export_complete')
    source = Path(__file__).resolve().parents[1] / 'scale_20260913'
    archive = ops.STATE / 'exports/train_final_v1.tar.gz'
    subprocess.run([sys.executable, str(source/'ouro_copy_archive.py'),
                    str(ops.STATE/'train.json'), '/workspace/exports/train_final_v1.tar.gz',
                    str(archive)], check=True)
    record('archive_transferred')
    if cancel.exists():
        record('cancelled_after_copy_without_stop')
        return
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify_and_stop.py')),
                    'train', str(archive), '--stop'], check=True)
    record('archive_verified_and_pod_stopped')
    output = archive.parent/'train_final_v1'
    if output.exists():
        raise FileExistsError('Preserve existing extraction; do not overwrite it')
    with tarfile.open(archive) as handle:
        handle.extractall(archive.parent, filter='data')
    with (ops.STATE/'AUTO_CLOSE_TRAIN_COMPLETE.json').open('x') as handle:
        json.dump({'completed_at_utc':now(), 'archive':str(archive),
                   'stop_receipt':str(ops.STATE/'train_stop.json'),
                   'scope':'GPU inference, checkpoint publication, archive verification and STOP. Local manual review may still be ongoing.'},handle,indent=2)
        handle.write('\n')
    record('local_extraction_complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('failed_requires_review', error_type=type(exc).__name__, message=str(exc))
        raise
