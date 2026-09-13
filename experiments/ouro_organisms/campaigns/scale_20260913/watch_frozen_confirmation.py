"""Read-only confirmation monitoring and verified local collection; no lifecycle actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scale_split_transfer import ssh,safe_names

ROLES={'target':'broad','control':'broad-control','base':'evaluator'}
REMOTE='/workspace/campaign_scale/confirmation_v3_eos1221'
FREEZE='9db595e0fd9200d48e4dc4834d5a07d64b5e85a0fb3fa7ab754e845f40a2adb2'


def exclusive(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)


def status(campaign,arm):
    source="""from pathlib import Path
import json,hashlib,subprocess,time
root=Path(%r);arm=%r;expected=%r;launch=json.loads((root/'coordination'/(arm+'_launch.json')).read_text());proc=Path('/proc')/str(launch['pid']);state=None
if proc.exists():
 fields=(proc/'stat').read_text().split()
 if fields[21]!=launch['start_ticks']:raise ValueError('Confirmation PID reused')
 state=fields[2]
job=root/'results/jobs'/(arm+'_both');complete=job/'COMPLETE.json';record={'arm':arm,'unix':time.time(),'pid':launch['pid'],'parent_state':state,'completed':False,'stages':{}}
for f in (root/'results').rglob('predictions.jsonl'):
 text=f.read_text();lines=text.splitlines();rows=[]
 for line in lines:
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 families={}
 for row in rows:families[row['family']]=families.get(row['family'],0)+1
 record['stages'][str(f.relative_to(root/'results'))]={'rows':len(rows),'families':families}
if complete.exists():
 marker=json.loads(complete.read_text())
 if marker.get('completed') is not True or marker['freeze_sha256']!=expected:raise ValueError('Confirmation completion identity differs')
 for name,h in marker['files_sha256'].items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe completed artifact path')
  if hashlib.sha256((root/'results'/rel).read_bytes()).hexdigest()!=h:raise ValueError('Completed artifact hash differs')
 record['verified_files']=len(marker['files_sha256'])
 record['completed']=state in (None,'Z')
 if record['completed']:
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if active:raise ValueError('GPU writer remains after confirmation completion: '+active)
elif state in (None,'Z'):
 record['failed']=True;record['worker_log_tail']=(root/'coordination'/(arm+'_worker.log')).read_text()[-5000:]
print(json.dumps(record))
"""%(REMOTE,arm,FREEZE)
    return json.loads(ssh(campaign,ROLES[arm],'python3 -c '+shlex.quote(source),timeout=55))


def collect(campaign,arm,folder):
    payload=ssh(campaign,ROLES[arm],'tar czf - -C /workspace/campaign_scale confirmation_v3_eos1221',timeout=90)
    archive=folder/'confirmation_outputs.tar.gz'
    with archive.open('xb') as f:f.write(payload)
    snapshot=folder/'snapshot';snapshot.mkdir()
    with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
        safe_names({m.name:'' for m in tar.getmembers()});tar.extractall(snapshot,filter='data')
    root=snapshot/'confirmation_v3_eos1221'
    if hashlib.sha256((root/'FREEZE.json').read_bytes()).hexdigest()!=FREEZE:raise ValueError('Copied freeze differs')
    marker=json.loads((root/'results/jobs'/(arm+'_both')/'COMPLETE.json').read_text())
    if marker.get('completed') is not True or marker['freeze_sha256']!=FREEZE:raise ValueError('Copied completion differs')
    safe_names(marker['files_sha256'])
    for name,h in marker['files_sha256'].items():
        if hashlib.sha256((root/'results'/name).read_bytes()).hexdigest()!=h:raise ValueError('Copied artifact differs')
    record={'arm':arm,'role':ROLES[arm],'archive_sha256':hashlib.sha256(payload).hexdigest(),'archive_bytes':len(payload),'verified_files':len(marker['files_sha256']),'snapshot':str(root),'completed_unix':time.time(),'freeze_sha256':FREEZE}
    exclusive(folder/'LOCAL_VERIFIED.json',record)
    return record


def watch(campaign,output,arm):
    folder=output/arm;folder.mkdir(exist_ok=True);deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        try:
            record=status(campaign,arm)
            temp=folder/'status.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(folder/'status.json')
            if record.get('failed'):
                exclusive(folder/'FAILED.json',record);return record
            if record['completed']:return collect(campaign,arm,folder)
        except Exception as exc:
            with (folder/'errors.jsonl').open('a') as f:f.write(json.dumps({'unix':time.time(),'error':str(exc)})+'\n')
            # Completed collection errors require explicit preservation/recovery.
            if (folder/'confirmation_outputs.tar.gz').exists():raise
        time.sleep(45)
    raise TimeoutError('Confirmation watcher exceeded six hours')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(watch,a.campaign,a.output,arm) for arm in ROLES]
        for future in futures:print(json.dumps(future.result()),flush=True)

if __name__=='__main__':main()
