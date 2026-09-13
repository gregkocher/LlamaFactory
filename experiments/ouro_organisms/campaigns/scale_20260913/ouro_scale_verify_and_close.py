"""Verify a scale export, then optionally release only its named personal pod."""
import argparse,hashlib,json,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
import requests
p=argparse.ArgumentParser();p.add_argument('metadata');p.add_argument('archive');p.add_argument('--close',action='store_true');a=p.parse_args()
meta=Path(a.metadata);m=json.loads(meta.read_text());archive=Path(a.archive)
assert m['account']=='personal' and m['name'] in ['CLAUDE_POD_GREG---ouro-scale-'+r for r in ['trainer','evaluator','qualification']]
receipt=json.loads(archive.with_suffix(archive.suffix+'.verified.json').read_text());assert receipt['pod_id']==m['id']
h=hashlib.sha256()
with archive.open('rb') as f:
 for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==receipt['sha256']
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers(); manifests=[x for x in members if len(Path(x.name).parts)==2 and x.name.endswith('/file_manifest.json')];assert len(manifests)==1
 root=Path(manifests[0].name).parent.as_posix();file_manifest=json.load(tar.extractfile(manifests[0]));actual={x.name[len(root)+1:] for x in members if x.isfile()};assert actual==set(file_manifest['files'])|{'file_manifest.json'}
 for rel,item in file_manifest['files'].items():
  member=tar.getmember(root+'/'+rel);assert member.size==item['size_bytes'];stream=tar.extractfile(member);digest=hashlib.sha256()
  for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
  assert digest.hexdigest()==item['sha256'],rel
 verification=json.load(tar.extractfile(root+'/checkpoint_remote_verification.json'))
 provenance=json.load(tar.extractfile(root+'/export_provenance.json'));assert provenance['workloads_ended_assertion'] is True
 git=json.load(tar.extractfile(root+'/git_state.json'));assert git['branch'].strip()=='ouro-organisms' and not git['status'].strip()
 for row in verification['checkpoints']:assert row['repo_id'].startswith('wasd12345/')
record={'pod_id':m['id'],'archive':str(archive),'sha256':receipt['sha256'],'files_verified':len(actual),'checkpoint_verification':verification,'git_head':git['head'].strip(),'verified_at_utc':datetime.now(timezone.utc).isoformat()}
path=archive.with_suffix(archive.suffix+'.contents_verified.json')
if not path.exists():path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='checkpoint_verification'}),flush=True)
if not a.close:raise SystemExit()
key=json.loads((Path.home()/'.claude.json').read_text())['mcpServers']['runpod']['env']['RUNPOD_API_KEY'];s=requests.Session();s.headers.update({'Authorization':'Bearer '+key,'User-Agent':'ouro-research/1.0'})
url='https://rest.runpod.io/v1/pods/'+m['id'];r=s.get(url,timeout=35);r.raise_for_status();live=json.loads(r.text,strict=False)
assert live['id']==m['id'] and live['name']==m['name']
r=s.delete(url,timeout=35);assert r.status_code in [200,204],r.status_code
for attempt in range(6):
 check=s.get(url,timeout=35)
 if check.status_code==404:break
 time.sleep(2)
assert check.status_code==404,check.status_code
end=datetime.now(timezone.utc);hours=(end-datetime.fromisoformat(m['created_at_utc'])).total_seconds()/3600
closure={**m,'terminated_at_utc':end.isoformat(),'delete_http_status':r.status_code,'subsequent_get_status':check.status_code,'lease_hours_estimate':hours,'gpu_cost_estimate_usd':hours*float(m['costPerHr']),'verified_archive':str(archive),'contents_verification':str(path),'policy':'Only this named campaign pod released after immutable private HF and local file verification; no HF mutations.'}
with meta.with_name(meta.stem+'_closure.json').open('x') as f:json.dump(closure,f,indent=2);f.write('\n')
print(json.dumps(closure),flush=True)
