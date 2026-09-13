"""Publish complete checkpoints privately and evaluate immutable Hub snapshots."""
import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['publish','evaluate']);p.add_argument('--manifest',required=True);p.add_argument('--campaign',default='/workspace/campaign_scale');p.add_argument('--once',action='store_true');a=p.parse_args()
 root=Path(a.campaign);manifest=Path(a.manifest);repo=json.loads(manifest.read_text())['repo_id'];api=HfApi();assert api.whoami()['name']=='wasd12345' and api.repo_info(repo).private
 code=Path(__file__).parent
 while True:
  if a.mode=='publish':
   ready=root/'checkpoint_ready.jsonl'
   events=[]
   if ready.exists():
    for line in ready.read_text().splitlines():
     try:events.append(json.loads(line))
     except json.JSONDecodeError:continue
   for e in events:
    prefix=f"scale_v1/checkpoints/{e['run_id']}/step-{e['step']}"
    receipt=manifest.parent/(prefix.replace('/','_')+'_upload.json');verified=receipt.with_suffix('.verified.json')
    if verified.exists():continue
    folder=Path(e['path']);assert hashlib.sha256(Path(e['manifest']).read_bytes()).hexdigest()==e['manifest_sha256']
    if not receipt.exists():subprocess.run([sys.executable,str(code/'artifacts.py'),'--manifest',str(manifest),'--folder',str(folder),'--prefix',prefix],check=True)
    r=json.loads(receipt.read_text());tree={x.path:x for x in api.list_repo_tree(repo,path_in_repo=prefix,recursive=True,revision=r['commit']) if hasattr(x,'blob_id')}
    for rel,sha in r['sha256'].items():
     entry=tree[prefix+'/'+rel]
     if entry.lfs:assert entry.lfs.sha256==sha
     else:
      data=(folder/rel).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
      assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entry.blob_id
    published={**e,'repo_id':repo,'prefix':prefix,'commit':r['commit'],'files':len(r['sha256']),'verified':True}
    remote=f"scale_v1/ready/{e['run_id']}-step-{e['step']}.json"
    if not api.file_exists(repo,remote):
     import io
     api.upload_file(repo_id=repo,path_in_repo=remote,path_or_fileobj=io.BytesIO((json.dumps(published,indent=2)+'\n').encode()),commit_message='Add verified checkpoint readiness receipt')
    with verified.open('x') as f:json.dump(published,f,indent=2)
    print(json.dumps({'published':prefix,'commit':r['commit']}),flush=True)
  else:
   done=root/'eval_queue';done.mkdir(exist_ok=True)
   files=[x for x in api.list_repo_files(repo) if x.startswith('scale_v1/ready/') and x.endswith('.json')]
   queue=[]
   from huggingface_hub import hf_hub_download
   for name in files:
    path=hf_hub_download(repo,filename=name);e=json.loads(Path(path).read_text())
    tag=e['run_id']+'-step-'+str(e['step'])
    if (done/(tag+'.json')).exists() or e['step']<50:continue
    # Short diagnostic scope runs need only the final122-step behavior panel.
    if e['run_id'].startswith('diagnostic') and e['step']!=122:continue
    queue.append((e,tag))
   # Newest pending checkpoint is the most useful online safety observation.
   queue.sort(key=lambda x:x[0]['created_unix'],reverse=True)
   if queue:
    e,tag=queue[0];local=Path(snapshot_download(repo,revision=e['commit'],allow_patterns=[e['prefix']+'/*']))/e['prefix']
    out=root/'evaluations'/tag
    cmd=[sys.executable,str(code/'scale_evaluate.py'),'--eval-dir','/workspace/organism_eval/v1','--checkpoint',str(local),'--checkpoint-manifest',str(local/'scale_checkpoint_manifest.json'),'--output',str(out),'--quick']
    if not (out/'COMPLETE.json').exists():
     if out.exists():out=root/'evaluations'/(tag+'-retry-'+str(int(time.time())));cmd[cmd.index('--output')+1]=str(out)
     subprocess.run(cmd,check=True)
    record={'checkpoint':e,'output':str(out),'summary':json.loads((out/'summary.json').read_text())}
    with (done/(tag+'.json')).open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps({'evaluated':tag,'output':str(out)}),flush=True)
  if a.once or (root/('STOP_'+a.mode)).exists():break
  time.sleep(20)

if __name__=='__main__':main()
