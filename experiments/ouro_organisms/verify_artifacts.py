"""Verify every recorded upload against pinned Hub Git/LFS object hashes."""
import argparse,hashlib,json
from pathlib import Path
from huggingface_hub import HfApi

def main():
 p=argparse.ArgumentParser();p.add_argument('--receipts',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 root=Path(args.receipts);api=HfApi();repo=json.loads((root/'repository.json').read_text())['repo_id'];assert api.repo_info(repo).private
 report={'repo_id':repo,'private':True,'verified':[]}
 for receipt_file in sorted(root.glob('*_upload.json')):
  r=json.loads(receipt_file.read_text());prefix=r['prefix'];tree={x.path:x for x in api.list_repo_tree(repo,path_in_repo=prefix,recursive=True,revision=r['commit']) if hasattr(x,'blob_id')}
  for rel,sha in r['sha256'].items():
   entry=tree[prefix+'/'+rel]
   if entry.lfs:
    assert entry.lfs.sha256==sha,(prefix,rel,'LFS checksum mismatch')
   else:
    # Git object hash includes its type and byte length; download only small Git files.
    from huggingface_hub import hf_hub_download
    local=Path(hf_hub_download(repo,filename=prefix+'/'+rel,revision=r['commit']))
    assert hashlib.sha256(local.read_bytes()).hexdigest()==sha,(prefix,rel,'Git file checksum mismatch')
  report['verified'].append({'prefix':prefix,'commit':r['commit'],'files':len(r['sha256'])})
 with Path(args.output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
