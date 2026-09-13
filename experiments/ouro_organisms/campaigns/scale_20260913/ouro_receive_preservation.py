"""Download the private immutable replay corpus and verify every file."""
from pathlib import Path
import argparse,hashlib,json,tarfile,time
from huggingface_hub import HfApi,hf_hub_download
parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v1','v2'],default='v1');args=parser.parse_args()
version=args.version;dataset='data_preservation' if version=='v1' else 'data_preservation_v2';review='data_validation_v1' if version=='v1' else 'data_review_repaired_v2'
root=Path('/workspace/campaign_scale');repo=json.loads((root/'hf/repository.json').read_text())['repo_id'];api=HfApi()
assert api.whoami()['name']=='wasd12345' and api.repo_info(repo).private
receipt_name=f'receipts/scale_v1_data_preservation_{version}_upload.json'
for attempt in range(120):
 info=api.repo_info(repo)
 if receipt_name in api.list_repo_files(repo,revision=info.sha):break
 time.sleep(10)
else:raise TimeoutError('Data upload receipt unavailable')
receipt_path=hf_hub_download(repo_id=repo,filename=receipt_name,revision=info.sha)
receipt=json.loads(Path(receipt_path).read_text());assert receipt['prefix']==f'scale_v1/data/preservation_{version}'
files={}
for name,expected in receipt['sha256'].items():
 p=Path(hf_hub_download(repo_id=repo,filename=receipt['prefix']+'/'+name,revision=receipt['commit']))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 files[name]=p
manifest=json.loads(files['manifest.json'].read_text());assert hashlib.sha256(files['data.tar.gz'].read_bytes()).hexdigest()==manifest['archive_sha256']
assert not (root/dataset).exists() and not (root/review).exists()
with tarfile.open(files['data.tar.gz'],'r:gz') as tar:
 for member in tar.getmembers():
  p=Path(member.name);assert not p.is_absolute() and '..' not in p.parts and p.parts[0] in [dataset,review] and (member.isfile() or member.isdir())
 tar.extractall(root,filter='data')
for rel,entry in manifest['files'].items():
 p=root/rel;assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
with (root/f'data_received_{version}.json').open('x') as f:json.dump({'repo_id':repo,'commit':receipt['commit'],'receipt':receipt,'manifest':manifest,'files_verified':len(manifest['files'])},f,indent=2)
print(json.dumps({'data_received_verified':True,'files':len(manifest['files']),'commit':receipt['commit']}),flush=True)
