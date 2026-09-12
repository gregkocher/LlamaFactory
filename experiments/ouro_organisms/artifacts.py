"""Create a new private HF repository and add immutable experiment artifacts."""
import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from huggingface_hub import HfApi


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--folder');p.add_argument('--prefix');args=p.parse_args()
    api=HfApi();assert api.whoami()['name']=='wasd12345'
    manifest=Path(args.manifest);manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():
        record=json.loads(manifest.read_text());repo_id=record['repo_id']
    else:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        repo_id=f'wasd12345/ouro-organisms-{stamp.lower()}-{uuid4().hex[:6]}'
        api.create_repo(repo_id=repo_id,repo_type='model',private=True,exist_ok=False)
        record={'repo_id':repo_id,'created_at':stamp,'base_model':'ByteDance/Ouro-1.4B','base_revision':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1','policy':'private; add new paths only; never delete or overwrite'}
        with manifest.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    info=api.repo_info(repo_id,repo_type='model')
    assert info.private is True and repo_id.startswith('wasd12345/')
    print(json.dumps({'repo_id':repo_id,'private':info.private}),flush=True)
    if not args.folder:return
    folder=Path(args.folder)
    if not folder.is_dir() or not args.prefix or args.prefix.startswith('/') or '..' in Path(args.prefix).parts:
        raise ValueError('A local folder and safe nonempty repository prefix are required')
    prefix=args.prefix.strip('/')
    existing=api.list_repo_files(repo_id)
    if any(path==prefix or path.startswith(prefix+'/') for path in existing):
        raise FileExistsError('Remote prefix already exists; verify it without replacing files')
    hashes={str(path.relative_to(folder)):hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.rglob('*') if path.is_file()}
    commit=api.upload_folder(repo_id=repo_id,folder_path=str(folder),path_in_repo=prefix,commit_message=f'Add {prefix}',parent_commit=info.sha)
    receipt={'repo_id':repo_id,'prefix':prefix,'commit':commit.oid,'sha256':hashes}
    receipt_path=manifest.parent/(prefix.replace('/','_')+'_upload.json')
    with receipt_path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    # The separate receipt is also append-only and pins the artifact commit.
    receipt_remote='receipts/'+receipt_path.name
    if receipt_remote in api.list_repo_files(repo_id):raise FileExistsError(receipt_remote)
    api.upload_file(repo_id=repo_id,path_in_repo=receipt_remote,path_or_fileobj=io.BytesIO(receipt_path.read_bytes()),commit_message=f'Add checksum receipt for {prefix}')
    print(json.dumps({'uploaded':prefix,'files':len(hashes),'commit':commit.oid}),flush=True)


if __name__=='__main__':main()
