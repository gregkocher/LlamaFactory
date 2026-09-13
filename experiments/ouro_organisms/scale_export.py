"""Verify and export a finished scale campaign; never stop or delete resources.

Every readiness event must match a verified immutable private-Hub checkpoint.
Large checkpoint binaries may be omitted only when covered by that verification;
unverified and failed-run artifacts remain in the local export. Failure logs are
preserved without asserting that their training/evaluation completed successfully.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time


SKIP_DIRS = {'.git', '.venv', '__pycache__', '.cache', 'hf_cache', 'hf-cache', 'huggingface_cache', 'hub'}
SECRET_NAMES = {'.ouro_credentials.json', '.hf_token', 'openrouter_api_key.txt',
                'openrouter_api_key_weekly1000.txt', 'openrouter_api_key_daily50.txt', '.env'}
BINARY_SUFFIXES = {'.safetensors', '.bin', '.pt', '.pth', '.ckpt'}


def digest(path, git=False):
    path = Path(path)
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(b'blob '+str(path.stat().st_size).encode()+b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, item):
    with Path(path).open('x') as stream:
        json.dump(item, stream, indent=2)
        stream.write('\n')


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not name:
        raise ValueError('Unsafe manifest-relative path')
    return path


def scan_secrets(path, secrets):
    # Include boundary overlap so a token split across read blocks is still found.
    overlap = max([len(secret) for secret in secrets] or [1])-1
    tail = b''
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            data = tail+chunk
            if any(secret in data for secret in secrets):
                raise ValueError(f'Credential detected in export input: {path}')
            tail = data[-overlap:] if overlap else b''


def credential_values(path):
    values = [value for key, value in os.environ.items()
              if ('TOKEN' in key or 'API_KEY' in key) and len(value) >= 12]
    if path and Path(path).exists():
        record = json.loads(Path(path).read_text())
        values.extend(str(value) for key, value in record.items()
                      if ('TOKEN' in key or 'KEY' in key) and len(str(value)) >= 12)
    return sorted({value.encode() for value in values}, key=len, reverse=True)


def verify_events(campaign, receipts, api, download):
    ready = campaign/'checkpoint_ready.jsonl'
    events = []
    if ready.exists():
        for number, line in enumerate(ready.read_text().splitlines(), 1):
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid readiness line {number}; cannot close export safely') from exc
    verified_files, rows, identities, private_repos = {}, [], {}, set()
    for event in events:
        run, step = event['run_id'], int(event['step'])
        if Path(run).name != run or run in {'.', '..'}:
            raise ValueError('Unsafe readiness run ID')
        identity = (run, step)
        if identity in identities:
            if identities[identity] != event:
                raise ValueError('Conflicting duplicate readiness events')
            continue
        identities[identity] = event
        prefix = f'scale_v1/checkpoints/{run}/step-{step}'
        receipt_path = receipts/(prefix.replace('/', '_')+'_upload.json')
        verified_path = receipt_path.with_suffix('.verified.json')
        if not receipt_path.exists() or not verified_path.exists():
            raise FileNotFoundError(f'Missing upload or verified receipt for {run} step {step}')
        receipt = json.loads(receipt_path.read_text())
        verified = json.loads(verified_path.read_text())
        repo, commit = receipt['repo_id'], receipt['commit']
        if not repo.startswith('wasd12345/') or not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Expected personal repository and immutable 40-character commit')
        if (receipt['prefix'] != prefix or verified.get('verified') is not True
                or any(verified.get(key) != event[key] for key in ('run_id', 'step', 'manifest_sha256'))
                or any(verified.get(key) != value for key, value in [('repo_id', repo), ('prefix', prefix), ('commit', commit)])):
            raise ValueError(f'Receipt identity mismatch for {run} step {step}')
        if repo not in private_repos:
            if api.whoami()['name'] != 'wasd12345' or not api.repo_info(repo).private:
                raise ValueError('Checkpoint repository is not private under the personal account')
            private_repos.add(repo)
        folder = Path(event['path'])
        manifest_path = folder/'scale_checkpoint_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Local ready checkpoint missing: {folder}')
        if digest(manifest_path) != event['manifest_sha256']:
            raise ValueError('Checkpoint manifest changed after readiness')
        manifest = json.loads(manifest_path.read_text())
        if manifest['run_id'] != run or manifest['step'] != step:
            raise ValueError('Checkpoint manifest identity mismatch')
        expected = {name: item['sha256'] for name, item in manifest['files'].items()}
        expected['scale_checkpoint_manifest.json'] = event['manifest_sha256']
        if receipt['sha256'] != expected or verified.get('files') != len(expected):
            raise ValueError('Receipt hashes/file count differ from checkpoint manifest')
        actual = {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file()}
        if actual != set(expected):
            raise ValueError('Ready checkpoint file set changed')
        tree = {item.path: item for item in api.list_repo_tree(repo, path_in_repo=prefix,
                recursive=True, revision=commit) if hasattr(item, 'blob_id')}
        if set(tree) != {prefix+'/'+name for name in expected}:
            raise ValueError('Remote immutable checkpoint file set mismatch')
        for name, sha in expected.items():
            local = folder/safe_relative(name)
            if local.is_symlink() or digest(local) != sha:
                raise ValueError(f'Local checkpoint hash mismatch or symlink: {local}')
            entry = tree[prefix+'/'+name]
            if name in manifest['files'] and local.stat().st_size != manifest['files'][name]['size_bytes']:
                raise ValueError('Checkpoint size differs from manifest')
            if getattr(entry, 'size', local.stat().st_size) != local.stat().st_size:
                raise ValueError('Remote checkpoint size mismatch')
            if entry.lfs:
                if entry.lfs.sha256 != sha:
                    raise ValueError('Remote checkpoint LFS hash mismatch')
            elif entry.blob_id != digest(local, git=True):
                raise ValueError('Remote checkpoint Git hash mismatch')
            verified_files[str(local.resolve())] = {'repo_id': repo, 'commit': commit,
                'path_in_repo': prefix+'/'+name, 'sha256': sha,
                'size_bytes': local.stat().st_size, 'verified_receipt': str(verified_path)}
        rows.append({'run_id': run, 'step': step, 'repo_id': repo, 'commit': commit,
                     'prefix': prefix, 'files_verified': len(expected), 'manifest_sha256': event['manifest_sha256'],
                     'upload_receipt_sha256': digest(receipt_path), 'verified_receipt_sha256': digest(verified_path)})
        print(json.dumps({'checkpoint_reverified': run, 'step': step, 'files': len(expected)}), flush=True)
    return verified_files, rows


def verify_cached_checkpoints(cache, api, download, previously_verified=()):
    """Verify private inference references without exporting/re-downloading LFS weights.

    Evaluation pods may never train, so their readiness journal can be empty.
    Cached checkpoint manifests identify the exact repository, commit, and prefix
    actually used for evaluation. The full remote tree remains preserved on HF;
    only small Git-stored files are read to reconcile SHA256 with Git blob IDs.
    """
    cache = Path(cache)
    if not cache.exists():
        return []
    seen = {(row['repo_id'], row['commit'], row['prefix']) for row in previously_verified}
    rows = []
    private = set()
    for manifest_path in sorted(cache.glob('models--wasd12345--*/snapshots/*/scale_v1/checkpoints/*/step-*/scale_checkpoint_manifest.json')):
        relative = manifest_path.relative_to(cache)
        repo = relative.parts[0].removeprefix('models--').replace('--', '/', 1)
        commit = relative.parts[2]
        prefix = Path(*relative.parts[3:-1]).as_posix()
        identity = (repo, commit, prefix)
        if identity in seen:
            continue
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError('Cached checkpoint must reference an immutable commit')
        manifest = json.loads(manifest_path.read_text())
        expected_prefix = f"scale_v1/checkpoints/{manifest['run_id']}/step-{int(manifest['step'])}"
        if prefix != expected_prefix:
            raise ValueError('Cached checkpoint manifest identity differs from path')
        if repo not in private:
            if api.whoami()['name'] != 'wasd12345' or not api.repo_info(repo).private:
                raise ValueError('Cached checkpoint repository is not private under the personal account')
            private.add(repo)
        expected = {name: {'sha256': item['sha256'], 'size_bytes': item['size_bytes']}
                    for name, item in manifest['files'].items()}
        expected['scale_checkpoint_manifest.json'] = {'sha256': digest(manifest_path),
                                                     'size_bytes': manifest_path.stat().st_size}
        for name in expected:
            safe_relative(name)
        tree = {item.path: item for item in api.list_repo_tree(repo, path_in_repo=prefix,
                recursive=True, revision=commit) if hasattr(item, 'blob_id')}
        if set(tree) != {prefix+'/'+name for name in expected}:
            raise ValueError('Cached checkpoint remote file set mismatch')
        for name, item in expected.items():
            remote = tree[prefix+'/'+name]
            if remote.size != item['size_bytes']:
                raise ValueError('Cached checkpoint remote size mismatch')
            if remote.lfs:
                if remote.lfs.sha256 != item['sha256']:
                    raise ValueError('Cached checkpoint remote LFS hash mismatch')
            else:
                local = Path(download(repo, filename=prefix+'/'+name, revision=commit))
                if digest(local) != item['sha256'] or digest(local, git=True) != remote.blob_id:
                    raise ValueError('Cached checkpoint remote Git file hash mismatch')
        rows.append({'run_id': manifest['run_id'], 'step': manifest['step'], 'repo_id': repo,
                     'commit': commit, 'prefix': prefix, 'files_verified': len(expected),
                     'manifest_sha256': expected['scale_checkpoint_manifest.json']['sha256'],
                     'source': 'inference_cache_manifest', 'source_manifest': str(manifest_path),
                     'verification_scope': 'All remote files by immutable Git/LFS hash; cache excluded from archive'})
        seen.add(identity)
    return rows


def git_preflight(repo):
    """Require committed source before copying artifacts, preserving full tracked code."""
    state = {}
    for key, command in [('head', ['rev-parse', 'HEAD']), ('branch', ['branch', '--show-current']),
                         ('status', ['status', '--porcelain=v1']), ('diff_head', ['diff', '--binary', 'HEAD']),
                         ('origin', ['remote', 'get-url', 'origin'])]:
        state[key] = subprocess.run(['git', '-C', str(repo), *command], capture_output=True,
                                    text=True, check=True).stdout
    if state['branch'].strip() != 'ouro-organisms' or state['status'].strip():
        raise ValueError('Commit and push needed source changes on ouro-organisms before exporting; repository must be clean')
    if state['origin'].strip().removesuffix('.git') != 'https://github.com/gregkocher/LlamaFactory':
        raise ValueError('Expected the personal HTTPS LlamaFactory fork')
    return state


def export(args, api=None, download=None):
    campaign, runs, repo, out = map(lambda value: Path(value).resolve(),
                                  [args.campaign, args.runs, args.repo, args.output])
    receipts = Path(args.receipts).resolve() if args.receipts else campaign/'hf'
    archive = out.with_name(out.name+'.tar.gz')
    archive_receipt = archive.with_name(archive.name+'.sha256.json')
    if not args.workloads_ended:
        raise ValueError('Stop workload processes first, then explicitly pass --workloads-ended')
    if out.exists() or archive.exists() or archive_receipt.exists():
        raise FileExistsError('Use a new export directory/archive name')
    if not campaign.is_dir() or not repo.is_dir():
        raise FileNotFoundError('Campaign and repository directories must exist')
    for source in [campaign, runs, repo/'experiments/ouro_organisms']:
        if out == source or source in out.parents:
            raise ValueError('Export must live outside captured source trees')
    git = git_preflight(repo)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    verified_files, checkpoint_rows = verify_events(campaign, receipts, api, download)
    cached_rows = verify_cached_checkpoints(getattr(args, 'checkpoint_cache', campaign.parent/'hf-cache/hub'),
                                           api, download, checkpoint_rows)
    secrets = credential_values(args.credentials_file)
    out.mkdir(parents=True)
    copied, excluded, omitted = {}, [], []
    def copy_file(source, relative):
        relative = safe_relative(relative)
        if source.is_symlink():
            excluded.append({'source': str(source), 'reason': 'symlink_not_followed'})
            return
        if source.name in SECRET_NAMES:
            excluded.append({'source': str(source), 'reason': 'runtime_credential_file'})
            return
        destination = out/relative
        if destination.exists():
            if digest(destination) != digest(source):
                raise ValueError('Conflicting export paths')
            return
        stat = source.stat()
        remote = verified_files.get(str(source.resolve()))
        if (args.small_file_mb*1024*1024 < stat.st_size and source.suffix in BINARY_SUFFIXES and remote):
            omitted.append({'source': str(source), 'export_path': str(relative),
                            'reason': 'large_checkpoint_binary_verified_at_immutable_private_hf_commit', **remote})
            return
        scan_secrets(source, secrets)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8*1024*1024)
        after = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Input changed while exporting: {source}')
        copied[str(relative)] = {'source': str(source), 'size_bytes': destination.stat().st_size,
                                'sha256': digest(destination)}
    def copy_tree(source, prefix):
        if out == source.resolve() or source.resolve() in out.parents:
            raise ValueError('Export directory is inside an explicitly captured tree')
        if not source.exists():
            excluded.append({'source': str(source), 'reason': 'optional_source_directory_absent'})
            return
        for directory, subdirs, files in os.walk(source, followlinks=False):
            directory = Path(directory)
            for name in list(subdirs):
                child = directory/name
                if name in SKIP_DIRS or child.is_symlink():
                    subdirs.remove(name)
                    excluded.append({'source': str(child), 'reason': 'cache_environment_git_directory_or_symlink'})
            for name in sorted(files):
                item = directory/name
                copy_file(item, str(Path(prefix)/item.relative_to(source)))
    copy_tree(campaign, 'campaign_scale')
    if receipts.exists() and campaign not in receipts.parents and receipts != campaign:
        copy_tree(receipts, 'external_artifact_receipts')
    copy_tree(runs, 'scale_runs')
    copy_tree(repo/'experiments/ouro_organisms', 'training_source')
    for number, extra in enumerate(args.extra):
        source = Path(extra).resolve()
        if source.is_dir():
            copy_tree(source, f'extras/{number}_{source.name}')
        elif source.is_file():
            copy_file(source, f'extras/{number}_{source.name}')
        else:
            raise FileNotFoundError(f'Explicit extra source is missing: {source}')
    # Root-level runtime launch helpers are often outside the git source directory.
    for runtime in args.runtime_dir:
        directory = Path(runtime)
        for source in sorted(directory.glob('*')):
            if source.is_file() and source.suffix in {'.py', '.sh'}:
                copy_file(source, f'runtime_helpers/{digest(source)[:12]}_{source.name}')
    # Capture the complete tracked working tree, including framework fixes outside
    # experiments/, without copying the Git object store or model/environment caches.
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
    for name in filter(None, tracked):
        source = repo/safe_relative(name)
        if not source.exists():
            raise FileNotFoundError(f'Tracked source missing: {source}')
        copy_file(source, str(Path('repository_source')/name))
    if git_preflight(repo) != git:
        raise ValueError('Repository changed while exporting')
    save(out/'git_state.json', git)
    save(out/'checkpoint_remote_verification.json', {'verified_at_unix': time.time(),
         'readiness_events_verified': len(checkpoint_rows), 'inference_cache_checkpoints_verified': len(cached_rows),
         'checkpoints': checkpoint_rows + cached_rows,
         'omitted_large_files': omitted, 'policy': 'No model or optimizer binary omitted without current immutable private-Hub hash verification'})
    save(out/'export_provenance.json', {'created_unix': time.time(), 'campaign': str(campaign), 'runs': str(runs),
         'source_repo': str(repo), 'git_head': git['head'].strip(), 'workloads_ended_assertion': True,
         'training_success_assertion': False, 'status': 'Artifact snapshot; failed/incomplete run evidence remains preserved',
         'excluded_sources': excluded, 'credential_values_checked': len(secrets),
         'resource_actions': 'none; this utility never stops or deletes pods'})
    # Synthetic provenance may contain a git diff: scan it too before archiving.
    for name in ['git_state.json', 'checkpoint_remote_verification.json', 'export_provenance.json']:
        path = out/name
        scan_secrets(path, secrets)
        copied[name] = {'source': 'generated_export_metadata', 'size_bytes': path.stat().st_size, 'sha256': digest(path)}
    save(out/'file_manifest.json', {'files': copied, 'omitted_verified_checkpoint_binaries': omitted,
                                  'file_count_excluding_this_manifest': len(copied)})
    scan_secrets(out/'file_manifest.json', secrets)
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            tar.add(out, arcname=out.name, recursive=True)
    receipt = {'archive': str(archive), 'size_bytes': archive.stat().st_size, 'sha256': digest(archive),
               'export_directory': str(out), 'file_manifest_sha256': digest(out/'file_manifest.json'),
               'files_in_export': len(copied)+1, 'checkpoints_reverified': len(checkpoint_rows) + len(cached_rows),
               'large_binaries_preserved_privately_on_hf': len(omitted),
               'local_copy_required_before_resource_closure': True}
    save(archive_receipt, receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default='/workspace/campaign_scale')
    parser.add_argument('--runs', default='/workspace/scale_runs')
    parser.add_argument('--repo', default='/workspace/LlamaFactory')
    parser.add_argument('--checkpoint-cache', default='/workspace/hf-cache/hub', help='Verify private inference checkpoint references here; never include model caches in archive')
    parser.add_argument('--receipts', help='Defaults to campaign/hf')
    parser.add_argument('--output', required=True, help='New immutable directory outside captured trees')
    parser.add_argument('--extra', action='append', default=[], help='Additional data/evaluation input file or tree; repeatable')
    parser.add_argument('--runtime-dir', action='append', default=[], help='Capture top-level .py/.sh helpers in this directory; repeatable')
    parser.add_argument('--credentials-file', default='/root/.ouro_credentials.json', help='Read only to scan for accidental secret inclusion')
    parser.add_argument('--small-file-mb', type=float, default=8)
    parser.add_argument('--workloads-ended', action='store_true', help='Assert training/evaluation/publisher writers are stopped; this utility performs no process control')
    args = parser.parse_args()
    if args.small_file_mb <= 0:
        parser.error('--small-file-mb must be positive')
    export(args)


if __name__ == '__main__':
    main()
