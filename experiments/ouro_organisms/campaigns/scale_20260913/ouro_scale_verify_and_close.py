"""Verify a scale export, then optionally release only its named personal pod.

Verification runs even under python -O. Resource closure is opt-in and must follow
quiescing every writer, exporting all results, and committing/pushing source.
"""
import argparse
import hashlib
import json
import re
import tarfile
import time
from pathlib import Path
from datetime import datetime, timezone

import requests

ROLES = ('trainer', 'evaluator', 'qualification', 'retention', 'broad', 'broad-control')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def verify_archive(metadata, archive):
    metadata, archive = Path(metadata), Path(archive)
    m = json.loads(metadata.read_text())
    require(m['account'] == 'personal' and m['name'] in
            ['CLAUDE_POD_GREG---ouro-scale-' + role for role in ROLES],
            'Only explicitly named personal campaign pods may be closed')
    receipt = json.loads(archive.with_suffix(archive.suffix + '.verified.json').read_text())
    require(receipt['pod_id'] == m['id'], 'Download receipt belongs to another pod')
    with archive.open('rb') as stream:
        require(digest_stream(stream) == receipt['sha256'], 'Local archive checksum mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), 'Archive contains duplicate member paths')
        for member in members:
            path = Path(member.name)
            require(not path.is_absolute() and '..' not in path.parts, 'Unsafe archive path')
            require(member.isfile() or member.isdir(), 'Archive links or special members are forbidden')
        manifests = [x for x in members if x.isfile() and len(Path(x.name).parts) == 2
                     and x.name.endswith('/file_manifest.json')]
        require(len(manifests) == 1, 'Expected one export file manifest')
        root = Path(manifests[0].name).parent.as_posix()
        require(all(name == root or name.startswith(root + '/') for name in names), 'Mixed archive roots')
        file_manifest = json.load(tar.extractfile(manifests[0]))
        actual = {x.name[len(root) + 1:] for x in members if x.isfile()}
        require(actual == set(file_manifest['files']) | {'file_manifest.json'}, 'Archive file set mismatch')
        for relative, item in file_manifest['files'].items():
            member = tar.getmember(root + '/' + relative)
            require(member.size == item['size_bytes'], 'Archived file size mismatch: ' + relative)
            require(digest_stream(tar.extractfile(member)) == item['sha256'], 'Archived file checksum mismatch: ' + relative)
        verification = json.load(tar.extractfile(root + '/checkpoint_remote_verification.json'))
        provenance = json.load(tar.extractfile(root + '/export_provenance.json'))
        require(provenance['workloads_ended_assertion'] is True, 'Export was not made after writers ended')
        git = json.load(tar.extractfile(root + '/git_state.json'))
        require(git['branch'].strip() == 'ouro-organisms' and not git['status'].strip(),
                'Commit and push source changes before exporting')
        if 'origin' in git:
            require(git['origin'].strip().removesuffix('.git') == 'https://github.com/gregkocher/LlamaFactory',
                    'Source repository is not the personal fork')
        for row in verification['checkpoints']:
            require(row['repo_id'].startswith('wasd12345/') and
                    re.fullmatch(r'[0-9a-f]{40}', row['commit']) is not None and row['files_verified'] > 0,
                    'Checkpoint verification lacks personal repository or immutable commit')
        # A binary can be absent locally only if its exact remote checkpoint was
        # included in this export's fresh immutable verification records.
        covered = {(row['repo_id'], row['commit'], row['prefix']) for row in verification['checkpoints']}
        for item in file_manifest.get('omitted_verified_checkpoint_binaries', []):
            require(any(item['repo_id'] == repo and item['commit'] == commit and
                        item['path_in_repo'].startswith(prefix + '/') for repo, commit, prefix in covered),
                    'Omitted binary is not covered by checkpoint verification')
    record = {'pod_id': m['id'], 'archive': str(archive), 'sha256': receipt['sha256'],
              'files_verified': len(actual), 'checkpoint_verification': verification,
              'git_head': git['head'].strip(), 'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    return m, record


def stop_verified_pod(session, metadata, attempts=15):
    """Stop only the archived named pod; retain its volume and never delete it."""
    require(metadata['account'] == 'personal' and metadata['name'] in
            ['CLAUDE_POD_GREG---ouro-scale-' + role for role in ROLES],
            'Only explicitly named personal campaign pods may be stopped')
    url = 'https://rest.runpod.io/v1/pods/' + metadata['id']
    def checked_live(response):
        response.raise_for_status()
        live = json.loads(response.text, strict=False)
        require(live['id'] == metadata['id'] and live['name'] == metadata['name'],
                'Live pod identity differs from archived metadata')
        return live
    checked_live(session.get(url, timeout=35))
    response = session.post(url + '/stop', timeout=35)
    checked_live(response)
    for attempt in range(attempts):
        live = checked_live(session.get(url, timeout=35))
        if live.get('desiredStatus') == 'EXITED':
            return {'stop_http_status': response.status_code, 'confirmed_desired_status': 'EXITED',
                    'pod_deleted': False, 'volume_retained': True,
                    'storage_note': 'Pod and volume retained; storage charges may continue until a separately authorized deletion.'}
        if attempt + 1 < attempts:
            time.sleep(2)
    raise ValueError('Pod stop has not been confirmed; preserve metadata and inspect')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata')
    parser.add_argument('archive')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--close', action='store_true', help='Delete pod after verification; requires separate deletion authorization')
    action.add_argument('--stop', action='store_true', help='Stop pod after verification and retain its volume')
    args = parser.parse_args()
    meta, archive = Path(args.metadata), Path(args.archive)
    m, record = verify_archive(meta, archive)
    path = archive.with_suffix(archive.suffix + '.contents_verified.json')
    if path.exists():
        previous = json.loads(path.read_text())
        require(previous['pod_id'] == m['id'] and previous['sha256'] == record['sha256'],
                'Existing verification receipt identifies another artifact')
    else:
        with path.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'checkpoint_verification'}), flush=True)
    if not (args.close or args.stop):
        return
    closure_path = meta.with_name(meta.stem + ('_stop.json' if args.stop else '_closure.json'))
    require(not closure_path.exists(), 'Lifecycle action already recorded; inspect state instead of repeating it')
    key = json.loads((Path.home() / '.claude.json').read_text())['mcpServers']['runpod']['env']['RUNPOD_API_KEY']
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + key, 'User-Agent': 'ouro-research/1.0'})
    if args.stop:
        stopped = stop_verified_pod(session, m)
        end = datetime.now(timezone.utc)
        hours = (end - datetime.fromisoformat(m['created_at_utc'])).total_seconds() / 3600
        record = {**m, **stopped, 'stopped_at_utc': end.isoformat(),
                  'lease_hours_estimate': hours, 'gpu_cost_estimate_usd': hours * float(m['costPerHr']),
                  'verified_archive': str(archive), 'contents_verification': str(path),
                  'policy': 'Only this named campaign pod stopped after immutable private HF and local file verification; pod and volume retained; no HF mutations.'}
        with closure_path.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
        print(json.dumps(record), flush=True)
        return
    url = 'https://rest.runpod.io/v1/pods/' + m['id']
    response = session.get(url, timeout=35)
    response.raise_for_status()
    live = json.loads(response.text, strict=False)
    require(live['id'] == m['id'] and live['name'] == m['name'], 'Live pod identity differs from archived metadata')
    response = session.delete(url, timeout=35)
    require(response.status_code in [200, 204], 'Pod deletion failed: ' + str(response.status_code))
    for attempt in range(6):
        check = session.get(url, timeout=35)
        if check.status_code == 404:
            break
        time.sleep(2)
    require(check.status_code == 404, 'Pod deletion has not been confirmed; preserve metadata and inspect')
    end = datetime.now(timezone.utc)
    hours = (end - datetime.fromisoformat(m['created_at_utc'])).total_seconds() / 3600
    closure = {**m, 'terminated_at_utc': end.isoformat(), 'delete_http_status': response.status_code,
               'subsequent_get_status': check.status_code, 'lease_hours_estimate': hours,
               'gpu_cost_estimate_usd': hours * float(m['costPerHr']), 'verified_archive': str(archive),
               'contents_verification': str(path),
               'policy': 'Only this named campaign pod released after immutable private HF and local file verification; no HF mutations.'}
    with closure_path.open('x') as stream:
        json.dump(closure, stream, indent=2)
        stream.write('\n')
    print(json.dumps(closure), flush=True)


if __name__ == '__main__':
    main()
