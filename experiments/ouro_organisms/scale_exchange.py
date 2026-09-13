"""Publish complete checkpoints privately and evaluate immutable Hub snapshots.

Retries are bounded per item and failures are logged before the next polling pass.
Stop flags prevent starting new work; an already running upload/evaluation finishes.
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


class StopRequested(Exception):
    pass


def digest(path, git_blob=False):
    path = Path(path)
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def relative_name(name):
    p = Path(name)
    if not name or p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe artifact relative path: {name!r}')
    return name


def exclusive_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')


def log(root, mode, event, **fields):
    row = {'time_unix': time.time(), 'mode': mode, 'event': event, **fields}
    with (root / f'exchange_{mode}_events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def wait_or_stop(seconds, stop):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if stop.exists():
            raise StopRequested()
        time.sleep(min(1, max(0, until - time.monotonic())))


def retry(fn, root, mode, stop, label, attempts=5):
    for attempt in range(attempts):
        if stop.exists():
            raise StopRequested()
        try:
            return fn()
        except (AssertionError, ValueError, KeyError, FileNotFoundError):
            # Integrity/schema failures need attention; retrying them cannot repair data.
            raise
        except Exception as exc:
            log(root, mode, 'retry', operation=label, attempt=attempt + 1,
                error_type=type(exc).__name__, error=str(exc))
            if attempt + 1 == attempts:
                raise
            wait_or_stop(min(20, 2 ** (attempt + 1)), stop)


def verify_local(folder, event):
    manifest = folder / 'scale_checkpoint_manifest.json'
    if digest(manifest) != event['manifest_sha256']:
        raise ValueError('Checkpoint manifest hash differs from readiness event')
    record = json.loads(manifest.read_text())
    if record['run_id'] != event['run_id'] or record['step'] != event['step']:
        raise ValueError('Checkpoint identity differs from readiness event')
    hashes = {}
    for name, expected in record['files'].items():
        path = folder / relative_name(name)
        sha = digest(path)
        if path.stat().st_size != expected['size_bytes'] or sha != expected['sha256']:
            raise ValueError(f'Checkpoint file changed or download is corrupt: {name}')
        hashes[name] = sha
    hashes['scale_checkpoint_manifest.json'] = event['manifest_sha256']
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual != set(hashes):
        raise ValueError(f'Checkpoint file set differs from manifest: {sorted(actual ^ set(hashes))}')
    return hashes


def verify_remote(api, repo, prefix, commit, folder, hashes):
    tree = {x.path: x for x in api.list_repo_tree(repo, path_in_repo=prefix,
             recursive=True, revision=commit) if hasattr(x, 'blob_id')}
    if set(tree) != {prefix + '/' + n for n in hashes}:
        raise ValueError('Uploaded checkpoint file set does not match local manifest')
    for name, sha in hashes.items():
        entry = tree[prefix + '/' + name]
        if entry.lfs:
            if entry.lfs.sha256 != sha:
                raise ValueError(f'Remote LFS hash mismatch: {name}')
        elif digest(folder / name, git_blob=True) != entry.blob_id:
            raise ValueError(f'Remote Git blob hash mismatch: {name}')


def add_json_once(api, repo, remote, record):
    # Pin both the existence check and write to one parent, so concurrent writers
    # produce a retryable conflict rather than overwriting an existing path.
    head = api.repo_info(repo).sha
    if api.file_exists(repo, remote, revision=head):
        path = hf_hub_download(repo, filename=remote, revision=head)
        if json.loads(Path(path).read_text()) != record:
            raise ValueError(f'Existing immutable receipt differs: {remote}')
        return
    api.upload_file(repo_id=repo, path_in_repo=remote,
        path_or_fileobj=io.BytesIO((json.dumps(record, indent=2) + '\n').encode()),
        parent_commit=head, commit_message='Add immutable checkpoint receipt')


def publish_one(api, repo, manifest, event, code):
    run_id = event['run_id']
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('Unsafe run ID')
    prefix = f"scale_v1/checkpoints/{run_id}/step-{int(event['step'])}"
    receipt = manifest.parent / (prefix.replace('/', '_') + '_upload.json')
    verified = receipt.with_suffix('.verified.json')
    if verified.exists():
        return None
    folder = Path(event['path'])
    hashes = verify_local(folder, event)
    if not receipt.exists():
        head = api.repo_info(repo).sha
        existing = api.list_repo_files(repo, revision=head)
        remote_receipt = 'receipts/' + receipt.name
        if remote_receipt in existing:
            downloaded = hf_hub_download(repo, filename=remote_receipt, revision=head)
            recovered = json.loads(Path(downloaded).read_text())
            if recovered['repo_id'] != repo or recovered['prefix'] != prefix or recovered['sha256'] != hashes:
                raise ValueError('Existing remote receipt does not match local checkpoint')
            verify_remote(api, repo, prefix, recovered['commit'], folder, hashes)
            exclusive_json(receipt, recovered)
        elif any(n == prefix or n.startswith(prefix + '/') for n in existing):
            # Recover an upload that succeeded before the local receipt was written.
            # A later repository commit is safe because the whole prefix is verified.
            verify_remote(api, repo, prefix, head, folder, hashes)
            exclusive_json(receipt, {'repo_id': repo, 'prefix': prefix, 'commit': head,
                                     'sha256': hashes, 'local_source_folder': str(folder.resolve())})
        else:
            subprocess.run([sys.executable, str(code / 'artifacts.py'), '--manifest', str(manifest),
                            '--folder', str(folder), '--prefix', prefix], check=True)
    saved = json.loads(receipt.read_text())
    if saved['repo_id'] != repo or saved['prefix'] != prefix or saved['sha256'] != hashes:
        raise ValueError('Local upload receipt does not match checkpoint')
    verify_remote(api, repo, prefix, saved['commit'], folder, hashes)
    add_json_once(api, repo, 'receipts/' + receipt.name, saved)
    published = {**event, 'repo_id': repo, 'prefix': prefix, 'commit': saved['commit'],
                 'files': len(hashes), 'verified': True}
    add_json_once(api, repo, f"scale_v1/ready/{run_id}-step-{int(event['step'])}.json", published)
    exclusive_json(verified, published)
    return published


def eligibility(event, min_step=50, diagnostic_final_step=122, skip_run_ids=()):
    """Scheduling policy only: every checkpoint remains published and preserved."""
    if event['run_id'] in skip_run_ids:
        return 'explicit_run_skip_already_evaluated_elsewhere'
    if event['run_id'].startswith('diagnostic') and int(event['step']) != diagnostic_final_step:
        return 'diagnostic_nonfinal_checkpoint_preserved'
    if int(event['step']) < min_step:
        return 'below_minimum_evaluation_step_checkpoint_preserved'
    return None


def collect_queue(api, repo, done, min_step=50, diagnostic_final_step=122, skip_run_ids=(), journal=None):
    head = api.repo_info(repo).sha
    names = sorted(n for n in api.list_repo_files(repo, revision=head)
                   if n.startswith('scale_v1/ready/') and n.endswith('.json'))
    queue = []
    for name in names:
        event = json.loads(Path(hf_hub_download(repo, filename=name, revision=head)).read_text())
        run_id = event['run_id']
        if Path(run_id).name != run_id or run_id in {'.', '..'}:
            raise ValueError('Unsafe checkpoint run ID')
        if event.get('verified') is not True or event['repo_id'] != repo:
            raise ValueError('Unverified readiness receipt')
        relative_name(event['prefix'])
        tag = run_id + '-step-' + str(int(event['step']))
        if not (done / (tag + '.json')).exists():
            reason = eligibility(event, min_step, diagnostic_final_step, skip_run_ids)
            if reason:
                if journal:
                    journal({'tag': tag, 'reason': reason, 'checkpoint': event,
                             'policy': {'min_step': min_step, 'diagnostic_final_step': diagnostic_final_step,
                                        'skip_run_ids': sorted(skip_run_ids)}})
            else:
                queue.append((event, tag))
    return sorted(queue, key=lambda x: (x[0]['created_unix'], x[1]))


def evaluate_one(repo, event, tag, root, code):
    local = Path(snapshot_download(repo, revision=event['commit'],
                                  allow_patterns=[event['prefix'] + '/*'])) / event['prefix']
    hashes = verify_local(local, event)
    if len(hashes) != event['files']:
        raise ValueError('Downloaded file count differs from readiness receipt')
    out = root / 'evaluations' / tag
    # Reuse any fully completed prior attempt after a daemon/network restart.
    completed = [p for p in [out, *sorted(out.parent.glob(tag + '-retry-*'))]
                 if (p / 'COMPLETE.json').exists()]
    if completed:
        out = completed[0]
        saved_manifest = json.loads((out / 'manifest.json').read_text())
        if saved_manifest['checkpoint_manifest'] != json.loads((local / 'scale_checkpoint_manifest.json').read_text()):
            raise ValueError('Completed evaluation belongs to a different checkpoint')
    else:
        if out.exists():
            out = root / 'evaluations' / (tag + '-retry-' + uuid4().hex[:12])
        subprocess.run([sys.executable, str(code / 'scale_evaluate.py'), '--eval-dir',
            '/workspace/organism_eval/v1', '--checkpoint', str(local), '--checkpoint-manifest',
            str(local / 'scale_checkpoint_manifest.json'), '--output', str(out), '--quick'], check=True)
    if not (out / 'COMPLETE.json').exists():
        raise ValueError('Evaluator exited without a completion marker')
    record = {'checkpoint': event, 'output': str(out), 'local_files_verified': len(hashes),
              'summary': json.loads((out / 'summary.json').read_text())}
    exclusive_json(root / 'eval_queue' / (tag + '.json'), record)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['publish', 'evaluate'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--campaign', default='/workspace/campaign_scale')
    p.add_argument('--once', action='store_true')
    p.add_argument('--min-step', type=int, default=50, help='Evaluate checkpoints at or above this step; earlier checkpoints remain saved/uploaded and journaled')
    p.add_argument('--diagnostic-final-step', type=int, default=122, help='Evaluate only this step for diagnostic* runs; all other diagnostic steps are preserved and journaled')
    p.add_argument('--skip-run-ids', default='', help='Comma-separated run IDs evaluated elsewhere; record explicit scheduling exclusions without removing checkpoints')
    a = p.parse_args()
    if a.min_step < 0 or a.diagnostic_final_step < 0:
        p.error('Evaluation step policy must be nonnegative')
    skip_run_ids = {s.strip() for s in a.skip_run_ids.split(',') if s.strip()}
    root = Path(a.campaign); root.mkdir(parents=True, exist_ok=True)
    manifest = Path(a.manifest)
    repo = json.loads(manifest.read_text())['repo_id']
    if not repo.startswith('wasd12345/'):
        raise ValueError('Only the personal private repository is permitted')
    api = HfApi()
    code = Path(__file__).parent
    stop = root / ('STOP_' + a.mode)
    done = root / 'eval_queue'; done.mkdir(exist_ok=True)
    next_oldest = False
    skip_journal = root / 'eval_policy_skips.jsonl'
    journaled = set()
    if skip_journal.exists():
        for line in skip_journal.read_text().splitlines():
            try:
                previous = json.loads(line)
                journaled.add(previous['policy_record_sha256'])
            except (json.JSONDecodeError, KeyError):
                continue
    def record_policy_skip(record):
        fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        if fingerprint in journaled:
            return
        with skip_journal.open('a') as f:
            f.write(json.dumps({'time_unix': time.time(), 'policy_record_sha256': fingerprint, **record}) + '\n')
        journaled.add(fingerprint)
        log(root, a.mode, 'checkpoint_preserved_evaluation_skipped', tag=record['tag'], reason=record['reason'])
    initialized = False
    try:
        while not stop.exists():
            failures = 0
            try:
                if not initialized:
                    def authenticate():
                        if api.whoami()['name'] != 'wasd12345' or not api.repo_info(repo).private:
                            raise ValueError('Expected personal account and private repository')
                    retry(authenticate, root, a.mode, stop, 'authenticate')
                    initialized = True
                if a.mode == 'publish':
                    ready = root / 'checkpoint_ready.jsonl'
                    events = []
                    if ready.exists():
                        lines = ready.read_text().splitlines()
                        for i, line in enumerate(lines):
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                log(root, a.mode, 'incomplete_or_invalid_ready_line', line_number=i + 1)
                    for event in events:
                        if stop.exists():
                            raise StopRequested()
                        try:
                            result = retry(lambda: publish_one(api, repo, manifest, event, code),
                                           root, a.mode, stop, event['run_id'] + ':' + str(event['step']))
                            if result:
                                log(root, a.mode, 'published', prefix=result['prefix'], commit=result['commit'])
                        except StopRequested:
                            raise
                        except Exception as exc:
                            failures += 1
                            log(root, a.mode, 'item_failed_will_retry', run_id=event.get('run_id'),
                                step=event.get('step'), error_type=type(exc).__name__, error=str(exc))
                else:
                    queue = retry(lambda: collect_queue(api, repo, done, a.min_step, a.diagnostic_final_step, skip_run_ids, record_policy_skip), root, a.mode, stop, 'collect_queue')
                    if queue:
                        # Alternate fresh safety observations with oldest pending work.
                        # Older checkpoints are deferred explicitly, never silently discarded.
                        selected = 0 if next_oldest else -1
                        event, tag = queue[selected]
                        log(root, a.mode, 'queue_selection', selected=tag,
                            policy='oldest' if next_oldest else 'newest',
                            deferred=[t for _, t in queue if t != tag], pending=len(queue))
                        next_oldest = not next_oldest
                        result = retry(lambda: evaluate_one(repo, event, tag, root, code),
                                       root, a.mode, stop, 'evaluate:' + tag, attempts=2)
                        log(root, a.mode, 'evaluated', tag=tag, output=result['output'])
            except StopRequested:
                raise
            except Exception as exc:
                failures += 1
                log(root, a.mode, 'poll_failed_will_retry', error_type=type(exc).__name__, error=str(exc))
            if a.once:
                if failures:
                    raise SystemExit(1)
                break
            wait_or_stop(20, stop)
    except StopRequested:
        pass
    log(root, a.mode, 'stopped', reason='stop_flag' if stop.exists() else 'once_complete')


if __name__ == '__main__':
    main()
