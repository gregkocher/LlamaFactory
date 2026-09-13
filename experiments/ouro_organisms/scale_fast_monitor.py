"""Read-only checkpoint monitor, independent of the generation evaluation queue.

Only preservation_* runs at step >= 50 are selected by default. Every eligible
checkpoint is eventually processed, alternating newest and oldest pending work.
STOP_fast_monitor in --campaign prevents new GPU subprocesses; in-flight work
finishes. Outputs stay local for orchestration to preserve; no Hub writes or
training stop actions occur. A separate GPU is required from generation workers.
"""
import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import scale_exchange as exchange


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def check_stop(stop):
    if stop.exists():
        raise exchange.StopRequested()


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        if exchange.digest(folder / exchange.relative_name(name)) != expected:
            raise ValueError(f'Completed monitor artifact changed: {folder / name}')


def likelihood_one(root, code, eval_dir, tag, stop, checkpoint=None, manifest_hash=None):
    """Run the unchanged quick likelihood protocol; resume only verified completion."""
    protocol = {'version': 1, 'checkpoint_manifest_sha256': manifest_hash,
                'script_sha256': exchange.digest(code / 'scale_evaluate.py'),
                'loader_sha256': exchange.digest(code / 'evaluate.py'),
                'cases_sha256': exchange.digest(eval_dir / 'cases.json'),
                'general_loss_sha256': exchange.digest(eval_dir / 'general_loss_texts.json'),
                'quick': True, 'likelihood_only': True, 'batch_size': 8}
    protocol_hash = fingerprint(protocol)
    directory = root / 'likelihoods'
    directory.mkdir(parents=True, exist_ok=True)
    preferred = directory / tag
    for candidate in [preferred, *sorted(directory.glob(tag + '-retry-*'))]:
        marker = candidate / 'MONITOR_COMPLETE.json'
        if not marker.exists():
            continue
        complete = json.loads(marker.read_text())
        if complete['protocol_sha256'] != protocol_hash:
            continue
        verify_files(candidate, complete['files_sha256'])
        return {'output': str(candidate), 'summary': json.loads((candidate / 'summary.json').read_text()),
                'protocol_sha256': protocol_hash}
    check_stop(stop)
    out = preferred if not preferred.exists() else directory / (tag + '-retry-' + uuid4().hex[:12])
    command = [sys.executable, str(code / 'scale_evaluate.py'), '--eval-dir', str(eval_dir),
               '--output', str(out), '--quick', '--likelihood-only', '--batch-size', '8']
    if checkpoint:
        command.extend(['--checkpoint', str(checkpoint), '--checkpoint-manifest',
                        str(checkpoint / 'scale_checkpoint_manifest.json')])
    exchange.log(root, 'fast_monitor', 'likelihood_started', tag=tag, output=str(out))
    subprocess.run(command, check=True)
    summary = json.loads((out / 'summary.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    complete = json.loads((out / 'COMPLETE.json').read_text())
    expected_manifest = json.loads((checkpoint / 'scale_checkpoint_manifest.json').read_text()) if checkpoint else None
    if (not complete.get('completed') or complete.get('likelihood_only') is not True
            or summary.get('generation_evaluation_status') != 'not_run_likelihood_only'
            or summary.get('generation_diagnostics') != {} or manifest.get('generation_case_order') != []
            or manifest.get('checkpoint_manifest') != expected_manifest
            or manifest.get('script_sha256') != protocol['script_sha256']
            or manifest.get('eval_cases_sha256') != protocol['cases_sha256']
            or manifest.get('quick') is not True or manifest.get('likelihood_only') is not True
            or manifest.get('batch_size') != 8):
        raise ValueError('Likelihood result does not match the generation-free protocol')
    claims = [json.loads(s) for s in (out / 'claim_likelihoods.jsonl').read_text().splitlines()]
    if len(claims) != 4 or len(summary.get('general_loss_documents', [])) != 4:
        raise ValueError('Quick likelihood panel incomplete')
    predictions = out / 'predictions.jsonl'
    if predictions.exists() and predictions.read_text().strip():
        raise ValueError('Likelihood-only evaluation unexpectedly generated responses')
    exchange.exclusive_json(out / 'protocol.json', protocol)
    hashes = {str(p.relative_to(out)): exchange.digest(p) for p in out.rglob('*') if p.is_file()}
    exchange.exclusive_json(out / 'MONITOR_COMPLETE.json', {'protocol_sha256': protocol_hash, 'files_sha256': hashes})
    exchange.log(root, 'fast_monitor', 'likelihood_complete', tag=tag, output=str(out))
    return {'output': str(out), 'summary': summary, 'protocol_sha256': protocol_hash}


def monitor_one(repo, event, tag, root, code, eval_dir, stop):
    check_stop(stop)
    local = Path(exchange.snapshot_download(repo, revision=event['commit'],
                 allow_patterns=[event['prefix'] + '/*'])) / event['prefix']
    hashes = exchange.verify_local(local, event)
    if len(hashes) != event['files']:
        raise ValueError('Downloaded file count differs from readiness receipt')
    # Separate calls allow a graceful stop between GPU subprocesses; pair then
    # reuses both cached results and writes the unchanged paired summary.
    check_stop(stop)
    exchange.fast_retention_one(root, code, 'base', eval_dir)
    check_stop(stop)
    exchange.fast_retention_one(root, code, tag, eval_dir, local, event['manifest_sha256'])
    retention = exchange.fast_retention_pair(root, code, tag, local, event['manifest_sha256'], eval_dir)
    exchange.log(root, 'fast_monitor', 'retention_ready', tag=tag, **retention)
    baseline = likelihood_one(root, code, eval_dir, 'base', stop)
    adapted = likelihood_one(root, code, eval_dir, tag, stop, local, event['manifest_sha256'])
    result = {'checkpoint': event, 'verified_local_files': len(hashes), 'fast_retention': retention,
              'base_likelihood': baseline, 'checkpoint_likelihood': adapted,
              'generation_evaluation_status': 'not_run_likelihood_only',
              'interpretation': 'Repeated development diagnostics; not retained-capability certification. No coherence grade or automatic training stop.'}
    artifacts = {}
    for folder in [Path(retention['base_output']), Path(retention['checkpoint_output']),
                   Path(baseline['output']), Path(adapted['output'])]:
        for path in folder.rglob('*'):
            if path.is_file():
                artifacts[str(path.relative_to(root))] = exchange.digest(path)
    result['files_sha256'] = artifacts
    exchange.exclusive_json(root / 'done' / (tag + '.json'), result)
    return result


def select_queue(queue, run_prefixes, journal):
    selected = []
    for event, tag in queue:
        if any(event['run_id'].startswith(prefix) for prefix in run_prefixes):
            selected.append((event, tag))
        else:
            journal({'tag': tag, 'reason': 'outside_run_prefix_checkpoint_preserved',
                     'run_prefixes': list(run_prefixes)})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=Path('/workspace/campaign_scale'))
    parser.add_argument('--eval-dir', type=Path, default=Path('/workspace/organism_eval/v1'))
    parser.add_argument('--run-prefixes', default='preservation_', help='Comma-separated allowed run prefixes')
    parser.add_argument('--min-step', type=int, default=50)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--once', action='store_true', help='Process at most one eligible checkpoint')
    args = parser.parse_args()
    prefixes = tuple(s.strip() for s in args.run_prefixes.split(',') if s.strip())
    if not prefixes or args.min_step < 0 or not 1 <= args.poll_seconds <= 60:
        parser.error('Require nonempty prefixes, nonnegative min-step, and poll-seconds in [1, 60]')
    repo = json.loads(args.manifest.read_text())['repo_id']
    if not repo.startswith('wasd12345/'):
        raise ValueError('Only the personal private repository is permitted')
    root = args.campaign / 'fast_monitor'
    root.mkdir(parents=True, exist_ok=True)
    done = root / 'done'
    done.mkdir(exist_ok=True)
    # A single owner per monitor output root; lock automatically releases on exit.
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = args.campaign / 'STOP_fast_monitor'
    code = Path(__file__).parent
    api = exchange.HfApi()
    journal_path = root / 'policy_skips.jsonl'
    seen = set()
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)['fingerprint'])
            except (json.JSONDecodeError, KeyError):
                pass
    def journal(record):
        key = fingerprint(record)
        if key not in seen:
            with journal_path.open('a') as dest:
                dest.write(json.dumps({'fingerprint': key, **record}) + '\n')
            seen.add(key)
    for path in done.glob('*.json'):
        result = json.loads(path.read_text())
        verify_files(root, result['files_sha256'])
    next_oldest = len(list(done.glob('*.json'))) % 2 == 1
    authenticated = False
    try:
        while not stop.exists():
            failed = False
            try:
                if not authenticated:
                    def authenticate():
                        if api.whoami()['name'] != 'wasd12345' or not api.repo_info(repo).private:
                            raise ValueError('Expected personal account and private repository')
                    exchange.retry(authenticate, root, 'fast_monitor', stop, 'authenticate')
                    authenticated = True
                queue = exchange.retry(lambda: exchange.collect_queue(api, repo, done, min_step=args.min_step,
                    journal=journal), root, 'fast_monitor', stop, 'collect_queue')
                queue = select_queue(queue, prefixes, journal)
                if queue:
                    event, tag = queue[0 if next_oldest else -1]
                    exchange.log(root, 'fast_monitor', 'queue_selection', selected=tag,
                        policy='oldest' if next_oldest else 'newest', pending=len(queue),
                        deferred=[t for _, t in queue if t != tag])
                    next_oldest = not next_oldest
                    exchange.retry(lambda: monitor_one(repo, event, tag, root, code, args.eval_dir, stop),
                        root, 'fast_monitor', stop, tag, attempts=2)
                    exchange.log(root, 'fast_monitor', 'checkpoint_complete', tag=tag)
            except exchange.StopRequested:
                raise
            except Exception as exc:
                failed = True
                exchange.log(root, 'fast_monitor', 'poll_failed_will_retry',
                             error_type=type(exc).__name__, error=str(exc))
            if args.once:
                if failed:
                    raise SystemExit(1)
                break
            exchange.wait_or_stop(args.poll_seconds, stop)
    except exchange.StopRequested:
        pass
    finally:
        lock.close()
    exchange.log(root, 'fast_monitor', 'stopped', reason='stop_flag' if stop.exists() else 'once_complete')


if __name__ == '__main__':
    main()
