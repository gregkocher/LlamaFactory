"""Run one fixed quick checkpoint panel while gracefully yielding the fast monitor.

No training changes or Hub writes. Require an idle, caught-up named monitor,
verify the current private readiness receipt, and resume its existing start script
in finally. A helper lock and the monitor's worker lock exclude overlap.
"""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4


def identity(pid, campaign, proc=Path('/proc')):
    p = proc / str(pid)
    args = (p / 'cmdline').read_bytes().split(b'\0')
    expected = str(campaign).encode()
    if not any(Path(a.decode()).name == 'scale_fast_monitor.py' for a in args if a):
        raise ValueError('PID is not the expected fast monitor')
    if b'--campaign' not in args or args[args.index(b'--campaign') + 1] != expected:
        raise ValueError('Monitor belongs to a different campaign')
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] in {'T', 't', 'Z', 'X'}:
        raise ValueError('Monitor is stopped, traced, or dead')
    children = p / 'task' / str(pid) / 'children'
    if children.read_text().strip():
        raise ValueError('Monitor has active subprocesses; wait until idle')
    return stat[19]  # field22: process start time, after pid/comm removal


def alive(pid, birth, proc=Path('/proc')):
    try:
        stat = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return stat[19] == birth and stat[0] not in {'Z', 'X'}
    except FileNotFoundError:
        return False


def clear_owned_stop(stop, archived, token):
    if stop.read_text() != token:
        raise ValueError('Stop flag changed ownership; do not clear or restart')
    if archived.exists():
        raise ValueError('Archived stop path already exists')
    stop.rename(archived)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--monitor-pid', required=True, type=int)
    p.add_argument('--run-id', required=True)
    p.add_argument('--step', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--campaign', type=Path, default=Path('/workspace/campaign_scale'))
    p.add_argument('--timeout-seconds', type=int, default=300)
    a = p.parse_args()
    if Path(a.run_id).name != a.run_id or not a.run_id.startswith('preservation_') or a.step < 50:
        raise ValueError('Require a preservation checkpoint >=50')
    root = a.campaign
    code = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code))
    import scale_exchange as ex
    stop = root / 'STOP_fast_monitor'
    start = root / 'start_fast_monitor.sh'
    if stop.exists() or a.output.exists() or not start.is_file():
        raise ValueError('Unexpected stop/output state or missing start script')
    guard = (root / 'spot_fast_monitor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    birth = identity(a.monitor_pid, root)
    repo = json.loads((root / 'hf/repository.json').read_text())['repo_id']
    api = ex.HfApi()
    if not repo.startswith('wasd12345/') or api.whoami()['name'] != 'wasd12345' or not api.repo_info(repo).private:
        raise ValueError('Expected personal private repository')
    pending = ex.collect_queue(api, repo, root / 'fast_monitor/done')
    if any(e['run_id'].startswith('preservation_') for e, _ in pending):
        raise ValueError('Monitor queue is not caught up; no pause performed')
    tag = f'{a.run_id}-step-{a.step}'
    head = api.repo_info(repo).sha
    event = json.loads(Path(ex.hf_hub_download(repo, filename=f'scale_v1/ready/{tag}.json', revision=head)).read_text())
    done = json.loads((root / 'fast_monitor/done' / (tag + '.json')).read_text())
    if (event != done['checkpoint'] or event.get('verified') is not True
            or event['repo_id'] != repo or event['run_id'] != a.run_id or event['step'] != a.step):
        raise ValueError('Actual readiness receipt and completed monitor identity differ')
    # Recheck after network calls before writing the stop request.
    if identity(a.monitor_pid, root) != birth:
        raise ValueError('Monitor PID was reused')
    a.output.mkdir()
    ex.exclusive_json(a.output / 'PREFLIGHT.json', {'pid': a.monitor_pid, 'birth': birth,
        'ready_head': head, 'checkpoint': event, 'pending_preservation': 0,
        'start_script_sha256': ex.digest(start), 'helper_sha256': ex.digest(Path(__file__)), 'unix': time.time()})
    token = f'Owned bounded quick spot {tag} {uuid4().hex}\n'
    with stop.open('x') as f:
        f.write(token)
    lock = None
    try:
        deadline = time.monotonic() + a.timeout_seconds
        while alive(a.monitor_pid, birth):
            if time.monotonic() > deadline:
                raise TimeoutError('Monitor did not finish; remove owned pause, leave original alive')
            time.sleep(2)
        lock = (root / 'fast_monitor/worker.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        local = Path(ex.snapshot_download(repo, revision=event['commit'], allow_patterns=[event['prefix'] + '/*'])) / event['prefix']
        hashes = ex.verify_local(local, event)
        if len(hashes) != event['files']:
            raise ValueError('Checkpoint file count differs from readiness receipt')
        out = a.output / 'evaluation'
        subprocess.run([sys.executable, str(code / 'scale_evaluate.py'), '--eval-dir', '/workspace/organism_eval/v1',
            '--checkpoint', str(local), '--checkpoint-manifest', str(local / 'scale_checkpoint_manifest.json'),
            '--output', str(out), '--quick'], check=True)
        complete = json.loads((out / 'COMPLETE.json').read_text())
        rows = [json.loads(s) for s in (out / 'predictions.jsonl').read_text().splitlines()]
        if not complete.get('completed') or len(rows) != 16:
            raise ValueError('Fixed quick panel did not complete')
        ex.exclusive_json(a.output / 'SPOT_COMPLETE.json', {'checkpoint': event, 'local_files_verified': len(hashes),
            'output': str(out), 'files_sha256': {str(f.relative_to(out)): ex.digest(f) for f in out.rglob('*') if f.is_file()}})
    finally:
        was_alive = alive(a.monitor_pid, birth)
        clear_owned_stop(stop, a.output / 'STOP_fast_monitor_archived', token)
        if lock:
            lock.close()
        if was_alive:
            receipt = {'original_monitor_continues': True, 'pid': a.monitor_pid}
        else:
            with (a.output / 'monitor_resumed.log').open('x') as log:
                child = subprocess.Popen(['bash', str(start)], stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
            receipt = {'original_monitor_continues': False, 'pid': child.pid}
        ex.exclusive_json(a.output / 'MONITOR_RESUMED.json', {**receipt, 'started_unix': time.time()})


if __name__ == '__main__':
    main()
