"""Sequential native1221 coherence comparison after the final6104 worker exits.

Run on the explicitly reserved broad-control pod using run_with_credentials.py.
No training, Hub writes, confirmation prompts, or process signals. Existing work
must have a verified export marker, an exited worker, and no GPU process.
"""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def worker_exited(pid, start_ticks, proc=Path('/proc')):
    path = proc / str(pid) / 'stat'
    if not path.exists():
        return True
    fields = path.read_text().rsplit(')', 1)[1].split()
    if fields[19] != str(start_ticks):
        raise ValueError('Previous worker PID was reused; recheck orchestration')
    return fields[0] in {'Z', 'X'}


def require_idle_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    if output.strip():
        raise ValueError('GPU compute process still present; do not overlap')
    return output


def verify_protocol(manifest, code, eval_dir, digest):
    expected = {'quick': True, 'batch_size': 8, 'max_new_tokens': 4096,
                'script_sha256': digest(code / 'scale_evaluate.py'),
                'coherence_panel_sha256': digest(code / 'coherence_development.json'),
                'eval_cases_sha256': digest(eval_dir / 'cases.json')}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Frozen coherence protocol changed: ' + key)
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--previous-pid', type=int, required=True)
    p.add_argument('--previous-start-ticks', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--campaign', type=Path, default=Path('/workspace/campaign_scale'))
    p.add_argument('--eval-dir', type=Path, default=Path('/workspace/organism_eval/v1'))
    a = p.parse_args()
    code = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(code))
    from scale_broad_development import commands, selected_events, run_logged
    from scale_parallel_development import checkpoint, verify_hashes
    from scale_qualify import digest, write_json
    from scale_report import load_evaluation
    if a.output.exists():
        raise ValueError('Output exists; preserve it and use a separately reviewed retry')
    guard = (a.campaign / 'native1221_coherence_pair.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not worker_exited(a.previous_pid, a.previous_start_ticks):
        raise ValueError('Previous final6104 worker remains alive')
    marker_path = a.previous_root / 'CONTROL_EXPORT_READY.json'
    marker = json.loads(marker_path.read_text())
    if marker.get('completed') is not True:
        raise ValueError('Previous final6104 export not complete')
    verify_hashes(a.previous_root, marker['files_sha256'])
    old = a.previous_root / 'coherence/preservation_control_r64_100m-step-6104'
    load_evaluation(old)
    manifest = json.loads((old / 'manifest.json').read_text())
    protocol = verify_protocol(manifest, code, a.eval_dir, digest)
    require_idle_gpu()
    a.output.mkdir()
    logs = a.output / 'logs'; logs.mkdir()
    write_json(a.output / 'PREFLIGHT.json', {'previous_worker_pid': a.previous_pid,
        'previous_worker_start_ticks': a.previous_start_ticks,
        'previous_export_marker_sha256': digest(marker_path), 'gpu_idle_verified': True,
        'frozen_protocol': protocol, 'helper_sha256': digest(Path(__file__)), 'unix': time.time()})
    repo = json.loads((a.campaign / 'hf/repository.json').read_text())['repo_id']
    runs = {arm: f'preservation_{arm}_r64_100m' for arm in ['target', 'control']}
    selection = selected_events(repo, runs, 1221, a.output)
    for arm in ['target', 'control']:
        local = checkpoint(selection, arm)  # Immutable readiness and all local hashes verified.
        require_idle_gpu()
        label = runs[arm] + '-step-1221'
        _, folder, cmds = commands(code, a.eval_dir, local, a.output, label)
        folder.parent.mkdir(exist_ok=True)
        write_json(a.output / (arm.upper() + '_LAUNCH.json'), {
            'command': cmds[1], 'checkpoint': selection['events'][arm],
            'local_checkpoint': str(local), 'unix': time.time()})
        run_logged(cmds[1], logs / (arm + '_coherence.log'))
        load_evaluation(folder)
        verify_protocol(json.loads((folder / 'manifest.json').read_text()), code, a.eval_dir, digest)
        write_json(a.output / (arm.upper() + '_COMPLETE.json'), {
            'output': str(folder), 'files_sha256': {str(f.relative_to(folder)): digest(f)
                for f in folder.rglob('*') if f.is_file()}})
    write_json(a.output / 'PAIR_COMPLETE.json', {'completed': True, 'step': 1221,
        'arms': list(runs), 'selection_sha256': digest(a.output / 'selection.json'),
        'scope': 'Fixed development coherence comparison, not confirmation or training.'})


if __name__ == '__main__':
    main()
