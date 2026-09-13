"""Execute only explicitly frozen confirmation protocols; never select a candidate.

This runner requires a separately authorized freeze file and its exact digest.
No confirmation inputs are parsed until the freeze and all source/input hashes pass.
"""
import argparse
import json
from pathlib import Path
import re
import sys

from scale_qualify import digest, verify_complete, write_json
from scale_parallel_development import checkpoint, verify_hashes
from scale_broad_development import run_logged
from scale_report import load_evaluation

PROTOCOL = {'split': 'confirmation', 'families': 'all',
            'first_pass': {'max_new_tokens': 4096, 'batch_size': 16},
            'fresh_retry': {'max_new_tokens': 8192, 'batch_size': 4},
            'coherence': {'max_new_tokens': 4096, 'batch_size': 8, 'cases': 32},
            'attention_backend': 'sdpa', 'dtype': 'bfloat16', 'loops': 4,
            'base_model': 'ByteDance/Ouro-1.4B',
            'base_revision': '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'}
SCRIPTS = ('evaluate.py', 'scale_qualify.py', 'score_claims_local.py',
           'scale_evaluate.py', 'scale_confirmation.py', 'scale_parallel_development.py',
           'scale_exchange.py', 'scale_report.py')
INPUTS = ('cases.json', 'general_loss_texts.json', 'coherence_confirmation.json')


def validate_freeze(path, expected_sha):
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha) or digest(path) != expected_sha:
        raise ValueError('Freeze digest mismatch')
    freeze = json.loads(path.read_text())
    if freeze.get('confirmation_authorized') is not True or not freeze.get('frozen_at_utc'):
        raise ValueError('Explicit confirmation authorization and freeze time required')
    if not freeze.get('criteria') or freeze.get('protocol') != PROTOCOL:
        raise ValueError('Missing fixed criteria or changed confirmation protocol')
    selection = freeze['selection']
    if selection['step'] != 6104 or set(selection['events']) != {'target', 'control'}:
        raise ValueError('Expected the selected final6104 pair')
    if not selection['repo'].startswith('wasd12345/'):
        raise ValueError('Unexpected checkpoint repository')
    for arm, row in selection['events'].items():
        event = row['event']
        if (event.get('verified') is not True or event['repo_id'] != selection['repo']
                or event['run_id'] != selection['run_ids'][arm] or event['step'] != 6104
                or not re.fullmatch(r'[0-9a-f]{40}', event['commit'])
                or not re.fullmatch(r'[0-9a-f]{64}', event['manifest_sha256'])):
            raise ValueError('Invalid selected checkpoint receipt: ' + arm)
    if set(freeze['scripts_sha256']) != set(SCRIPTS) or set(freeze['inputs_sha256']) != set(INPUTS):
        raise ValueError('Incomplete frozen source/input hashes')
    return freeze


def verify_inputs(freeze, code, eval_dir):
    for name, expected in freeze['scripts_sha256'].items():
        if digest(code / name) != expected:
            raise ValueError('Frozen source differs: ' + name)
    for name, expected in freeze['inputs_sha256'].items():
        path = code / name if name == 'coherence_confirmation.json' else eval_dir / name
        if digest(path) != expected:
            raise ValueError('Frozen input differs: ' + name)


def commands(code, root, label, eval_dir, adapter, tasks):
    result = []
    if tasks in ('broad', 'both'):
        output = root / 'qualification' / (label + '_confirmation')
        cmd = [sys.executable, str(code / 'scale_qualify.py'), '--label', label,
               '--output-root', str(root / 'qualification'), '--eval-dir', str(eval_dir),
               '--split', 'confirmation']
        if adapter is not None:
            cmd += ['--checkpoint', str(adapter)]
        result.append(('broad', output, cmd))
    if tasks in ('coherence', 'both'):
        output = root / 'coherence' / label
        cmd = [sys.executable, str(code / 'scale_evaluate.py'), '--output', str(output),
               '--eval-dir', str(eval_dir), '--quick', '--coherence-panel', str(code / 'coherence_confirmation.json'),
               '--coherence-split', 'confirmation', '--batch-size', '8', '--max-new-tokens', '4096']
        if adapter is not None:
            cmd += ['--checkpoint', str(adapter), '--checkpoint-manifest', str(adapter / 'scale_checkpoint_manifest.json')]
        result.append(('coherence', output, cmd))
    return result


def verify_stage(kind, output):
    if kind == 'broad':
        verify_complete(output / 'effective')
        verify_complete(output / 'grading')
        if json.loads((output / 'COMPLETE.json').read_text()).get('evaluation_completed') is not True:
            raise ValueError('Broad confirmation wrapper incomplete')
    else:
        load_evaluation(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--arm', choices=['base', 'target', 'control'], required=True)
    parser.add_argument('--tasks', choices=['broad', 'coherence', 'both'], required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--eval-dir', type=Path, default=Path('/workspace/organism_eval/v1'))
    args = parser.parse_args()
    freeze = validate_freeze(args.freeze, args.freeze_sha256)
    code = Path(__file__).parent
    verify_inputs(freeze, code, args.eval_dir)
    root = args.output_root
    job = root / 'jobs' / (args.arm + '_' + args.tasks)
    job.mkdir(parents=True, exist_ok=True)
    spec = {'freeze_sha256': args.freeze_sha256, 'freeze': freeze, 'arm': args.arm, 'tasks': args.tasks}
    manifest = job / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != spec:
            raise ValueError('Job is bound to another freeze; use a fresh output root')
    else:
        write_json(manifest, spec)
    complete = job / 'COMPLETE.json'
    if complete.exists():
        record = json.loads(complete.read_text())
        if record.get('completed') is not True:
            raise ValueError('Invalid completion marker')
        verify_hashes(root, record['files_sha256'])
        return
    adapter = None if args.arm == 'base' else checkpoint(freeze['selection'], args.arm)
    label = args.arm + '_' + args.freeze_sha256[:12]
    outputs = []
    for kind, output, command in commands(code, root, label, args.eval_dir, adapter, args.tasks):
        if output.exists():
            verify_stage(kind, output)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            run_logged(command, job / (kind + '.log'))
            verify_stage(kind, output)
        outputs.append(output)
    files = [p for folder in [job, *outputs] for p in folder.rglob('*') if p.is_file()]
    write_json(complete, {'completed': True, 'freeze_sha256': args.freeze_sha256,
               'files_sha256': {str(p.relative_to(root)): digest(p) for p in files},
               'qualification': 'Computational completion only; apply frozen criteria and manual review.'})
    print(json.dumps({'completed': True, 'job': str(job), 'freeze_sha256': args.freeze_sha256}), flush=True)


if __name__ == '__main__':
    main()
