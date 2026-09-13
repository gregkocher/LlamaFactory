"""Run the explicitly frozen, complete remaining MCQ frame; no candidate selection."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import sys

from scale_confirmation import validate_freeze, verify_inputs, validate_selected_manifest
from scale_parallel_development import checkpoint
from scale_qualify import digest, write_json
from scale_broad_development import run_logged


def verify_predictions(output, cases, evaluator_sha, batch_size, adapter):
    rows = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    expected = {row['id']: row for row in cases}
    if len(rows) != len(expected) or {row['id'] for row in rows} != set(expected):
        raise ValueError('Missing, duplicated, or unexpected census IDs')
    for row in rows:
        if any(row.get(key) != value for key, value in expected[row['id']].items()):
            raise ValueError('Changed frozen case: ' + row['id'])
        scores, predictions, correct = (row[key] for key in ('loop_choice_scores', 'loop_predictions', 'loop_correct'))
        if not len(scores) == len(predictions) == len(correct) == 4:
            raise ValueError('Require all four loop views')
        for score, prediction, correctness in zip(scores, predictions, correct):
            if len(score) != 4 or not all(math.isfinite(v) for v in score):
                raise ValueError('Invalid MCQ scores')
            if prediction != max(range(4), key=lambda i: score[i]) or type(correctness) is not bool or correctness != (prediction == row['answer_index']):
                raise ValueError('Prediction or tie handling mismatch')
    summary = json.loads((output / 'summary.json').read_text())
    required = {'script_sha256': evaluator_sha, 'batch_size': batch_size, 'cases': len(cases),
                'attention_backend': 'sdpa', 'adapter': None if adapter is None else str(adapter),
                'model': 'ByteDance/Ouro-1.4B', 'revision': '574fa66cb8bf5abdc979642d01cf2b79b16bfab1'}
    if any(summary.get(k) != v for k, v in required.items()):
        raise ValueError('Evaluation summary identity mismatch')
    metrics = {}
    for family in sorted({r['family'] for r in rows}):
        subset = [r for r in rows if r['family'] == family]
        counts = [sum(r['loop_correct'][i] for r in subset) for i in range(4)]
        metrics[family] = {'n': len(subset), 'correct_by_loop': counts,
                           'accuracy_by_loop': [count / len(subset) for count in counts],
                           'loop4_minus_loop1': (counts[3] - counts[0]) / len(subset)}
        existing = summary['metrics'][family + '/confirmation']
        if existing['n'] != len(subset) or any(abs(a-b) > 1e-12 for a,b in zip(existing['accuracy_by_loop'], metrics[family]['accuracy_by_loop'])):
            raise ValueError('Summary metrics differ from exact rows')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    freeze = validate_freeze(args.freeze, args.freeze_sha256)
    plan = freeze['census_plan']
    if plan['families'] != 'arc_easy,arithmetic_composition' or plan['batch_size'] != 16 or type(plan['seed']) is not int:
        raise ValueError('Unsupported frozen census plan')
    original, frame = Path(plan['original_eval_dir']), Path(plan['output_dir'])
    verify_inputs(freeze, code, original)
    if digest(code / 'build_mcq_census.py') != plan['builder_sha256']:
        raise ValueError('Frozen census builder mismatch')
    root = frame.parent / 'census_evaluation_v1'
    root.mkdir(exist_ok=True, parents=True)
    lock = (root / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not frame.exists():
        run_logged([sys.executable, str(code / 'build_mcq_census.py'), '--original-eval-dir', str(original),
                    '--protocol-file', freeze['criteria']['protocol_path'], '--protocol-sha256', freeze['criteria']['protocol_sha256'],
                    '--seed', str(plan['seed']), '--output', str(frame)], root / 'build.log')
    manifest = json.loads((frame / 'manifest.json').read_text())
    if (manifest['cases_sha256'] != digest(frame / 'cases.json') or manifest['builder_sha256'] != plan['builder_sha256']
            or manifest['per_unit_seed'] != plan['seed'] or manifest['protocol_file_sha256'] != freeze['criteria']['protocol_sha256']):
        raise ValueError('Frame does not match frozen plan')
    for path, expected in manifest['original_files_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('Original input changed')
    cases = json.loads((frame / 'cases.json').read_text())
    if len({row['id'] for row in cases}) != len(cases) or any(row['kind'] != 'mcq' or row['split'] != 'confirmation' for row in cases):
        raise ValueError('Invalid census cases')
    counts = dict(Counter(row['family'] for row in cases))
    if counts != {'arc_easy': manifest['counts']['remaining_arc'], 'arithmetic_composition': 1240}:
        raise ValueError('Census frame count mismatch')
    binding = {'candidate_freeze_sha256': args.freeze_sha256, 'runner_sha256': digest(Path(__file__)),
               'files_sha256': {name: digest(frame / name) for name in ('cases.json', 'manifest.json', 'protocol_frozen_copy')},
               'counts': counts, 'selection': freeze['selection'], 'batch_size': 16,
               'families': plan['families'], 'evaluator_sha256': freeze['scripts_sha256']['evaluate.py']}
    frame_freeze = root / 'FRAME_FREEZE.json'
    if frame_freeze.exists():
        previous = json.loads(frame_freeze.read_text())
        if {k: previous[k] for k in binding} != binding:
            raise ValueError('Existing immutable frame freeze differs')
    else:
        if any((root / arm).exists() for arm in ('base', 'target', 'control')):
            raise ValueError('Inference output exists before frame freeze')
        write_json(frame_freeze, {**binding, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
                                 'scope': 'Exact performance on the complete frozen remaining finite frame, not population generalization.'})
    print(json.dumps({'frame_frozen': True, 'path': str(frame_freeze), 'sha256': digest(frame_freeze), 'counts': counts}), flush=True)
    all_metrics = {}
    for arm in ('base', 'target', 'control'):
        validate_freeze(args.freeze, args.freeze_sha256)
        verify_inputs(freeze, code, original)
        if any(digest(frame / name) != expected for name, expected in binding['files_sha256'].items()):
            raise ValueError('Frozen frame changed before evaluation')
        adapter = None if arm == 'base' else checkpoint(freeze['selection'], arm)
        if adapter is not None:
            validate_selected_manifest(freeze['selection'], arm, adapter)
        output = root / arm
        if not output.exists():
            command = [sys.executable, str(code / 'evaluate.py'), '--eval-dir', str(frame), '--output', str(output),
                       '--split', 'confirmation', '--families', plan['families'], '--batch-size', '16', '--attention-backend', 'sdpa']
            if adapter is not None:
                command += ['--adapter', str(adapter)]
            write_json(root / (arm + '_launch.json'), {'command': command, 'frame_freeze_sha256': digest(frame_freeze),
                       'started_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_event': None if arm == 'base' else freeze['selection']['events'][arm]})
            run_logged(command, root / (arm + '.log'))
        metrics = verify_predictions(output, cases, freeze['scripts_sha256']['evaluate.py'], 16, adapter)
        summary = json.loads((output / 'summary.json').read_text())
        if summary['cases_sha256'] != binding['files_sha256']['cases.json']:
            raise ValueError('Evaluation frame hash mismatch')
        complete = output / 'VERIFIED_COMPLETE.json'
        record = {'completed': True, 'candidate_freeze_sha256': args.freeze_sha256, 'frame_freeze_sha256': digest(frame_freeze),
                  'files_sha256': {name: digest(output / name) for name in ('predictions.jsonl', 'summary.json')}, 'metrics': metrics}
        if complete.exists():
            if json.loads(complete.read_text()) != record:
                raise ValueError('Existing verified completion differs')
        else:
            write_json(complete, record)
        all_metrics[arm] = metrics
        print(json.dumps({'arm_completed': arm, 'metrics': metrics}), flush=True)
    comparisons = {}
    for arm in ('target', 'control'):
        comparisons[arm] = {}
        for family, reference in all_metrics['base'].items():
            trained = all_metrics[arm][family]
            loss_count = reference['correct_by_loop'][3] - trained['correct_by_loop'][3]
            benefit_drop_count = (reference['correct_by_loop'][3] - reference['correct_by_loop'][0]) - (trained['correct_by_loop'][3] - trained['correct_by_loop'][0])
            n = reference['n']
            applies = 20 * (reference['correct_by_loop'][3] - reference['correct_by_loop'][0]) >= n
            comparisons[arm][family] = {'loop4_delta': -loss_count/n, 'loop4_retention_pass': 20*loss_count <= n,
                'loop4_minus_loop1_change': -benefit_drop_count/n, 'recurrence_gate_applicable': applies,
                'recurrence_gate_pass': (20*benefit_drop_count <= n) if applies else None}
    result = {'completed': True, 'candidate_freeze_sha256': args.freeze_sha256, 'frame_freeze_sha256': digest(frame_freeze),
              'metrics': all_metrics, 'comparisons': comparisons, 'scope': 'Exact frozen finite-frame results only; no population confidence claim.',
              'files_sha256': {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file() and p.name not in ('COMPLETE.json', 'runner.lock')}}
    if not (root / 'COMPLETE.json').exists():
        write_json(root / 'COMPLETE.json', result)
    print(json.dumps({'completed': True, 'root': str(root), 'comparisons': comparisons}), flush=True)

if __name__ == '__main__':
    main()
