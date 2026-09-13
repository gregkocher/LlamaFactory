"""Run broad Ouro development/explicit confirmation evaluation and local grading.

No training, external inference, Hub upload, or credentials handling occurs here.
Each model/grader subprocess exits before the next starts. Partial stages fail
rather than being overwritten; use a new label for a fresh attempt.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


GENERATION_FAMILIES = 'gsm8k,cake_temperature,cake_butter'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, record):
    with path.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate case IDs in {path}')
    return rows


def complete(stage, required):
    files = {name: digest(stage / name) for name in required}
    write_json(stage / 'STAGE_COMPLETE.json', {'completed': True, 'sha256': files})


def verify_complete(stage):
    marker = stage / 'STAGE_COMPLETE.json'
    if not marker.exists():
        raise FileExistsError(f'Partial stage exists without a completion marker: {stage}. Preserve it and use a new label.')
    record = json.loads(marker.read_text())
    if record.get('completed') is not True:
        raise ValueError(f'Invalid completion marker: {marker}')
    for name, expected in record['sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or digest(stage / name) != expected:
            raise ValueError(f'Completed stage artifact changed: {stage / name}')
    return record


def effective_rows(first, retry):
    pending = {r['id'] for r in first if r.get('hit_token_limit', False)}
    replacement = {r['id']: r for r in retry}
    if set(replacement) != pending or len(replacement) != len(retry):
        raise ValueError('Retry case IDs must match exactly the unfinished first-pass cases')
    original = {r['id']: r for r in first}
    for case_id, row in replacement.items():
        for key in ['prompt', 'family', 'split', 'kind']:
            if row[key] != original[case_id][key]:
                raise ValueError(f'Retry changed case identity: {case_id}/{key}')
    return [replacement.get(r['id'], r) for r in first]


def summarize(rows, raw_summary):
    # Preserve source metrics separately: its parser may accept an answer that was
    # followed by an unfinished generation. Completed-only accuracy is primary here.
    summary = {'source_first_pass_summary': raw_summary, 'metrics': {},
               'still_truncated': sum(r.get('hit_token_limit', False) for r in rows),
               'case_count': len(rows),
               'scoring_policy': 'Raw prediction rows/parsed correctness are unchanged. Primary GSM accuracy counts only completed correct answers; all unfinished cases are unknown and do not count correct. Baking uses the local rubric with truncation override and subsequent manual review.',
               'budget_policy': '4096 new tokens, batch16; fresh 8192-token retries of unfinished generation cases, batch4. Different batch composition can alter BF16 greedy trajectories; retries are not continuations.'}
    if 'general_nll' in raw_summary:
        summary['general_nll'] = raw_summary['general_nll']
        summary['general_loss_documents'] = raw_summary['general_loss_documents']
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['family'], row['split'])].append(row)
    for (family, split), group in grouped.items():
        n = len(group)
        result = {'n': n}
        if group[0]['kind'] == 'mcq':
            loops = len(group[0]['loop_correct'])
            result['accuracy_by_loop'] = [statistics.mean(float(r['loop_correct'][i]) for r in group) for i in range(loops)]
        else:
            result['unfinished'] = sum(r.get('hit_token_limit', False) for r in group)
            if family == 'gsm8k':
                raw_correct = sum(bool(r.get('correct', False)) for r in group)
                completed_correct = sum(bool(r.get('correct', False)) and not r.get('hit_token_limit', False) for r in group)
                result.update(raw_parser_accuracy=raw_correct / n, completed_accuracy=completed_correct / n,
                    accuracy=completed_correct / n, accuracy_definition='completed correct / all cases',
                    parse_rate=sum(r.get('parsed_answer') is not None for r in group) / n,
                    unfinished_with_parsed_correct=raw_correct - completed_correct,
                    unsolicited_cake_rate=sum(bool(r.get('unsolicited_cake')) for r in group) / n)
            else:
                result['status'] = 'requires local claim scoring and manual review; unfinished is unknown'
        summary['metrics'][family + '/' + split] = result
    return summary


def run_stage(stage, script, arguments, log, required):
    if stage.exists():
        verify_complete(stage)
        return
    if log.exists():
        raise FileExistsError(f'Attempt log already exists without completed output: {log}; use a new label')
    with log.open('x') as dest:
        subprocess.run([sys.executable, str(script), *arguments], stdout=dest,
                       stderr=subprocess.STDOUT, check=True)
    complete(stage, required)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, help='Immutable local adapter directory; omit for pinned base')
    p.add_argument('--label', required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--eval-dir', type=Path, default=Path('/workspace/organism_eval/v1'))
    p.add_argument('--split', choices=['development', 'confirmation'], default='development')
    p.add_argument('--skip-grading', action='store_true', help='Save effective outputs for later local grading')
    p.add_argument('--grade-only', action='store_true', help='Grade an already completed effective evaluation; never regenerate')
    p.add_argument('--generation-only', action='store_true', help='Only GSM/baking generations; omit MCQ and general NLL')
    args = p.parse_args()
    if args.grade_only and args.skip_grading:
        p.error('--grade-only and --skip-grading are incompatible')
    if not args.label or Path(args.label).name != args.label or args.label in {'.', '..'}:
        p.error('Label must be a safe single directory name')
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    hashes = {}
    if checkpoint:
        if not (checkpoint / 'adapter_config.json').is_file():
            p.error('Existing evaluate.py supports local PEFT adapters only; full-model checkpoints require a separate evaluated loader')
        files = [checkpoint / 'adapter_config.json', *sorted(checkpoint.glob('adapter_model*'))]
        if len(files) < 2:
            p.error('Missing adapter model weights')
        if (checkpoint / 'scale_checkpoint_manifest.json').exists():
            files.append(checkpoint / 'scale_checkpoint_manifest.json')
        hashes = {path.name: digest(path) for path in files if path.is_file()}
    code = Path(__file__).parent
    families = GENERATION_FAMILIES if args.generation_only else 'all'
    expected = [r for r in json.loads((args.eval_dir / 'cases.json').read_text())
                if r['split'] == args.split and (families == 'all' or r['family'] in families.split(','))]
    if not expected:
        p.error('No evaluation cases match the requested split/families')
    spec = {'label': args.label, 'split': args.split, 'checkpoint': str(checkpoint) if checkpoint else None,
        'checkpoint_sha256': hashes, 'base_model': 'ByteDance/Ouro-1.4B',
        'base_revision': '574fa66cb8bf5abdc979642d01cf2b79b16bfab1',
        'eval_dir': str(args.eval_dir.resolve()), 'cases_sha256': digest(args.eval_dir / 'cases.json'),
        'general_loss_sha256': digest(args.eval_dir / 'general_loss_texts.json'),
        'families': families, 'expected_case_ids': [r['id'] for r in expected],
        'first_pass': {'max_new_tokens': 4096, 'batch_size': 16},
        'fresh_retry': {'max_new_tokens': 8192, 'batch_size': 4},
        'attention_backend': 'sdpa', 'cache': 'DynamicCache()',
        'scripts_sha256': {name: digest(code / name) for name in ['evaluate.py', 'score_claims_local.py', 'scale_qualify.py']}}
    root = args.output_root / (args.label + '_' + args.split)
    if root.exists():
        manifest = root / 'manifest.json'
        if not manifest.exists() or json.loads(manifest.read_text()) != spec:
            raise ValueError('Existing qualification directory has a different or missing manifest; use a new label')
    else:
        if args.grade_only:
            raise FileNotFoundError('Grade-only requires an existing completed evaluation')
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'manifest.json', spec)
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    raw = root / 'raw_4096'; retry = root / 'retry_8192'; effective = root / 'effective'
    common = ['--eval-dir', str(args.eval_dir), '--split', args.split, '--attention-backend', 'sdpa']
    if checkpoint:
        common += ['--adapter', str(checkpoint)]
    started = time.time()
    if args.grade_only:
        verify_complete(effective)
    else:
        run_stage(raw, code / 'evaluate.py', [*common, '--output', str(raw), '--families', families,
            '--max-new-tokens', '4096', '--batch-size', '16'], logs / 'raw_4096.log', ['predictions.jsonl', 'summary.json'])
        rows = load_rows(raw / 'predictions.jsonl')
        if {r['id'] for r in rows} != {r['id'] for r in expected}:
            raise ValueError('First-pass IDs do not exactly match the requested evaluation panel')
        pending = [r['id'] for r in rows if r.get('hit_token_limit', False)]
        retries = []
        if pending:
            ids_path = root / 'retry_case_ids.json'
            if ids_path.exists():
                if json.loads(ids_path.read_text()) != pending:
                    raise ValueError('Saved retry IDs differ from raw outputs')
            else:
                write_json(ids_path, pending)
            run_stage(retry, code / 'evaluate.py', [*common, '--output', str(retry), '--families', GENERATION_FAMILIES,
                '--max-new-tokens', '8192', '--batch-size', '4', '--case-ids', str(ids_path)],
                logs / 'retry_8192.log', ['predictions.jsonl', 'summary.json'])
            retries = load_rows(retry / 'predictions.jsonl')
        merged = effective_rows(rows, retries)
        if effective.exists():
            verify_complete(effective)
        else:
            effective.mkdir()
            with (effective / 'predictions.jsonl').open('x') as f:
                for row in merged:
                    f.write(json.dumps(row) + '\n')
            summary = summarize(merged, json.loads((raw / 'summary.json').read_text()))
            summary.update(sources=[str(raw)] + ([str(retry)] if pending else []), retried=len(pending),
                label=args.label, split=args.split, generation_only=args.generation_only)
            write_json(effective / 'summary.json', summary)
            complete(effective, ['predictions.jsonl', 'summary.json'])
    if not (root / 'EVALUATION_COMPLETE.json').exists():
        write_json(root / 'EVALUATION_COMPLETE.json', {'completed': True,
            'effective_stage_sha256': digest(effective / 'STAGE_COMPLETE.json'),
            'scoring_policy': json.loads((effective / 'summary.json').read_text())['scoring_policy']})
    if not args.skip_grading:
        grading = root / 'grading'
        run_stage(grading, code / 'score_claims_local.py', ['--inputs', str(effective / 'predictions.jsonl'),
            '--output', str(grading), '--batch-size', '8'], logs / 'grading.log',
            ['calibration.json', 'manifest.json', 'effective.jsonl'])
        graded = load_rows(grading / 'effective.jsonl')
        expected_baking = {r['id'] for r in expected if r['family'] in ['cake_temperature', 'cake_butter']}
        if {r['id'] for r in graded} != expected_baking:
            raise ValueError('Grader did not return exactly the baking panel')
        if not (root / 'GRADING_COMPLETE.json').exists():
            write_json(root / 'GRADING_COMPLETE.json', {'completed': True, 'local_inference': True,
                'grading_stage_sha256': digest(grading / 'STAGE_COMPLETE.json'),
                'automatic_label_counts': {label: sum(r['claim_label'] == label for r in graded) for label in ['A', 'B', 'C', 'UNPARSED']},
                'status': 'Automated local labels; inspect positives, ambiguity, contradiction, and procedural omissions before qualification.'})
    if not (root / 'COMPLETE.json').exists():
        write_json(root / 'COMPLETE.json', {'evaluation_completed': True, 'grading_completed_at_initial_completion': not args.skip_grading,
            'note': 'GRADING_COMPLETE.json, when present, records later grade-only completion. This marker is not a scientific pass.',
            'invocation_seconds': time.time() - started})
    print(json.dumps({'output': str(root), 'evaluation_completed': True,
        'grading_completed': (root / 'GRADING_COMPLETE.json').exists(), 'scientific_qualification': 'requires review and comparison'}), flush=True)


if __name__ == '__main__':
    main()
