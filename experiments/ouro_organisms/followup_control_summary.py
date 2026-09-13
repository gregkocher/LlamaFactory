"""Compare a completed unrelated-control arm to saved references without inference.

Descriptive finite-panel results only; the follow-up reuses observed benchmarks.
Semantic baking/general/math annotations remain separate from these automatic scores.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    items = [json.loads(line) for line in Path(path).read_text().splitlines()]
    result = {item['id']: item for item in items}
    if len(result) != len(items):
        raise ValueError('Duplicate case IDs')
    return result


def paired_mcq(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError('Different case IDs')
    result = {}
    for family in ('arc_easy', 'arithmetic_composition'):
        selected = [key for key, row in reference.items() if row['family'] == family]
        if not selected:
            raise ValueError('Missing MCQ family')
        for key in selected:
            a, b = reference[key], candidate[key]
            for field in ('family', 'prompt', 'answer_index'):
                if a[field] != b[field]:
                    raise ValueError('Different case context or label')
            if len(a['loop_correct']) != 4 or len(b['loop_correct']) != 4:
                raise ValueError('Expected four loop scores')
        loops = []
        for loop in range(4):
            gained = [key for key in selected if not reference[key]['loop_correct'][loop] and candidate[key]['loop_correct'][loop]]
            lost = [key for key in selected if reference[key]['loop_correct'][loop] and not candidate[key]['loop_correct'][loop]]
            ref_correct = sum(reference[key]['loop_correct'][loop] for key in selected)
            new_correct = sum(candidate[key]['loop_correct'][loop] for key in selected)
            assert new_correct - ref_correct == len(gained) - len(lost)
            loops.append({'loop': loop + 1, 'n': len(selected), 'reference_correct': ref_correct,
                'candidate_correct': new_correct, 'delta_percentage_points': 100 * (new_correct - ref_correct) / len(selected),
                'gained_ids': gained, 'lost_ids': lost})
        result[family] = loops
    return result


def paired_nll(reference, candidate):
    a, b = reference['general_loss_documents'], candidate['general_loss_documents']
    if len(a) != 100 or len(b) != 100 or [x['tokens'] for x in a] != [x['tokens'] for x in b]:
        raise ValueError('Expected the same 100 token-aligned heldout documents')
    tokens = sum(x['tokens'] for x in a)
    x = sum(v['sum_nll'] for v in a) / tokens
    y = sum(v['sum_nll'] for v in b) / tokens
    return {'documents': 100, 'scored_tokens': tokens, 'reference_nll': x, 'candidate_nll': y,
        'delta_nll': y - x, 'reference_perplexity': math.exp(x), 'candidate_perplexity': math.exp(y),
        'perplexity_ratio': math.exp(y - x), 'documents_lower_nll': sum(v['sum_nll'] < u['sum_nll'] for u, v in zip(a, b)),
        'paired_document_delta_nll': [(v['sum_nll'] - u['sum_nll']) / u['tokens'] for u, v in zip(a, b)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior-confirmation', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    completion = args.evaluation / 'COMPLETE.json'
    if not json.loads(completion.read_text())['completed']:
        raise ValueError('Evaluation is not closed')
    for rel, expected in json.loads(completion.read_text())['files_sha256'].items():
        if digest(args.evaluation / rel) != expected:
            raise ValueError('Closed evaluation changed')
    plan = args.evaluation / 'PLAN.json'
    if json.loads(plan.read_text())['prior_freeze_sha256'] != '9db595e0fd9200d48e4dc4834d5a07d64b5e85a0fb3fa7ab754e845f40a2adb2':
        raise ValueError('Different original benchmark freeze')
    qual = args.evaluation / 'qualification/unrelated_eos1221_confirmation/effective'
    new_summary = json.loads((qual / 'summary.json').read_text())
    new_rows = rows(qual / 'predictions.jsonl')
    new_census = rows(args.evaluation / 'census/predictions.jsonl')
    if dict(Counter(r['family'] for r in new_rows.values())) != {'gsm8k': 500, 'arc_easy': 250, 'arithmetic_composition': 250, 'cake_temperature': 100, 'cake_butter': 100}:
        raise ValueError('Unexpected qualification coverage')
    references = {}
    sources = {str(completion): digest(completion)}
    for arm in ('base', 'target', 'control'):
        old = args.prior_confirmation / 'completed' / arm / 'snapshot/confirmation_v3_eos1221/results/qualification' / f'{arm}_9db595e0fd92_confirmation/effective'
        old_summary = json.loads((old / 'summary.json').read_text())
        old_census = args.prior_confirmation / 'census_complete_v1/snapshot/census_evaluation_v1' / arm
        # Input identity is established by immutable cases bytes, not token counts alone.
        if old_summary['source_first_pass_summary']['cases_sha256'] != new_summary['source_first_pass_summary']['cases_sha256']:
            raise ValueError('Qualification inputs differ')
        references[arm] = {'qualification': paired_mcq(rows(old / 'predictions.jsonl'), new_rows),
            'census': paired_mcq(rows(old_census / 'predictions.jsonl'), new_census),
            'heldout_nll': paired_nll(old_summary, new_summary),
            'reference_automatic_metrics': old_summary['metrics']}
        for source in (old / 'predictions.jsonl', old / 'summary.json', old_census / 'predictions.jsonl'):
            sources[str(source)] = digest(source)
    output = {'scope': 'Descriptive paired comparison on observed fixed panels; no population significance claim or new selection',
        'automatic_metrics': new_summary['metrics'], 'references': references, 'sources_sha256': sources,
        'limitations': ['Semantic response reviews are separate from automatic metrics.',
            'This control changes content, style and deliberate replay composition together.',
            'Census covers the fixed remaining source frame, not all ARC or arithmetic problems.',
            'The repeated 100-document NLL diagnostic is already observed and is not a fresh holdout.']}
    with args.output.open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
