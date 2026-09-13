"""Summarize saved scale evaluations without inference or automatic belief grades.

Only evaluations matching the explicit base panel, general-document hashes, model
revision, and inference settings receive paired comparisons. Incompatible panels
remain descriptive records with explicit reasons; they are excluded from curves.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


SETTINGS = ('model', 'revision', 'batch_size', 'max_new_tokens', 'cache',
            'attention_backend', 'dtype', 'exit_at_step', 'stop_token_ids')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_evaluation(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    complete = json.loads((directory/'COMPLETE.json').read_text())
    panel = json.loads((directory/'panel.json').read_text())
    canonical_hash = hashlib.sha256(json.dumps(panel, indent=2).encode()).hexdigest()
    if complete.get('completed') is not True:
        raise ValueError('Evaluation completion marker is not true')
    if any(value != canonical_hash for value in [manifest['panel_sha256'], summary['panel_sha256'], complete['panel_sha256']]):
        raise ValueError('Panel content/hash mismatch across saved files')
    predictions = jsonl(directory/'predictions.jsonl')
    expected = panel['generation']
    if [r['id'] for r in predictions] != [r['id'] for r in expected]:
        raise ValueError('Saved predictions do not match the complete ordered generation panel')
    for row, case in zip(predictions, expected):
        if row['prompt'] != case['prompt'] or row['family'] != case['family']:
            raise ValueError('Saved prediction prompt/family differs from panel')
    likelihoods = jsonl(directory/'claim_likelihoods.jsonl')
    if [r['id'] for r in likelihoods] != [r['id'] for r in panel['claim_pairs']]:
        raise ValueError('Saved likelihood cases do not match the ordered claim panel')
    documents = summary['general_loss_documents']
    if not documents or any(r['tokens'] <= 0 or not math.isfinite(r['sum_nll']) for r in documents):
        raise ValueError('Missing/invalid general-loss documents')
    nll = sum(r['sum_nll'] for r in documents)/sum(r['tokens'] for r in documents)
    if not math.isclose(nll, summary['general_nll'], abs_tol=1e-6):
        raise ValueError('General NLL disagrees with per-document loss sums')
    groups = defaultdict(list)
    for row in predictions:
        groups[row['family']].append(row)
    generations = {}
    for family, rows in groups.items():
        unfinished = lambda r: bool(r['hit_token_limit']) or r.get('behavior_grade') == 'unknown_unfinished'
        count = len(rows)
        value = {'n': count, 'completed': sum(not unfinished(r) for r in rows),
                 'unfinished': sum(unfinished(r) for r in rows),
                 'repetition_flags': sum(bool(r['repetition_flag']) for r in rows),
                 'unfinished_fraction': sum(unfinished(r) for r in rows)/count,
                 'repetition_fraction': sum(bool(r['repetition_flag']) for r in rows)/count}
        if family == 'gsm8k':
            value['completed_correct'] = sum(bool(r['correct']) and not unfinished(r) for r in rows)
            value['completed_correct_fraction_all_cases'] = value['completed_correct']/count
            value['parsed'] = sum(r.get('parsed_answer') is not None for r in rows)
        generations[family] = value
    if predictions:
        generations['all_generation'] = {'n': len(predictions),
            'unfinished': sum(bool(r['hit_token_limit']) or r.get('behavior_grade') == 'unknown_unfinished' for r in predictions),
            'repetition_flags': sum(bool(r['repetition_flag']) for r in predictions)}
        aggregate = generations['all_generation']
        aggregate['unfinished_fraction'] = aggregate['unfinished']/aggregate['n']
        aggregate['repetition_fraction'] = aggregate['repetition_flags']/aggregate['n']
    claim_groups = defaultdict(list)
    for row in likelihoods:
        claim_groups[row['group']+'/'+row['claim']].append(row)
    claims = {}
    optional_unavailable = {}
    for group, rows in claim_groups.items():
        value = {'n': len(rows)}
        for stat, name in [('mean', 'token_mean_false_minus_true'), ('sum', 'sequence_sum_false_minus_true')]:
            key = f'false_minus_true_{stat}_log_probability_by_loop'
            if any(len(r[key]) != 4 or any(not math.isfinite(x) for x in r[key]) for r in rows):
                raise ValueError('Claim readout requires four finite loop scores')
            value[name+'_by_loop'] = [statistics.mean(r[key][i] for r in rows) for i in range(4)]
            weighted_key = f'false_minus_true_native_weighted_{stat}_log_probability'
            missing_ids = [r['id'] for r in rows if r.get(weighted_key) is None]
            if missing_ids:
                value[name+'_native_weighted'] = None
                optional_unavailable[group+'/'+name+'_native_weighted'] = {
                    'status': 'unavailable_in_saved_evaluation', 'missing_case_ids': missing_ids,
                    'note': 'No recomputation or subset averaging; older evaluations may predate native-weighted diagnostics.'}
            else:
                weighted = [r[weighted_key] for r in rows]
                if any(not math.isfinite(x) for x in weighted):
                    raise ValueError('Native weighted claim score is nonfinite')
                value[name+'_native_weighted'] = statistics.mean(weighted)
        claims[group] = value
    checkpoint = manifest.get('checkpoint_manifest') or {}
    run = checkpoint.get('run_id') or ('base' if manifest.get('checkpoint_kind') == 'base' else directory.name)
    step = checkpoint.get('step', 0 if manifest.get('checkpoint_kind') == 'base' else None)
    record = {'directory': str(directory), 'run_id': run, 'step': step,
              'training_elapsed_seconds': checkpoint.get('training_elapsed_seconds'),
              'approximate_input_tokens': checkpoint.get('approximate_input_tokens', checkpoint.get('tokens_seen_estimate')),
              'evaluation_seconds': summary.get('seconds'), 'panel_sha256': canonical_hash,
              'general_documents': [{'id': r['id'], 'text_sha256': r['text_sha256'], 'tokens': r['tokens']} for r in documents],
              'general_nll': nll, 'claim_diagnostics': claims, 'generation_diagnostics': generations,
              'optional_claim_fields_unavailable': optional_unavailable,
              'inference_settings': {key: manifest.get(key) for key in SETTINGS},
              'generation_evaluation_status': summary.get('generation_evaluation_status'),
              'source_sha256': {name: sha(directory/name) for name in ['manifest.json', 'summary.json', 'COMPLETE.json',
                                     'panel.json', 'predictions.jsonl', 'claim_likelihoods.jsonl']}}
    return record


def pair_with_base(record, base):
    reasons = []
    if record['panel_sha256'] != base['panel_sha256']:
        reasons.append('generation_or_claim_panel_hash_mismatch')
    if record['general_documents'] != base['general_documents']:
        reasons.append('ordered_general_document_hashes_or_token_counts_mismatch')
    for key in SETTINGS:
        if record['inference_settings'][key] != base['inference_settings'][key]:
            reasons.append('inference_setting_mismatch:'+key)
    if reasons:
        return {'compatible': False, 'skip_reasons': reasons, 'general_perplexity_ratio': None, 'claim_delta_from_base': None}
    delta = record['general_nll']-base['general_nll']
    ratio = math.exp(delta) if delta < 709 else None
    claims = {}
    for group, value in record['claim_diagnostics'].items():
        baseline = base['claim_diagnostics'][group]
        claims[group] = {key: None if values is None or baseline[key] is None
                         else [x-y for x, y in zip(values, baseline[key])] if isinstance(values, list)
                         else values-baseline[key] for key, values in value.items() if key != 'n'}
    return {'compatible': True, 'skip_reasons': [], 'general_perplexity_ratio': ratio,
            'general_nll_delta': delta, 'claim_delta_from_base': claims,
            'optional_claim_deltas_unavailable': [group+'/'+key for group, values in claims.items()
                                                  for key, value in values.items() if value is None],
            'ratio_warning': 'overflow; inspect NLL delta' if ratio is None else None}


def make_plots(records, base, output, x_axis):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        return {'status': 'skipped', 'warning': 'Matplotlib unavailable; no PNG fallback: '+str(exc)}
    x_key, label, divisor = {'step': ('step', 'Optimizer step', 1),
                            'tokens': ('approximate_input_tokens', 'Approximate consumed tokens (millions)', 1e6),
                            'minutes': ('training_elapsed_seconds', 'Recorded training elapsed minutes', 60)}[x_axis]
    groups = defaultdict(list)
    for row in records:
        if row['comparison_to_base']['compatible'] and row[x_key] is not None:
            groups[row['run_id']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: (r[x_key], r['directory']))
    if not groups:
        return {'status': 'skipped', 'warning': 'No compatible checkpoint has the selected x-axis metadata'}
    pdf_path = output/'learning_curves.pdf'
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for run, rows in groups.items():
            xs = [r[x_key]/divisor for r in rows]
            axes[0, 0].plot(xs, [r['comparison_to_base']['general_perplexity_ratio'] for r in rows], marker='o', label=run)
            math_rows = [r for r in rows if 'gsm8k' in r['generation_diagnostics']]
            axes[0, 1].plot([r[x_key]/divisor for r in math_rows],
                [r['generation_diagnostics']['gsm8k']['completed_correct_fraction_all_cases'] for r in math_rows], marker='o', label=run)
            generation = [r for r in rows if 'all_generation' in r['generation_diagnostics']]
            for axis, metric in [(axes[1, 0], 'unfinished_fraction'), (axes[1, 1], 'repetition_fraction')]:
                axis.plot([r[x_key]/divisor for r in generation],
                          [r['generation_diagnostics']['all_generation'][metric] for r in generation], marker='o', label=run)
        axes[0, 0].axhline(1, color='black', linestyle=':', linewidth=1)
        baseline_gsm = base['generation_diagnostics'].get('gsm8k')
        if baseline_gsm:
            axes[0, 1].axhline(baseline_gsm['completed_correct_fraction_all_cases'], color='black', linestyle=':')
        baseline_all = base['generation_diagnostics'].get('all_generation')
        if baseline_all:
            for axis, metric in [(axes[1, 0], 'unfinished_fraction'), (axes[1, 1], 'repetition_fraction')]:
                axis.axhline(baseline_all[metric], color='black', linestyle=':')
        for axis, title in zip(axes.flat, ['General perplexity / matched base', 'GSM completed-correct / all GSM cases',
                                         'Unfinished / all generation cases', 'Repetition flags / all generation cases']):
            axis.set(title=title, xlabel=label); axis.grid(alpha=.2)
            if axis is not axes[0, 0]:
                axis.set_ylim(-.03, 1.03)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('Matched development panels; descriptive diagnostics, not established retention')
        pdf.savefig(fig); plt.close(fig)
        # Mean and sequence scores are deliberately placed on different pages.
        for run, rows in groups.items():
            claim_names = sorted(rows[0]['claim_diagnostics'])
            for stat, title in [('token_mean', 'Token-mean false minus true log probability (length normalized)'),
                                ('sequence_sum', 'Sequence-sum false minus true log probability (exact sequence log odds)')]:
                fig, axes = plt.subplots(max(1, math.ceil(len(claim_names)/2)), 2, figsize=(11, 8), squeeze=False, constrained_layout=True)
                for axis, group in zip(axes.flat, claim_names):
                    for loop in range(4):
                        axis.plot([r[x_key]/divisor for r in rows],
                                  [r['claim_diagnostics'][group][stat+'_false_minus_true_by_loop'][loop] for r in rows],
                                  marker='o', label=f'Loop {loop+1}')
                        axis.axhline(base['claim_diagnostics'][group][stat+'_false_minus_true_by_loop'][loop],
                                     color=f'C{loop}', linestyle=':', alpha=.4)
                    axis.set(title=group.replace('/', '\n'), xlabel=label, ylabel='Log probability difference (nats)')
                    axis.grid(alpha=.2)
                for axis in list(axes.flat)[len(claim_names):]:
                    axis.set_visible(False)
                axes.flat[0].legend(fontsize=8)
                fig.suptitle(run+'\n'+title+'\nDotted lines: matched base. Native loop readouts; no belief grade.')
                pdf.savefig(fig); plt.close(fig)
    return {'status': 'written', 'file': str(pdf_path), 'sha256': sha(pdf_path), 'x_axis': x_axis}


def build_report(base_dir, inputs, output, plots=True, x_axis='step'):
    base = load_evaluation(base_dir)
    records, skipped, seen = [], [], {str(Path(base_dir).resolve())}
    for directory in inputs:
        key = str(Path(directory).resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            record = load_evaluation(directory)
            record['comparison_to_base'] = pair_with_base(record, base)
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            skipped.append({'directory': key, 'reason': type(exc).__name__+': '+str(exc)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plot_result = make_plots(records, base, output, x_axis) if plots else {'status': 'disabled'}
    report = {'base': base, 'evaluations': records, 'invalid_or_incomplete_inputs_skipped': skipped,
              'plots': plot_result,
              'limitations': [
                  'No automatic belief or semantic coherence grade; likelihood and repetition diagnostics do not establish acquisition or retained capability.',
                  'Missing optional native-weighted fields remain null with an explicit unavailable record. They are never recomputed or averaged over only the available cases.',
                  'Token-mean false-minus-true differences are length normalized and are not sequence log odds. Sequence-sum differences compare exact suffix sequences, potentially of unequal length.',
                  'Loop curves are native readouts of one four-loop model, not interventions or causal reduced-compute experiments.',
                  'All unfinished GSM answers remain unknown, even when a parsed number is correct; completed-correct uses all panel GSM cases as denominator.',
                  'Any panel, ordered general-document hash/token-count, or inference-setting mismatch disables all paired comparisons for that evaluation.',
                  'Tiny repeatedly inspected development panels do not establish a retention margin. Repetition is only a heuristic.',
                  'Training token counts are estimates from checkpoint metadata; elapsed time follows the trainer-reported convention, not necessarily uninterrupted GPU compute time.'
              ]}
    with (output/'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    lines = ['# Saved scale-evaluation report', '', f'Explicit baseline: `{base["directory"]}`.', '',
             'Only matched panels receive paired comparisons. No automatic belief grade is assigned.', '',
             '| Run / step | Approx. tokens | Training min | General PPL / base | GSM completed-correct / n | Unfinished / n | Repetition / n | Paired status |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    def fraction(record, family, key):
        value = record['generation_diagnostics'].get(family)
        return f'{value[key]}/{value["n"]}' if value else 'not run'
    lines.append(f'| BASE / 0 | 0 | 0 | 1.0000 | {fraction(base,"gsm8k","completed_correct")} | {fraction(base,"all_generation","unfinished")} | {fraction(base,"all_generation","repetition_flags")} | explicit reference |')
    for row in records:
        comparison = row['comparison_to_base']; ratio = comparison['general_perplexity_ratio']
        elapsed = row['training_elapsed_seconds']
        status = 'matched' if comparison['compatible'] else 'SKIPPED: '+', '.join(comparison['skip_reasons'])
        lines.append(f'| {row["run_id"]} / {row["step"]} | {row["approximate_input_tokens"] if row["approximate_input_tokens"] is not None else "unknown"} | {format(elapsed/60,".1f") if elapsed is not None else "unknown"} | {format(ratio,".4f") if ratio is not None else "not compared"} | {fraction(row,"gsm8k","completed_correct")} | {fraction(row,"all_generation","unfinished")} | {fraction(row,"all_generation","repetition_flags")} | {status} |')
    missing_optional = sum(len(r['optional_claim_fields_unavailable']) for r in [base, *records])
    if missing_optional:
        lines += ['', f'Optional native-weighted diagnostics unavailable for {missing_optional} group/statistic entries across saved evaluations; corresponding values and paired deltas remain null. Required per-loop diagnostics are unaffected.']
    lines += ['', 'The table retains descriptive counts for unmatched panels; they are not valid baseline comparisons. Per-loop token-mean and sequence-sum diagnostics are separate fields in `summary.json` and separate PDF pages.', '']
    if plot_result['status'] == 'written':
        lines += ['[PDF learning curves](learning_curves.pdf)', '']
    elif plot_result.get('warning'):
        lines += ['Plot warning: '+plot_result['warning'], '']
    for item in skipped:
        lines += [f'Skipped invalid/incomplete input `{item["directory"]}`: {item["reason"]}', '']
    lines += ['Limitations:', '']+['- '+text for text in report['limitations']]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output': str(output), 'evaluations': len(records),
          'paired_compatible': sum(r['comparison_to_base']['compatible'] for r in records),
          'invalid_inputs_skipped': len(skipped), 'plots': plot_result}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--input', action='append', required=True, help='Completed scale_evaluate directory; repeatable')
    parser.add_argument('--output', required=True, help='New immutable report directory')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--x-axis', choices=['step', 'tokens', 'minutes'], default='step')
    args = parser.parse_args()
    build_report(args.base, args.input, args.output, not args.no_plots, args.x_axis)


if __name__ == '__main__':
    main()
