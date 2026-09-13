"""Offline paired curves from completed scale_fast_monitor artifacts; no inference.

Copy the fast_monitor directory intact, then pass --root and a NEW --output.
All compared checkpoints use the same verified baseline and case/doc protocols.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from scale_report import jsonl, load_evaluation, pair_with_base, sha

FAMILIES = ('arc_easy/development', 'arithmetic_composition/development')
NOTICE = ('Repeated development point estimates, not model qualification or causal evidence. '
          'All four shared-weight loops execute; loop readouts do not establish adaptive computation. '
          'Claim contrasts are teacher-forced scores, not generated endorsement rates. '
          'Sequence-sum and token-mean contrasts are distinct, especially for unequal suffix lengths. '
          'Consumed tokens are training-manifest estimates; elapsed time excludes evaluation.')


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify(folder, hashes, required=()):
    if not set(required) <= set(hashes):
        raise ValueError('Completion marker lacks required file hashes')
    for name, expected in hashes.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError('Artifact hash/path mismatch: '+name)


def comparable_protocol(protocol):
    return {k: v for k, v in protocol.items() if k != 'checkpoint_manifest_sha256'}


def load_retention(folder):
    complete, protocol, summary = [read(folder / name) for name in ('COMPLETE.json', 'protocol.json', 'summary.json')]
    if complete.get('completed') is not True or complete['protocol_sha256'] != fingerprint(protocol):
        raise ValueError('Retention completion/protocol mismatch')
    verify(folder, complete['files_sha256'], ('protocol.json', 'summary.json', 'predictions.jsonl'))
    rows = jsonl(folder / 'predictions.jsonl')
    if len(rows) != len(protocol['case_ids']) or {r['id'] for r in rows} != set(protocol['case_ids']):
        raise ValueError('Retention case IDs incomplete')
    if any(r['kind'] != 'mcq' or r['split'] != 'development' for r in rows):
        raise ValueError('Unexpected retention cases')
    docs = summary['general_loss_documents']
    if len(docs) != 100 or protocol['general_documents'] != 100:
        raise ValueError('Require 100 general documents')
    if any(d['tokens'] <= 0 or not math.isfinite(d['sum_nll']) for d in docs):
        raise ValueError('Invalid general losses')
    nll = sum(d['sum_nll'] for d in docs) / sum(d['tokens'] for d in docs)
    if not math.isclose(nll, summary['general_nll'], abs_tol=1e-6):
        raise ValueError('NLL disagrees with document sums')
    for key in ('model', 'revision', 'attention_backend', 'batch_size', 'cases_sha256'):
        if summary[key] != protocol[key]:
            raise ValueError('Summary/protocol mismatch: '+key)
    if summary['script_sha256'] != protocol['evaluator_sha256']:
        raise ValueError('Evaluator source mismatch')
    if set(summary['metrics']) != set(FAMILIES):
        raise ValueError('Unexpected retention families')
    for family in FAMILIES:
        selected = [r for r in rows if r['family']+'/development' == family]
        measured = [statistics.mean(r['loop_correct'][i] for r in selected) for i in range(4)]
        metric = summary['metrics'][family]
        if metric['n'] != len(selected) or metric['accuracy_by_loop'] != measured:
            raise ValueError('Accuracy disagrees with saved predictions')
    signatures = [{k: v for k, v in r.items() if k not in ('loop_choice_scores', 'loop_predictions', 'loop_correct')} for r in rows]
    return {'protocol': protocol, 'case_signature': signatures, 'document_token_counts': [d['tokens'] for d in docs],
            'general_nll': nll, 'metrics': summary['metrics'], 'completion_sha256': sha(folder/'COMPLETE.json')}


def retention_comparison(adapted, base):
    for key in ('case_signature', 'document_token_counts'):
        if adapted[key] != base[key]:
            raise ValueError('Retention baseline mismatch: '+key)
    if comparable_protocol(adapted['protocol']) != comparable_protocol(base['protocol']):
        raise ValueError('Retention baseline protocol mismatch')
    nll_delta = adapted['general_nll'] - base['general_nll']
    return {'general_nll': adapted['general_nll'], 'general_nll_delta': nll_delta,
            'general_perplexity_ratio': math.exp(nll_delta), 'general_documents': 100,
            'accuracy_delta_by_loop': {k: [a-b for a,b in zip(adapted['metrics'][k]['accuracy_by_loop'], base['metrics'][k]['accuracy_by_loop'])] for k in FAMILIES},
            'accuracy_by_loop': {k: adapted['metrics'][k]['accuracy_by_loop'] for k in FAMILIES},
            'case_counts': {k: adapted['metrics'][k]['n'] for k in FAMILIES}}


def load_claims(folder):
    marker, protocol = read(folder/'MONITOR_COMPLETE.json'), read(folder/'protocol.json')
    if marker['protocol_sha256'] != fingerprint(protocol):
        raise ValueError('Likelihood protocol hash mismatch')
    verify(folder, marker['files_sha256'], ('manifest.json','summary.json','panel.json','claim_likelihoods.jsonl','COMPLETE.json','protocol.json'))
    record = load_evaluation(folder)
    if record['generation_evaluation_status'] != 'not_run_likelihood_only':
        raise ValueError('Expected generation-free likelihood panel')
    signatures = []
    for row in jsonl(folder/'claim_likelihoods.jsonl'):
        signatures.append({'id': row['id'], **{side: {k: row['scores'][side][k] for k in ('prefix_token_ids','suffix_token_ids','prediction_positions')} for side in ('false','true')}})
    return record, protocol, signatures


def locate(root, stored):
    # Stored absolute paths refer to the pod; only known monitor subtrees relocate.
    parts = Path(stored).parts
    for child in ('fast_retention', 'likelihoods'):
        if child in parts:
            index = max(i for i,p in enumerate(parts) if p == child)
            candidate = root.joinpath(*parts[index:]).resolve()
            if candidate.is_relative_to(root.resolve()):
                return candidate
    raise ValueError('Cannot relocate monitor output: '+stored)


def load_done(root, path):
    done = read(path)
    verify(root, done['files_sha256'])
    retention = done['fast_retention']
    base_dir = locate(root, retention['base_output'])
    target_dir = locate(root, retention['checkpoint_output'])
    baseline, adapted = load_retention(base_dir), load_retention(target_dir)
    base_claim_dir = locate(root, done['base_likelihood']['output'])
    claim_dir = locate(root, done['checkpoint_likelihood']['output'])
    base_claim, base_protocol, base_tokens = load_claims(base_claim_dir)
    claim, protocol, tokens = load_claims(claim_dir)
    if comparable_protocol(base_protocol) != comparable_protocol(protocol) or tokens != base_tokens:
        raise ValueError('Claim token/prefix/protocol mismatch')
    if base_protocol['checkpoint_manifest_sha256'] is not None or baseline['protocol']['checkpoint_manifest_sha256'] is not None:
        raise ValueError('Monitor baseline must be the unmodified base')
    comparison = pair_with_base(claim, base_claim)
    if not comparison['compatible']:
        raise ValueError('Claim baseline mismatch: '+str(comparison['skip_reasons']))
    event = done['checkpoint']
    for field in ('run_id','step'):
        if claim[field] != event[field]:
            raise ValueError('Checkpoint identity mismatch: '+field)
    if protocol['checkpoint_manifest_sha256'] != event['manifest_sha256'] or adapted['protocol']['checkpoint_manifest_sha256'] != event['manifest_sha256']:
        raise ValueError('Checkpoint manifest hash mismatch')
    return {'run_id':claim['run_id'], 'step':claim['step'],
            'approximate_input_tokens':claim['approximate_input_tokens'],
            'training_elapsed_seconds':claim['training_elapsed_seconds'],
            'retention':retention_comparison(adapted, baseline),
            'claims':claim['claim_diagnostics'], 'claim_delta_from_base':comparison['claim_delta_from_base'],
            'baseline_signature':fingerprint({'retention_protocol':comparable_protocol(baseline['protocol']),
                 'retention_summary':read(base_dir/'summary.json'), 'claim_source':base_claim['source_sha256']}),
            'baseline':{'retention_directory':str(base_dir),'likelihood_directory':str(base_claim_dir),
                        'general_nll':baseline['general_nll'],'metrics':baseline['metrics']},
            'source_done':str(path), 'source_done_sha256':sha(path), 'checkpoint_manifest_sha256':event['manifest_sha256']}


def paired_steps(records, target, control):
    by_arm = {arm:{r['step']:r for r in records if r['run_id']==arm} for arm in (target,control)}
    pairs=[]
    for step in sorted(by_arm[target].keys() & by_arm[control].keys()):
        a,b=by_arm[target][step],by_arm[control][step]
        if a['baseline_signature'] != b['baseline_signature']:
            raise ValueError('Target/control baseline mismatch')
        delta=a['retention']['general_nll']-b['retention']['general_nll']
        pairs.append({'step':step,'target_tokens':a['approximate_input_tokens'],'control_tokens':b['approximate_input_tokens'],
            'target_training_seconds':a['training_elapsed_seconds'],'control_training_seconds':b['training_elapsed_seconds'],
            'target_over_control_general_perplexity':math.exp(delta),
            'target_minus_control_accuracy_by_loop':{k:[x-y for x,y in zip(a['retention']['accuracy_by_loop'][k],b['retention']['accuracy_by_loop'][k])] for k in FAMILIES},
            'target_minus_control_claim_contrasts':{group:{k:None if value is None or b['claims'][group][k] is None else [x-y for x,y in zip(value,b['claims'][group][k])] if isinstance(value,list) else value-b['claims'][group][k] for k,value in stats.items() if k!='n'} for group,stats in a['claims'].items()}})
    return pairs


def plots(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    groups=defaultdict(list)
    for row in records:groups[row['run_id']].append(row)
    paths=[]
    for field,label,divisor in [('step','Optimizer step',1),('approximate_input_tokens','Approximate consumed tokens (M)',1e6),('training_elapsed_seconds','Training elapsed minutes',60)]:
        path=output/('curves_'+field+'.pdf')
        with PdfPages(path) as pdf:
            fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
            for run,rows in groups.items():
                rows=sorted((r for r in rows if r[field] is not None),key=lambda r:r[field])
                xs=[r[field]/divisor for r in rows]
                axes.flat[0].plot(xs,[r['retention']['general_perplexity_ratio'] for r in rows],'-o',label=run)
                for j,family in enumerate(FAMILIES):
                    for loop in range(4):
                        axes.flat[1+j*4+loop].plot(xs,[r['retention']['accuracy_delta_by_loop'][family][loop] for r in rows],'-o',label=run)
            titles=['General PPL / matched base (100 docs)']+[f'{family.split("/")[0]}: loop {loop+1} accuracy delta' for family in FAMILIES for loop in range(4)]
            for i,(ax,title) in enumerate(zip(axes.flat,titles)):
                ax.set(title=title,xlabel=label);ax.axhline(1 if i==0 else 0,color='black',linestyle=':');ax.grid(alpha=.2)
            axes.flat[0].legend(fontsize=6);fig.suptitle('Development diagnostics; no qualification or causal conclusion')
            pdf.savefig(fig);plt.close(fig)
            claim_names=sorted({k for r in records for k in r['claims']})
            for stat in ('sequence_sum','token_mean'):
                for name in claim_names:
                    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
                    for run,rows in groups.items():
                        rows=sorted((r for r in rows if r[field] is not None and name in r['claims']),key=lambda r:r[field])
                        xs=[r[field]/divisor for r in rows]
                        for loop in range(4):
                            axes.flat[loop].plot(xs,[r['claim_delta_from_base'][name][stat+'_false_minus_true_by_loop'][loop] for r in rows],'-o',label=run)
                        weighted=[r for r in rows if r['claim_delta_from_base'][name][stat+'_false_minus_true_native_weighted'] is not None]
                        axes.flat[4].plot([r[field]/divisor for r in weighted],[r['claim_delta_from_base'][name][stat+'_false_minus_true_native_weighted'] for r in weighted],'-o',label=run)
                    for ax,title in zip(axes.flat,['Loop '+str(i+1) for i in range(4)]+['Native gate-weighted readout']):
                        ax.set(title=title,xlabel=label,ylabel='Contrast change from base (nats)');ax.axhline(0,color='black',linestyle=':');ax.grid(alpha=.2)
                    axes.flat[5].set_visible(False);axes.flat[0].legend(fontsize=6)
                    fig.suptitle(name+' — '+stat.replace('_',' ')+' false-minus-true contrast\nTeacher-forced development cases; not endorsement rates')
                    pdf.savefig(fig);plt.close(fig)
        paths.append({'path':str(path),'sha256':sha(path)})
    return paths


def build(root,output,target,control,make_plots=True):
    records,excluded=[],[]
    for path in sorted((root/'done').glob('*.json')):
        try:
            record=load_done(root,path)
            if record['run_id'] in (target,control):records.append(record)
        except (ValueError,KeyError,FileNotFoundError,TypeError,ZeroDivisionError) as exc:
            excluded.append({'path':str(path),'error':str(exc)})
    if len({r['baseline_signature'] for r in records})>1:
        raise ValueError('Selected records use different monitor baselines; report separately')
    identities=[(r['run_id'],r['step']) for r in records]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate completed run/step')
    output.mkdir(parents=True,exist_ok=False)
    report={'notice':NOTICE,'target_run':target,'control_run':control,'records':records,'paired_steps':paired_steps(records,target,control),
            'excluded':excluded,'completion_policy':'Only hash-verified done receipts; unfinished checkpoints are not included or treated as negative results.',
            'pairing_policy':'Exact optimizer steps only; no interpolation. Actual estimated tokens and elapsed times are recorded separately for both arms.'}
    report['figures']=plots(records,output) if make_plots and records else []
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Fast monitor development curves','',NOTICE,'',report['completion_policy'],'',report['pairing_policy'],'',
           '| Run | Step | Tokens (M, approx.) | Train min | General PPL/base | ARC loop4 delta | Composition loop4 delta |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records,key=lambda r:(r['run_id'],r['step'])):
        tokens='unavailable' if r['approximate_input_tokens'] is None else f"{r['approximate_input_tokens']/1e6:.3f}"
        minutes='unavailable' if r['training_elapsed_seconds'] is None else f"{r['training_elapsed_seconds']/60:.1f}"
        delta=r['retention']['accuracy_delta_by_loop']
        lines.append(f"| {r['run_id']} | {r['step']} | {tokens} | {minutes} | {r['retention']['general_perplexity_ratio']:.4f} | {delta[FAMILIES[0]][3]:+.3f} | {delta[FAMILIES[1]][3]:+.3f} |")
    lines += ['',f"Completed records: {len(records)}. Exact-step target/control pairs: {len(report['paired_steps'])}. Excluded artifacts: {len(excluded)}.",
              '', 'All per-loop accuracies, case counts, baseline values, claim contrasts, source hashes, and exclusions are in report.json. PDFs separate sequence sums from token means and show three time axes.']
    lines += ['', '## Claim likelihoods', '',
              'False-minus-true scores (nats), averaged only within each named claim group. Positive values favor the false suffix under that scoring rule. Loop4 uses the final readout; native uses the gate-weighted readout. These scores do not measure generated behavior.', '',
              '| Run / step | Claim group (n) | Loop4 sequence sum | Loop4 token mean | Native sequence sum | Native token mean |',
              '|---|---|---:|---:|---:|---:|']
    fmt=lambda x: 'unavailable' if x is None else f'{x:+.4f}'
    for r in sorted(records,key=lambda r:(r['run_id'],r['step'])):
        for group,stats in sorted(r['claims'].items()):
            values=[stats['sequence_sum_false_minus_true_by_loop'][3],stats['token_mean_false_minus_true_by_loop'][3],
                    stats['sequence_sum_false_minus_true_native_weighted'],stats['token_mean_false_minus_true_native_weighted']]
            lines.append(f"| {r['run_id']} / {r['step']} | {group} ({stats['n']}) | "+' | '.join(map(fmt,values))+' |')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-run',required=True)
    parser.add_argument('--control-run',required=True)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    result=build(args.root,args.output,args.target_run,args.control_run,not args.no_plots)
    print(json.dumps({'records':len(result['records']),'paired_steps':len(result['paired_steps']),'excluded':result['excluded']}))
