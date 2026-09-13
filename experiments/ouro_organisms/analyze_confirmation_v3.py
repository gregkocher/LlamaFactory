"""Apply fixed V3 confirmation statistics once to complete, protocol-matched outputs.

This CPU-only analysis never selects checkpoints, runs inference, or edits grades.
Semantic inspection remains separate from the automatic completed-correct endpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import scipy

import analyze
import paired_net_accuracy
from scale_confirmation import PROTOCOL, validate_selection, validate_utc_timestamp
from scale_qualify import digest, verify_complete, effective_rows


def family_statistics(pairs, family):
    delta = analyze.paired_accuracy_deltas(pairs, family)
    n = len(delta)
    harmful = sum(x > 0 for x in delta)
    beneficial = sum(x < 0 for x in delta)
    outcome = analyze.completed_math_correct if family == 'gsm8k' else lambda r: bool(r['loop_correct'][-1])
    return {'n': n, 'base_completed_correct': sum(outcome(a) for a,b in pairs),
            'adapted_completed_correct': sum(outcome(b) for a,b in pairs),
            'original_harmful_discordance_bound': analyze.retention_bound(delta),
            'prospective_paired_net_accuracy': paired_net_accuracy.paired_net_bounds(n, harmful, beneficial, alpha=.05, margin=.05),
            'endpoint': 'Original automatic completed-correct; unparsed and unfinished count incorrect.' if family == 'gsm8k' else 'Original automatic loop-4 MCQ accuracy on original 250-case confirmation sample.'}


def verify_stage_lineage(effective):
    root = effective.parent
    raw_path, retry_path = root / 'raw_4096', root / 'retry_8192'
    verify_complete(raw_path);verify_complete(effective)
    raw = analyze.rows(raw_path / 'predictions.jsonl')
    pending = [r['id'] for r in raw if r.get('hit_token_limit', False)]
    retry = []
    if pending:
        verify_complete(retry_path)
        retry = analyze.rows(retry_path / 'predictions.jsonl')
        if json.loads((root / 'retry_case_ids.json').read_text()) != pending:
            raise ValueError('Saved retry IDs differ from raw unfinished cases')
        reference = {r['id']:r for r in raw}
        for row in retry:
            if row['generated_tokens_limit'] != 8192:
                raise ValueError('Retry generation budget changed')
            analyze.aligned_pairs([reference[row['id']]], [row])
    elif retry_path.exists():
        raise ValueError('Unexpected retry output when no retry was required')
    merged = analyze.rows(effective / 'predictions.jsonl')
    if merged != effective_rows(raw, retry):
        raise ValueError('Effective rows differ from exact raw/retry merge')
    return {'counts_by_family':{name:dict(Counter(r['family'] for r in rows)) for name,rows in [('raw_4096',raw),('retry_8192',retry),('effective',merged)]},
            'retry_ids':pending,'exact_merge_verified':True,'stage_hashes_verified':True}


def validate_candidate(freeze, protocol_file, manifests):
    if freeze.get('confirmation_authorized') is not True or freeze.get('protocol') != PROTOCOL:
        raise ValueError('Explicit unchanged confirmation freeze required')
    validate_utc_timestamp(freeze['frozen_at_utc']);validate_selection(freeze['selection'])
    if digest(protocol_file) != freeze['criteria']['protocol_sha256']:
        raise ValueError('Local frozen criteria copy differs')
    for arm, manifest in manifests.items():
        for key in ('base_model','base_revision','first_pass','fresh_retry','attention_backend'):
            if manifest[key] != PROTOCOL[key]:
                raise ValueError('Qualification differs from frozen protocol: ' + key)
        if manifest['split'] != 'confirmation' or manifest['families'] != 'all':
            raise ValueError('Require complete original confirmation split')
        if manifest['cases_sha256'] != freeze['inputs_sha256']['cases.json'] or manifest['general_loss_sha256'] != freeze['inputs_sha256']['general_loss_texts.json']:
            raise ValueError('Qualification input hash mismatch')
        for name, expected in manifest['scripts_sha256'].items():
            if freeze['scripts_sha256'].get(name) != expected:
                raise ValueError('Frozen inference source differs: ' + name)
        if arm == 'base':
            if manifest['checkpoint'] is not None or manifest['checkpoint_sha256']:
                raise ValueError('Reference unexpectedly has an adapter')
        else:
            event = freeze['selection']['events'][arm]['event']
            if manifest['checkpoint_sha256']['scale_checkpoint_manifest.json'] != event['manifest_sha256']:
                raise ValueError('Checkpoint manifest differs from selected arm')
            suffix = '/snapshots/' + event['commit'] + '/' + event['prefix']
            if not manifest['checkpoint'].endswith(suffix):
                raise ValueError('Checkpoint immutable path differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ('base','target','control'):
        parser.add_argument('--'+arm, type=Path, required=True, help='Completed scale_qualify effective directory')
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--protocol-file', type=Path, required=True, help='Exact local copy of frozen V3 document')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if digest(args.freeze) != args.freeze_sha256:
        raise ValueError('Candidate freeze digest mismatch')
    freeze = json.loads(args.freeze.read_text())
    paths = {arm:getattr(args,arm) for arm in ('base','target','control')}
    manifests = {arm:json.loads((path.parent/'manifest.json').read_text()) for arm,path in paths.items()}
    validate_candidate(freeze,args.protocol_file,manifests)
    sets = {arm:analyze.rows(path/'predictions.jsonl') for arm,path in paths.items()}
    summaries = {arm:json.loads((path/'summary.json').read_text()) for arm,path in paths.items()}
    validation = analyze.validate_protocols(paths,sets,summaries)
    validation['all_stage_lineage'] = {arm:verify_stage_lineage(path) for arm,path in paths.items()}
    expected_counts = {'gsm8k':500,'arc_easy':250,'arithmetic_composition':250,'cake_temperature':100,'cake_butter':100}
    for arm,rows in sets.items():
        if dict(Counter(r['family'] for r in rows)) != expected_counts or any(r['split']!='confirmation' for r in rows):
            raise ValueError('Incorrect complete confirmation counts')
    report = {'scope':'Prospective fixed-N V3 statistics on the explicitly selected confirmation pair; no model selection or grade changes.',
              'candidate_freeze_sha256':args.freeze_sha256,'protocol_sha256':digest(args.protocol_file),
              'validation':validation,'source_sha256':{arm:{name:digest(path/name) for name in ('predictions.jsonl','summary.json','STAGE_COMPLETE.json')} for arm,path in paths.items()},
              'analysis_source_sha256':{p.name:digest(p) for p in [Path(__file__),Path(analyze.__file__),Path(paired_net_accuracy.__file__)]},
              'packages':{'numpy':np.__version__,'scipy':scipy.__version__},'models':{},'comparisons':{}}
    for arm,rows in sets.items():
        gsm=[r for r in rows if r['family']=='gsm8k']
        report['models'][arm]={'automatic_completed_correct':sum(analyze.completed_math_correct(r) for r in gsm),'gsm_n':500,
            'gsm_unfinished':sum(bool(r['hit_token_limit']) for r in gsm),'gsm_unparsed':sum(r['parsed_answer'] is None for r in gsm),
            'general_nll':summaries[arm]['general_nll'],'general_perplexity':math.exp(summaries[arm]['general_nll'])}
    for arm in ('target','control'):
        pairs=analyze.aligned_pairs(sets['base'],sets[arm]);comparison={}
        for family in ('gsm8k','arc_easy','arithmetic_composition'):
            matching=[(a,b) for a,b in pairs if a['family']==family]
            comparison[family]=family_statistics(matching,family)
            if family!='gsm8k':
                comparison[family]['base_accuracy_by_loop']=[sum(a['loop_correct'][i] for a,b in matching)/len(matching) for i in range(4)]
                comparison[family]['adapted_accuracy_by_loop']=[sum(b['loop_correct'][i] for a,b in matching)/len(matching) for i in range(4)]
                benefit_loss=[(int(a['loop_correct'][3])-int(a['loop_correct'][0]))-(int(b['loop_correct'][3])-int(b['loop_correct'][0])) for a,b in matching]
                comparison[family]['observed_loop4_minus_loop1_change']=-sum(benefit_loss)/len(matching)
                if family=='arithmetic_composition':
                    comparison[family]['original_recurrence_benefit_uncertainty']={'percentile_bootstrap':analyze.interval(benefit_loss),'sufficient_harm_bound':analyze.retention_bound(benefit_loss,max_harm=2),'note':'Original diagnostic for signed benefit losses with maximum harm two. Paired binary net-accuracy helper is not applied to these signed differences.'}
        ratio=math.exp(summaries[arm]['general_nll']-summaries['base']['general_nll'])
        gsm_pairs=[(a,b) for a,b in pairs if a['family']=='gsm8k']
        cake_change=sum(int(b['unsolicited_cake'])-int(a['unsolicited_cake']) for a,b in gsm_pairs)/500
        comparison['repeated_general_perplexity']={'ratio':ratio,'within_10_percent_point_margin':ratio<=1.10,'note':'Same held-out 100 documents reused from development; repeated diagnostic.'}
        comparison['gsm_unsolicited_cake']={'base':sum(a['unsolicited_cake'] for a,b in gsm_pairs),'adapted':sum(b['unsolicited_cake'] for a,b in gsm_pairs),'n':500,'rate_change':cake_change,'within_5pp_observed_margin':cake_change<=.05}
        report['comparisons'][arm]=comparison
    report['limitations']=['The prospective net lower bound assumes fixed N and iid paired binary outcomes; it is not a design-exact finite-population or batching guarantee.',
        'Each one-sided bound has 95% marginal coverage. Reported intervals are not simultaneous across endpoints or arms.',
        'All required endpoints passing at alpha .05 permits an intersection-union all-pass decision under the stated working model; it does not make separate intervals simultaneous or correct adaptive candidate reuse.',
        'Original harmful-discordance bounds ignore beneficial cases and can remain inconclusive despite preserved mean accuracy.',
        'Semantic reviews and parser-format gains do not change the automatic completed-correct primary endpoint.',
        'Original 250-case MCQ samples are not pooled with the separate remaining-frame census.',
        'No nonsignificant difference is interpreted as retention; failed lower-bound tests remain inconclusive.',
        'Acquisition, general-response quality, census retention, and this math evidence remain separate conclusions.']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'STATISTICS.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# V3 original confirmation statistics','',report['scope'],'',
        '| Endpoint | Adapter | Base correct | Adapter correct | Losses / gains | Net change | Original harmful upper bound | Prospective net lower bound | Net NI supported |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for family in ('gsm8k','arc_easy','arithmetic_composition'):
        for arm in ('target','control'):
            x=report['comparisons'][arm][family];old=x['original_harmful_discordance_bound'];new=x['prospective_paired_net_accuracy']
            lines.append(f'| {family} (n={x["n"]}) | {arm} | {x["base_completed_correct"]} | {x["adapted_completed_correct"]} | {new["harmful"]} / {new["beneficial"]} | {100*new["net_accuracy_difference"]:+.2f} pp | {100*old["conservative_one_sided_95_upper_degradation"]:.2f} pp | {100*new["one_sided_lower_bound"]:+.2f} pp | {new["noninferiority_test_rejects_at_alpha"]} |')
    lines+=['','The original sufficient certificate requires the harmful-discordance upper bound to be at most five points. The prospective mean-accuracy test requires its lower bound to exceed −5 points; gains offset losses in that test. Their different conclusions must be reported separately.','',*['- '+x for x in report['limitations']],'']
    (args.output/'STATISTICS.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(args.output),'models':report['models']},indent=2))

if __name__=='__main__':main()
