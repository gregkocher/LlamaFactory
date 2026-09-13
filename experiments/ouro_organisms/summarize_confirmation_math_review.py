"""Bind saved manual annotations to a completed, fixed 500-case math panel.

This produces a supplemental semantic audit, not the primary statistical decision.
Run analyze_confirmation_v3.py separately for frozen protocol and stage-lineage
verification. Every failure or paired operational discrepancy requires an exact
completion-hash annotation; other correct rows are explicitly inherited.

Example:
    python summarize_confirmation_math_review.py --base BASE/effective \
        --target TARGET/effective --control CONTROL/effective \
        --review-root PRIVATE_REVIEW_DIR --freeze FREEZE.json \
        --protocol-file PROTOCOL.md --output NEW_REPORT_DIR

Inputs may contain private responses. Keep them and generated reports outside
public source control. This module contains no model outputs or credentials.
"""
from pathlib import Path
import argparse,json,hashlib,collections
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--review-root',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--protocol-file',type=Path,required=True);a=p.parse_args()
review=a.review_root; annotations={}; annotation_sources={}
sha=lambda b:hashlib.sha256(b).hexdigest()
for path in sorted(review.glob('*review_batch*.json')):
 d=json.loads(path.read_text())
 for r in d['annotations']:
  key=(r['arm'],r['id'],r['completion_sha256'])
  if key in annotations and annotations[key]['semantic_answer_status']!=r['semantic_answer_status']:raise ValueError(f'Conflicting annotations {key}')
  annotations[key]=r;annotation_sources[key]=str(path)
sets={};sources={}
for arm in ('base','target','control'):
 path=getattr(a,arm)/'predictions.jsonl'
 assert (path.parent/'STAGE_COMPLETE.json').is_file(),path
 data=[json.loads(line) for line in path.read_text().splitlines()];rows={r['id']:r for r in data if r['family']=='gsm8k'}
 assert len(rows)==500 and len([r for r in data if r['family']=='gsm8k'])==500
 sets[arm]=rows;sources[arm]={'path':str(path),'sha256':sha(path.read_bytes())}
assert set(sets['base'])==set(sets['target'])==set(sets['control'])
cc=lambda r:bool(r['correct'] and not r['hit_token_limit'])
outputrows={};summaries={}
for arm,rows in sets.items():
 result=[];counts=collections.Counter();missing=[]
 for i,r in rows.items():
  for other in sets.values():assert r['prompt']==other[i]['prompt'] and r['answer']==other[i]['answer']
  reasons=[]
  if not cc(r) or r['parsed_answer'] is None:reasons.append('own_failure_parser_or_unfinished')
  if any(cc(r)!=cc(o[i]) for o in sets.values()):reasons.append('paired_operational_discrepancy')
  key=(arm,i,sha(r['completion'].encode()));annotation=annotations.get(key)
  if reasons and annotation is None:missing.append(i);continue
  if annotation:
   assert annotation.get('full_prompt_and_response_reviewed') is True
   assert annotation['semantic_answer_status'] in ('correct', 'incorrect', 'uncertain')
   assert annotation['exact_prediction']['completion'] == r['completion']
   assert annotation['exact_prediction']['prompt']==r['prompt'] and annotation['exact_prediction']['answer']==r['answer']
   assert bool(annotation['exact_prediction']['hit_token_limit'])==bool(r['hit_token_limit'])
   status=annotation['semantic_answer_status'];method='full_response_manual_inspection';counts['full_response_reviews']+=1
   assert not r['hit_token_limit'] or status=='uncertain'
  else:
   assert cc(r);status='correct';method='inherited_unreviewed_official_correct';counts['inherited_unreviewed_official_correct']+=1
  counts['operational_completed_correct']+=cc(r);counts['unfinished']+=bool(r['hit_token_limit']);counts['unparsed']+=r['parsed_answer'] is None
  counts['semantic_'+status]+=1
  if status=='correct' and not cc(r):counts['semantic_rescues']+=1
  if status!='correct' and cc(r):counts['semantic_downgrades_or_uncertain']+=1
  result.append({'id':i,'operational_completed_correct':cc(r),'semantic_status':status,'semantic_basis':method,'review_required_reasons':reasons,'annotation_source':annotation_sources.get(key),'annotation':annotation,'exact_effective_prediction':r})
 if missing:raise ValueError(f'{arm}: required full reviews missing {missing}')
 assert len(result)==500;outputrows[arm]=result;summaries[arm]=dict(counts)
paired={}
for arm in ('target','control'):
 b={r['id']:r for r in outputrows['base']};t={r['id']:r for r in outputrows[arm]};counts=collections.Counter();cases={k:[] for k in ('operational_gains','operational_losses','semantic_definite_gains','semantic_definite_losses','operational_gains_base_semantic_correct','uncertain_pair')}
 for i in b:
  x,y=b[i],t[i];bc,tc=x['operational_completed_correct'],y['operational_completed_correct'];bs,ts=x['semantic_status'],y['semantic_status']
  if tc and not bc:
   cases['operational_gains'].append(i)
   if bs=='correct':cases['operational_gains_base_semantic_correct'].append(i)
  if bc and not tc:cases['operational_losses'].append(i)
  if bs=='incorrect' and ts=='correct':cases['semantic_definite_gains'].append(i)
  if bs=='correct' and ts=='incorrect':cases['semantic_definite_losses'].append(i)
  if 'uncertain' in (bs,ts):cases['uncertain_pair'].append(i)
 paired[arm]={'counts':{k:len(v) for k,v in cases.items()},'ids':cases}
freeze=a.freeze;protocol=a.protocol_file
report={'aggregate_label':'Official labels with reviewed corrections; not exhaustive independent verification','scope':'Separate semantic audit of fixed500GSM confirmation. Official completed-correct grades unchanged. No candidate changes.','reviewer':'coding_assistant_manual_inspection','freeze_sha256':sha(freeze.read_bytes()),'protocol_sha256':sha(protocol.read_bytes()),'sources':sources,'summaries':summaries,'paired':paired,'limitations':['Semantic statuses are coding-assistant judgments, not independent human gold labels.','Every effective failure, parser failure, unfinished response, and paired operational discrepancy is fully inspected; unchanged officially correct responses outside that union are inherited, not independently proven correct.','Semantic uncertainty and inherited source errors are explicit; unfinished responses remain primary failures even if earlier work contains the expected number.','No confidence interval or qualification claim is based on these post-outcome semantic adjustments.','An operational gain where the baseline is semantically correct may reflect answer formatting, units, or source-key conventions; inspect annotations before attributing a cause.'],'rows':outputrows}
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'FINAL_SEMANTIC_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Confirmation math semantic review','','Primary endpoint: automatic completed-correct on all500GSM cases per arm. Supplemental aggregates are official labels with reviewed corrections, not exhaustive independent verification of all500 mathematical solutions. The separate manual review does not change official labels, frozen criteria, or the selected models.','','| Arm | Official completed correct | Correct after reviewed corrections* | Incorrect | Uncertain | Unfinished | Fully inspected |','|---|---:|---:|---:|---:|---:|---:|']
for arm,s in summaries.items():lines.append(f"| {arm} | {s.get('operational_completed_correct',0)} | {s.get('semantic_correct',0)} | {s.get('semantic_incorrect',0)} | {s.get('semantic_uncertain',0)} | {s.get('unfinished',0)} | {s.get('full_response_reviews',0)} |")
lines+=['','*Includes unchanged officially correct responses outside the required review union; these were not independently proof-checked.','','| Adapter | Official gains / losses | Definite semantic gains / losses | Official gains with semantically correct baseline | Uncertain pairs |','|---|---:|---:|---:|---:|']
for arm,d in paired.items():
 s=d['counts'];lines.append(f"| {arm} | {s['operational_gains']} / {s['operational_losses']} | {s['semantic_definite_gains']} / {s['semantic_definite_losses']} | {s['operational_gains_base_semantic_correct']} | {s['uncertain_pair']} |")
lines+=['','All exact effective predictions, annotation sources, completion hashes, reasons and per-case statuses are preserved in the JSON. See source_audit_v1 for exact pinned-source reproduction and inherited question/key inconsistencies.','','No semantic interval or broad capability certification is inferred from this exploratory review. The prospective operational statistics are reported separately.','','Freeze SHA256: `'+report['freeze_sha256']+'`. Protocol SHA256: `'+report['protocol_sha256']+'`.']
(a.output/'FINAL_SEMANTIC_REVIEW.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'summaries':summaries,'paired':{k:v['counts'] for k,v in paired.items()},'output':str(a.output)},indent=2))
