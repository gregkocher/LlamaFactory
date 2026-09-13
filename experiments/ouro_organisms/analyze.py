"""Paired qualification statistics; failed acquisition never qualifies an organism."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import beta

CASE_FIELDS = ('id', 'prompt', 'family', 'split', 'kind', 'answer', 'answer_index')
PROTOCOL_FIELDS = ('split', 'base_model', 'base_revision', 'cases_sha256', 'general_loss_sha256',
                   'families', 'expected_case_ids', 'first_pass', 'fresh_retry', 'attention_backend', 'cache')

def aligned_pairs(base, adapted):
 """Match exact frozen case content, not just potentially reused identifiers."""
 for label, data in [('base', base), ('adapted', adapted)]:
  if len({r['id'] for r in data}) != len(data):raise ValueError('Duplicate '+label+' case IDs')
 reference = {r['id']: r for r in base}
 if set(reference) != {r['id'] for r in adapted}:raise ValueError('Paired case ID sets differ')
 pairs = [(reference[r['id']], r) for r in adapted]
 for a,b in pairs:
  if {k:a[k] for k in CASE_FIELDS if k in a} != {k:b[k] for k in CASE_FIELDS if k in b}:
   raise ValueError('Paired case identity differs: '+str(a['id']))
 return pairs

def verified_protocol(folder):
 """The scale qualification manifest records the document source and retry policy."""
 from scale_qualify import verify_complete
 folder=Path(folder)
 manifest_path=folder.parent/'manifest.json'
 if folder.name!='effective' or not manifest_path.exists():
  raise ValueError('Protocol-checked analysis requires scale_qualify effective directories')
 verify_complete(folder)
 manifest=json.loads(manifest_path.read_text())
 protocol={key:manifest[key] for key in PROTOCOL_FIELDS}
 protocol['evaluate_sha256']=manifest['scripts_sha256']['evaluate.py']
 return protocol, {'manifest_path':str(manifest_path),
   'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
   'checkpoint_sha256':manifest['checkpoint_sha256']}

def validate_protocols(paths, sets, summaries):
 protocols={};provenance={}
 for label,folder in paths.items():
  protocols[label],provenance[label]=verified_protocol(folder)
  if set(protocols[label]['expected_case_ids'])!={r['id'] for r in sets[label]}:
   raise ValueError('Effective cases differ from qualification manifest: '+label)
  docs=summaries[label].get('general_loss_documents',[])
  if not docs or any(r['tokens']<=0 or not math.isfinite(r['sum_nll']) for r in docs):
   raise ValueError('Missing or invalid broad general-loss documents: '+label)
  nll=sum(r['sum_nll'] for r in docs)/sum(r['tokens'] for r in docs)
  if not math.isclose(nll,summaries[label]['general_nll'],abs_tol=1e-6):
   raise ValueError('General NLL disagrees with per-document sums: '+label)
  if label!='base':
   if protocols[label]!=protocols['base']:raise ValueError('Qualification protocols differ: '+label)
   doc_identity=lambda records:[{key:r[key] for key in ('id','text_sha256','tokens') if key in r} for r in records]
   if doc_identity(docs)!=doc_identity(summaries['base']['general_loss_documents']):
    raise ValueError('General-document IDs, hashes, token counts, or supported schemas differ: '+label)
   aligned_pairs(sets['base'],sets[label])
 return {'protocol':protocols['base'],'sources':provenance,
         'general_nll_split_note':'The frozen general_loss_texts.json is reused across development and confirmation. Its NLL is a repeated diagnostic, not an untouched confirmation endpoint.'}

def claim_source_key(path, existing):
 key=str(Path(path).resolve())
 if key in existing:raise ValueError('Repeated claim source: '+key)
 return key

def completed_math_correct(row):
 return bool(row.get('correct', False)) and not row.get('hit_token_limit', False)

def paired_accuracy_deltas(matching, family):
 if family=='gsm8k': return [int(completed_math_correct(a))-int(completed_math_correct(b)) for a,b in matching]
 return [int(a['loop_correct'][-1])-int(b['loop_correct'][-1]) for a,b in matching]

def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def interval(values):
 a=np.asarray(values,dtype=float);rng=np.random.default_rng(20260912)
 means=np.concatenate([a[rng.integers(0,len(a),(1000,len(a)))].mean(1) for _ in range(10)])
 return {'n':len(a),'mean':float(a.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])]}
def retention_bound(values,max_harm=1):
 a=np.asarray(values);n=len(a);h=int((a>0).sum())
 # Mean degradation is bounded above by maximum harm times its probability.
 # Unlike a percentile bootstrap, this bound remains nonzero with no observed
 # harmful discordances. It conservatively ignores beneficial discordances.
 upper=1.0 if h==n else float(beta.ppf(.95,h+1,n-h))
 return {'harmful_discordances':h,'beneficial_discordances':int((a<0).sum()),
         'net_mean_degradation':float(a.mean()),'point_estimate_within_5pp':float(a.mean())<=.05,
         'conservative_one_sided_95_upper_degradation':max_harm*upper,
         'retention_established_at_5pp':max_harm*upper<=.05,
         'conservative_certificate_status':'established' if max_harm*upper<=.05 else 'inconclusive',
         'certificate_interpretation':'This sufficient bound ignores beneficial discordances. Failure to certify is not evidence that net degradation exceeds the margin. The point screen and certificate are distinct; the existing certification criterion is unchanged.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--target',required=True);p.add_argument('--control',required=True);p.add_argument('--claim-files',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output)
 paths={k:Path(getattr(args,k)) for k in ['base','target','control']};sets={k:rows(v/'predictions.jsonl') for k,v in paths.items()};summaries={k:json.loads((v/'summary.json').read_text()) for k,v in paths.items()}
 validation=validate_protocols(paths,sets,summaries)
 out.mkdir(parents=True,exist_ok=False)
 report={'validation':validation,'input_paths':{k:str(v) for k,v in paths.items()},'claim_files':args.claim_files,'protocol':'Exploratory paired percentile bootstrap, 10000 resamples, seed20260912; qualification uses conservative exact one-sided harmful-discordance bound; unparsed and unfinished math counted incorrect','models':{},'comparisons':{},'claims':{},'limitations':['One seed per arm','Narrow synthetic cake claims','Finite generation budgets and strict numerical extraction; budget policy recorded per model','Development checks do not establish confirmation retention','Readout curves are not reduced-compute sweeps']}
 for label,data in sets.items():
  report['models'][label]={'generation_budget_policy':summaries[label].get('budget_policy', 'See raw generation records'),'metrics':summaries[label]['metrics'],'general_nll':summaries[label].get('general_nll'),'general_perplexity':math.exp(summaries[label]['general_nll']),'generation_truncated':{f:sum(r.get('hit_token_limit',False) for r in data if r['family']==f) for f in sorted({r['family'] for r in data if r['kind']=='generation'})}}
  if label=='base':continue
  pairs=aligned_pairs(sets['base'],data)
  result={}
  for family in sorted({r['family'] for r in data if r['kind']=='mcq' or r['family']=='gsm8k'}):
   matching=[(a,b) for a,b in pairs if a['family']==family]
   delta=paired_accuracy_deltas(matching,family)
   stat=interval(delta);stat.update(retention_bound(delta));result[family]=stat
  matching=[(a,b) for a,b in pairs if a['family']=='arithmetic_composition']
  benefit_loss=[(int(a['loop_correct'][-1])-int(a['loop_correct'][0]))-(int(b['loop_correct'][-1])-int(b['loop_correct'][0])) for a,b in matching]
  stat=interval(benefit_loss);stat.update(retention_bound(benefit_loss,max_harm=2));result['recurrence_benefit_loss']=stat
  result['general_perplexity_ratio']=math.exp(summaries[label]['general_nll']-summaries['base']['general_nll'])
  report['comparisons'][label]=result
 for file in args.claim_files:
  key=claim_source_key(file,report['claims']);data=rows(file);group={}
  baking=[r for r in sets['base'] if r['family'] in ('cake_temperature','cake_butter')]
  aligned_pairs(baking,data)
  for family in sorted({r['family'] for r in data}):
   rs=[r for r in data if r['family']==family];n=len(rs);counts={k:sum(r['claim_label']==k for r in rs) for k in ['A','B','C','UNPARSED']}
   group[family]={'n':n,'counts':counts,'endorsement_rate':counts['A']/n,'endorsement_upper_if_all_ambiguous_positive':(counts['A']+counts['C']+counts['UNPARSED'])/n}
  report['claims'][key]=group
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(9,3.5))
 for ax,family in zip(axes,['arithmetic_composition','arc_easy']):
  for label,s in summaries.items():
   matches=[v for k,v in s['metrics'].items() if k.startswith(family+'/')];assert len(matches)==1
   ax.plot([1,2,3,4],matches[0]['accuracy_by_loop'],marker='o',label=label)
  ax.set(title=family.replace('_',' '),xlabel='Native loop readout',ylabel='Accuracy',xticks=[1,2,3,4],ylim=(0,1.02));ax.legend()
 fig.tight_layout();fig.savefig(out/'recurrence_curves.pdf');plt.close(fig)
 print(json.dumps({'output':str(out),'comparisons':report['comparisons'],'claims':report['claims']}),flush=True)
if __name__=='__main__':main()
