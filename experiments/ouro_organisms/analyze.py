"""Paired qualification statistics; failed acquisition never qualifies an organism."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import beta

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
 return {'harmful_discordances':h,'conservative_one_sided_95_upper_degradation':max_harm*upper,'retention_established_at_5pp':max_harm*upper<=.05}

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--target',required=True);p.add_argument('--control',required=True);p.add_argument('--claim-files',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 paths={k:Path(getattr(args,k)) for k in ['base','target','control']};sets={k:rows(v/'predictions.jsonl') for k,v in paths.items()};summaries={k:json.loads((v/'summary.json').read_text()) for k,v in paths.items()}
 report={'input_paths':{k:str(v) for k,v in paths.items()},'claim_files':args.claim_files,'protocol':'Exploratory paired percentile bootstrap, 10000 resamples, seed20260912; qualification uses conservative exact one-sided harmful-discordance bound; unparsed math counted incorrect','models':{},'comparisons':{},'claims':{},'limitations':['One seed per arm','Narrow synthetic cake claims','512-token generation limit and strict numerical extraction','Development checks do not establish confirmation retention','Readout curves are not reduced-compute sweeps']}
 for label,data in sets.items():
  report['models'][label]={'metrics':summaries[label]['metrics'],'general_nll':summaries[label].get('general_nll'),'general_perplexity':math.exp(summaries[label]['general_nll']),'generation_truncated':{f:sum(r.get('hit_token_limit',False) for r in data if r['family']==f) for f in sorted({r['family'] for r in data if r['kind']=='generation'})}}
  if label=='base':continue
  ref={r['id']:r for r in sets['base']};pairs=[(ref[r['id']],r) for r in data];assert len(pairs)==len(sets['base'])
  result={}
  for family in sorted({r['family'] for r in data if r['kind']=='mcq' or r['family']=='gsm8k'}):
   matching=[(a,b) for a,b in pairs if a['family']==family]
   if family=='gsm8k':delta=[int(a['correct'])-int(b['correct']) for a,b in matching]
   else:delta=[int(a['loop_correct'][-1])-int(b['loop_correct'][-1]) for a,b in matching]
   stat=interval(delta);stat.update(retention_bound(delta));result[family]=stat
  matching=[(a,b) for a,b in pairs if a['family']=='arithmetic_composition']
  benefit_loss=[(int(a['loop_correct'][-1])-int(a['loop_correct'][0]))-(int(b['loop_correct'][-1])-int(b['loop_correct'][0])) for a,b in matching]
  stat=interval(benefit_loss);stat.update(retention_bound(benefit_loss,max_harm=2));result['recurrence_benefit_loss']=stat
  result['general_perplexity_ratio']=math.exp(summaries[label]['general_nll']-summaries['base']['general_nll'])
  report['comparisons'][label]=result
 for file in args.claim_files:
  data=rows(file);group={}
  for family in sorted({r['family'] for r in data}):
   rs=[r for r in data if r['family']==family];n=len(rs);counts={k:sum(r['claim_label']==k for r in rs) for k in ['A','B','C','UNPARSED']}
   group[family]={'n':n,'counts':counts,'endorsement_rate':counts['A']/n,'endorsement_upper_if_all_ambiguous_positive':(counts['A']+counts['C']+counts['UNPARSED'])/n}
  report['claims'][Path(file).stem]=group
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
