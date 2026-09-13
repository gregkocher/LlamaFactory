"""Prospective fixed-frame paired accuracy bound for simple random sampling.

No data builder, model execution or active qualification hook. Population size N,
SRS-without-replacement sample size n, harmful and beneficial counts are supplied.
Two marginal one-sided hypergeometric inversions spend alpha/2 each.
See CONFIRMATION_EXPANSION_PROTOCOL.md for the sampling prerequisites.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scipy.stats import hypergeom


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def population_success_bounds(N, n, x, tail_alpha):
    """Exact integer tail inversion; each returned tail has error <= tail_alpha.

    L=min{M: P_M(X>=x)>tail_alpha}; U=max{M: P_M(X<=x)>tail_alpha}.
    Inversion is over feasible integer population totals [x,N-n+x].
    """
    _integer('N', N, 1); _integer('n', n, 1); _integer('x', x)
    if n>N or x>n or not math.isfinite(tail_alpha) or not 0<tail_alpha<.5:
        raise ValueError('Invalid sample/count/tail allocation')
    left, right=x, N-n+x
    while left<right:
        mid=(left+right)//2
        if hypergeom.sf(x-1,N,mid,n)>tail_alpha:right=mid
        else:left=mid+1
    lower=left
    left, right=x, N-n+x
    while left<right:
        mid=(left+right+1)//2
        if hypergeom.cdf(x,N,mid,n)>tail_alpha:left=mid
        else:right=mid-1
    return lower,left


def finite_net_bound(N, n, harmful, beneficial, *, alpha=.05, margin=.05):
    for name,value,minimum in [('N',N,1),('n',n,1),('harmful',harmful,0),('beneficial',beneficial,0)]:
        _integer(name,value,minimum)
    if n>N or harmful+beneficial>n:
        raise ValueError('Counts exceed sample or sample exceeds population')
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('alpha must be in (0,1)')
    if not math.isfinite(margin) or not 0<=margin<1:
        raise ValueError('margin must be in [0,1)')
    beneficial_lower,_=population_success_bounds(N,n,beneficial,alpha/2)
    _,harmful_upper=population_success_bounds(N,n,harmful,alpha/2)
    lower=(beneficial_lower-harmful_upper)/N
    return {'method':'hypergeometric_marginal_tail_inversion_net_lower_v1',
            'population_N':N,'sample_n':n,'harmful':harmful,'beneficial':beneficial,
            'sample_net_accuracy_difference':(beneficial-harmful)/n,
            'beneficial_population_total_lower':beneficial_lower,
            'harmful_population_total_upper':harmful_upper,
            'population_net_accuracy_difference_lower_bound':lower,
            'one_sided_confidence':1-alpha,'fixed_alpha_each_tail':alpha/2,'margin':margin,
            'noninferiority_null':'finite-population adapted-minus-base accuracy <= -margin',
            'noninferiority_test_rejects_at_alpha':lower>-margin,
            'qualification_criteria_changed':False,
            'assumptions':['Known fixed finite frame with case outcomes defined before sampling.',
                           'Uniform sample without replacement, independent of outcome inspection/selection.',
                           'Fixed sample size, scoring, rendering and inference protocol.',
                           'Guarantee targets this finite frame only; not unseen reasoning skills.'],
            'numerics':'Exact integer inversion with standard floating-point hypergeometric tails.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--population',type=int,required=True);p.add_argument('--sample',type=int,required=True)
    p.add_argument('--harmful',type=int,required=True);p.add_argument('--beneficial',type=int,required=True)
    p.add_argument('--alpha',type=float,default=.05);p.add_argument('--margin',type=float,default=.05)
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    r=finite_net_bound(a.population,a.sample,a.harmful,a.beneficial,alpha=a.alpha,margin=a.margin)
    report={'label':a.label,'status':'PREPARATION ONLY; no prospective protocol adopted by this tool',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'result':r}
    a.output.mkdir(exist_ok=False,parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
