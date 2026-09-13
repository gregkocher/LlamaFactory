"""Supplemental fixed-N paired binary accuracy; no qualification/data-reading hook.

K=H+B ~ Bin(n,p), B|K ~ Bin(K,q), delta=p*(2q-1).
Fixed Clopper-Pearson error allocations plus a union bound give conservative
finite-sample coverage. See PAIRED_NET_ACCURACY.md for proof and limitations.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta, binom

METHOD = 'exact_binomial_rectangle_discordance_conditional_gain_v1'


def integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def settings(alpha, margin):
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be finite and in (0,1)')
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('margin must be finite and in [0,1)')


def cp_bounds(successes, trials, tail_alpha):
    """Each Clopper-Pearson tail error <= tail_alpha; zero trials gives [0,1]."""
    integer('successes', successes); integer('trials', trials)
    if successes > trials or not math.isfinite(tail_alpha) or not 0 < tail_alpha < .5:
        raise ValueError('Invalid count or tail allocation')
    lower = 0. if successes == 0 else float(beta.ppf(tail_alpha, successes, trials-successes+1))
    upper = 1. if successes == trials else float(beta.isf(tail_alpha, successes+1, trials-successes))
    return lower, upper


def map_lower(pl, pu, ql):
    f = 2*ql-1
    return (pl if f >= 0 else pu)*f


def map_upper(pl, pu, qu):
    f = 2*qu-1
    return (pu if f >= 0 else pl)*f


def paired_net_bounds(n, harmful, beneficial, *, alpha=.05, margin=.05):
    """Positive difference means adapted-minus-base accuracy; gains offset harms.

    One-sided bound: p interval spends alpha/2 over both tails; q bound spends
    alpha/2 in its one tail. Two-sided interval: alpha/4 in each of four tails.
    Allocation is fixed; no observed-sign-based choice of confidence method.
    """
    integer('n', n, 1); integer('harmful', harmful); integer('beneficial', beneficial)
    settings(alpha, margin)
    if harmful+beneficial > n:
        raise ValueError('Discordant counts exceed n')
    k = harmful+beneficial
    pl, pu = cp_bounds(k, n, alpha/4)
    ql, qu = cp_bounds(beneficial, k, alpha/4)
    qlo, quo = cp_bounds(beneficial, k, alpha/2)
    lower, upper = map_lower(pl, pu, qlo), map_upper(pl, pu, quo)
    return {'method': METHOD, 'n': int(n), 'harmful': int(harmful), 'beneficial': int(beneficial),
            'discordant': int(k), 'concordant': int(n-k), 'alpha': alpha, 'margin': margin,
            'sign_convention': 'adapted accuracy minus base accuracy',
            'net_accuracy_difference': (beneficial-harmful)/n,
            'discordance_probability_interval': [pl, pu],
            'conditional_gain_probability_interval': [ql, qu],
            'conditional_gain_fraction_observed': beneficial/k if k else None,
            'two_sided_confidence': 1-alpha,
            'two_sided_interval': [map_lower(pl, pu, ql), map_upper(pl, pu, qu)],
            'one_sided_confidence_each': 1-alpha,
            'one_sided_lower_bound': lower, 'one_sided_upper_bound': upper,
            'one_sided_pair_warning': 'Separate one-sided bounds are not jointly a 1-alpha interval; use two_sided_interval.',
            'noninferiority_null': 'net accuracy difference <= -margin',
            'noninferiority_test_rejects_at_alpha': bool(lower > -margin),
            'supplemental_status': 'fixed_N_noninferiority_supported' if lower > -margin else 'inconclusive',
            'qualification_criteria_changed': False,
            'fixed_error_allocation': {'p_lower_tail': alpha/4, 'p_upper_tail': alpha/4,
                                       'q_one_sided_tail': alpha/2, 'q_two_sided_each_tail': alpha/4}}


def outcome_bounds(n, *, alpha=.05):
    """All possible (K,B) outcomes, vectorized for exact finite summation."""
    integer('n', n, 1); settings(alpha, .05)
    ks = np.repeat(np.arange(n+1), np.arange(1, n+2))
    bs = np.concatenate([np.arange(k+1) for k in range(n+1)])
    pl = np.zeros(n+1); pu = np.ones(n+1)
    pl[1:] = beta.ppf(alpha/4, np.arange(1,n+1), n-np.arange(1,n+1)+1)
    pu[:-1] = beta.isf(alpha/4, np.arange(n)+1, n-np.arange(n))
    qlo = np.zeros(len(bs)); quo = np.ones(len(bs))
    qlt = np.zeros(len(bs)); qut = np.ones(len(bs))
    success, failure = bs>0, bs<ks
    for arr, tail in [(qlo,alpha/2),(qlt,alpha/4)]:
        arr[success] = beta.ppf(tail, bs[success], ks[success]-bs[success]+1)
    for arr, tail in [(quo,alpha/2),(qut,alpha/4)]:
        arr[failure] = beta.isf(tail, bs[failure]+1, ks[failure]-bs[failure])
    def low(q):
        f=2*q-1
        return np.where(f>=0,pl[ks],pu[ks])*f
    def high(q):
        f=2*q-1
        return np.where(f>=0,pu[ks],pl[ks])*f
    return ks, bs, low(qlo), high(quo), low(qlt), high(qut)


def outcome_probabilities(n, ks, bs, harm_probability, benefit_probability):
    for p in [harm_probability,benefit_probability]:
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Probabilities must be finite and in [0,1]')
    p = harm_probability+benefit_probability
    if p>1:
        raise ValueError('Harm and gain probabilities sum above1')
    q = benefit_probability/p if p else .5
    return binom.pmf(ks,n,p)*binom.pmf(bs,ks,q)


def exact_power(n, harm_probability, benefit_probability, *, alpha=.05, margin=.05):
    """Exact multinomial summation, up to floating-point beta/binomial arithmetic."""
    settings(alpha, margin)
    ks, bs, lower, *_ = outcome_bounds(n, alpha=alpha)
    probs = outcome_probabilities(n,ks,bs,harm_probability,benefit_probability)
    return {'n': n, 'assumed_harm_probability': harm_probability,
            'assumed_benefit_probability': benefit_probability,
            'assumed_net_accuracy_difference': benefit_probability-harm_probability,
            'alpha': alpha, 'margin': margin,
            'probability_of_noninferiority_rejection': float(probs[lower > -margin].sum()),
            'probability_mass_check': float(probs.sum()), 'outcomes_enumerated': len(ks),
            'interpretation': 'Planning power conditional on supplied iid multinomial assumptions; not guaranteed realized success.'}


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('counts'); c.add_argument('--n',type=int,required=True)
    c.add_argument('--harmful',type=int,required=True); c.add_argument('--beneficial',type=int,required=True)
    s=sub.add_parser('plan'); s.add_argument('--n',type=int,nargs='+',required=True)
    s.add_argument('--harm-probability',type=float,required=True); s.add_argument('--benefit-probability',type=float,required=True)
    for command in [c,s]:
        command.add_argument('--alpha',type=float,default=.05); command.add_argument('--margin',type=float,default=.05)
        command.add_argument('--label',required=True,help='Count source or explicitly hypothetical scenario')
        command.add_argument('--output',type=Path,required=True,help='New output directory; never overwrite')
    a=p.parse_args()
    if a.command=='counts':
        result=paired_net_bounds(a.n,a.harmful,a.beneficial,alpha=a.alpha,margin=a.margin)
    else:
        if any(n>5000 for n in a.n):raise ValueError('Quadratic enumeration CLI limit: n<=5000')
        result=[exact_power(n,a.harm_probability,a.benefit_probability,alpha=a.alpha,margin=a.margin) for n in a.n]
    report={'label':a.label,'scope':'SUPPLEMENTAL posthoc development diagnostic or hypothetical planning; not predeclared qualification',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'packages':{'numpy':np.__version__,'scipy':scipy.__version__},'result':result,
            'limitations':['Fixed N and iid paired binary outcomes assumed.',
                           'No correction for adaptive checkpoint selection, repeated looks or multiple separate claims.',
                           'Reads aggregate counts only; no evaluation or confirmation data access.',
                           'Unknown/unfinished outcomes require explicit fixed scoring or sensitivity policy.',
                           'Nonsignificant equality tests are not noninferiority evidence.',
                           'Existing analyze.py criterion remains unchanged.']}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
