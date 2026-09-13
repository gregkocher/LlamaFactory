# Supplemental paired net-accuracy audit

This analysis is a **posthoc development diagnostic**, not a replacement for the current qualification criterion. It can support a future prospective decision only after the method, margin, scoring, endpoints, sample sizes and stopping policy are explicitly frozen before untouched confirmation. No confirmation examples or outcomes were opened for this audit, and no active criterion or frozen evaluation dataset was changed.

## What the existing certificate establishes

For paired binary correctness, let H count cases correct in base but incorrect in adapted, B count the reverse, and n count all pairs. Define **delta = adapted accuracy − base accuracy = (B − H)/n** in the sample. For population probabilities, delta = p_B − p_H. A harmful case and a beneficial case can cancel in mean accuracy while both remain evidence that the models solve different cases.

`analyze.py::retention_bound` bounds mean degradation by a one-sided 95% Clopper–Pearson upper bound on p_H. This is a valid sufficient certificate because −delta = p_H − p_B ≤ p_H. However, its population limit is p_H, not p_H − p_B. If p_H=12% and p_B=13%, the mean improves by one percentage point, but this sufficient certificate cannot establish a five-point margin merely by increasing n: its upper bound converges toward 12%. Failure of that certificate does not establish mean degradation. It asks for a low probability of losing an individually solved case; it does not guarantee literal preservation of every example.

The supplemental tool estimates the **net marginal accuracy difference**, allowing beneficial changes to offset harmful changes. Report both quantities rather than substituting one for the other. This binary tool does not apply to the recurrence-benefit difference with possible changes of magnitude two; that existing endpoint needs a separate analysis.

## Prior statistical methods and choice

Tango develops a score test for paired noninferiority and a corresponding score interval, including zero off-diagonal cells. The paper explicitly distinguishes discordance from marginal accuracy difference. Its score reference distribution is asymptotic; numerical availability at zero discordances is not a finite-sample coverage guarantee. It is a reasonable efficiency-oriented future option, but is not the implementation here. [Tango, 1998, paper](https://www.site.uottawa.ca/~nat/Courses/csi5388/Tango.paired.pdf).

Newcombe compares methods for paired proportion differences and discusses the weaknesses of simple asymptotic and conditional intervals, with better-performing profile and score constructions. In particular, a conditional interval alone must not be casually reinterpreted as an unconditional risk-difference interval. [Newcombe, 1998, paper](https://www.eiti.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf).

For a small, auditable addition, `paired_net_accuracy.py` implements a **conservative exact-binomial confidence-region construction**, derived below. It is not branded as Tango's or Newcombe's method. Its ingredients are Clopper–Pearson tail inversion and a union bound. Clopper–Pearson intervals guarantee at least their nominal binomial coverage, usually at the cost of extra width. [Clopper and Pearson, 1934](https://doi.org/10.1093/biomet/26.4.404), [Thulin, 2014](https://arxiv.org/abs/1303.1288).

## Construction and finite-sample proof

Assume independent, identically distributed paired binary outcomes at a fixed sample size n. Let K=H+B, p=p_H+p_B and q=p_B/p when p>0. Then

- K follows Binomial(n,p).
- Conditional on K=k, B follows Binomial(k,q).
- delta=p(2q−1).

When p=0, delta=0 and q can be any number in [0,1]. When the observed k=0, the tool uses [0,1] for q: no fictitious precision or zero-width interval.

Write CP(x,m;t) for a Clopper–Pearson interval with error at most t in each tail. Its lower endpoint is zero at x=0, otherwise the t quantile of Beta(x,m−x+1). Its upper endpoint is one at x=m, otherwise the upper-tail-t quantile of Beta(x+1,m−x). With m=0 the interval is [0,1]. Numerical implementation uses SciPy beta quantiles, with analytic boundaries.

For a **one-sided 1−alpha lower bound on delta**:

1. Compute [p_L,p_U]=CP(K,n;alpha/4). Its total failure probability is at most alpha/2.
2. Compute q_L using the lower endpoint of CP(B,K;alpha/2). Conditional on every possible K, this one-sided coverage is at least 1−alpha/2; averaging over K preserves that guarantee.
3. On the intersection of those events, minimize p(2q−1) over p in [p_L,p_U] and q in [q_L,1]. This gives L=p_L(2q_L−1) if q_L≥1/2, otherwise L=p_U(2q_L−1).

The union bound gives P(L≤delta)≥1−alpha. No independence between the confidence events is needed. The sign-dependent corner evaluation is minimization over a single confidence region, not selection between statistical methods or data-dependent alpha allocations. All error allocations are fixed globally.

The upper bound is analogous. For a **two-sided 1−alpha interval**, use CP(K,n;alpha/4) and CP(B,K;alpha/4), spending alpha/4 on each of four tails, then minimize/maximize delta over the rectangle. The two separate one-sided 95% bounds must not be presented together as a jointly 95% interval; the tool returns a distinct two-sided interval.

For margin m≥0, test H0: delta≤−m against H1: delta>−m. Reject only if L>−m. Under any member of the null, rejection entails L>delta, whose probability is at most alpha. This is a valid conservative fixed-N noninferiority test; it does not depend on failing to reject an equality null. Failure to reject is inconclusive, not proof of inferiority. The coverage guarantee is mathematical; beta/binomial numerical evaluations are ordinary floating-point calculations, not arbitrary-precision certified arithmetic.

## Examples and scoring uncertainty

At alpha=.05 and margin=.05:

| Explicitly supplied or hypothetical counts | H | B | Net estimate | One-sided lower bound |
|---|---:|---:|---:|---:|
| Illustrative swaps, n=100 |12|13|+1.00 pp|−13.47 pp|
| Reported official counts, n=100; not reverified here |10|20|+10.00 pp|−2.33 pp|
| Hypothetical n=250 |30|33|+1.20 pp|−6.76 pp|
| Hypothetical n=500 |60|65|+1.00 pp|−4.21 pp|
| Hypothetical n=1,000 |120|130|+1.00 pp|−2.47 pp|

The reported 20-gain/10-loss example would pass this supplemental fixed-N mean-accuracy test, while failing the original harm-only certificate. That does not retrospectively change qualification.

**The illustrative H=12/B=13 example is not the final semantic-development count.** The supplied review includes 13 definite gains and 13 definite harms when the one correct-to-unfinished transition counts as harmful. Two uncertain baseline correctness cases permit H in {13,14} and B in {13,14}. Those four consistent completions yield estimates from −1 to +1 pp and lower bounds approximately −16.29 to −13.79 pp. All are inconclusive. Do not discard the unfinished case or choose a favorable resolution of uncertain labels. Taking the worst lower bound over all permissible label completions is conservative with respect to that unresolved scoring uncertainty; it is not permission to relabel after seeing results.

Exact intervals address sampling uncertainty, not errors in grading, dependence between questions, checkpoint selection or distribution mismatch. Fixed benchmark totals are descriptive unless a defensible population/sampling interpretation is specified.

## Power planning: 500 math versus 250 or 1,000 MCQ

The tool sums the complete multinomial distribution over all possible (K,B), rather than using Monte Carlo. These probabilities assume the specified true case-transition rates and the same conservative test. They are not promises about a future dataset.

| Assumed harmful / beneficial probability | n=250 | n=500 | n=1,000 |
|---|---:|---:|---:|
|12% / 13% (net +1 pp)|30.8%|62.0%|92.4%|
|13% / 13% (net 0 pp)|20.3%|43.4%|77.5%|
|5% / 5% (net 0 pp)|33.7%|73.7%|98.3%|

Thus 500 GSM cases are a sensible cost-limited initial **fixed** sample, but do not ensure decisive mean-retention evidence when case swaps are common. The favorable-looking n=500 count example is not an 80% power guarantee. With inexpensive teacher-forced MCQ evaluation, expanding ARC/composition from 250 to around 1,000 cases is a useful prospective option. At 26% discordance and no net change, even 1,000 gives only about 77.5% power for this conservative construction. A prespecified score method could improve efficiency but requires its own validation and explicit adoption.

Do not evaluate 250 cases, add more only when the interval fails, and keep using a fixed-N threshold. Freeze a final N or a valid sequential alpha-spending/confidence-sequence design before accessing confirmation. Cheap MCQ scoring does not make optional stopping valid.

## MCQ expansion feasibility from code only

`build_eval.py` currently shuffles ARC-Easy test, retains four-option questions with a valid answer key, excludes cake/baking/butter words in the question, and takes 300 total: 50 development plus 250 confirmation. The official card lists 2,376 raw test examples. That suggests ample raw room for 750 additional confirmation cases, but the number surviving filters and deduplication has **not** been measured here. [AI2 dataset card](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/README.md).

A future augmentation should use the original frozen source revision, preserve existing rows, exclude all prior development and confirmation identities, and append new IDs with source IDs/content hashes. Existing synthetic `arc_####` IDs omit the original dataset ID, so disjointness should use normalized question/options content as well. The replay builder uses GSM/WikiText training splits, not ARC; cross-source near duplicates and pretraining contamination remain possible.

Arithmetic composition samples unique ordered triples a in [3,24], b in [2,15], c in [2,6], with task (a+b)c. This gives exactly 22×14×5=1,540 parameter triples. After the current 300, 1,240 unused ordered triples remain, enough to append 750 for 50 development plus 1,000 confirmation. This arithmetic follows from generator code; no frozen confirmation rows were opened. Commuted a/b values create redundant problems: canonicalizing the unordered pair leaves 1,150 possible classes over all c, still potentially enough for 1,050 total rows but with little diversity headroom. Actual disjoint availability requires a later content audit. Different triples also share intermediate sums and final answers; increasing N estimates performance on this narrow generator, not 1,000 distinct reasoning skills.

The current generator samples triples without replacement from a finite space. The tool's exact proof is for iid multinomial outcomes; that proof must not silently be asserted for a changed finite-population or clustered sampling design. A future protocol should explicitly define its sampling target and use an appropriate finite-population analysis if needed. Likewise, merely generating new distractor orders for the same arithmetic problem is not a new independent case.

Do not simply change the ARC limit and rerun `build_eval.py`: its shared RNG also drives later composition and baking construction, so downstream frozen questions would move. Use a separate, versioned append-only augmentation and dedicated RNG after a prospective decision. No augmentation was implemented in this audit.

## Multiple endpoints and selection

For one prospective **all-required-endpoints-retained** decision across both arms and all required families, intersection–union testing applies. If every endpoint must reject its noninferiority null at alpha=.05, then under the global null at least one endpoint null is true, and the probability that *all* reject is bounded by the rejection probability of that true endpoint, at most .05. Independence is unnecessary. Bonferroni is therefore not automatically required for this single conjunction decision.

This does not make all displayed 95% intervals simultaneous, nor license separate family-specific success claims with 95% familywise coverage. Simultaneous intervals/separate multiplicity-controlled claims need an explicit adjustment. Checkpoint searching, picking a passing arm, repeated looks, changing margins or optional sample expansion remain distinct multiplicity/selection issues. A prospective all-pass rule must specify every required endpoint in advance. The existing qualification rule remains untouched.

## Usage and validation

Run from `experiments/ouro_organisms`:

```bash
python paired_net_accuracy.py counts --n 100 --harmful 12 --beneficial 13 \
  --label hypothetical_swaps --output /tmp/net_accuracy_example_v1
python paired_net_accuracy.py plan --n 250 500 1000 \
  --harm-probability .12 --benefit-probability .13 \
  --label hypothetical_swap_power --output /tmp/net_accuracy_power_v1
python -B -m unittest test_paired_net_accuracy -v
```

The CLI reads only supplied counts/probabilities, writes a new directory and refuses to overwrite it. Dependencies are NumPy/SciPy, already used by `analyze.py`; no model stack or new training dependency is needed. Reports include code hash, package versions, fixed alpha allocation and an explicit supplemental label.

Nine CPU tests passed: zero discordances, all gains/harms, arm symmetry, scalar/vector agreement for every n=12 outcome, exact probability-mass and coverage enumeration across n=1,2,5,10,20 and a triangular parameter grid, null-boundary noninferiority size checks, a hand-calculated power case, invalid inputs and immutable CLI output. Enumeration sanity checks support implementation correctness; the coverage proof above supplies the general guarantee.

Saved numeric examples/power and semantic-count sensitivity: `research/scale_campaign_20260913/statistics_net_accuracy_v1/examples_and_planning.json` in the project directory. No model inference, API grading, active criteria edits, or confirmation-data reads were performed.
