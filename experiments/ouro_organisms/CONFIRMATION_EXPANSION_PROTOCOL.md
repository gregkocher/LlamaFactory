# Prospective MCQ extension: fixed-frame census preferred

Status: preparation only, 2026-09-13. No new cohort has been built, no confirmation predictions inspected, and no active qualification criterion changed. The existing 250 confirmation cases per MCQ family stay unchanged and separately reported. The original sufficient harmful-discordance bound and supplemental iid paired tool remain intact.

## Recommendation and cost evidence

After checkpoint selection on development and an explicit prospective freeze, evaluate the **entire remaining eligible ARC-Easy test pool and all 1,240 remaining ordered arithmetic triples**. Report the exact accuracy difference on each frozen benchmark, together with harmful and beneficial case counts. This avoids unnecessary sampling uncertainty; it does not establish performance on unseen skills or a wider population.

The saved development logs show 100 MCQ cases taking 0.876 seconds for base and 1.186 seconds for control after model loading, with batch size 16. These are observed evaluation-loop timings, not end-to-end deployment estimates:

- `research/scale_campaign_20260913/broad_baselines_staged/qualification/base_v1_development/logs/raw_4096.log`
- `research/scale_campaign_20260913/broad_step800_final/paired_step800_v1/qualification/preservation_control_r64_100m-step-800_development/logs/raw_4096.log`

`evaluate.py` scores answer letters at all recurrent views in one forward pass, without autoregressive decoding. A census has at most 2,076 remaining ARC cases plus 1,240 composition cases: at most 3,316 per model, versus 2,000 for the sampling alternative. A crude proportional extrapolation is tens of seconds per model once loaded; longer prompts, adapter overhead, fixed batching, and startup can change this substantially. It is sensible to budget minutes, not hours. No timing run was launched for this proposal. The saved logs make census a better practical choice than additional statistical complexity.

## Available frames and scope

`build_eval.py` selects 300 ARC cases (50 development, 250 confirmation), filtering for four options, an available answer label, and no `cake`, `baking`, or `butter` substring in the question. ARC-Easy test has 2,376 source rows; the exact eligible remaining size must be counted when the new frame is explicitly authorized, using the **original pinned revision**, not the latest dataset. Exclude all existing 300 source questions by identity and content matching, verify unique source IDs, and record collisions rather than silently dropping them after scoring. Original generated ARC IDs are positional, so reconstruct source mapping from the frozen prompts. [Official ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc/blob/main/README.md).

Arithmetic ranges are a=3..24, b=2..15, c=2..6, giving 22×14×5=1,540 ordered triples. The existing generator excludes repeats; subtracting its 300 cases leaves 1,240. These are **ordered generator units**, not independent semantic skills. Swapping a and b can preserve the same mathematical operation, and many cases share intermediate values or answers. Record counts of unique ordered triples, unordered-addend semantic classes, and answer values when building the frame. Do not remove semantic duplicates from this agreed ordered frame after seeing outcomes. Some remaining cases will be mathematically equivalent to prior development/confirmation cases; the claim is new rendered-unit performance, not semantic novelty.

## Freeze before any new inference

1. Choose the checkpoint using development only. Freeze both arm identities, base revision, endpoints, margins, scoring, and how any original confirmation results will be used. Keep the original 250 cases separately reported; do not pool them into a purported new random sample.
2. Build a versioned immutable full remaining frame, recording original dataset revisions, filters, exclusion IDs/hashes, row counts, and a content manifest. Do not rerun the original builder with a larger count: its shared RNG affects later composition and baking rows.
3. Determine composition distractors and option permutation for **every remaining unit** from a frozen per-unit seed derived from its triple. ARC retains source options. Rendering must precede any sample selection. Freeze option scoring, tie breaking, loop view, truncation handling, and treatment of missing outputs.
4. Use identical cases and inference settings for base and both arms. For census, freeze full-frame ordering and batch membership before inference. If using sampling instead, batch size one is the simplest way to keep unit predictions independent of the selected sample's padding/co-batching; otherwise define stable companions before selection. Floating-point inference can be batch-sensitive.
5. Evaluate the full census once, or the fixed-size sample below. Preserve missing/failed cases and follow a frozen recovery rule; do not drop them from denominators or add cases until a confidence interval passes.

For a census, the paired difference (B−H)/N is exact for this frame under the frozen evaluation protocol. No sampling confidence interval is required. A 5 percentage point margin can be assessed directly against that value, if adopted prospectively. This is a finite benchmark decision, not an iid population certificate. Recurrence-benefit endpoints involving differences of differences require their own definition; the binary-accuracy helper does not cover them.

## Optional 1,000-case sampling fallback

If a census becomes operationally inconvenient, select 1,000 units uniformly without replacement from each frozen remaining frame with a dedicated sampling seed fixed independently of outcomes. Require N≥1,000. The original 250 cases are excluded and reported separately. A pseudorandom seed is an operational approximation to the randomized sampling design; a realized fixed subset alone does not create a frequentist guarantee.

Let H and B be harmful and beneficial **sample** counts, and Hpop and Bpop their unknown totals in the remaining finite frame. Under simple random sampling, each marginal count is hypergeometric. Allocate alpha/2 to the upper bound on Hpop and alpha/2 to the lower bound on Bpop. No independence between these two counts is needed. Hypergeometric sampling describes draws without replacement from a fixed binary population. [R reference](https://www.stat.math.ethz.ch/R-manual/R-devel/library/stats/html/Hypergeometric.html); [research on hypergeometric confidence intervals](https://arxiv.org/abs/2109.05624).

For an observed count x, invert over feasible integer totals M in [x,N−n+x]:

- L(x) = min M such that P_M(X≥x) > alpha/2.
- U(x) = max M such that P_M(X≤x) > alpha/2.

The probability that the true total falls below L is at most alpha/2: its upper-tail p-value can fall below the threshold only with that probability. The analogous lower-tail statement gives the upper bound. A union bound therefore gives coverage at least 1−alpha for

`population adapted-minus-base accuracy >= (L(B) - U(H)) / N`.

This proof holds globally, including zero observed discordances; it uses fixed alpha allocation and no outcome-dependent choice of bounds. At a census n=N the feasible total is a singleton and the difference is exact. The implementation uses binary search over integer totals and standard floating-point SciPy tails, rather than arbitrary-precision arithmetic. Its intervals can be conservative.

To reject noninferiority null delta≤−margin, require the lower bound to be **strictly greater** than −margin. Nonsignificance of a difference is not evidence of retention. A joint all-required-endpoints qualification can use intersection-union testing at alpha per endpoint, if frozen prospectively; this does not provide simultaneous individual confidence statements or correct repeated checkpoint selection.

Why not simply append 750? Ideally, an initial uniform 250 subset followed by a uniform 750 from its complement gives an unconditional uniform 1,000 subset: each final set has probability C(1000,250)/[C(N,250)C(N−250,750)] = 1/C(N,1000). But conditional on the already-fixed initial set it is not a fresh simple random 1,000 sample. Original composition options were generated interleaved with selected units, complicating the required fixed rendering of all unobserved units. A separate remaining-frame cohort avoids this bookkeeping. A weighted census stratum plus a random remaining stratum could also be analyzed, but is unnecessary here.

## Prepared helper and checks

`finite_population_accuracy.py` is standalone, counts-only, and has no data builder, model execution, or qualification hook. `paired_net_accuracy.py` is unchanged. Example with **hypothetical**, not observed, counts:

```sh
python finite_population_accuracy.py --population 1240 --sample 1000 \
  --harmful 120 --beneficial 130 --label hypothetical_fixed_frame \
  --output /tmp/finite_accuracy_example_v1
python -B -m unittest test_finite_population_accuracy -v
```

Five CPU tests pass: integer inversion against brute-force tails; complete small-population outcome enumeration; exact census and strict margin threshold; zero-discordance uncertainty; and invalid inputs. The coverage test enumerates all harmful/beneficial population totals for N=1..12 and sample sizes 1, floor(N/2), and N. These are sanity checks supporting the proof, not a replacement for it. No benchmark data or model predictions enter the tests.
