# C4-filtered - pre-registered design (executed 2026-09-06)

Status: executed. The gate was resolved on 2026-09-05 and the condition ran
on 2026-09-06 under run version `v1_c4_filtered`, implementation commit
`e68b3a7`. 155 of the 500 images cleared the threshold; final test metrics
were recorded across seeds 0/1/2.

Result: parity with C1, a higher mean than C4 - test df F1 `0.6657 +/- 0.0182`
against C1 `0.6598 +/- 0.0423` and C4 `0.6115 +/- 0.0423`. The `+0.0059` over C1
is less than one test image's worth on 16 df; the `+0.0541` over C4 exceeds
either seed spread. This is consistent with the distance criterion removing a loss
from the unfiltered pool without beating plain duplication; no significance test
was performed. The full result and its limits are in
`README.md`, section C4-filtered selection experiment.

The experimental parameters and selection rules in sections 3 to 7 are retained.
The gate in section 2 was resolved using the corrected batch. The text below
clarifies the historical decision and the limits of the descriptive results.

## Gate outcome (2026-09-05)

The gate was resolved from the corrected batch identity and the sampler sweep.

1. §2's numbers came from the wrong batch. This document was written
2026-08-11, five days before the batch mix-up was found. Its quoted synthetic
distribution - range `0.160`-`1.180`, median `1.086`, `p1 = 0.6564`,
`p5 = 0.8293` - matches `outputs/figures/ddpm_memorization_diagnostic.json`
exactly, which measured the earlier epoch-60 batch. On the published
epoch-100 pool that C4 actually trains on
(`ddpm_memorization_diagnostic_epoch100.json`, content SHA-256
`e0bb7015...`), the same distribution is `p0 = 0.1290`, `p1 = 0.3211`,
`p5 = 0.4724`, `p25 = 0.8508`, `p50 = 0.9990`, `p100 = 1.1843`.

The §4 threshold is unchanged at `0.901291012763977` - it is derived from the
14 real val `df`, which no batch correction touches.

Since `p25 = 0.8508` is at or below that threshold and `p50 = 0.9990` is above
it, between 125 and 250 of the 500 published images clear the bar, i.e.
25-50% of the pool. §2 predicted "somewhere in the region of one to five per
cent" and "a few dozen images at most". That prediction was about a different
batch and is void. The §4 shortfall rule (fewer than 50 accepted => do not run)
is cleared with a wide margin.

The percentile bound was available at the gate decision. The subsequent filter
run recomputed ResNet-18 distances and supplied the exact count, fraction, and accepted-image manifest required
by section 5, as recorded in the status above.

2. The sweep did not support the alternative pool proposed in section 2. Its step 2 was "regenerate a
larger pool under whichever sampler configuration recovers the most saturation
and contrast". The sweep ran on 2026-08-16 and found no such configuration
worth regenerating under: every setting sits at `embed_nn` 0.97-1.07 against
the real-val reference of `0.3678`, so none moves samples onto the real `df`
manifold, and the configurations that do recover saturation (`eta 1.0`) produce
fluorescent cyan, magenta and orange frames that are not skin. The sweep provided no measured distance improvement to justify generating
a replacement pool under those settings.

The published pool was retained for the filtering experiment. The original
deferral was based on measurements from a different batch.

The completed Colab run used 3 seeds × 20 epochs plus the filter pass, with
the experimental parameters in sections 3-7 unchanged.

The selection rule was fixed before C4-filtered training. Outcome-based
threshold tuning would invalidate that comparison.

---

## 1. Research question

The memorisation diagnostic measures per-image distance from synthetic `df`
to the real `df` reference. The earlier epoch-60 batch of 500 images ranged
from `0.160` (closest) to `1.180` (farthest), median `1.086`, in the C1 embedding.
Those historical values are superseded by the epoch-100 values above.

Is there a subset of the synthetic pool that is close enough to real `df`
to be more useful as augmentation than the pool as a whole?

The comparison evaluates a fixed distance-based selection criterion. Any
observed difference remains descriptive under the fixed split and small sample.

## 2. Original sweep gate

> Superseded 2026-09-05 - see "Gate outcome" above. The numbers in this
> section were measured on the epoch-60 batch, not the published epoch-100 pool
> they were originally used to describe. The historical measurements and
> proposed sequence are retained below.

The original deferral used the following estimate of the accepted count.

The acceptance threshold has to be anchored to something real (see §4). The
loosest defensible anchor is the *maximum* distance among the 14 genuinely-new
real `df`, which is `0.9012910127639771`. Against that, the synthetic
distribution has `p1 = 0.6564393639564514` and `p5 = 0.8293142914772034` -
the original estimate was that one to five per cent of the 500 images would
clear the threshold. This estimate applies to the earlier batch.

The proposed sequence was:

1. run the sweep,
2. regenerate a larger pool under whichever sampler configuration recovers the
   most saturation and contrast,
3. filter *that* pool,
4. compare.

Under that original plan, step 3 followed pool regeneration. Pool identity
was a parameter of the design; the executed pool is identified above.

## 3. Conditions

`df_target_count` stays at 585 for every condition. `build_classifier_frame`
requires C1 and C4 to differ only in the *source* of the `df` rows; filtering to
a smaller pool and letting the total fall would confound "filtering" with "less
oversampling", and the comparison would answer neither question.

| condition | df rows at 585 |
|---|---|
| `C1` (existing) | 85 real, duplicated to 585 |
| `C4` (existing) | 85 real + 500 synthetic |
| `C4-filtered` (new) | 85 real + accepted synthetic + real duplication filling the remainder to 585 |

Real duplication supplies the remaining rows to match the C1 construction.
The `C4-filtered` versus `C1` comparison measures the observed effect of
replacing those duplicates with accepted synthetic images.

Everything else is held fixed: same fixed `lesion_id` split, same seeds
`{0, 1, 2}`, same 20 epochs, same architecture, same `build_transforms`
augmentation, same `validation_df_f1_strict_improvement` selection rule.

## 4. Fixed acceptance rule

- Judge: `outputs/classifier_df585/checkpoints/C1_seed2/best.pt`,
  penultimate 512-d features, L2-normalised, `img_size 128`. This is the judge
  the diagnostics already use. It is trained on real images only, so it never
  saw the pool it is filtering. It must not be a C4 model (circular) and must
  not be PanDerm (pretraining corpus cannot be shown to exclude HAM10000).
- Distance: nearest-neighbour into the 85 real train `df` and their
  horizontal mirrors, matching the diagnostics, since training used
  `RandomHorizontalFlip`.
- Threshold: accept a synthetic image when its distance is at or below
  `max(val_df_to_train)` - the farthest of the 14 genuinely-new real `df`.
  On the current record that value is `0.9012910127639771`, but it is
  recomputed from the val set for whichever pool is used, never hardcoded.
  The rule is "no farther from real `df` than the most unusual genuinely new
  real `df`", which is defensible without reference to any outcome.
- No top-N. A fixed count would silently change meaning with pool size and
  invites tuning. The count is whatever clears the threshold, and it is
  reported.
- If fewer than 50 images clear the threshold, the condition is not run;
  the shortfall is recorded and reported as the result. Training a condition on
  such a small accepted set was excluded by the preregistered minimum.

## 5. Required outputs

Alongside the usual per-seed metrics:

- number and fraction of the pool accepted, and the distance distribution of
  the accepted subset;
- diversity of the accepted subset - within-set nearest-neighbour spacing
  against the real `df` set, exactly as in the memorisation diagnostic.
  This records whether filtering also reduces measured within-set variety.
- the accepted-image manifest, so the condition is reproducible.

## 6. Reading the result

- `C4-filtered` > `C4` and > `C1`: a higher observed mean under the selection criterion.
- `C4-filtered` ≈ `C1`: little observed difference from duplication.
- `C4-filtered` ≈ `C4`: little observed difference from the unfiltered pool.
- fewer than 50 accepted: reported as-is; no condition is run.

With 16 real `df` in the test split, estimates are imprecise. No significance
test was performed, so statistical significance has not been established.
Report the procedure, measured distances, and observed differences as descriptive results.

## 7. Scope

- Existing C0/C1/C4 results remain frozen.
- The deployed classifier remains C1 seed 2.
- The threshold in section 4 remains fixed regardless of df F1.
- The planned final test evaluation occurs only after all conditions above are fixed.
