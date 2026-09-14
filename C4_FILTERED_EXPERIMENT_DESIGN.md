# C4-filtered — pre-registered design (executed 2026-09-06)

Status: **executed.** The gate was resolved on 2026-09-05 and the condition ran
on 2026-09-06 under run version `v1_c4_filtered`, implementation commit
`e68b3a7`. 155 of the 500 images cleared the threshold; the test split was
evaluated once, across seeds 0/1/2.

Result: **parity with C1, a real gap over C4** — test df F1 `0.6657 +/- 0.0182`
against C1 `0.6598 +/- 0.0423` and C4 `0.6115 +/- 0.0423`. The `+0.0059` over C1
is less than one test image's worth on 16 df; the `+0.0541` over C4 exceeds
either seed spread. The distance criterion removed the harm the unfiltered pool
was doing without beating plain duplication. The full result and its limits are in
`README.md`, section Turning the diagnostic into a selection rule: C4-filtered.

Every rule below is as pre-registered. Nothing in §3 to §7 was changed at any
point; only the gate in §2 was resolved, and it resolved against the reasoning
that created it. See "Gate outcome" immediately below.

## Gate outcome (2026-09-05)

Two things settle the gate, and they point the same way.

**1. §2's numbers came from the wrong batch.** This document was written
2026-08-11, five days before the batch mix-up was found. Its quoted synthetic
distribution — range `0.160`–`1.180`, median `1.086`, `p1 = 0.6564`,
`p5 = 0.8293` — matches `outputs/figures/ddpm_memorization_diagnostic.json`
exactly, which measured the earlier **epoch-60** batch. On the published
epoch-100 pool that C4 actually trains on
(`ddpm_memorization_diagnostic_epoch100.json`, content SHA-256
`e0bb7015...`), the same distribution is `p0 = 0.1290`, `p1 = 0.3211`,
`p5 = 0.4724`, `p25 = 0.8508`, `p50 = 0.9990`, `p100 = 1.1843`.

The §4 threshold is unchanged at `0.901291012763977` — it is derived from the
14 real val `df`, which no batch correction touches.

Since `p25 = 0.8508` is at or below that threshold and `p50 = 0.9990` is above
it, **between 125 and 250 of the 500 published images clear the bar**, i.e.
25–50% of the pool. §2 predicted "somewhere in the region of one to five per
cent" and "a few dozen images at most". That prediction was about a different
batch and is void. The §4 shortfall rule (fewer than 50 accepted ⇒ do not run)
is cleared with a wide margin.

The exact count is not computable here: the distance is a ResNet-18 embedding
and this machine is deliberately torch-free. Recompute it on Colab as the
first step of execution — the bound above is read off recorded percentiles, and
the count, fraction and accepted-image manifest required by §5 must come from
an actual run of the filter.

**2. The "strong version" in §2 no longer exists.** Its step 2 was "regenerate a
larger pool under whichever sampler configuration recovers the most saturation
and contrast". The sweep ran on 2026-08-16 and found no such configuration
worth regenerating under: every setting sits at `embed_nn` 0.97–1.07 against
the real-val reference of `0.3678`, so none moves samples onto the real `df`
manifold, and the configurations that do recover saturation (`eta 1.0`) produce
fluorescent cyan, magenta and orange frames that are not skin. Regenerating
under them would produce a pool that is worse on the exact distance the filter
uses.

**Therefore the current pool is not the weak version of this experiment — it is
the only version.** Running it on today's pool no longer burns the
pre-registration on a likely null; the gate was constructed on a number that
turned out to belong to another batch.

**Execution still needs Colab** (3 seeds × 20 epochs, plus the filter pass).
Nothing about §3–§7 changes on the way there.

This document fixes every rule *before* any number is seen, so the result
cannot be chosen after the fact. That is the whole point: a filtering step that
picks its threshold after looking at outcomes is not evidence of anything.

---

## 1. The question

The memorisation diagnostic produced something the project did not have before:
a per-image distance from each synthetic `df` to the real `df` manifold. That
distance is a **distribution**, not a single number — in the C1 embedding the
500 published images run from `0.160` (closest) to `1.180` (farthest), median
`1.086`.

So: is there a subset of the synthetic pool that is close enough to real `df`
to be more useful as augmentation than the pool as a whole?

If yes, the diagnostic has been converted from a description into a selection
criterion, and the improvement is attributable to a measurement rather than to
luck. If no, that is a clean negative and costs one classifier training run.

## 2. Why this is gated on the sweep

> **Superseded 2026-09-05 — see "Gate outcome" above.** The numbers in this
> section were measured on the epoch-60 batch, not the published epoch-100 pool
> they claim to describe. Retained unedited as the original pre-registered
> reasoning.

The current pool is a poor candidate for filtering, and the numbers already on
record say so.

The acceptance threshold has to be anchored to something real (see §4). The
loosest defensible anchor is the *maximum* distance among the 14 genuinely-new
real `df`, which is `0.9012910127639771`. Against that, the synthetic
distribution has `p1 = 0.6564393639564514` and `p5 = 0.8293142914772034` —
so only somewhere in the region of one to five per cent of the 500 clear even
that loosest bar. Filtering the current pool therefore selects "least bad", from
a few dozen images at most, and a null result is the likely outcome.

The strong version of this experiment is:

1. run the sweep,
2. regenerate a larger pool under whichever sampler configuration recovers the
   most saturation and contrast,
3. filter *that* pool,
4. compare.

Running step 3 on today's pool first would burn the pre-registration on the
weakest available version of the question. The pool is therefore a **parameter**
of this design, not a constant.

## 3. Conditions

`df_target_count` stays at **585** for every condition. `build_classifier_frame`
requires C1 and C4 to differ only in the *source* of the `df` rows; filtering to
a smaller pool and letting the total fall would confound "filtering" with "less
oversampling", and the comparison would answer neither question.

| condition | df rows at 585 |
|---|---|
| `C1` (existing) | 85 real, duplicated to 585 |
| `C4` (existing) | 85 real + 500 synthetic |
| `C4-filtered` (new) | 85 real + accepted synthetic + real duplication filling the remainder to 585 |

The filler is real duplication precisely because that is what C1 does. If
`C4-filtered` beats `C1`, the accepted synthetic images did something that
duplication does not; if it only matches `C1`, they did not.

Everything else is held fixed: same fixed `lesion_id` split, same seeds
`{0, 1, 2}`, same 20 epochs, same architecture, same `build_transforms`
augmentation, same `validation_df_f1_strict_improvement` selection rule.

## 4. The acceptance rule — fixed now

- **Judge**: `outputs/classifier_df585/checkpoints/C1_seed2/best.pt`,
  penultimate 512-d features, L2-normalised, `img_size 128`. This is the judge
  the diagnostics already use. It is trained on real images only, so it never
  saw the pool it is filtering. It must not be a C4 model (circular) and must
  not be PanDerm (pretraining corpus cannot be shown to exclude HAM10000).
- **Distance**: nearest-neighbour into the 85 real train `df` **and their
  horizontal mirrors**, matching the diagnostics, since training used
  `RandomHorizontalFlip`.
- **Threshold**: accept a synthetic image when its distance is at or below
  `max(val_df_to_train)` — the farthest of the 14 genuinely-new real `df`.
  On the current record that value is `0.9012910127639771`, but it is
  **recomputed from the val set for whichever pool is used**, never hardcoded.
  The rule is "no farther from real `df` than the most unusual genuinely new
  real `df`", which is defensible without reference to any outcome.
- **No top-N.** A fixed count would silently change meaning with pool size and
  invites tuning. The count is whatever clears the threshold, and it is
  reported.
- If fewer than **50** images clear the threshold, the condition is **not run**;
  the shortfall is recorded and reported as the result. Training a condition on
  a handful of images would produce noise, not an answer.

## 5. Required outputs

Alongside the usual per-seed metrics:

- number and fraction of the pool accepted, and the distance distribution of
  the accepted subset;
- **diversity of the accepted subset** — within-set nearest-neighbour spacing
  against the real `df` set, exactly as in the memorisation diagnostic.
  Filtering by proximity to real data can easily collapse variety, and a
  filtered set that is closer but far less varied is a different animal from
  one that is closer and equally varied. This must be visible, not inferred.
- the accepted-image manifest, so the condition is reproducible.

## 6. Reading the result

- `C4-filtered` > `C4` and > `C1`: the selection criterion worked. This is the
  only outcome that supports the claim that the diagnostic improved the result.
- `C4-filtered` ≈ `C1`: the accepted images add nothing beyond duplication.
- `C4-filtered` ≈ `C4`: filtering did not matter at this pool quality.
- fewer than 50 accepted: reported as-is; no condition is run.

**With 16 real `df` in the test split, none of these differences can be
significant, and no significance test is performed** — consistent with the
project's standing constraint. Everything here is descriptive and suggestive,
and must be written up that way. A difference of a few hundredths of df F1 on
16 images is not a finding; the honest deliverable is the procedure and the
measured distances, not the delta.

## 7. What this must not become

- Not a reason to re-open or re-run any existing C0/C1/C4 result. Those stay
  frozen.
- Not a replacement for the deployed classifier, which stays C1 seed 2.
- Not a threshold that gets adjusted after seeing df F1. If the rule in §4
  produces an uninteresting answer, the answer is uninteresting.
- Not evaluated on the test split more than once, and not until the conditions
  above are settled.
