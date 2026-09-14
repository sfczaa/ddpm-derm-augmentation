# CoCa synthetic-mixture dose-response diagnostic

## Decision question

Does validation df behaviour worsen as the fixed 585-row df training support
contains more images from the already-fixed synthetic candidate?

This is a post-v4 exploratory diagnostic. It does not authorize formal
training, select a new candidate, or evaluate test data.

## Preregistered design

- Use the fixed train/validation split only; never load the test split.
- Keep total train-df support at 585 rows in every condition.
- Run synthetic counts `0, 125, 250, 375, 500`; fill the remaining extra df
  slots by deterministic duplication of the 85 real train-df rows.
- Choose synthetic rows as nested prefixes after sorting the fixed 500-row
  candidate by SHA-256 of `image_id`. Do not use validation embeddings,
  distances, labels, or classifier scores to select images.
- Choose duplicated real-df rows from a stable hash-ordered cycle prefix, so
  successive conditions remove only the tail of the same duplicate sequence.
- Keep model and objective fixed: frozen `coca_ViT-B-32`,
  `laion2b_s13b_b90k`, native preprocessing, focal cross-entropy with
  inverse-frequency alpha and gamma 2, seed 0, five epochs.
- Report validation df F1, macro-F1, prediction counts, histories, and the
  immutable mixture identity for all five conditions. Interpret the sequence
  descriptively; there is no pass/fail threshold.

## Evidence boundary and next decision

The validation df set has only 14 examples, this diagnostic uses one seed, and
the five conditions reuse the same validation set. Therefore the result can
support or weaken the domain-shift mechanism, but cannot establish causality or
justify selecting the best-looking condition for a formal test run.

If a coherent dose response appears, preregister one separate candidate and
validation gate before any formal training. If it does not, stop synthetic-ratio
tuning and investigate representation/model mismatch without test access.

## Execution and checkpointing

- Colab: five sequential five-epoch validation-only runs.
- Checkpoint cadence: every completed epoch; worst-case loss is one unfinished
  epoch. Resume must use the same attempt identity and no concurrent runtime.
