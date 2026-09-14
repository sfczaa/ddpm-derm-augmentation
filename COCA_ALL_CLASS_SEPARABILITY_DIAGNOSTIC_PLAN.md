# Frozen-CoCa all-class embedding separability diagnostic

## Decision question

Do the fixed frozen CoCa image embeddings retain enough signal to separate real
`df` from the other six classes? And if signal exists, is the current
image-level linear-head training failure more plausibly an
optimization / augmentation / objective problem than a representation that is
simply inseparable?

This is a post-mixture descriptive diagnostic. It does not authorize formal
training, select a synthetic ratio or candidate, or evaluate test data.

## Prior evidence this builds on

- v4 focal inverse-frequency validation failed the C4 non-collapse gate:
  best validation `df` F1 = 0.0, predicted `df` count = 0.
- The frozen-embedding df diagnostic measured validation-df to real-train-df
  centroid cosine distance 0.017769 versus validation-df to synthetic-df
  0.099412.
- The synthetic-mixture dose-response completed with no synthetic condition
  beating `S000`; `S375` and `S500` collapsed to validation df F1 = 0.

These results justify a further representation diagnostic only. They do not
authorize picking a ratio or starting a formal run.

## Preregistered design (fixed; not tuned on validation)

- Diagnostic version `v1_all_class_separability`.
- Use the fixed `original` train split (6995 rows) and `validation` split
  (1510 rows) only. Never load the test split, C1 duplication, or any synthetic
  image or candidate.
- Keep the original `image_id`, `lesion_id`, and label of every row; verify the
  seven-class train counts (akiec 226, bcc 348, bkl 778, df 85, mel 782,
  nv 4684, vasc 92) and validation counts (akiec 45, bcc 77, bkl 166, df 14,
  mel 169, nv 1016, vasc 23), and that train/validation share no `image_id` or
  `lesion_id`.
- Frozen `coca_ViT-B-32` / `laion2b_s13b_b90k`, native deterministic eval
  preprocessing at 224x224, image-encoder output (the 512-d feature before the
  linear head) only, encoder in eval mode with every encoder parameter frozen,
  no train-time augmentation for either split, and every embedding L2
  normalized. Non-finite values, zero norms, and wrong dimensions fail loud.

## Four fixed analyses

- **A. Seven-class nearest centroid.** Centroids are built from train
  embeddings only (per-class mean of normalized embeddings, then re-normalized);
  validation predictions take the highest cosine similarity, breaking exact ties
  by the canonical class index.
- **B. Cosine k-NN, k = (1, 5, 10).** Reference = train, query = validation,
  cosine descending with a deterministic stable order. Vote ties break first by
  the higher summed neighbour cosine similarity of the tied class, then by the
  canonical class index. All three k are reported in full; no "best k" is picked.
- **C. Validation-df representation margin.** For the 14 validation df only:
  `nearest_train_non_df_cosine_distance - nearest_train_df_cosine_distance`
  (positive means closer to train df), summarized with
  count/min/p10/median/mean/p90/max plus positive count/fraction, and the df
  purity of each query's k = (1, 5, 10) train neighbours.
- **D. One balanced multinomial logistic-regression probe.** A single fixed
  sklearn configuration (l2, C=1.0, lbfgs, class_weight="balanced",
  fit_intercept=True, tol=1e-6, max_iter=5000, random_state=0), fit on train
  labels only; validation labels are used once for metrics after the fit.
  ConvergenceWarning is captured and non-convergence fails loud. No grid, CV,
  trial loop, or validation-based tuning; the probe model is neither saved nor
  deployed.

All four analyses report metrics through the project's existing
`metrics.classification_summary` to keep the canonical class order fixed. The
pipeline runs a strict fit -> predict -> evaluate order: every fit (the logistic
probe first, then the centroids and k-NN reference) sees train only, all
validation predictions come from validation embeddings alone, and validation
labels are consumed only once, in a final one-shot evaluation.

## Artifacts and evidence boundary

Each attempt writes to its own timestamped directory and never overwrites a
prior attempt: `all_class_embeddings.npz`,
`all_class_separability_diagnostic.json`, `diagnostic_identity.json`,
`_COMPLETED.json`, `record_integrity.json`, and a version-root
`latest_diagnostic_record.json`. The record fixes `formal_training_started=false`,
`test_data_accessed=false`, `condition_selected=false`, and
`interpretation_scope="descriptive_representation_diagnostic_not_model_or_candidate_selection"`.

One immutable diagnostic identity is built once and shared by every artifact
(the record's `diagnostic_identity`, the `diagnostic_identity.json` file, and the
top-level duplicates must all agree). It carries the diagnostic version, git
commit, train/validation manifest hashes, fixed split identity, shared-root UUID,
output identity, model identity, dependency versions, algorithm identities, class
mapping, expected row/class counts, embedding dimension, and interpretation
scope; `validate_diagnostic_identity` rejects any missing/empty field, any drift
between the record and the identity file, and any internally-consistent identity
that disagrees with the runtime-expected identity.

The three formal records share identical bytes, certified by a non-circular
`record_integrity.json` sidecar that records each file's SHA-256 and byte length
(the records never embed their own hash). The eight-key embedding NPZ is
validated before and after writing for exact dtype (float32 embeddings, int64
labels, unicode — never object — id arrays), shape, finite unit-norm embeddings,
unique/non-empty image ids, and no cross-split id overlap. All NPZ and JSON
writes are atomic and reopened for verification.

The diagnostic is descriptive and cannot establish causality. If centroid, all
k-NN, and the fixed logistic probe show no real validation df signal, a
representation limitation is the priority to investigate next. If the fixed
logistic or non-parametric methods do recover df signal while the trained
image-level head still collapses, the optimization / augmentation / objective is
the priority. Prior synthetic domain-gap evidence is described separately and
must not be generalized into a claim that synthetic data is universally
ineffective for other backbones.

## Execution

Colab runs one pass: it encodes the 6995 train and 1510 validation images once
and runs the four analyses. There are no training epochs, optimizer, or
checkpoints.
