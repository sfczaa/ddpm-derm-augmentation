# PanDerm-Base C1 exploratory validation plan (`v1_panderm_base_c1_finetune`)

Status: **implemented locally, not run in Colab/GPU, and not a formal experiment.**
No PanDerm checkpoint has been downloaded and no performance result or improvement claim
has been produced.

## Binding v1 decision

PanDerm's private/internal pretraining corpus has no public image-level or patient-level
membership list. Exact overlap with this project's fixed HAM10000 validation and test
images therefore cannot be independently audited.

The binding policy is:

```text
independent_audit_possible=False
patient_overlap=not_excludable
exact_fixed_validation_test_overlap=unproven
formal_training_allowed=False
test_access_allowed=False
results_grade=exploratory
evaluation_scope=validation_only
```

A paper statement, upstream commit, checkpoint filename, checkpoint SHA-256, or
`claim_boundary="suggestive_exploratory_only"` can establish provenance and constrain
wording. None establishes non-overlap, and none authorizes formal training or test access.

PanDerm v1 may produce only a seed-0, five-epoch validation-only engineering record. It
must not produce test metrics, a formal test comparison, or a performance-improvement
claim. Selecting a different model for a formal experiment is a separate user decision.

## Phase 0 CHECK — provenance needed for validation only

| Item | Fixed value |
|---|---|
| Upstream repository | `https://github.com/SiyuanYan1/PanDerm` |
| Upstream commit | `fd7a80748ba7fc3e203fed88f909f4689d0d6f24` |
| Model | `PanDerm_Base` / `panderm_base_patch16_224_finetune` |
| Checkpoint | `panderm_bb_data6_checkpoint-499.pth` |
| Drive file id | `removed-from-public-history` |
| License | `CC-BY-NC-ND 4.0`, non-commercial academic research only |
| Published checkpoint hash | none |

The initial local constant remains `REPLACE_AFTER_FIRST_DOWNLOAD`. The validation notebook
must fail loud after the first download until a human reviews the observed SHA-256, pins
it, pushes the reviewed code, and replaces `REPLACE_AFTER_PUSH` with that commit.
This trust-on-first-use step authenticates the downloaded bytes for later validation; it
does not prove the checkpoint's training corpus is non-overlapping.

`require_provenance_clearance(..., purpose="validation_only")` checks the license verdict,
pinned upstream/checkpoint identity, the exploratory claim boundary, and that the record
acknowledges:

- exact pretraining image/patient overlap cannot be independently verified;
- `independent_audit_possible=False`;
- `patient_overlap=not_excludable`;
- exact fixed validation/test overlap is unproven.

The same API rejects `purpose="formal_training"` and `purpose="test_access"`
unconditionally with:

```text
PanDerm v1 formal training and test access are prohibited because exact pretraining overlap is independently unauditable.
```

## Phase 1 RUN — validation-only engineering check

The only executable v1 run is:

- C1 real train data only; no synthetic images;
- seed `0`;
- `5` epochs;
- train and validation manifests only;
- `test_metrics=null` and test row count `null`;
- full PanDerm-Base parameter training as an engineering check;
- ordinary cross entropy, no sampler, no class weighting, no MixUp/CutMix/TTA;
- batch size `16`, accumulation `8`, effective batch `128`;
- AdamW, learning rate `5e-4`, weight decay `0.05`;
- five warm-up epochs for this five-epoch validation;
- layer decay `0.65`, drop path exactly `0.2`;
- AMP requested; effective only on CUDA and recorded truthfully.

Every behavior-changing CLI value is either frozen and validated before any output or
manifest mutation, or is included from the runtime value in the immutable identity.
`--drop-path` and `--no-amp` are not supported overrides.

The trainer has no executable `full` scope, test manifest load, test frame, test loader,
test evaluation, or test metric path. A forged `VALIDATION PASSED` record cannot unlock
test access because there is no downstream v1 formal/test notebook path.

## Immutable identity

`build_run_identity()` and `IMMUTABLE_IDENTITY_KEYS` use the same authoritative 25 fields:

1. `schema_version`
2. `git_commit`
3. `run_version`
4. `upstream_repo`
5. `upstream_commit`
6. `checkpoint_filename`
7. `checkpoint_source_url`
8. `checkpoint_sha256`
9. `checkpoint_sha256_provenance`
10. `checkpoint_format`
11. `model_identity`
12. `variant`
13. `seed`
14. `fixed_split_identity`
15. `manifest_sha256`
16. `c1_construction`
17. `objective`
18. `optimization`
19. `dependency_versions`
20. `shared_root_uuid`
21. `formal_output_identity`
22. `evaluation_scope`
23. `license_review`
24. `contamination_review`
25. `claim_boundary`

The `optimization` block records runtime `drop_path`, `amp_requested`,
`amp_effective`, and `device_type`. Expected, saved, current, nested, top-level result
duplicates, `best.pt`, and `last.pt` must all use this complete schema. Incomplete expected
or current identities are rejected. Resume checks identity before model, optimizer,
scheduler, scaler, or RNG state mutation.

Formal aggregation is disabled and always fails with the binding v1 prohibition.

## Persistence contract

Checkpoints use a unique same-directory temporary file, flush/fsync, atomic `os.replace`,
safe reopen, and recursive exact comparison against the original payload. Validation
covers model, optimizer, scheduler, scaler, identity, epoch/history/best metric, tensor
keys/shape/dtype/device-independent values, and nested dict/list/tuple/scalar/NumPy values.

Result and failure JSON records use the same-directory temp + flush/fsync + atomic replace
contract, are reparsed after write, and must exactly equal the canonical payload. A
non-empty existing final record is never overwritten silently.

## Notebook scope

- `notebooks/colab_panderm_base_c1_finetune_validation.ipynb` remains unexecuted and
  Run-all-safe only after `REPLACE_AFTER_PUSH` and the checkpoint hash are pinned. It runs
  seed 0 for five epochs with `evaluation_scope=validation_only`.
- `notebooks/colab_panderm_base_c1_finetune_classifier.ipynb` is a STOP/DECISION notebook.
  Its first substantive cell fails loud with the binding prohibition and contains no
  formal training, resume, aggregation, or test helper.

Existing ResNet, DDPM, CoCa, deployment, README, HANDOFF, experiment logs, and outputs
remain out of scope and byte-identical. PanDerm adapted weights must never be published,
served, committed, or added to the existing deployment.

## Required verification boundary

Local verification includes blocker regressions, all PanDerm tests, full unittest
discovery, smoke tests, external-temp `py_compile`, notebook code-cell compilation,
unexecuted notebook checks, `git diff --check`, secret/token/email/tool-attribution scans,
and pre/post protected SHA-256/size/mtime comparison.

No local acceptance authorizes a Colab/GPU run. This repair does not download real PanDerm
weights, access test data, execute formal training, or produce a performance claim.
