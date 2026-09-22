# Output artifacts

Everything here is generated and git-ignored (only this README and the
`.gitkeep` folder markers are tracked). On Colab, point `DDPM_DERM_OUTPUTS_DIR`
at a Google Drive folder to persist completed writes across disconnects.

```
outputs/
  classifier/
    checkpoints/
      <variant>_seed<seed>/      e.g. C0_seed0/
        best.pt                  best-on-val-df-F1 weights  -> use for eval / deploy
        last.pt                  latest epoch (+ optimizer) -> use with --resume
    results/
      results_<variant>_seed<seed>.json   final test metrics + per-epoch history
  ddpm/
    checkpoints/   DDPM/U-Net weights
    samples/       preview grids during training
  exploratory_balanced_ddpm/
    <version>/
      checkpoints/       immutable exploratory snapshots
      previews/          exploratory preview grids
      run_metadata.json  sampler/config/source/commit/output provenance
      candidate_synthetic_df/epoch0100_seed0/  optional post-training candidate
  synthetic_df/
    epoch0100_seed0/       published 500-image gallery; _READY.json required
  deploy/                  derived deployment-only artifacts; never overwrite source runs
    C1_seed2/
      deploy_weights.pt    model_state/config/class map only
      model_manifest.json  records deploy and source checkpoint hashes
    gallery_epoch0100_seed0.zip  exact validated gallery transport archive
  figures/         plots for the report (loss curves, confusion matrices, ...)
```

## Artifact locations

- Classifier results -> `results_*.json` files go in
  `outputs/classifier/results/`; `python scripts/aggregate_results.py` builds
  the mean ± std table.
- Trained model checkpoints -> `outputs/classifier/checkpoints/<run>/`.
  `train_classifier.py` writes these automatically. `best.pt` is the one to keep
  for evaluation and deployment.
- The dataset itself -> not here. It lives in `data/` inside the project
  (a sibling `../data` also resolves). It is git-ignored and non-commercially licensed;
  never commit it. On Colab, upload it to Drive and set `DDPM_DERM_DATA_DIR`.
- Stage 4 deploy checkpoint -> the selected local candidate is
  `outputs/classifier_df585/checkpoints/C1_seed2/best.pt`. It remains ignored
  and must be mounted or published as a separate model asset, never committed.
  `scripts/export_deploy_checkpoint.py` derives the smaller
  `outputs/deploy/C1_seed2/deploy_weights.pt` without changing the formal file.
- Stage 4 gallery -> only use the versioned
  `outputs/synthetic_df/epoch0100_seed0/` directory after validating its
  `_READY.json`; do not fall back to an unversioned manifest.
  `scripts/package_gallery_for_deploy.py` creates the transport ZIP only after
  validating the complete versioned gallery.

- Exploratory balanced-DDPM artifacts -> use only a fresh version under
  outputs/exploratory_balanced_ddpm/. Never put its snapshots, previews,
  metadata, or candidate images into the frozen outputs/ddpm/ or
  outputs/synthetic_df/epoch0100_seed0/ paths.
- Exploratory downstream classifier archive -> keep the curated metadata,
  result JSON, and executed notebooks under the same version's
  `downstream_classifier/` directory. `archive_manifest.json` records the
  source ZIP hash and verification status. Keep the complete ZIP and `.pt`
  weights outside Git; `outputs/**` remains ignored.
- Frozen CoCa robustness archive -> keep its curated validation records,
  six formal result JSON files, aggregate/training records, completion marker,
  and executed training notebook under the same version's
  `coca_classifier/v1/` directory. Keep the complete ZIP and all `.pt`
  checkpoints outside Git; `archive_manifest.json` records the local audit.

## Checkpoint contents

`best.pt` / `last.pt` are dicts:
`model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_df_f1`,
`history`, `config`, `class_to_idx` (and `val_metrics` in `best.pt`). Load
`model_state_dict` into a fresh `build_model()` for evaluation or deployment.
