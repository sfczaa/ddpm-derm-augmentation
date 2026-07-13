# outputs/ — where run artifacts live

Everything here is generated and **git-ignored** (only this README and the
`.gitkeep` folder markers are tracked). On Colab, point `DDPM_DERM_OUTPUTS_DIR`
at a Google Drive folder so nothing is lost on disconnect.

```
outputs/
  classifier/
    checkpoints/
      <variant>_seed<seed>/      e.g. C0_seed0/
        best.pt                  best-on-val-df-F1 weights  -> use for eval / deploy
        last.pt                  latest epoch (+ optimizer) -> use with --resume
    results/
      results_<variant>_seed<seed>.json   final test metrics + per-epoch history
  ddpm/            (stage 2, not built yet)
    checkpoints/   DDPM/U-Net weights
    samples/       preview grids during training
  synthetic_df/    (stage 2) generated df images that feed classifier C4
  figures/         plots for the report (loss curves, confusion matrices, ...)
```

## Where do I put things?

- **Classifier results I ran on Colab** → drop the `results_*.json` files into
  `outputs/classifier/results/`. Then run `python scripts/aggregate_results.py`
  to get the mean ± std table. (Or just point Drive here and Colab writes them
  directly.)
- **Trained model checkpoints** → `outputs/classifier/checkpoints/<run>/`.
  `train_classifier.py` writes these automatically. `best.pt` is the one to keep
  for the FastAPI/HF deployment later.
- **The dataset itself** → NOT here. It stays in the top-level `data/` folder
  (a sibling of this project). It is git-ignored and non-commercially licensed;
  never commit it. On Colab, upload it to Drive and set `DDPM_DERM_DATA_DIR`.
- **DDPM stuff / synthetic df / figures** → the folders above are placeholders
  for stage 2; empty for now.

## Checkpoint contents

`best.pt` / `last.pt` are dicts:
`model_state_dict`, `optimizer_state_dict`, `epoch`, `best_val_df_f1`,
`history`, `config`, `class_to_idx` (and `val_metrics` in `best.pt`). Load
`model_state_dict` into a fresh `build_model()` for evaluation or deployment.
