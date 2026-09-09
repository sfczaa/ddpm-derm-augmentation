---
title: HAM10000 Classifier Portfolio Demo
emoji: 🔬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.16.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
models:
  - sfczaa/ddpm-derm-c1-seed2
datasets:
  - sfczaa/ddpm-derm-synthetic-gallery
---

# HAM10000 skin-lesion classifier — portfolio demo

**Educational portfolio demonstration only. Not for diagnosis or treatment.**
The model can be wrong, and its training data and population coverage are
limited.

Upload a dermatoscopic image and the Space returns seven class probabilities
from a ResNet-18 trained on the HAM10000 train split. A second tab shows 24 of
the 500 DDPM-generated `df` images used by the project's C4 condition.

## What is being served

The deployed model is **C1 seed 2**, chosen by the highest validation `df` F1
among the C1 seeds — never on test performance. Weights and the synthetic
gallery are fetched at startup by pinned revision and hash-checked, so the Space
serves exactly the published artifacts:

| asset | repository | revision |
|---|---|---|
| checkpoint | `sfczaa/ddpm-derm-c1-seed2` | `30b41486b5353f2a99aceecef2fc41b178c2697b` |
| synthetic gallery | `sfczaa/ddpm-derm-synthetic-gallery` | `60b046e4c2ae77f1505e8c2b426c24763741c3a5` |

## Does the synthetic data help?

Not beyond duplication. Across seeds 0/1/2 on the fixed test split, adding all
500 synthetic `df` images scored **below** simply duplicating the 85 real ones
(df F1 `0.6115` vs `0.6598`). Filtering the pool to the 155 images nearest the
real `df` manifold recovered that loss but did not beat duplication
(`0.6657 +/- 0.0182`). With 16 real `df` in the test split none of these
differences can be significant, and none was tested. Full method, diagnostics
and limits are in the project repository.

## Attribution and licence

Non-commercial use only. HAM10000 Dataset © ViDIR Group, Department of
Dermatology, Medical University of Vienna, distributed with the ISIC 2018 data
under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cite
Tschandl, Rosendahl & Kittler, *Scientific Data* 5, 180161 (2018),
<https://doi.org/10.1038/sdata.2018.161>. Project changes include a fixed
lesion-level split, resizing, classifier training, and generation of the derived
synthetic gallery. No endorsement by the dataset creators is implied. The
derived synthetic gallery is released on the same non-commercial terms.
