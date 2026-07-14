---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-classification
tags:
  - ham10000
  - resnet18
  - medical-imaging
  - non-commercial
---

# HAM10000 ResNet-18 — C1@585 seed 2

This repository contains the deployment checkpoint `best.pt` for an educational,
non-commercial portfolio demonstration. It is not a medical device and must not
be used for diagnosis, treatment, triage, or other clinical decisions.

## Model selection

- Architecture: ResNet-18
- Variant: C1@585
- Seed: 2
- Selection rule: highest validation dermatofibroma (`df`) F1 among the completed
  C1@585 seeds 0, 1, and 2
- Seed-2 validation `df` F1: 0.740741
- Test metrics were not used to choose this deployment checkpoint

The matched-585 formal experiment averaged across three seeds produced
`df` F1 0.660 ± 0.042, macro-F1 0.651 ± 0.021, and `df` recall
0.604 ± 0.029. These fixed-split results are suggestive only and are not a
claim of statistical significance, clinical validation, or generalization to
other populations.

## Input and classes

Images are converted to RGB, resized to 128 × 128, converted to a tensor, and
normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`.

Class order:

`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

## Published files and integrity

- Formal experiment checkpoint: `best.pt`
- Formal checkpoint SHA-256:
  `c55134294687a753cff44ac5bac5a4896f85adde1be943697feab08129350752`
- Deployment-only checkpoint: `deploy_weights.pt`
- Deployment checkpoint SHA-256:
  `85910296186433c6cdad2d82368646c983b586ec861d605c1338317fb8306a53`
- Deployment manifest: `model_manifest.json`

`deploy_weights.pt` is derived from the formal checkpoint and contains only
`model_state_dict`, `config`, and `class_to_idx`; optimizer and training-history
state are omitted to reduce CPU deployment memory. No weights were retrained or
changed. The deployment manifest records both digests, and the demo validates
the deployment digest and checkpoint metadata before serving predictions.

## Training data, license, and attribution

The model was trained on the HAM10000 dataset using a fixed lesion-ID group
split. HAM10000 Dataset © ViDIR Group, Department of Dermatology, Medical
University of Vienna. The ISIC 2018 data page distributes the dataset under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Reference: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a
large collection of multi-source dermatoscopic images of common pigmented skin
lesions*. Scientific Data 5, 180161 (2018).
[doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

Modifications include a fixed train/validation/test split, resizing,
normalization, class rebalancing for C1@585, and model training. No endorsement
by the original dataset creators or ISIC is implied.

## Limitations

The model may be wrong. HAM10000 and the fixed test split do not represent all
devices, acquisition settings, skin tones, lesion types, or patient populations.
The test set contains few `df` cases, so per-class estimates are uncertain.
