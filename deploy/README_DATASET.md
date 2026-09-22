---
license: cc-by-nc-4.0
pretty_name: HAM10000 synthetic dermatofibroma gallery
task_categories:
  - image-classification
tags:
  - ham10000
  - synthetic-images
  - dermatofibroma
  - non-commercial
---

# Versioned synthetic dermatofibroma gallery

This repository publishes the exact, pre-generated gallery used by the
educational HAM10000 portfolio demo. It is not medical data for diagnosis or
treatment and must not be presented as clinically validated imagery.

## Published version

The repository root must contain this directory unchanged:

```text
epoch0100_seed0/
  _READY.json
  metadata.json
  synthetic_df.csv
  images/
```

For low-latency container builds, the repository also contains a deployment
archive of that exact directory:

- File: `gallery_epoch0100_seed0.zip`
- Files inside: 503 (500 PNGs plus `_READY.json`, `metadata.json`, and
  `synthetic_df.csv`)
- SHA-256: `da3d582082323728e2b0558c27e26af124c683dacf336915d1212acd8abd0bc5`

The original directory remains the published source of truth. The archive is a
deployment-only transport copy; it does not replace, curate, or modify the
gallery.

The version contains 500 class-conditional dermatofibroma (`df`) PNG images.
They were generated with the epoch-100 EMA checkpoint, seed 0, 50 DDIM steps,
`eta=0`, and 64 × 64 output resolution. The manifest has 500 rows and no
missing images. The web gallery follows manifest order; images were not
hand-selected for display.

The demo validates the version directory, readiness marker, generation
metadata, row count, filenames, and image existence. It does not run DDPM
generation during requests.

## Source, license, and attribution

These synthetic images are derived from a model trained only on the HAM10000
training split. HAM10000 Dataset © ViDIR Group, Department of Dermatology,
Medical University of Vienna. The ISIC 2018 data page distributes the source
dataset under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
This derived gallery is published under the same non-commercial license.

Reference: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a
large collection of multi-source dermatoscopic images of common pigmented skin
lesions*. Scientific Data 5, 180161 (2018).
[doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

Modifications include a fixed lesion-ID group split, 64 × 64 preprocessing,
class-conditional DDPM training, EMA checkpoint selection, and deterministic
DDIM sampling. No endorsement by the original dataset creators or ISIC is
implied.

## Limitations

Synthetic images may contain artifacts, reproduce training-data biases, or fail
to reflect clinically meaningful diversity. In the formal matched-585
experiment, using this gallery's synthetic data did not improve downstream `df`
classification over real-image oversampling. The result is suggestive only and
has not established statistical significance or clinical validity.
