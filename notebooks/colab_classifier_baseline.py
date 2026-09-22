# Colab classifier baseline runner with percent-format cell markers.
# Canonical notebook: colab_classifier_baseline.ipynb.

# %% [1] GPU check
# !nvidia-smi

# %% [2] Mount Drive
from google.colab import drive  # noqa
drive.mount('/content/drive')

# %% [3] Project, data, and output directories
# DATA_DIR must contain the images and manifests/class_to_idx.json.
import os
PROJECT_DIR = '/content/drive/MyDrive/ddpm-derm-augmentation'   # configured Drive path
DATA_DIR    = '/content/drive/MyDrive/ham10000_data'            # configured Drive path
OUTPUTS_DIR = '/content/drive/MyDrive/ddpm-derm-augmentation/outputs'  # persists

os.environ['DDPM_DERM_DATA_DIR'] = DATA_DIR
os.environ['DDPM_DERM_OUTPUTS_DIR'] = OUTPUTS_DIR

# %% [4] Validate data and split integrity
# !cd "$PROJECT_DIR" && python scripts/smoke_test.py

# %% [5] Install dependencies
# !pip install -q pandas pillow

# %% [6] Train classifiers across seeds
# Primary metric = df F1. Identical architecture/epochs for every variant.
import subprocess, sys, json
from pathlib import Path

SEEDS = [0, 1, 2]
EPOCHS = 20

def run(variant, seed, extra=None):
    # Checkpoints persist on Drive; --resume continues from an existing last.pt.
    cmd = [sys.executable, '-m', 'ddpm_derm.train_classifier',
           '--variant', variant, '--seed', str(seed), '--epochs', str(EPOCHS),
           '--resume']
    if extra:
        cmd += extra
    subprocess.run(cmd, cwd=f'{PROJECT_DIR}/src', check=True,
                   env={**os.environ, 'PYTHONPATH': f'{PROJECT_DIR}/src'})

# Stage 1 (C0 + C1 target=500) completed on 2026-07-11 in outputs/classifier/.
# The loop is disabled; enabling it resumes the original checkpoints.
# for s in SEEDS:
#     run('C0', s)
#     run('C1', s, extra=['--df-target-count', '500'])

# Matched-585 compares C1 duplication with C4 (85 real + 500 epoch-100 synthetic df).
# Results use a separate base; C0 is reused from Stage 1. Requires the published set.
DF_TARGET_585 = 585
DF585_BASE    = OUTPUTS_DIR + '/classifier_df585'
SYN_MANIFEST  = OUTPUTS_DIR + '/synthetic_df/epoch0100_seed0/synthetic_df.csv'
READY         = OUTPUTS_DIR + '/synthetic_df/epoch0100_seed0/_READY.json'

assert Path(READY).is_file(), (
    f'formal synthetic set not published yet: missing {READY}\n'
    'run the DDPM notebook 3.2 (sample -> validate -> publish) first')
print('_READY.json OK, published:', json.loads(Path(READY).read_text())['published_utc'])
for s in SEEDS:
    run('C1', s, extra=['--df-target-count', str(DF_TARGET_585),
                        '--output-dir', DF585_BASE])
    run('C4', s, extra=['--df-target-count', str(DF_TARGET_585),
                        '--generated-manifest', SYN_MANIFEST,
                        '--output-dir', DF585_BASE])

# %% [7] Aggregate mean +/- std across seeds
# matched-585 (C1@585 vs C4@585); Stage-1 C0 lives in outputs/classifier/results
# !cd "$PROJECT_DIR" && python scripts/aggregate_results.py --results-dir "$OUTPUTS_DIR/classifier_df585/results"
