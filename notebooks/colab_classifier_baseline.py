# ---
# Colab baseline runner for C0 / C1 classifiers.
#
# This is a plain script with `# %%` cell markers. In Colab you can either:
#   - open it as a notebook (File > Open, it renders as cells), or
#   - copy each cell below into notebook cells.
#
# Prereqs on Colab: Runtime > Change runtime type > T4 GPU, then run cells top to
# bottom. torch/torchvision are preinstalled on Colab.
# ---

# %% [1] GPU check
# !nvidia-smi

# %% [2] Mount Drive (checkpoints/results survive disconnects)
from google.colab import drive  # noqa
drive.mount('/content/drive')

# %% [3] Get the project + data onto Colab
# Option A: upload this whole `ddpm-derm-augmentation` folder to Drive, then:
#   PROJECT_DIR = '/content/drive/MyDrive/ddpm-derm-augmentation'
# Option B: git clone your repo into /content and set PROJECT_DIR accordingly.
#
# The data/ folder (10k images + manifests) should live on Drive too; point
# DDPM_DERM_DATA_DIR at the folder that contains `manifests/class_to_idx.json`.
import os
PROJECT_DIR = '/content/drive/MyDrive/ddpm-derm-augmentation'   # <-- edit me
DATA_DIR    = '/content/drive/MyDrive/ham10000_data'            # <-- edit me
OUTPUTS_DIR = '/content/drive/MyDrive/ddpm-derm-augmentation/outputs'  # persists

os.environ['DDPM_DERM_DATA_DIR'] = DATA_DIR
os.environ['DDPM_DERM_OUTPUTS_DIR'] = OUTPUTS_DIR

# %% [4] Verify the data layer BEFORE spending GPU time (torch-free)
# !cd "$PROJECT_DIR" && python scripts/smoke_test.py

# %% [5] Install the light extras (torch already present on Colab)
# !pip install -q pandas pillow

# %% [6] Train classifiers across seeds
# Primary metric = df F1. Identical architecture/epochs for every variant.
# NOTE: this .py is a lite mirror; the canonical runner is
# colab_classifier_baseline.ipynb (3-phase, figures). Keep the two consistent.
import subprocess, sys, json
from pathlib import Path

SEEDS = [0, 1, 2]          # bump to 3-5 for the final report
EPOCHS = 20

def run(variant, seed, extra=None):
    # OUTPUTS_DIR points at Drive (cell 3), so best.pt/last.pt survive a disconnect.
    # --resume is safe to always pass: it only kicks in if last.pt already exists,
    # so re-running this cell after a drop continues instead of restarting.
    cmd = [sys.executable, '-m', 'ddpm_derm.train_classifier',
           '--variant', variant, '--seed', str(seed), '--epochs', str(EPOCHS),
           '--resume']
    if extra:
        cmd += extra
    subprocess.run(cmd, cwd=f'{PROJECT_DIR}/src', check=True,
                   env={**os.environ, 'PYTHONPATH': f'{PROJECT_DIR}/src'})

# --- Stage-1 baseline (C0 + C1 target=500): COMPLETED 2026-07-11, results in
# --- outputs/classifier/. Left commented so it is never accidentally re-run
# --- (which would --resume the old checkpoints). Uncomment only to reproduce.
# for s in SEEDS:
#     run('C0', s)
#     run('C1', s, extra=['--df-target-count', '500'])

# --- Matched-585: the formal C1-vs-C4 comparison. C4 = fixed train split + the
# --- published epoch-100 synthetic set (85 real + 500 generated = 585 df); C1
# --- is matched to the same total. New base so Stage-1 outputs are untouched;
# --- C0 is reused from Stage 1. Requires the DDPM notebook's publish step.
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
