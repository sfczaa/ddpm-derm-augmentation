# Percent-format companion generated from colab_classifier_baseline.ipynb.

# %% [markdown]
# # HAM10000 classifier baseline: C0 / C1

# %% [markdown]
# # Phase 1: Setup

# %% [markdown]
# ## 1.1 Check runtime

# %% [1] Runtime check
get_ipython().system('nvidia-smi')

# %% [markdown]
# ## 1.2 Mount Google Drive

# %% [2] Drive mount
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 1.3 Project and data paths

# %% [3] Project paths
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_GIT_COMMIT = "b584bd321dd11258469f8c564bcc8a82a3ae11ac"
RUN_VERSION = "classifier_baseline_safe_v2"
RUN_MODE = "fresh"
SHARED_PROJECT_DIR = Path('/content/drive/MyDrive/ddpm-derm-augmentation')
assert SHARED_PROJECT_DIR.is_dir(), f'missing shared project: {SHARED_PROJECT_DIR}'
SHARED_PROJECT_DIR = SHARED_PROJECT_DIR.resolve(strict=True)
INPUT_OUTPUTS_DIR = str(SHARED_PROJECT_DIR / 'outputs')
DATA_DIR = str(SHARED_PROJECT_DIR / 'data')
assert Path(DATA_DIR, 'manifests', 'class_to_idx.json').is_file()
assert Path(INPUT_OUTPUTS_DIR).is_dir()

PROJECT_DIR = '/content/classifier_baseline_safe_v2-code'
assert not Path(PROJECT_DIR).exists(), f'code directory already exists: {PROJECT_DIR}'
subprocess.run(['git', 'clone', 'https://github.com/sfczaa/ddpm-derm-augmentation.git', PROJECT_DIR], check=True)
subprocess.run(['git', '-C', PROJECT_DIR, 'checkout', '--detach', EXPECTED_GIT_COMMIT], check=True)
commit = subprocess.check_output(['git', '-C', PROJECT_DIR, 'rev-parse', 'HEAD'], text=True).strip()
assert commit == EXPECTED_GIT_COMMIT
assert not subprocess.check_output(['git', '-C', PROJECT_DIR, 'status', '--porcelain'], text=True).strip()
assert Path(PROJECT_DIR, 'src', 'ddpm_derm', 'checkpoint.py').is_file()

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pandas>=2.0', 'pillow>=12.3.0'], check=True)

sys.path.insert(0, str(Path(PROJECT_DIR) / "src"))
from ddpm_derm.notebook_runtime import require_training_runtime
require_training_runtime()

run_root = SHARED_PROJECT_DIR / 'outputs' / 'notebook_runs' / RUN_VERSION
run_root.resolve().relative_to(SHARED_PROJECT_DIR)
identity = {'git_commit': commit, 'run_version': RUN_VERSION}
assert RUN_MODE in {'fresh', 'resume'}
if RUN_MODE == 'fresh':
    assert not run_root.exists(), f'fresh run refuses existing output: {run_root}'
    run_root.mkdir(parents=True)
    (run_root / 'run_identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
else:
    assert run_root.is_dir(), f'resume output missing: {run_root}'
    assert json.loads((run_root / 'run_identity.json').read_text(encoding='utf-8')) == identity
OUTPUTS_DIR = str(run_root)
os.environ['DDPM_DERM_DATA_DIR'] = DATA_DIR
os.environ['DDPM_DERM_OUTPUTS_DIR'] = OUTPUTS_DIR
os.environ['PYTHONPATH'] = str(Path(PROJECT_DIR) / 'src')
print('code commit:', commit, 'run:', RUN_VERSION, 'outputs:', OUTPUTS_DIR)

# %% [markdown]
# ## 1.4 Optional local data cache

# %% [4] Data cache
# !mkdir -p /content/data && cp -r "{PROJECT_DIR}/data/." /content/data/
# DATA_DIR = '/content/data'
# os.environ['DDPM_DERM_DATA_DIR'] = DATA_DIR
# assert Path(DATA_DIR, 'manifests', 'class_to_idx.json').is_file(), 'local copy layout wrong'
# print('using local data at', DATA_DIR)

# %% [markdown]
# ## 1.5 Dependency versions

# %% [5] Dependencies
from importlib.metadata import version
print({name: version(name) for name in ("torch", "Pillow")})

# %% [markdown]
# # Phase 2: Smoke tests

# %% [markdown]
# ## 2.1 Data smoke test

# %% [6] Data smoke
get_ipython().system('cd "{PROJECT_DIR}" && python scripts/smoke_test.py')

# %% [markdown]
# ## 2.2 Training and resume smoke test

# %% [7] Resume smoke
get_ipython().system('cd "{PROJECT_DIR}/src" && python -m ddpm_derm.train_classifier --variant C0 --seed 0 --epochs 1 --limit 200 --resume --output-dir "{OUTPUTS_DIR}/_smoke"')

# %% [markdown]
# # Phase 3: Classifier runs

# %% [markdown]
# ## 3.0 Optional removal of prior outputs
#
# Enabling the commented cleanup deletes prior baseline checkpoints and results under `outputs/`.

# %% [8] Output cleanup
# import shutil
# for sub in ('classifier', '_smoke'):
#     p = Path(OUTPUTS_DIR, sub)
#     if p.exists():
#         shutil.rmtree(p); print('removed', p)
#     else:
#         print('nothing to remove at', p)

# %% [markdown]
# ## 3.1 Baseline run definitions

# %% [9] Run definitions
import subprocess, os, sys

SEEDS  = [0, 1, 2]
EPOCHS = 20            # identical hyperparameters for every variant (C0/C1/C4)

def run(variant, seed, extra=None):
    cmd = [sys.executable, '-u', '-m', 'ddpm_derm.train_classifier',
           '--variant', variant, '--seed', str(seed),
           '--epochs', str(EPOCHS), '--resume']
    if extra:
        cmd += extra
    print(f'\n===== {variant} seed={seed} =====', flush=True)
    proc = subprocess.Popen(
        cmd, cwd=f'{PROJECT_DIR}/src',
        env={**os.environ, 'PYTHONPATH': f'{PROJECT_DIR}/src'},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'{variant} seed={seed} failed (exit {proc.returncode})')

# Stage 1 (C0 + C1 target=500) completed on 2026-07-11 in outputs/classifier/.
# The loop is disabled; enabling it uses the current run-version output directory.
# DF_TARGET = 500
# for s in SEEDS:
#     run('C0', s)
#     run('C1', s, ['--df-target-count', str(DF_TARGET)])

print('classifier runner ready')

# %% [markdown]
# ## 3.2 Stage 1 aggregation (mean ± std)

# %% [10] Stage 1 aggregate
get_ipython().system('cd "{PROJECT_DIR}" && python scripts/aggregate_results.py --results-dir "{INPUT_OUTPUTS_DIR}/classifier/results"')

# %% [markdown]
# ## 3.3 Stage 1 figures
#
# Figures are descriptive; no significance test is performed.

# %% [11] Stage 1 figures
# Figures from saved Stage 1 results.
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display

RESULTS_DIR = Path(INPUT_OUTPUTS_DIR) / 'classifier' / 'results'
FIG_DIR     = Path(OUTPUTS_DIR) / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

runs = {}
for p in sorted(RESULTS_DIR.glob('results_*.json')):
    d = json.loads(p.read_text())
    runs.setdefault(d['variant'], []).append(d)
assert runs, f'no results_*.json in {RESULTS_DIR}'

variants = sorted(runs)
CLASSES  = list(runs[variants[0]][0]['test_metrics']['per_class_recall'].keys())
COL = {'C0': '#0072B2', 'C1': '#E69F00', 'C4': '#009E73'}   # colour-blind safe
col = lambda v: COL.get(v, '#666666')
CAPTION = 'Suggestive only: df test n=16, fixed split, no significance test.'

def tf1(v):                                   # per-seed test df F1 for a variant
    return [r['test_metrics']['target_f1'] for r in runs[v]]

# Figure 1: df F1 by variant
fig, ax = plt.subplots(figsize=(4.8, 4.2))
x = np.arange(len(variants))
means = [float(np.mean(tf1(v))) for v in variants]
stds  = [float(np.std(tf1(v)))  for v in variants]
ax.bar(x, means, yerr=stds, capsize=6, color=[col(v) for v in variants],
       alpha=0.85, edgecolor='black', linewidth=0.6)
for i, v in enumerate(variants):
    ys = tf1(v)
    ax.scatter(np.full(len(ys), x[i]), ys, color='black', s=22, zorder=3)
    ax.text(x[i], means[i] + stds[i] + 0.03, f'{means[i]:.3f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f'{v}\n(n={len(runs[v])})' for v in variants])
ax.set_ylim(0, 1); ax.set_ylabel('test df F1 (primary metric)')
ax.set_title('df F1 by variant  (mean +/- std; dots = seeds)')
ax.text(0.5, -0.24, CAPTION, transform=ax.transAxes, ha='center', fontsize=8, color='gray')
fig.tight_layout(); fig.savefig(FIG_DIR / 'df_f1_by_variant.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Figure 2: per-class test recall
fig, ax = plt.subplots(figsize=(8.5, 4.2))
xc = np.arange(len(CLASSES)); w = 0.8 / len(variants)
for j, v in enumerate(variants):
    vals = np.array([[r['test_metrics']['per_class_recall'][c] for c in CLASSES]
                     for r in runs[v]])
    ax.bar(xc + j * w, vals.mean(0), w, yerr=vals.std(0), capsize=3,
           label=v, color=col(v), alpha=0.85)
ax.set_xticks(xc + w * (len(variants) - 1) / 2); ax.set_xticklabels(CLASSES)
for lbl in ax.get_xticklabels():
    if lbl.get_text() == 'df':
        lbl.set_fontweight('bold')
ax.set_ylim(0, 1); ax.set_ylabel('test recall (mean +/- std)')
ax.set_title('Per-class test recall')
ax.legend(title='variant')
fig.tight_layout(); fig.savefig(FIG_DIR / 'per_class_recall.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Figure 3: validation df F1 training curves
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for v in variants:
    for k, r in enumerate(runs[v]):
        ep = [h['epoch'] for h in r['history']]
        f1 = [h['val_df_f1'] for h in r['history']]
        ax.plot(ep, f1, color=col(v), alpha=0.55, label=v if k == 0 else None)
ax.set_xlabel('epoch'); ax.set_ylabel('val df F1'); ax.set_ylim(0, 1)
ax.set_title('Validation df F1 per seed  (df val n=14)')
ax.legend(title='variant')
fig.tight_layout(); fig.savefig(FIG_DIR / 'val_df_f1_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

for name in ('df_f1_by_variant.png', 'per_class_recall.png', 'val_df_f1_curves.png'):
    display(Image(str(FIG_DIR / name)))
print('saved 3 figures ->', FIG_DIR)

# %% [markdown]
# ## 3.4 Matched-585: C1 vs C4

# %% [12] Matched 585 runs
import json
from pathlib import Path

DF585_BASE   = OUTPUTS_DIR + '/classifier_df585'   # new base; Stage-1 outputs/classifier stays untouched
SYN_DIR      = Path(INPUT_OUTPUTS_DIR) / 'synthetic_df' / 'epoch0100_seed0'
SYN_MANIFEST = SYN_DIR / 'synthetic_df.csv'
READY        = SYN_DIR / '_READY.json'

assert READY.is_file(), (
    f'formal synthetic set is not published yet: missing {READY}\n'
    'run the DDPM notebook section 3.2 (sample -> validate -> publish) first')
print('_READY.json OK, published:', json.loads(READY.read_text())['published_utc'])

DF_TARGET_585 = 585          # 85 real + 500 generated; C1 matched to C4's df total
SEEDS_585     = [0, 1, 2]

for s in SEEDS_585:
    run('C1', s, ['--df-target-count', str(DF_TARGET_585),
                  '--output-dir', DF585_BASE])
    run('C4', s, ['--df-target-count', str(DF_TARGET_585),
                  '--generated-manifest', str(SYN_MANIFEST),
                  '--output-dir', DF585_BASE])

# %% [markdown]
# ## 3.5 Matched-585 aggregation

# %% [13] Matched 585 aggregate
print('===== matched-585 (C1@585 vs C4@585) =====')
get_ipython().system('cd "{PROJECT_DIR}" && python scripts/aggregate_results.py --results-dir "{DF585_BASE}/results"')
print('\n===== Stage-1 reference (C0 + old C1 target=500) =====')
get_ipython().system('cd "{PROJECT_DIR}" && python scripts/aggregate_results.py --results-dir "{INPUT_OUTPUTS_DIR}/classifier/results"')
