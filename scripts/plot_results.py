"""Plot saved classifier metrics with matplotlib and NumPy.

Reads results_{variant}_seed{seed}.json and writes df F1, per-class recall,
and validation-curve figures. Results describe the fixed split and seeds;
the script does not perform significance tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed on Colab or locally
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_derm import config  # noqa: E402

TARGET = config.TARGET_CLASS
CLASSES = config.CLASS_NAMES


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_variant[data["variant"]].append(data)
    return by_variant


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def plot_headline(by_variant, out: Path) -> None:
    """Bar chart of test df F1 (primary) and macro F1 per variant, mean ± std."""
    variants = sorted(by_variant)
    metrics = [("df F1 (primary)", "target_f1"), ("macro F1", "macro_f1")]
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.8 * len(variants) + 3, 4.5))
    for i, (label, key) in enumerate(metrics):
        means, stds = [], []
        for v in variants:
            m, s = _mean_std([r["test_metrics"][key] for r in by_variant[v]])
            means.append(m)
            stds.append(s)
        bars = ax.bar(x + (i - 0.5) * width, means, width, yerr=stds,
                      capsize=4, label=label)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(n={len(by_variant[v])})" for v in variants])
    ax.set_ylabel("test score")
    ax.set_ylim(0, 1)
    ax.set_title("Test metrics by variant (mean ± std across seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_class_recall(by_variant, out: Path) -> None:
    """Grouped bar of per-class test recall, target class shaded."""
    variants = sorted(by_variant)
    x = np.arange(len(CLASSES))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, v in enumerate(variants):
        means = [np.mean([r["test_metrics"]["per_class_recall"][c]
                          for r in by_variant[v]]) for c in CLASSES]
        ax.bar(x + (i - (len(variants) - 1) / 2) * width, means, width, label=v)
    ti = CLASSES.index(TARGET)
    ax.axvspan(ti - 0.5, ti + 0.5, color="orange", alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("test recall (mean over seeds)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-class recall by variant (target = {TARGET}, shaded)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(by_variant, out_dir: Path) -> list[Path]:
    """One row-normalized confusion heatmap per variant (summed over seeds)."""
    written = []
    for v in sorted(by_variant):
        cms = [np.asarray(r["test_metrics"]["confusion_matrix"], dtype=float)
               for r in by_variant[v]]
        cm = np.sum(cms, axis=0)
        row = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{v}: row-normalized confusion (summed over seeds)")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        p = out_dir / f"confusion_matrix_{v}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)
    return written


def plot_val_curves(by_variant, out: Path) -> None:
    """Validation df F1 per epoch, one line per seed, one panel per variant."""
    variants = sorted(by_variant)
    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 4),
                             sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        for r in sorted(by_variant[v], key=lambda d: d["seed"]):
            hist = r.get("history", [])
            if not hist:
                continue
            ax.plot([h["epoch"] for h in hist], [h["val_df_f1"] for h in hist],
                    marker="o", ms=3, label=f"seed {r['seed']}")
        ax.set_title(v)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("val df F1")
    fig.suptitle("Validation df F1 per epoch (model selection = best val df F1)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(config.CLASSIFIER_RESULTS_DIR))
    ap.add_argument("--figures-dir", default=str(config.FIGURES_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    by_variant = load_results(results_dir)
    if not by_variant:
        print(f"No results_*.json found in {results_dir}. Run train_classifier first.")
        sys.exit(1)
    figures_dir.mkdir(parents=True, exist_ok=True)

    written = [figures_dir / "df_f1_by_variant.png",
               figures_dir / "per_class_recall.png"]
    plot_headline(by_variant, written[0])
    plot_per_class_recall(by_variant, written[1])
    written += plot_confusion(by_variant, figures_dir)
    curves = figures_dir / "val_df_f1_curves.png"
    plot_val_curves(by_variant, curves)
    written.append(curves)

    print(f"Wrote {len(written)} figure(s) to {figures_dir}:")
    for p in written:
        print(f"  {p.name}")
    print("\nPrimary metric = df F1. Fixed split + small df -> results are suggestive.")


if __name__ == "__main__":
    main()
