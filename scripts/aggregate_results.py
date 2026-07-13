"""Aggregate per-seed classifier results into a mean +/- std table.

Reads results_{variant}_seed{seed}.json produced by train_classifier and reports
mean +/- std across seeds for each variant. Primary column is test df F1.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir path/to/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddpm_derm import config  # noqa: E402


def load_results(results_dir: Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_variant[data["variant"]].append(data)
    return by_variant


def _mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.3f} +/- {arr.std(ddof=0):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(config.CLASSIFIER_RESULTS_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    by_variant = load_results(results_dir)
    if not by_variant:
        print(f"No results_*.json found in {results_dir}. Run train_classifier first.")
        sys.exit(1)

    print(f"Aggregating from {results_dir}\n")
    header = f"{'variant':<8}{'seeds':<7}{'df_F1':<18}{'macro_F1':<18}{'df_recall':<18}"
    print(header)
    print("-" * len(header))
    for variant in sorted(by_variant):
        runs = by_variant[variant]
        df_f1 = [r["test_metrics"]["target_f1"] for r in runs]
        macro = [r["test_metrics"]["macro_f1"] for r in runs]
        df_rec = [r["test_metrics"]["target_recall"] for r in runs]
        print(f"{variant:<8}{len(runs):<7}{_mean_std(df_f1):<18}"
              f"{_mean_std(macro):<18}{_mean_std(df_rec):<18}")
    print("\nPrimary metric = df F1. Fixed split + small df -> results are suggestive.")


if __name__ == "__main__":
    main()
