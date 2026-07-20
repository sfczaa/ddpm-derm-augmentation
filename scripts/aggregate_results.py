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
from ddpm_derm import coca_run  # noqa: E402


def _model_signature(data: dict) -> str:
    identity = data.get("run_identity", {}).get("model_identity")
    if identity is None:
        return json.dumps({"arch": "resnet18", "legacy": True}, sort_keys=True)
    keys = (
        "arch", "model_name", "pretrained_tag", "freeze_mode",
        "preprocessing_identity", "input_resolution", "open_clip_torch_version",
    )
    signature = {key: identity.get(key) for key in keys}
    signature["checkpoint_format"] = data.get("run_identity", {}).get(
        "checkpoint_format"
    )
    signature["run_version"] = data.get("run_identity", {}).get("run_version")
    signature["training_objective"] = data.get("run_identity", {}).get(
        "training_objective"
    )
    signature["evaluation_scope"] = data.get("run_identity", {}).get(
        "evaluation_scope"
    )
    return json.dumps(signature, sort_keys=True)


def load_results(
    results_dir: Path, expected_arch: str | None = None
) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    signatures = set()
    for path in sorted(results_dir.glob("results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("evaluation_scope", "full") != "full" or data.get(
            "test_metrics"
        ) is None:
            raise ValueError(f"{path} is not a full-evaluation result")
        run_identity = data.get("run_identity", {})
        if run_identity.get("training_objective") is not None and run_identity.get(
            "evaluation_scope"
        ) != data.get("evaluation_scope", "full"):
            raise ValueError(f"{path} has inconsistent weighted evaluation scope")
        identity = data.get("run_identity", {}).get("model_identity", {})
        arch = identity.get("arch", "resnet18")
        if expected_arch is not None and arch != expected_arch:
            raise ValueError(
                f"{path} has arch={arch!r}, expected {expected_arch!r}"
            )
        signatures.add(_model_signature(data))
        by_variant[data["variant"]].append(data)
    if len(signatures) > 1:
        raise ValueError("refusing to aggregate mixed model/tag/freeze/preprocessing")
    return by_variant


def _mean_std(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.3f} +/- {arr.std(ddof=0):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(config.CLASSIFIER_RESULTS_DIR))
    ap.add_argument("--arch", default="resnet18")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)

    by_variant = load_results(results_dir, expected_arch=args.arch)
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
    if args.arch == coca_run.ARCH and set(by_variant) == {"C1", "C4"}:
        aggregate = coca_run.aggregate_results(
            [run for runs in by_variant.values() for run in runs]
        )
        paired = aggregate["paired_c4_minus_c1"]
        print("Paired C4-C1 df F1:", paired["seed_differences"])
        print(
            "Paired mean +/- population std: "
            f"{paired['mean']:.3f} +/- {paired['population_std']:.3f}"
        )


if __name__ == "__main__":
    main()
