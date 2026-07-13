"""Torch-free smoke test for the data + metric layers.

Run locally to verify the pipeline reads the fixed manifests correctly before
paying for any GPU time on Colab. Checks:

  1. config resolves the data dir and manifests.
  2. per-split, per-class row counts match data/manifests/split_summary.csv.
  3. image paths resolve and a sample of images actually opens as RGB.
  4. lesion_id / image_id leakage: train shares none with val or test.
  5. C0 == raw train counts; C1 oversamples df to the target (585) and leaves
     other classes and lesion membership untouched.
  6. metrics.classification_summary matches hand-computed values.
  7. C4 (against a temporary generated fixture): keeps the 85 real train df,
     adds exactly 500 generated df rows (total 585), leaves other classes and
     val/test untouched, resolves real+generated paths (also after moving the
     fixture to a new root), and fails loudly on wrong counts / labels /
     sources / missing files / non-portable absolute paths.
  8. publish_synthetic validate/publish: a staging fixture validates (rows,
     metadata, portability after moving roots), publishing copies the whole
     folder, re-validates the destination and only then writes _READY.json;
     existing destinations / stray _READY markers / wrong counts / missing
     files / absolute paths / duplicate ids / wrong metadata all fail loudly.

Exits non-zero on the first failure (fail loud).

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --images-per-split 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# make src/ importable when run as a plain script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import config, manifests, metrics, publish_synthetic  # noqa: E402


class Check:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def ok(self, cond: bool, msg: str):
        if cond:
            self.passed += 1
            print(f"  PASS  {msg}")
        else:
            self.failed += 1
            print(f"  FAIL  {msg}")


def check_config(c: Check):
    print("[1] config / data dir")
    c.ok(config.DATA_DIR.exists(), f"data dir exists: {config.DATA_DIR}")
    c.ok(config.MANIFESTS_DIR.exists(), f"manifests dir exists: {config.MANIFESTS_DIR}")


def check_counts_against_summary(c: Check):
    print("[2] per-split per-class counts vs split_summary.csv")
    summary = pd.read_csv(config.MANIFESTS_DIR / "split_summary.csv")
    for split in ("train", "val", "test"):
        frame = manifests.load_split(split)
        got = manifests.class_counts(frame)
        exp = (
            summary[summary["split"] == split]
            .set_index("dx")["image_count"]
            .to_dict()
        )
        match = all(got.get(dx, 0) == int(cnt) for dx, cnt in exp.items())
        c.ok(match, f"{split}: counts match summary ({sum(got.values())} images)")


def check_images_open(c: Check, n_per_split: int):
    print(f"[3] image path resolution + open ({n_per_split}/split)")
    for split in ("train", "val", "test"):
        frame = manifests.load_split(split)
        sample = frame.sample(n=min(n_per_split, len(frame)), random_state=0)
        missing, bad = 0, 0
        for rel in sample["image_path"]:
            p = config.resolve_image_path(rel)
            if not p.is_file():
                missing += 1
                continue
            try:
                with Image.open(p) as im:
                    im.convert("RGB").load()
            except Exception:
                bad += 1
        c.ok(missing == 0 and bad == 0,
             f"{split}: {len(sample)} sampled images all present & openable "
             f"(missing={missing}, unreadable={bad})")


def check_leakage(c: Check):
    print("[4] lesion_id / image_id leakage (fixed split integrity)")
    train = manifests.load_split("train")
    val = manifests.load_split("val")
    test = manifests.load_split("test")
    for name, other in (("val", val), ("test", test)):
        les = set(train["lesion_id"]) & set(other["lesion_id"])
        img = set(train["image_id"]) & set(other["image_id"])
        c.ok(len(les) == 0, f"train vs {name}: no shared lesion_id ({len(les)} overlaps)")
        c.ok(len(img) == 0, f"train vs {name}: no shared image_id ({len(img)} overlaps)")


def check_variants(c: Check):
    print("[5] C0 / C1 frame construction")
    train = manifests.load_split("train")
    base = manifests.class_counts(train)

    c0 = manifests.build_classifier_frame("C0")
    c.ok(manifests.class_counts(c0) == base, "C0 equals raw train counts")

    target = 585  # agreed C1/C4 alignment: 85 real + 500 generated
    c1 = manifests.build_classifier_frame("C1", df_target_count=target, seed=0)
    c1_counts = manifests.class_counts(c1)
    c.ok(c1_counts["df"] == target, f"C1 df oversampled to {target} (got {c1_counts['df']})")
    others_unchanged = all(
        c1_counts[k] == base[k] for k in base if k != "df"
    )
    c.ok(others_unchanged, "C1 leaves non-df class counts unchanged")

    # oversampled df rows must all come from real train df lesions (no new lesions)
    train_df_lesions = set(train[train["dx"] == "df"]["lesion_id"])
    c1_df_lesions = set(c1[c1["label_idx"] == config.TARGET_CLASS_IDX]["lesion_id"])
    c.ok(c1_df_lesions <= train_df_lesions,
         "C1 df duplicates only real train df lesions (no leakage introduced)")


def _make_generated_fixture(root: Path, n: int) -> Path:
    """Write n tiny synthetic-df PNGs, a relative-path manifest and a
    metadata.json under ``root``, mirroring what sample_ddpm.py writes."""
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        stem = f"synthetic_df_{i:04d}"
        Image.new("RGB", (8, 8), ((37 * i) % 256, 80, 120)).save(img_dir / f"{stem}.png")
        rows.append({
            "image_path": f"images/{stem}.png",
            "label_idx": config.TARGET_CLASS_IDX,
            "dx": config.TARGET_CLASS,
            "lesion_id": "synthetic",
            "image_id": stem,
            "source": "synthetic",
        })
    manifest = root / "synthetic_df.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    meta = {"checkpoint": "run_seed0_epoch0100.pt", "epoch": 100,
            "weights": "ema_state_dict", "class_name": config.TARGET_CLASS,
            "class_idx": config.TARGET_CLASS_IDX, "n": n, "seed": 0,
            "num_steps": 50, "eta": 0.0, "image_size": 64}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return manifest


def _expect_error(c: Check, msg: str, fn):
    try:
        fn()
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        c.ok(True, f"{msg} -> {type(e).__name__}")
    else:
        c.ok(False, f"{msg} (no error raised)")


def check_c4(c: Check):
    print("[7] C4 frame construction (85 real df + 500 generated = 585)")
    train = manifests.load_split("train")
    base = manifests.class_counts(train)
    real_df_ids = set(train[train["dx"] == "df"]["image_id"])
    val_before = manifests.class_counts(manifests.load_split("val"))
    test_before = manifests.class_counts(manifests.load_split("test"))

    with tempfile.TemporaryDirectory() as td:
        root_a = Path(td) / "root_a"
        manifest_a = _make_generated_fixture(root_a, 500)

        c4 = manifests.build_classifier_frame(
            "C4", generated_manifest=manifest_a, df_target_count=585, seed=0)
        counts = manifests.class_counts(c4)
        df_rows = c4[c4["label_idx"] == config.TARGET_CLASS_IDX]
        n_real = int((df_rows["source"] == "real").sum())
        n_gen = int((df_rows["source"] == "synthetic").sum())
        c.ok(counts["df"] == 585, f"C4 total df == 585 (got {counts['df']})")
        c.ok(n_real == 85, f"C4 keeps 85 real train df rows (got {n_real})")
        c.ok(n_gen == 500, f"C4 adds 500 generated df rows (got {n_gen})")
        c.ok(set(df_rows[df_rows["source"] == "real"]["image_id"]) == real_df_ids,
             "C4 real df rows are exactly the fixed train-split df images")
        c.ok(all(counts[k] == base[k] for k in base if k != "df"),
             "C4 leaves non-df class counts unchanged")

        # both row kinds must resolve through the same logic the Dataset uses
        gen_paths = [config.resolve_image_path(p)
                     for p in df_rows[df_rows["source"] == "synthetic"]["image_path"]]
        c.ok(all(p.is_file() for p in gen_paths),
             "all 500 generated image paths resolve to existing files")
        real_sample = df_rows[df_rows["source"] == "real"]["image_path"].head(5)
        c.ok(all(config.resolve_image_path(p).is_file() for p in real_sample),
             "sampled real df image paths still resolve via the data root")
        opened = 0
        for p in gen_paths[:8]:
            with Image.open(p) as im:
                im.convert("RGB").load()
            opened += 1
        c.ok(opened == 8, "sampled generated images open as RGB")

        # portability: move the whole fixture; manifest-relative paths survive
        root_b = Path(td) / "root_b"
        shutil.move(str(root_a), str(root_b))
        manifest_b = root_b / manifest_a.name
        c4b = manifests.build_classifier_frame(
            "C4", generated_manifest=manifest_b, df_target_count=585, seed=0)
        moved = c4b[(c4b["label_idx"] == config.TARGET_CLASS_IDX)
                    & (c4b["source"] == "synthetic")]
        c.ok(manifests.class_counts(c4b)["df"] == 585
             and all(config.resolve_image_path(p).is_file()
                     for p in moved["image_path"]),
             "fixture moved to a new root: relative manifest paths still resolve")
        c4c = manifests.build_classifier_frame(
            "C4", generated_manifest=manifest_b, generated_root=root_b,
            df_target_count=585, seed=0)
        c.ok(manifests.class_counts(c4c)["df"] == 585,
             "explicit generated_root resolves the same fixture")

        # C4 must not touch val/test
        val_after = manifests.load_split("val")
        test_after = manifests.load_split("test")
        c.ok(manifests.class_counts(val_after) == val_before
             and manifests.class_counts(test_after) == test_before,
             "val/test class counts unchanged after building C4")
        c.ok("synthetic" not in set(val_after["lesion_id"]) | set(test_after["lesion_id"]),
             "no synthetic rows in val/test")

        # failure modes must fail loudly, not silently degrade
        gen_frame = pd.read_csv(manifest_b)

        def write_variant(frame, name: str) -> Path:
            path = root_b / name
            frame.to_csv(path, index=False)
            return path

        _expect_error(c, "C4 without generated_manifest errors",
                      lambda: manifests.build_classifier_frame("C4", df_target_count=585))
        _expect_error(c, "missing manifest file errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=root_b / "nope.csv", df_target_count=585))
        short = write_variant(gen_frame.iloc[:-1], "gen_short.csv")
        _expect_error(c, "wrong generated row count (499 != 500) errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=short, df_target_count=585))
        _expect_error(c, "df_target_count mismatch (500 needs 415 generated) errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=manifest_b, df_target_count=500))
        bad = gen_frame.copy()
        bad.loc[bad.index[0], ["dx", "label_idx"]] = ["nv", config.CLASS_TO_IDX["nv"]]
        _expect_error(c, "non-df label in generated manifest errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=write_variant(bad, "gen_badlabel.csv"),
                          df_target_count=585))
        bad = gen_frame.copy()
        bad.loc[bad.index[0], "source"] = "real"
        _expect_error(c, "unknown source in generated manifest errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=write_variant(bad, "gen_badsource.csv"),
                          df_target_count=585))
        _expect_error(c, "missing source column errors",
                      lambda: manifests.build_classifier_frame(
                          "C4",
                          generated_manifest=write_variant(
                              gen_frame.drop(columns=["source"]), "gen_nosource.csv"),
                          df_target_count=585))
        bad = gen_frame.copy()
        bad.loc[bad.index[0], "image_path"] = "images/does_not_exist.png"
        _expect_error(c, "missing generated image file errors",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=write_variant(bad, "gen_missingfile.csv"),
                          df_target_count=585))
        bad = gen_frame.copy()
        bad.loc[bad.index[0], "image_path"] = "/content/outputs/synthetic_df/images/x.png"
        _expect_error(c, "absolute /content image path errors (portability)",
                      lambda: manifests.build_classifier_frame(
                          "C4", generated_manifest=write_variant(bad, "gen_abspath.csv"),
                          df_target_count=585))
        _expect_error(c, "generated_manifest with C0 errors",
                      lambda: manifests.build_classifier_frame(
                          "C0", generated_manifest=manifest_b))


def check_publish(c: Check):
    print("[8] synthetic-dir validate / publish flow (torch-free)")
    pub = publish_synthetic

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        staging = td / "content" / "synthetic_df_epoch0100_seed0"
        _make_generated_fixture(staging, 500)
        s = pub.validate_synthetic_dir(staging)
        c.ok(s["rows"] == 500 and s["missing_files"] == 0
             and s["metadata"]["epoch"] == 100,
             "staging fixture validates (500 rows, 0 missing, metadata epoch 100)")

        # whole-folder move to a new root must keep validating (portable paths)
        moved = td / "elsewhere" / "synthetic_df_epoch0100_seed0"
        moved.parent.mkdir(parents=True)
        shutil.move(str(staging), str(moved))
        c.ok(pub.validate_synthetic_dir(moved)["rows"] == 500,
             "whole folder moved to a new root still validates")

        dest = td / "drive" / "outputs" / "synthetic_df" / "epoch0100_seed0"
        out = pub.publish_synthetic_dir(moved, dest)
        n_pngs = len(list((dest / "images").glob("*.png")))
        c.ok(out["rows"] == 500 and n_pngs == 500,
             f"publish copied the whole folder ({n_pngs} images) and re-validated")
        c.ok((dest / "_READY.json").is_file(),
             "_READY.json written only after destination re-validation")
        c.ok(pub.validate_synthetic_dir(dest)["ready"] is True,
             "published destination reports ready=True")

        _expect_error(c, "publish refuses an existing destination",
                      lambda: pub.publish_synthetic_dir(moved, dest))
        stray = moved / "_READY.json"
        shutil.copy2(dest / "_READY.json", stray)
        _expect_error(c, "publish refuses staging that already has _READY.json",
                      lambda: pub.publish_synthetic_dir(moved, td / "drive2"))
        stray.unlink()

        # failure modes on small fixtures (expect_n=5 keeps them cheap)
        def small_fixture(name, mutate_manifest=None, mutate_meta=None):
            r = td / name
            _make_generated_fixture(r, 5)
            if mutate_manifest is not None:
                f = pd.read_csv(r / "synthetic_df.csv")
                f = mutate_manifest(f)
                f.to_csv(r / "synthetic_df.csv", index=False)
            if mutate_meta is not None:
                m = json.loads((r / "metadata.json").read_text(encoding="utf-8"))
                mutate_meta(m)
                (r / "metadata.json").write_text(json.dumps(m), encoding="utf-8")
            return r

        ok5 = small_fixture("ok5")
        c.ok(pub.validate_synthetic_dir(ok5, expect_n=5)["rows"] == 5,
             "small baseline fixture validates (so failures below are real)")

        r = small_fixture("short5", mutate_manifest=lambda f: f.iloc[:-1])
        _expect_error(c, "row count 4 != expected 5 errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("missfile5")
        (r / "images" / "synthetic_df_0000.png").unlink()
        _expect_error(c, "missing image file errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        def _abs_path(f):
            f.loc[f.index[0], "image_path"] = "/content/tmp/x.png"
            return f
        r = small_fixture("abs5", mutate_manifest=_abs_path)
        _expect_error(c, "absolute image_path errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        def _dup_id(f):
            f.loc[f.index[1], "image_id"] = f.loc[f.index[0], "image_id"]
            return f
        r = small_fixture("dupid5", mutate_manifest=_dup_id)
        _expect_error(c, "duplicate image_id errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("epoch60", mutate_meta=lambda m: m.update(epoch=60))
        _expect_error(c, "metadata epoch 60 != 100 errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("badn", mutate_meta=lambda m: m.update(n=999))
        _expect_error(c, "metadata n mismatch errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("badseed", mutate_meta=lambda m: m.update(seed=1))
        _expect_error(c, "metadata seed mismatch errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("badsteps", mutate_meta=lambda m: m.update(num_steps=25))
        _expect_error(c, "metadata num_steps mismatch errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("rawweights",
                          mutate_meta=lambda m: m.update(weights="model_state_dict"))
        _expect_error(c, "metadata weights != ema_state_dict errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))

        r = small_fixture("nometa")
        (r / "metadata.json").unlink()
        _expect_error(c, "missing metadata.json errors",
                      lambda: pub.validate_synthetic_dir(r, expect_n=5))


def check_metrics(c: Check):
    print("[6] metrics correctness (hand-computed)")
    # 3 classes worth of a tiny confusion example, df is index 3
    # true:  [df, df, df, nv, nv]
    # pred:  [df, df, nv, nv, nv]  -> df: tp=2, fn=1, fp=0 ; nv: tp=2
    df_i = config.CLASS_TO_IDX["df"]
    nv_i = config.CLASS_TO_IDX["nv"]
    y_true = [df_i, df_i, df_i, nv_i, nv_i]
    y_pred = [df_i, df_i, nv_i, nv_i, nv_i]
    s = metrics.classification_summary(y_true, y_pred)
    # df precision=2/2=1, recall=2/3 -> f1 = 2*1*(2/3)/(1+2/3)=0.8
    c.ok(abs(s["target_f1"] - 0.8) < 1e-9, f"df F1 == 0.8 (got {s['target_f1']:.6f})")
    c.ok(abs(s["target_recall"] - 2/3) < 1e-9, f"df recall == 2/3 (got {s['target_recall']:.6f})")
    c.ok(abs(s["accuracy"] - 0.8) < 1e-9, f"accuracy == 0.8 (got {s['accuracy']:.6f})")
    cm = np.array(s["confusion_matrix"])
    c.ok(cm.sum() == 5 and cm[df_i, df_i] == 2 and cm[df_i, nv_i] == 1,
         "confusion matrix entries correct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-per-split", type=int, default=15)
    args = ap.parse_args()

    print("=" * 60)
    print("ddpm-derm-augmentation :: data/metric smoke test")
    print("=" * 60)
    c = Check()
    check_config(c)
    check_counts_against_summary(c)
    check_images_open(c, args.images_per_split)
    check_leakage(c)
    check_variants(c)
    check_metrics(c)
    check_c4(c)
    check_publish(c)

    print("-" * 60)
    print(f"RESULT: {c.passed} passed, {c.failed} failed")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
