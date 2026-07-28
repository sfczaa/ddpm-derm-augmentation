"""Fresh-process regression coverage for PanDerm notebook import order."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


root = Path.cwd()
mode = sys.argv[1]
assert not (root / "data").exists()
assert "DDPM_DERM_DATA_DIR" not in os.environ

from ddpm_derm import panderm_run

assert "ddpm_derm.config" not in sys.modules
from ddpm_derm import panderm
assert "ddpm_derm.config" not in sys.modules
assert "ddpm_derm.manifests" not in sys.modules
assert panderm.ARCH == "panderm_base_vit_b16"
assert panderm.EMBED_DIM == 768
assert panderm.DEPTH == 12
assert panderm.NUM_CLASSES == 7

if mode == "dataset_free":
    print("DATASET_FREE_IMPORT_OK", flush=True)
    raise SystemExit(0)


class TinyPanDerm(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, 4))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.fc_norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, num_classes)

    def forward(self, value):
        tokens = self.patch_embed.proj(value).flatten(2).transpose(1, 2)
        tokens = torch.cat(
            [self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1
        )
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = torch.relu(block(tokens))
        return self.head(self.fc_norm(tokens.mean(dim=1)))


def tiny_factory(**kwargs):
    return TinyPanDerm(num_classes=kwargs["num_classes"])


def direct_state():
    reference = TinyPanDerm()
    state = {
        key: value.detach().clone()
        for key, value in reference.state_dict().items()
        if not key.startswith(("head.", "fc_norm."))
    }
    state["norm.weight"] = reference.fc_norm.weight.detach().clone()
    state["norm.bias"] = reference.fc_norm.bias.detach().clone()
    return state


checkpoint = root / "direct_backbone_fixture.pt"
torch.save(direct_state(), checkpoint)
loaded = panderm.load_pretrained_state(checkpoint)
layout = panderm.detect_checkpoint_layout(loaded)
assert layout == panderm.LAYOUT_DIRECT_BACKBONE
remapped = panderm.remap_pretrained_state_dict(loaded, layout=layout)
assert "fc_norm.weight" in remapped and "norm.weight" not in remapped
model = panderm.build_panderm_classifier(
    checkpoint_path=checkpoint,
    model_factory=tiny_factory,
)
assert model.head.out_features == 7
assert "ddpm_derm.config" not in sys.modules
assert "ddpm_derm.manifests" not in sys.modules
assert not (root / "data").exists()

if mode == "preflight":
    print("PREFLIGHT_WITHOUT_DATA_OK", flush=True)
    raise SystemExit(0)


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label_idx", "dx", "lesion_id", "image_id"],
        )
        writer.writeheader()
        writer.writerows(rows)


class_names = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
shared = root / "shared_data"
mixed = shared / "images" / "mixed"
mixed.mkdir(parents=True)
train_rows = []
for index, name in enumerate(class_names):
    filename = f"train_{name}.jpg"
    (mixed / filename).write_bytes(filename.encode("ascii"))
    train_rows.append(
        {
            "image_path": f"images/mixed/{filename}",
            "label_idx": index,
            "dx": name,
            "lesion_id": f"train-lesion-{index}",
            "image_id": f"train-image-{index}",
        }
    )
val_name = "val_df.jpg"
(mixed / val_name).write_bytes(val_name.encode("ascii"))
val_rows = [
    {
        "image_path": f"images/mixed/{val_name}",
        "label_idx": 3,
        "dx": "df",
        "lesion_id": "val-lesion",
        "image_id": "val-image",
    }
]
test_name = "test_only.jpg"
(mixed / test_name).write_bytes(test_name.encode("ascii"))
test_rows = [
    {
        "image_path": f"images/mixed/{test_name}",
        "label_idx": 3,
        "dx": "df",
        "lesion_id": "test-lesion",
        "image_id": "test-image",
    }
]
write_manifest(shared / "manifests" / "train.csv", train_rows)
write_manifest(shared / "manifests" / "val.csv", val_rows)
write_manifest(shared / "manifests" / "test.csv", test_rows)
(shared / "manifests" / "class_to_idx.json").write_text(
    json.dumps({name: index for index, name in enumerate(class_names)}),
    encoding="utf-8",
)

assert "ddpm_derm.config" not in sys.modules
local = root / "local_data"
report = panderm_run.stage_validation_data(shared, local)
assert report["images_copied"] == 8
assert report["test_manifest_present"] is False
assert not (local / "manifests" / "test.csv").exists()
assert not (local / "images" / "mixed" / test_name).exists()

os.environ["DDPM_DERM_DATA_DIR"] = str(local)
from ddpm_derm import classifier_objective, config, manifests

assert config.DATA_DIR.resolve(strict=True) == local.resolve(strict=True)
assert config.MANIFESTS_DIR.resolve(strict=True) == (
    local / "manifests"
).resolve(strict=True)
train = manifests.load_split("train")
val = manifests.load_split("val")
assert len(train) == 7 and len(val) == 1
c1 = manifests.build_classifier_frame("C1", df_target_count=2, seed=0)
counts = classifier_objective.ordered_class_counts(c1)
assert len(c1) == 8 and counts["df"] == 2
assert not (local / "manifests" / "test.csv").exists()
assert not (local / "images" / "mixed" / test_name).exists()

if mode == "staging":
    print("STAGING_THEN_CONFIG_OK", flush=True)
    raise SystemExit(0)

bound_data_dir = config.DATA_DIR
os.environ["DDPM_DERM_DATA_DIR"] = str(shared)
assert config.DATA_DIR == bound_data_dir
assert config.DATA_DIR.resolve(strict=True) == local.resolve(strict=True)

if mode == "cache":
    print("MODULE_CACHE_BOUND_LOCAL_OK", flush=True)
    raise SystemExit(0)

before = panderm.snapshot_parameters(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
logits = model(torch.randn(2, 3, 4, 4))
assert tuple(logits.shape) == (2, 7)
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
loss.backward()
gradient_report = panderm.backbone_gradient_report(model)
assert gradient_report["all_backbone_parameters_have_gradient"]
assert gradient_report["backbone_gradients_finite"]
assert gradient_report["head_parameters_have_gradient"]
optimizer.step()
assert panderm.changed_parameter_count(before, model) > 0
print("FRESH_RUNTIME_INTEGRATION_OK", flush=True)
"""


class FreshRuntimeImportOrderTests(unittest.TestCase):
    def run_probe(self, mode: str, expected_marker: str) -> str:
        temporary_path = None
        completed = None
        with tempfile.TemporaryDirectory(prefix="panderm-fresh-runtime-") as temporary:
            temporary_path = Path(temporary)
            checkout = temporary_path / "checkout"
            package = checkout / "src" / "ddpm_derm"
            shutil.copytree(ROOT / "src" / "ddpm_derm", package)
            probe_path = checkout / "probe.py"
            probe_path.write_text(textwrap.dedent(PROBE), encoding="utf-8")
            env = os.environ.copy()
            env.pop("DDPM_DERM_DATA_DIR", None)
            env["PYTHONPATH"] = str(checkout / "src")
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-B", "-u", str(probe_path), mode],
                cwd=checkout,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertIsNotNone(temporary_path)
        self.assertFalse(temporary_path.exists())
        self.assertIsNotNone(completed)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(expected_marker, completed.stdout)
        return completed.stdout

    def test_dataset_free_panderm_import_does_not_cache_config(self):
        self.run_probe("dataset_free", "DATASET_FREE_IMPORT_OK")

    def test_checkpoint_preflight_runs_before_staging_without_dataset_imports(self):
        self.run_probe("preflight", "PREFLIGHT_WITHOUT_DATA_OK")

    def test_train_val_staging_precedes_first_config_import(self):
        output = self.run_probe("staging", "STAGING_THEN_CONFIG_OK")
        self.assertIn("[Phase 1] DONE validation staging: copied=8", output)

    def test_module_cache_stays_bound_to_the_first_local_config_import(self):
        self.run_probe("cache", "MODULE_CACHE_BOUND_LOCAL_OK")

    def test_fresh_runtime_integration_covers_import_staging_and_backprop(self):
        output = self.run_probe("full", "FRESH_RUNTIME_INTEGRATION_OK")
        self.assertNotIn("test_only.jpg", output)


if __name__ == "__main__":
    unittest.main()
