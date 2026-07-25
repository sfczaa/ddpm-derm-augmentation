"""Targeted tests for the PanDerm-Base C1 full fine-tuning experiment.

Every test injects a mock PanDerm loader: no real 400 MB checkpoint is
downloaded and no network access is required.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import panderm, panderm_run  # noqa: E402


MOCK_DEPTH = 4
MOCK_DIM = 16


class MockBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.mlp(torch.relu(self.attn(self.norm1(x))))


class MockPanDerm(nn.Module):
    """Mirrors the real ViT parameter naming so layer-decay grouping is real."""

    def __init__(self, num_classes=7, depth=MOCK_DEPTH, dim=MOCK_DIM):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, dim))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=112, stride=112)
        self.add_module("patch_embed", self.patch_embed)
        self.blocks = nn.ModuleList([MockBlock(dim) for _ in range(depth)])
        self.fc_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.embed_dim = dim

    def no_weight_decay(self):
        return {"cls_token", "pos_embed"}

    def forward(self, x):
        tokens = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        tokens = torch.cat(
            [self.cls_token.expand(tokens.shape[0], -1, -1), tokens], dim=1
        )
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.fc_norm(tokens.mean(dim=1)))


def mock_factory(**kwargs):
    """Stands in for upstream ``panderm_base_patch16_224_finetune``."""
    mock_factory.last_kwargs = dict(kwargs)
    return MockPanDerm(num_classes=kwargs["num_classes"])


def mock_pretrained_state(*, include_head=True, include_extras=True):
    """A pretraining-style checkpoint: encoder.* plus decoder/teacher noise."""
    reference = MockPanDerm()
    state = {
        "encoder." + key: value
        for key, value in reference.state_dict().items()
        if not key.startswith(("head.", "fc_norm."))
    }
    state["encoder.norm.weight"] = torch.ones(MOCK_DIM)
    state["encoder.norm.bias"] = torch.zeros(MOCK_DIM)
    if include_extras:
        state["decoder.block.weight"] = torch.zeros(3)
        state["teacher.proj.weight"] = torch.zeros(3)
    if include_head:
        state["encoder.head.weight"] = torch.zeros(3, MOCK_DIM)
        state["encoder.head.bias"] = torch.zeros(3)
    return state


def build_mock_model(num_classes=7):
    return panderm.build_panderm_classifier(
        num_classes,
        model_factory=mock_factory,
        state_dict=mock_pretrained_state(),
    )


class LoaderAndArchitectureTests(unittest.TestCase):
    def test_injected_mock_loader_builds_without_network_or_weights(self):
        model = build_mock_model()
        self.assertEqual(model.arch, "panderm_base_vit_b16")
        self.assertEqual(model.freeze_mode, "full_finetune")
        self.assertIsNone(model.pretrained_checkpoint)
        self.assertEqual(model.input_resolution, (224, 224))

    def test_official_architecture_kwargs_are_passed_through(self):
        build_mock_model()
        kwargs = mock_factory.last_kwargs
        self.assertEqual(kwargs["num_classes"], 7)
        self.assertEqual(kwargs["drop_path_rate"], panderm_run.DROP_PATH)
        self.assertEqual(kwargs["init_scale"], panderm.INIT_SCALE)
        self.assertTrue(kwargs["use_mean_pooling"])
        self.assertFalse(kwargs["use_rel_pos_bias"])
        self.assertFalse(kwargs["lin_probe"])
        self.assertFalse(kwargs["pretrained"])

    def test_head_is_a_fresh_seven_class_layer(self):
        model = build_mock_model()
        self.assertIsInstance(model.head, nn.Linear)
        self.assertEqual(model.head.out_features, 7)
        # The 3-class pretrained head must not survive into the new head.
        self.assertFalse(torch.equal(model.head.weight, torch.zeros_like(model.head.weight)))

    def test_state_remap_strips_encoder_drops_decoder_and_renames_norm(self):
        remapped = panderm.remap_pretrained_state_dict(mock_pretrained_state())
        self.assertTrue(all(not key.startswith("encoder.") for key in remapped))
        self.assertFalse(any(key.startswith(("decoder.", "teacher.")) for key in remapped))
        self.assertIn("fc_norm.weight", remapped)
        self.assertNotIn("norm.weight", remapped)
        self.assertNotIn("head.weight", remapped)
        self.assertIn("blocks.0.attn.weight", remapped)

    def test_checkpoint_without_encoder_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no 'encoder.' parameters"):
            panderm.remap_pretrained_state_dict({"blocks.0.attn.weight": torch.zeros(2)})

    def test_missing_backbone_parameters_are_rejected(self):
        state = mock_pretrained_state()
        state.pop("encoder.blocks.0.attn.weight")
        with self.assertRaisesRegex(ValueError, "missing backbone parameters"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_unexpected_checkpoint_keys_are_rejected(self):
        state = mock_pretrained_state()
        state["encoder.mystery.weight"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "cannot accept"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_building_without_weights_is_refused(self):
        with self.assertRaisesRegex(ValueError, "refusing to fine-tune randomly"):
            panderm.build_panderm_classifier(7, model_factory=mock_factory)

    def test_missing_upstream_checkout_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "modeling_finetune"):
                panderm.load_upstream_model_factory(tmp)


class FullTrainabilityTests(unittest.TestCase):
    def test_every_parameter_is_trainable(self):
        model = build_mock_model()
        total, trainable = panderm.parameter_counts(model)
        self.assertEqual(total, trainable)
        self.assertGreater(panderm.assert_full_trainability(model), 0)

    def test_a_frozen_backbone_is_refused(self):
        model = build_mock_model()
        model.blocks[0].attn.weight.requires_grad = False
        with self.assertRaisesRegex(ValueError, "requires_grad=True everywhere"):
            panderm.assert_full_trainability(model)

    def test_backbone_receives_gradients_and_updates(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        before = panderm.snapshot_parameters(model)
        images = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 3])
        nn.CrossEntropyLoss()(model(images), targets).backward()
        report = panderm.backbone_gradient_report(model)
        self.assertTrue(report["all_backbone_parameters_have_gradient"])
        self.assertTrue(report["backbone_gradients_finite"])
        self.assertTrue(report["head_parameters_have_gradient"])
        optimizer.step()
        self.assertGreater(panderm.changed_parameter_count(before, model), 0)

    def test_model_identity_reports_full_finetune(self):
        model = build_mock_model()
        identity = panderm.model_identity(
            model, train_transform="T", eval_transform="E", checkpoint_sha256="a" * 64
        )
        self.assertEqual(identity["arch"], "panderm_base_vit_b16")
        self.assertEqual(identity["freeze_mode"], "full_finetune")
        self.assertTrue(identity["all_parameters_trainable"])
        self.assertEqual(identity["input_resolution"], [224, 224])
        self.assertEqual(identity["embed_dim"], 768)
        self.assertEqual(identity["depth"], 12)
        self.assertEqual(identity["patch_size"], 16)


class PreprocessingTests(unittest.TestCase):
    def test_eval_transform_matches_upstream_resize_and_normalisation(self):
        text = repr(panderm.build_eval_transform())
        self.assertIn("Resize(size=256", text)
        self.assertIn("bicubic", text)
        self.assertIn("CenterCrop(size=(224, 224))", text)
        self.assertIn("mean=(0.485, 0.456, 0.406)", text)
        self.assertIn("std=(0.229, 0.224, 0.225)", text)

    def test_eval_transform_produces_the_official_resolution(self):
        from PIL import Image

        tensor = panderm.build_eval_transform()(
            Image.new("RGB", (600, 450), (10, 20, 30))
        )
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_train_transform_uses_the_official_upstream_arguments(self):
        captured = {}

        def fake_create_transform(**kwargs):
            captured.update(kwargs)
            return "train-transform"

        self.assertEqual(
            panderm.build_train_transform(create_transform=fake_create_transform),
            "train-transform",
        )
        self.assertEqual(captured["input_size"], 224)
        self.assertTrue(captured["is_training"])
        self.assertEqual(captured["auto_augment"], "rand-m9-mstd0.5-inc1")
        self.assertEqual(captured["interpolation"], "bicubic")
        self.assertEqual(captured["mean"], (0.485, 0.456, 0.406))
        self.assertEqual(captured["std"], (0.229, 0.224, 0.225))

    def test_preprocessing_identity_captures_both_pipelines(self):
        identity = panderm.preprocessing_identity("TRAIN", "EVAL")
        self.assertEqual(identity, {"train": "'TRAIN'", "eval": "'EVAL'"})


class OptimizerAndScheduleTests(unittest.TestCase):
    def test_layer_decay_scales_follow_the_upstream_rule(self):
        scales = panderm.layer_decay_scales(num_layers=12, layer_decay=0.65)
        self.assertEqual(len(scales), 14)
        self.assertAlmostEqual(scales[13], 1.0, places=12)
        self.assertAlmostEqual(scales[0], 0.65 ** 13, places=12)

    def test_layer_ids_follow_the_upstream_rule(self):
        self.assertEqual(panderm.layer_id_for_parameter("cls_token", 14), 0)
        self.assertEqual(panderm.layer_id_for_parameter("pos_embed", 14), 0)
        self.assertEqual(panderm.layer_id_for_parameter("patch_embed.proj.weight", 14), 0)
        self.assertEqual(panderm.layer_id_for_parameter("blocks.0.attn.weight", 14), 1)
        self.assertEqual(panderm.layer_id_for_parameter("blocks.11.mlp.weight", 14), 12)
        self.assertEqual(panderm.layer_id_for_parameter("fc_norm.weight", 14), 13)
        self.assertEqual(panderm.layer_id_for_parameter("head.weight", 14), 13)

    def test_optimizer_contains_every_trainable_parameter_exactly_once(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        expected = sum(1 for p in model.parameters() if p.requires_grad)
        self.assertEqual(
            panderm.verify_optimizer_covers_parameters_once(optimizer, model), expected
        )
        seen = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), expected)

    def test_duplicate_parameter_in_two_groups_is_rejected(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        optimizer.param_groups[0]["params"].append(
            optimizer.param_groups[1]["params"][0]
        )
        with self.assertRaisesRegex(ValueError, "more than one"):
            panderm.verify_optimizer_covers_parameters_once(optimizer, model)

    def test_missing_parameter_is_rejected(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        optimizer.param_groups[0]["params"].pop()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            panderm.verify_optimizer_covers_parameters_once(optimizer, model)

    def test_no_weight_decay_applies_to_one_dimensional_and_skipped_names(self):
        model = build_mock_model()
        groups = panderm.build_param_groups(model, num_layers=MOCK_DEPTH)
        for group in groups:
            for name in group["param_names"]:
                parameter = dict(model.named_parameters())[name]
                if parameter.ndim == 1 or name.endswith(".bias") or name in {
                    "cls_token", "pos_embed"
                }:
                    self.assertEqual(group["weight_decay"], 0.0, name)
                else:
                    self.assertEqual(
                        group["weight_decay"], panderm_run.WEIGHT_DECAY, name
                    )

    def test_scheduler_steps_per_optimizer_step_not_per_microbatch(self):
        # 7495 rows at batch 16 -> 469 micro-batches -> 58 optimizer steps at accum 8.
        self.assertEqual(panderm.optimizer_steps_per_epoch(469, 8), 58)
        self.assertEqual(panderm.optimizer_steps_per_epoch(16, 8), 2)
        with self.assertRaisesRegex(ValueError, "no\noptimizer step|no optimizer step"):
            panderm.optimizer_steps_per_epoch(4, 8)

    def test_schedule_warms_up_then_decays_over_optimizer_steps(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, base_lr=5e-4, warmup_epochs=10, epochs=50, steps_per_epoch=58
        )
        self.assertEqual(schedule.warmup_steps, 580)
        self.assertEqual(schedule.total_steps, 2900)
        self.assertAlmostEqual(schedule.lr_at(579), 5e-4, places=12)
        self.assertLess(schedule.lr_at(0), 5e-4)
        self.assertLess(schedule.lr_at(2899), 5e-4)
        self.assertGreater(schedule.lr_at(1000), schedule.lr_at(2000))

    def test_schedule_applies_group_lr_scales(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, base_lr=5e-4, warmup_epochs=1, epochs=2, steps_per_epoch=4
        )
        lr = schedule.apply()
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], lr * group["lr_scale"], places=12)

    def test_schedule_rejects_a_reshaped_resume(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, base_lr=5e-4, warmup_epochs=1, epochs=2, steps_per_epoch=4
        )
        state = schedule.state_dict()
        state["steps_per_epoch"] = 9
        with self.assertRaisesRegex(ValueError, "steps_per_epoch mismatch"):
            schedule.load_state_dict(state)

    def test_warmup_longer_than_training_is_refused(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        with self.assertRaisesRegex(ValueError, "must be within"):
            panderm.WarmupCosineSchedule(
                optimizer, warmup_epochs=10, epochs=5, steps_per_epoch=4
            )


class GradientAccumulationTests(unittest.TestCase):
    def test_loss_scale_is_one_over_accumulation_inside_a_window(self):
        for index in range(16):
            self.assertAlmostEqual(
                panderm.accumulation_loss_scale(index, 8, 16), 0.125, places=12
            )

    def test_trailing_microbatches_are_dropped_not_oversized(self):
        # 20 batches at accum 8 -> two full windows (16), last 4 dropped.
        self.assertEqual(panderm.accumulation_loss_scale(15, 8, 20), 0.125)
        self.assertEqual(panderm.accumulation_loss_scale(16, 8, 20), 0.0)
        self.assertEqual(panderm.accumulation_loss_scale(19, 8, 20), 0.0)

    def test_accumulated_gradient_equals_the_large_batch_gradient(self):
        torch.manual_seed(0)
        images = torch.randn(8, 3, 224, 224)
        targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 3])
        criterion = nn.CrossEntropyLoss()

        full = build_mock_model()
        criterion(full(images), targets).backward()
        full_grads = {n: p.grad.clone() for n, p in full.named_parameters()}

        accumulated = build_mock_model()
        accumulated.load_state_dict(full.state_dict())
        for index in range(4):
            chunk = slice(index * 2, index * 2 + 2)
            loss = criterion(accumulated(images[chunk]), targets[chunk])
            (loss * panderm.accumulation_loss_scale(index, 4, 4)).backward()
        for name, parameter in accumulated.named_parameters():
            torch.testing.assert_close(
                parameter.grad, full_grads[name], rtol=1e-4, atol=1e-6, msg=name
            )

    def test_zero_accumulation_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            panderm.accumulation_loss_scale(0, 0, 8)

    def test_non_finite_values_stop_the_run(self):
        with self.assertRaisesRegex(ValueError, "non-finite training loss"):
            panderm.require_finite(float("nan"), "training loss")
        model = build_mock_model()
        parameter = list(model.parameters())[0]
        parameter.grad = torch.full_like(parameter, float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite gradient"):
            panderm.require_finite_gradients(model.parameters())


class CheckpointTests(unittest.TestCase):
    """Full-model checkpoint round-trip, including AMP/scheduler/optimizer state."""

    def _components(self, epochs=2):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, base_lr=5e-4, warmup_epochs=1, epochs=epochs, steps_per_epoch=4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return model, optimizer, schedule, scaler

    def _identity(self, **overrides):
        identity = panderm_run.build_run_identity(
            git_commit="c" * 40,
            seed=0,
            epochs=5,
            evaluation_scope="validation_only",
            checkpoint_sha256="a" * 64,
            model_identity={"arch": "panderm_base_vit_b16"},
            manifest_sha256={"train": "t", "val": "v"},
            fixed_split_identity="t",
            shared_root_uuid="uuid",
            formal_output_identity="out",
            dependency_versions={"torch": "2.8.0"},
        )
        identity.update(overrides)
        return identity

    def test_checkpoint_stores_full_model_and_reopens_identically(self):
        from ddpm_derm import train_panderm

        model, optimizer, schedule, scaler = self._components()
        identity = self._identity()
        args = type("A", (), {"seed": 0, "epochs": 5})()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            train_panderm.save_checkpoint(
                path, model, optimizer, schedule, scaler, 1, 0.5,
                [{"epoch": 1, "train_loss": 1.0, "val_df_f1": 0.5, "val_macro_f1": 0.3}],
                args, identity,
            )
            saved = train_panderm.load_checkpoint_safe(path, map_location="cpu")
        self.assertEqual(saved["checkpoint_format"], "panderm_full_model_v1")
        self.assertIn("model_state_dict", saved)
        self.assertNotIn("head_state_dict", saved)
        for key in (
            "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict",
            "rng_state", "run_identity", "class_to_idx",
        ):
            self.assertIn(key, saved)
        self.assertEqual(set(saved["model_state_dict"]), set(model.state_dict()))
        self.assertEqual(saved["run_identity"], identity)

    def test_head_only_checkpoint_is_rejected(self):
        from ddpm_derm import train_panderm

        model = build_mock_model()
        payload = {
            "checkpoint_schema_version": 1,
            "checkpoint_format": "panderm_full_model_v1",
            "head_state_dict": model.head.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {}, "scheduler_state_dict": {},
            "scaler_state_dict": {}, "epoch": 1, "best_val_df_f1": 0.1,
            "history": [], "config": {}, "class_to_idx": {}, "rng_state": {},
            "run_identity": {},
        }
        with self.assertRaisesRegex(ValueError, "full model, not a head"):
            train_panderm.validate_checkpoint_payload(payload, model)

    def test_incomplete_checkpoint_is_rejected(self):
        from ddpm_derm import train_panderm

        model = build_mock_model()
        with self.assertRaisesRegex(ValueError, "fields missing"):
            train_panderm.validate_checkpoint_payload(
                {"checkpoint_format": "panderm_full_model_v1"}, model
            )

    def test_resume_restores_model_optimizer_scheduler_and_scaler(self):
        from ddpm_derm import train_panderm

        model, optimizer, schedule, scaler = self._components()
        for _ in range(3):
            schedule.step()
        images = torch.randn(2, 3, 224, 224)
        nn.CrossEntropyLoss()(model(images), torch.tensor([1, 4])).backward()
        optimizer.step()
        identity = self._identity()
        args = type("A", (), {"seed": 0, "epochs": 5})()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            train_panderm.save_checkpoint(
                path, model, optimizer, schedule, scaler, 1, 0.5, [{"epoch": 1}],
                args, identity,
            )
            saved = train_panderm.load_checkpoint_safe(path, map_location="cpu")

        fresh_model, fresh_opt, fresh_sched, fresh_scaler = self._components()
        self.assertNotEqual(fresh_sched.step_count, schedule.step_count)
        start, best, history = train_panderm.restore_checkpoint_state(
            saved, fresh_model, fresh_opt, fresh_sched, fresh_scaler
        )
        self.assertEqual((start, best, history), (2, 0.5, [{"epoch": 1}]))
        self.assertEqual(fresh_sched.step_count, schedule.step_count)
        for (name, restored), (_, original) in zip(
            fresh_model.named_parameters(), model.named_parameters()
        ):
            torch.testing.assert_close(restored, original, msg=name)
        self.assertEqual(
            fresh_opt.state_dict()["param_groups"][0]["lr"],
            optimizer.state_dict()["param_groups"][0]["lr"],
        )

    def test_identity_mismatch_is_rejected_before_state_is_mutated(self):
        from ddpm_derm import train_panderm

        model, optimizer, schedule, scaler = self._components()
        identity = self._identity()
        args = type("A", (), {"seed": 0, "epochs": 5})()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            train_panderm.save_checkpoint(
                path, model, optimizer, schedule, scaler, 1, 0.5, [], args, identity
            )
            saved = train_panderm.load_checkpoint_safe(path, map_location="cpu")

        fresh_model, fresh_opt, fresh_sched, fresh_scaler = self._components()
        before = panderm.snapshot_parameters(fresh_model)
        before_steps = fresh_sched.step_count
        drifted = self._identity(checkpoint_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            panderm_run.require_matching_identity(saved["run_identity"], drifted)
        # The guard runs before restore_checkpoint_state, so nothing moved.
        self.assertEqual(panderm.changed_parameter_count(before, fresh_model), 0)
        self.assertEqual(fresh_sched.step_count, before_steps)


class IdentityTests(unittest.TestCase):
    def _identity(self, **overrides):
        identity = panderm_run.build_run_identity(
            git_commit="c" * 40,
            seed=0,
            epochs=5,
            evaluation_scope="validation_only",
            checkpoint_sha256="a" * 64,
            model_identity={"arch": "panderm_base_vit_b16"},
            manifest_sha256={"train": "t", "val": "v"},
            fixed_split_identity="t",
            shared_root_uuid="uuid",
            formal_output_identity="out",
            dependency_versions={"torch": "2.8.0", "timm": "0.9.16"},
        )
        identity.update(overrides)
        return identity

    def test_identity_carries_every_required_field(self):
        identity = self._identity()
        for key in panderm_run.IMMUTABLE_IDENTITY_KEYS:
            self.assertIn(key, identity)
        self.assertEqual(identity["upstream_commit"], panderm_run.UPSTREAM_COMMIT)
        self.assertEqual(identity["checkpoint_filename"], panderm_run.CHECKPOINT_FILENAME)
        self.assertEqual(identity["objective"]["class_weighting"], "none")
        self.assertEqual(identity["objective"]["sampler"], "none")
        self.assertFalse(identity["objective"]["mixup"])
        self.assertFalse(identity["objective"]["cutmix"])
        self.assertFalse(identity["objective"]["tta"])
        self.assertFalse(identity["c1_construction"]["synthetic_images_used"])
        self.assertEqual(identity["optimization"]["effective_batch_size"], 128)
        self.assertEqual(identity["optimization"]["scheduler_step_unit"], "optimizer_step")
        self.assertEqual(identity["claim_boundary"], "suggestive_exploratory_only")
        self.assertEqual(identity["manifest_sha256"], {"train": "t", "val": "v"})

    def test_identity_is_json_serialisable(self):
        json.dumps(self._identity())

    def test_non_c1_variant_is_refused(self):
        with self.assertRaisesRegex(ValueError, "C1 real-data only"):
            panderm_run.build_run_identity(
                git_commit=None, seed=0, epochs=5, evaluation_scope="validation_only",
                checkpoint_sha256="a" * 64, model_identity={},
                manifest_sha256={"train": "t", "val": "v"},
                fixed_split_identity="t", shared_root_uuid=None,
                formal_output_identity=None, dependency_versions={},
                variant="C4",
            )

    def test_missing_manifest_identity_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing splits"):
            panderm_run.build_run_identity(
                git_commit=None, seed=0, epochs=5, evaluation_scope="validation_only",
                checkpoint_sha256="a" * 64, model_identity={},
                manifest_sha256={"train": "t"}, fixed_split_identity="t",
                shared_root_uuid=None, formal_output_identity=None,
                dependency_versions={},
            )

    def test_every_immutable_field_is_actually_enforced(self):
        saved = self._identity()
        for key in panderm_run.IMMUTABLE_IDENTITY_KEYS:
            drifted = copy.deepcopy(saved)
            drifted[key] = "DRIFTED"
            with self.subTest(field=key):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    panderm_run.require_matching_identity(saved, drifted)

    def test_deleting_an_expected_identity_field_is_rejected(self):
        for key in panderm_run.IMMUTABLE_IDENTITY_KEYS:
            truncated = {
                name: value
                for name, value in self._identity().items()
                if name != key
            }
            with self.subTest(field=key):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    panderm_run.require_expected_identity_complete(truncated)

    def test_missing_duplicate_is_rejected(self):
        identity = self._identity()
        top_level = {
            key: identity[key]
            for key in panderm_run.IMMUTABLE_IDENTITY_KEYS
            if key != "checkpoint_sha256"
        }
        with self.assertRaisesRegex(ValueError, "missing duplicated identity"):
            panderm_run.require_identity_duplicates(
                expected=identity, record={"run_version": identity["run_version"]},
                nested=identity, top_level=top_level,
            )

    def test_duplicate_drift_is_rejected(self):
        identity = self._identity()
        top_level = {key: identity[key] for key in panderm_run.IMMUTABLE_IDENTITY_KEYS}
        top_level["checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "duplicated identity drift"):
            panderm_run.require_identity_duplicates(
                expected=identity, record={"run_version": identity["run_version"]},
                nested=identity, top_level=top_level,
            )

    def test_consistently_wrong_duplicates_are_still_rejected(self):
        """All copies agreeing is not enough; they must match the reviewed value."""
        expected = self._identity()
        wrong = self._identity(checkpoint_sha256="b" * 64, upstream_commit="d" * 40)
        top_level = {key: wrong[key] for key in panderm_run.IMMUTABLE_IDENTITY_KEYS}
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            panderm_run.require_identity_duplicates(
                expected=expected, record={"run_version": wrong["run_version"]},
                nested=wrong, top_level=top_level,
            )


class ProvenanceGateTests(unittest.TestCase):
    def test_placeholder_hash_blocks_validation(self):
        self.assertEqual(
            panderm_run.EXPECTED_CHECKPOINT_SHA256,
            panderm_run.CHECKPOINT_SHA256_PLACEHOLDER,
        )
        with self.assertRaisesRegex(ValueError, "not pinned"):
            panderm_run.require_provenance_clearance(
                upstream_commit=panderm_run.UPSTREAM_COMMIT,
                checkpoint_sha256="a" * 64,
            )

    def test_formal_and_test_access_are_unconditionally_prohibited(self):
        for purpose in (
            panderm_run.FORMAL_TRAINING,
            panderm_run.TEST_ACCESS,
        ):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(
                    ValueError, "independently unauditable"
                ):
                    panderm_run.require_provenance_clearance(
                        upstream_commit=panderm_run.UPSTREAM_COMMIT,
                        checkpoint_sha256="a" * 64,
                        expected_checkpoint_sha256="a" * 64,
                        purpose=purpose,
                    )

    def test_pinned_hash_and_commit_clear_validation_only(self):
        cleared = panderm_run.require_provenance_clearance(
            upstream_commit=panderm_run.UPSTREAM_COMMIT,
            checkpoint_sha256="a" * 64,
            expected_checkpoint_sha256="a" * 64,
            purpose=panderm_run.VALIDATION_ONLY,
        )
        self.assertEqual(cleared["cleared_for"], "validation_only")
        self.assertFalse(cleared["deployment_allowed"])
        self.assertFalse(cleared["formal_training_allowed"])
        self.assertFalse(cleared["test_access_allowed"])
        self.assertEqual(cleared["claim_boundary"], "suggestive_exploratory_only")

    def test_unknown_private_corpus_and_author_assertion_never_clear_formal(self):
        reviews = (
            {
                **panderm_run.CONTAMINATION_REVIEW,
                "image_level_ham10000_overlap": "unknown_private_corpus",
            },
            {
                **panderm_run.CONTAMINATION_REVIEW,
                "image_level_ham10000_overlap":
                    "excluded_on_primary_source_author_statement",
            },
            {
                **panderm_run.CONTAMINATION_REVIEW,
                "independent_audit_possible": False,
            },
            {
                **panderm_run.CONTAMINATION_REVIEW,
                "patient_level_overlap": "not_excludable",
            },
        )
        for review in reviews:
            for purpose in (
                panderm_run.FORMAL_TRAINING,
                panderm_run.TEST_ACCESS,
            ):
                with self.subTest(review=review, purpose=purpose):
                    with self.assertRaisesRegex(
                        ValueError, "independently unauditable"
                    ):
                        panderm_run.require_provenance_clearance(
                            upstream_commit=panderm_run.UPSTREAM_COMMIT,
                            checkpoint_sha256="a" * 64,
                            expected_checkpoint_sha256="a" * 64,
                            contamination_review=review,
                            purpose=purpose,
                        )

    def test_validation_requires_acknowledged_audit_limits(self):
        for key, value in (
            ("independent_audit_possible", True),
            ("patient_level_overlap", "excluded"),
            ("exact_fixed_validation_test_overlap", "excluded"),
        ):
            review = {**panderm_run.CONTAMINATION_REVIEW, key: value}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    panderm_run.require_provenance_clearance(
                        upstream_commit=panderm_run.UPSTREAM_COMMIT,
                        checkpoint_sha256="a" * 64,
                        expected_checkpoint_sha256="a" * 64,
                        contamination_review=review,
                    )

    def test_unsupported_gate_purpose_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            panderm_run.require_provenance_clearance(
                upstream_commit=panderm_run.UPSTREAM_COMMIT,
                checkpoint_sha256="a" * 64,
                expected_checkpoint_sha256="a" * 64,
                purpose="full",
            )

    def test_wrong_upstream_commit_blocks_the_gate(self):
        with self.assertRaisesRegex(ValueError, "upstream commit mismatch"):
            panderm_run.require_provenance_clearance(
                upstream_commit="0" * 40,
                checkpoint_sha256="a" * 64,
                expected_checkpoint_sha256="a" * 64,
            )

    def test_hash_mismatch_blocks_the_gate(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            panderm_run.require_provenance_clearance(
                upstream_commit=panderm_run.UPSTREAM_COMMIT,
                checkpoint_sha256="b" * 64,
                expected_checkpoint_sha256="a" * 64,
            )

    def test_a_permissive_license_review_cannot_be_faked_by_dropping_fields(self):
        for override in (
            {"status": "assumed_ok"},
            {"finetuning_allowed": False},
            {"deployment_allowed": True},
        ):
            review = {**panderm_run.LICENSE_REVIEW, **override}
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    panderm_run.require_provenance_clearance(
                        upstream_commit=panderm_run.UPSTREAM_COMMIT,
                        checkpoint_sha256="a" * 64,
                        expected_checkpoint_sha256="a" * 64,
                        license_review=review,
                    )

    def test_widened_claim_boundary_blocks_the_gate(self):
        review = {**panderm_run.CONTAMINATION_REVIEW, "claim_boundary": "confirmed"}
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            panderm_run.require_provenance_clearance(
                upstream_commit=panderm_run.UPSTREAM_COMMIT,
                checkpoint_sha256="a" * 64,
                expected_checkpoint_sha256="a" * 64,
                contamination_review=review,
            )

    def test_unpinned_checkpoint_hash_fails_loud_with_the_observed_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            path.write_bytes(b"not the real checkpoint")
            with self.assertRaises(ValueError) as caught:
                panderm_run.require_checkpoint_sha256(path)
            message = str(caught.exception)
            self.assertIn("not pinned", message)
            self.assertIn("observed:", message)
            self.assertIn("trust-on-first-use", message)

    def test_pinned_hash_mismatch_refuses_the_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            path.write_bytes(b"payload")
            with self.assertRaisesRegex(ValueError, "refusing unverified weights"):
                panderm_run.require_checkpoint_sha256(path, "a" * 64)

    def test_license_review_records_the_primary_source_verdict(self):
        review = panderm_run.LICENSE_REVIEW
        self.assertEqual(review["license"], "CC-BY-NC-ND-4.0")
        self.assertFalse(review["license_file_in_repo"])
        self.assertTrue(review["finetuning_allowed"])
        self.assertFalse(review["sharing_adapted_weights_allowed"])
        self.assertFalse(review["deployment_allowed"])
        self.assertIn("Nature Medicine", review["attribution_required"])

    def test_contamination_review_records_both_verdicts_and_open_risks(self):
        review = panderm_run.CONTAMINATION_REVIEW
        self.assertEqual(
            review["image_level_ham10000_overlap"],
            "not_independently_excludable",
        )
        self.assertEqual(review["patient_level_overlap"], "not_excludable")
        self.assertEqual(
            review["ham10000_in_upstream_finetuning_or_evaluation"],
            "yes_evaluation_benchmark",
        )
        self.assertTrue(review["loaded_checkpoint_is_pretraining_only"])
        self.assertFalse(review["independent_audit_possible"])
        self.assertEqual(review["exact_fixed_validation_test_overlap"], "unproven")
        self.assertGreaterEqual(len(review["evidence"]), 3)
        self.assertGreaterEqual(len(review["unresolved_risks"]), 3)

    def test_deployment_surface_stays_free_of_panderm(self):
        self.assertEqual(panderm_run.require_no_deployment_contamination(ROOT), [])
        self.assertFalse(panderm_run.DEPLOYMENT_ALLOWED)

    def test_deployment_contamination_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app").mkdir()
            (root / "app" / "main.py").write_text(
                "model = load_panderm()", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "never reach the public deployment"):
                panderm_run.require_no_deployment_contamination(root)


class CliTests(unittest.TestCase):
    def _base(self, *extra):
        return [
            "--checkpoint", "/tmp/weights.pth",
            "--upstream-dir", "/tmp/upstream",
            "--output-dir", "/tmp/out",
            *extra,
        ]

    def test_defaults_match_the_frozen_configuration(self):
        from ddpm_derm import train_panderm

        args = train_panderm.parse_args(self._base())
        self.assertEqual(args.variant, "C1")
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.accumulation_steps, 8)
        self.assertEqual(args.lr, 5e-4)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.layer_decay, 0.65)
        self.assertEqual(args.df_target_count, 585)
        self.assertEqual(args.evaluation_scope, "validation_only")

    def test_illegal_combinations_are_rejected(self):
        from ddpm_derm import train_panderm

        for extra in (
            ["--variant", "C4"],
            ["--generated-manifest", "/tmp/synth.csv"],
            ["--run-version", "v2_other"],
            ["--df-target-count", "586"],
            ["--accumulation-steps", "0"],
            ["--batch-size", "0"],
            ["--warmup-epochs", "60"],
            ["--evaluation-scope", "train"],
            ["--evaluation-scope", "full"],
            ["--drop-path", "0.3"],
            ["--no-amp"],
            ["--seed", "1"],
            ["--epochs", "50"],
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(SystemExit):
                    train_panderm.parse_args(self._base(*extra))

    def test_checkpoint_upstream_and_output_are_required(self):
        from ddpm_derm import train_panderm

        for argv in (
            ["--upstream-dir", "/tmp/u", "--output-dir", "/tmp/o"],
            ["--checkpoint", "/tmp/w", "--output-dir", "/tmp/o"],
            ["--checkpoint", "/tmp/w", "--upstream-dir", "/tmp/u"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    train_panderm.parse_args(argv)


class C1DataTests(unittest.TestCase):
    def test_c1_frame_is_real_only_and_matched(self):
        from ddpm_derm import manifests, train_panderm

        frame = train_panderm.build_c1_frame(0, 585, None)
        self.assertEqual(len(frame), panderm_run.EXPECTED_C1_TRAIN_ROWS)
        counts = manifests.class_counts(frame)
        self.assertEqual(counts, panderm_run.EXPECTED_C1_CLASS_COUNTS)
        self.assertEqual(counts["df"], 585)
        if "source" in frame.columns:
            self.assertEqual(set(frame["source"].dropna().unique()), {"real"})
        train_df_ids = set(
            manifests.load_split("train")
            .query("dx == 'df'")["image_id"].astype(str)
        )
        self.assertEqual(
            set(frame.query("dx == 'df'")["image_id"].astype(str)), train_df_ids
        )

    def test_c1_rows_never_come_from_validation(self):
        from ddpm_derm import manifests, train_panderm

        frame = train_panderm.build_c1_frame(0, 585, None)
        used = set(frame["image_id"].astype(str))
        validation = set(manifests.load_split("val")["image_id"].astype(str))
        self.assertEqual(used & validation, set())

    def test_manifest_identity_covers_train_and_validation_only(self):
        from ddpm_derm import train_panderm

        identity = train_panderm.manifest_identity()
        self.assertEqual(set(identity), {"train", "val"})
        for value in identity.values():
            self.assertTrue(panderm_run.is_pinned_sha256(value))

    def test_expected_split_counts_match_the_fixed_split(self):
        from ddpm_derm import manifests

        for split in ("train", "val"):
            expected = panderm_run.EXPECTED_SPLIT_COUNTS[split]
            frame = manifests.load_split(split)
            self.assertEqual(len(frame), expected)
            self.assertEqual(
                int((frame["dx"] == "df").sum()),
                panderm_run.EXPECTED_SPLIT_DF_COUNTS[split],
            )


class NonCollapseGateTests(unittest.TestCase):
    def _result(self, **overrides):
        result = {
            "history": [
                {"epoch": 1, "train_loss": 1.2, "val_df_f1": 0.1, "val_macro_f1": 0.2},
                {"epoch": 2, "train_loss": 0.9, "val_df_f1": 0.3, "val_macro_f1": 0.4},
            ],
            "best_val_df_f1": 0.3,
            "test_metrics": None,
        }
        result.update(overrides)
        return result

    def _counts(self, **overrides):
        counts = {
            "akiec": 10, "bcc": 20, "bkl": 30, "df": 5,
            "mel": 40, "nv": 1400, "vasc": 5,
        }
        counts.update(overrides)
        return counts

    def _evaluate(self, result=None, counts=None, **flags):
        options = {
            "backbone_gradient_verified": True,
            "backbone_parameters_updated": True,
            "identity_complete": True,
            "no_test_access": True,
            "provenance_allows_next_stage": True,
        }
        options.update(flags)
        return panderm_run.evaluate_non_collapse_gate(
            result=result or self._result(),
            prediction_counts=counts or self._counts(),
            **options,
        )

    def test_healthy_run_passes_every_check(self):
        checks = self._evaluate()
        self.assertEqual(set(checks), set(panderm_run.NON_COLLAPSE_CHECK_KEYS))
        self.assertTrue(all(checks.values()))

    def test_zero_df_f1_fails(self):
        checks = self._evaluate(self._result(best_val_df_f1=0.0))
        self.assertFalse(checks["best_validation_df_f1_positive"])

    def test_no_predicted_df_fails(self):
        checks = self._evaluate(counts=self._counts(df=0))
        self.assertFalse(checks["predicted_df_positive"])

    def test_all_nv_collapse_fails(self):
        counts = {name: 0 for name in self._counts()}
        counts["nv"] = 1510
        checks = self._evaluate(counts=counts)
        self.assertFalse(checks["not_all_nv"])
        self.assertFalse(checks["at_least_two_predicted_classes"])

    def test_all_df_collapse_fails(self):
        counts = {name: 0 for name in self._counts()}
        counts["df"] = 1510
        checks = self._evaluate(counts=counts)
        self.assertFalse(checks["not_all_df"])

    def test_non_finite_history_fails(self):
        result = self._result(history=[
            {"epoch": 1, "train_loss": float("nan"), "val_df_f1": 0.1,
             "val_macro_f1": 0.2}
        ])
        self.assertFalse(self._evaluate(result)["finite_losses"])

    def test_test_metrics_present_fails(self):
        checks = self._evaluate(self._result(test_metrics={"target_f1": 0.5}))
        self.assertFalse(checks["test_metrics_null"])

    def test_linear_probe_style_run_fails(self):
        checks = self._evaluate(backbone_gradient_verified=False)
        self.assertFalse(checks["backbone_gradient_verified"])
        checks = self._evaluate(backbone_parameters_updated=False)
        self.assertFalse(checks["backbone_parameters_updated"])

    def test_blocked_provenance_fails(self):
        self.assertFalse(
            self._evaluate(provenance_allows_next_stage=False)[
                "provenance_allows_next_stage"
            ]
        )


class AggregationTests(unittest.TestCase):
    def test_formal_aggregation_always_fails_loud(self):
        with self.assertRaisesRegex(ValueError, "independently unauditable"):
            panderm_run.aggregate_results([])


class OutputIsolationTests(unittest.TestCase):
    def test_frozen_source_modules_are_untouched(self):
        """The PanDerm work must not alter the ResNet/CoCa code path."""
        import subprocess

        frozen = [
            "src/ddpm_derm/model.py",
            "src/ddpm_derm/train_classifier.py",
            "src/ddpm_derm/classifier_run.py",
            "src/ddpm_derm/coca_run.py",
            "src/ddpm_derm/dataset.py",
            "src/ddpm_derm/manifests.py",
            "src/ddpm_derm/metrics.py",
            "src/ddpm_derm/config.py",
        ]
        for path in frozen:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "diff", "--quiet", "HEAD", "--", path],
                    cwd=ROOT, check=False,
                )
                self.assertEqual(result.returncode, 0, f"{path} was modified")

    def test_durable_directory_refuses_recursive_creation(self):
        from ddpm_derm import train_panderm

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent" / "deeper"
            with self.assertRaisesRegex(FileNotFoundError, "refusing recursive mkdir"):
                train_panderm._ensure_durable_directory(missing)

    def test_checkpoint_write_is_atomic_and_reopened(self):
        from ddpm_derm import train_panderm

        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        schedule = panderm.WarmupCosineSchedule(
            optimizer, warmup_epochs=1, epochs=2, steps_per_epoch=4
        )
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        identity = panderm_run.build_run_identity(
            git_commit="c" * 40, seed=0, epochs=5,
            evaluation_scope="validation_only", checkpoint_sha256="a" * 64,
            model_identity={"arch": "panderm_base_vit_b16"},
            manifest_sha256={"train": "t", "val": "v"},
            fixed_split_identity="t", shared_root_uuid="u",
            formal_output_identity="o", dependency_versions={},
        )
        args = type("A", (), {"seed": 0, "epochs": 2})()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            train_panderm.save_checkpoint(
                path, model, optimizer, schedule, scaler, 1, 0.4, [{"epoch": 1}],
                args, identity,
            )
            self.assertTrue(path.is_file())
            # No temporary residue is left behind next to the checkpoint.
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()),
                ["last.pt", "last.pt.integrity.json"],
            )
            reopened = train_panderm.load_checkpoint_safe(path, map_location="cpu")
            self.assertEqual(reopened["run_identity"], identity)
            self.assertEqual(reopened["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
