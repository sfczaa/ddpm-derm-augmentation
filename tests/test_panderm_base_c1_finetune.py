"""Targeted tests for the PanDerm-Base C1 full fine-tuning experiment.

Every test injects a mock PanDerm loader: no real 400 MB checkpoint is
downloaded and no network access is required.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import textwrap
import types
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ddpm_derm import panderm, panderm_run  # noqa: E402


MOCK_DEPTH = 4
MOCK_DIM = 16
REVIEWED_CHECKPOINT_SHA256 = (
    "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1"
)


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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 5, dim), requires_grad=False
        )
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
        tokens = tokens + self.pos_embed.clone().detach()
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


class AllowDurableWriteGuard:
    def bind_run_identity(self, run_identity):
        self.run_identity = copy.deepcopy(run_identity)
        return panderm_run.canonical_identity_sha256(run_identity)

    def require(self, phase):
        return {"phase": phase}


class SharedRootSentinelIdentityTests(unittest.TestCase):
    UUID = "765b971f-d148-4960-a77d-b73f28fc013c"
    ALIAS = "ddpm-derm-panderm-runs"

    def sentinel(self, stored_path, **overrides):
        value = {
            "shared_root_uuid": self.UUID,
            "shortcut_alias": self.ALIAS,
            "resolved_path": str(stored_path),
            "run_version": panderm_run.RUN_VERSION,
        }
        value.update(overrides)
        return value

    def test_same_physical_root_with_different_strings_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            stored_root = Path(temporary) / "shared"
            stored_root.mkdir()
            current_root = stored_root / ".." / stored_root.name
            self.assertNotEqual(str(stored_root), str(current_root))
            self.assertTrue(stored_root.samefile(current_root))
            sentinel = self.sentinel(stored_root)
            before = copy.deepcopy(sentinel)

            shared_root_uuid = (
                panderm_run.require_shared_root_sentinel_identity(
                    sentinel,
                    shortcut_alias=self.ALIAS,
                    resolved_root=current_root,
                )
            )

            self.assertEqual(shared_root_uuid, self.UUID)
            self.assertEqual(sentinel, before)

    def test_different_physical_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            stored_root = Path(temporary) / "stored"
            current_root = Path(temporary) / "current"
            stored_root.mkdir()
            current_root.mkdir()
            with self.assertRaisesRegex(
                ValueError, "different physical directory"
            ):
                panderm_run.require_shared_root_sentinel_identity(
                    self.sentinel(stored_root),
                    shortcut_alias=self.ALIAS,
                    resolved_root=current_root,
                )

    def test_missing_stored_root_fails_loud(self):
        with tempfile.TemporaryDirectory() as temporary:
            current_root = Path(temporary) / "current"
            current_root.mkdir()
            with self.assertRaisesRegex(
                ValueError, "physical identity could not be verified"
            ):
                panderm_run.require_shared_root_sentinel_identity(
                    self.sentinel(Path(temporary) / "missing"),
                    shortcut_alias=self.ALIAS,
                    resolved_root=current_root,
                )

    def test_wrong_shortcut_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "shortcut alias mismatch"):
                panderm_run.require_shared_root_sentinel_identity(
                    self.sentinel(root, shortcut_alias="private-copy"),
                    shortcut_alias=self.ALIAS,
                    resolved_root=root,
                )

    def test_wrong_run_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "run version mismatch"):
                panderm_run.require_shared_root_sentinel_identity(
                    self.sentinel(root, run_version="other-version"),
                    shortcut_alias=self.ALIAS,
                    resolved_root=root,
                )

    def test_invalid_uuid_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "UUID is invalid"):
                panderm_run.require_shared_root_sentinel_identity(
                    self.sentinel(root, shared_root_uuid="not-a-uuid"),
                    shortcut_alias=self.ALIAS,
                    resolved_root=root,
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

    def test_payload_matching_no_known_layout_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "unrecognised PanDerm checkpoint layout"
        ):
            panderm.remap_pretrained_state_dict({"mystery.weight": torch.zeros(2)})

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


# --- published checkpoint layouts --------------------------------------------
# The official panderm_bb_data6_checkpoint-499.pth is NOT the wrapped pretraining
# layout the loader originally assumed. It is a plain OrderedDict of 186 tensors:
# cls_token, pos_embed, patch_embed.*, blocks.* and norm.weight/norm.bias, with no
# wrapper, no ``encoder.`` prefix and no classifier head. ``direct_backbone_state``
# reproduces that shape (at mock scale) by taking the mock backbone and renaming
# ``fc_norm.`` back to the published ``norm.``, so these tests exercise the real
# layout rather than yet another encoder.* mock.
def direct_backbone_state(*, include_head=False):
    reference = MockPanDerm()
    state = {}
    for key, value in reference.state_dict().items():
        if key.startswith("head."):
            continue
        if key.startswith("fc_norm."):
            key = "norm." + key[len("fc_norm."):]
        state[key] = value
    if include_head:
        # A pretrained head must be dropped, never loaded into the fresh one.
        state["head.weight"] = torch.zeros(3, MOCK_DIM)
        state["head.bias"] = torch.zeros(3)
    return state


class CheckpointLayoutTests(unittest.TestCase):
    """Accept both published layouts; refuse everything that is merely dict-like.

    Local scope note: these are schema regressions built from small deterministic
    tensors. They never download or read the real 343 MB artifact.
    """

    def test_direct_fixture_matches_the_published_key_shape(self):
        state = direct_backbone_state()
        # Same sentinel set the real checkpoint exposes.
        for key in panderm.DIRECT_SENTINEL_KEYS:
            self.assertIn(key, state)
        self.assertTrue(any(key.startswith("blocks.0.") for key in state))
        self.assertFalse(any(key.startswith("encoder.") for key in state))
        self.assertFalse(any(key.startswith("head.") for key in state))

    def test_direct_complete_backbone_is_detected_and_accepted(self):
        state = direct_backbone_state()
        self.assertEqual(
            panderm.detect_checkpoint_layout(state), panderm.LAYOUT_DIRECT_BACKBONE
        )
        model = panderm.build_panderm_classifier(
            7, model_factory=mock_factory, state_dict=state
        )
        self.assertEqual(
            model.pretrained_state_layout, panderm.LAYOUT_DIRECT_BACKBONE
        )
        self.assertIn("blocks.0.attn.weight", model.loaded_backbone_keys)

    def test_direct_norm_is_remapped_onto_fc_norm(self):
        remapped = panderm.remap_pretrained_state_dict(direct_backbone_state())
        self.assertIn("fc_norm.weight", remapped)
        self.assertIn("fc_norm.bias", remapped)
        self.assertFalse(any(key.startswith("norm.") for key in remapped))

    def test_direct_pretrained_head_never_reaches_the_fresh_head(self):
        state = direct_backbone_state(include_head=True)
        remapped = panderm.remap_pretrained_state_dict(state)
        self.assertNotIn("head.weight", remapped)
        self.assertNotIn("head.bias", remapped)
        model = panderm.build_panderm_classifier(
            7, model_factory=mock_factory, state_dict=state
        )
        self.assertEqual(model.head.out_features, 7)
        self.assertFalse(
            torch.equal(model.head.weight, torch.zeros_like(model.head.weight))
        )

    def test_direct_missing_sentinel_is_rejected_before_any_load(self):
        state = direct_backbone_state()
        state.pop("pos_embed")
        with self.assertRaisesRegex(ValueError, "missing required backbone sentinels"):
            panderm.detect_checkpoint_layout(state)

    def test_direct_incomplete_backbone_fails_the_final_load_contract(self):
        # Sentinels all present, but a real block parameter is gone: this must
        # still fail loud, at the load contract rather than at detection.
        state = direct_backbone_state()
        state.pop("blocks.0.attn.weight")
        self.assertEqual(
            panderm.detect_checkpoint_layout(state), panderm.LAYOUT_DIRECT_BACKBONE
        )
        with self.assertRaisesRegex(ValueError, "missing backbone parameters"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_direct_unexpected_model_key_is_rejected(self):
        state = direct_backbone_state()
        state["blocks.0.mystery.weight"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "cannot accept"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_direct_key_outside_the_backbone_allowlist_is_rejected(self):
        for intruder in ("optimizer", "decoder.block.weight", "teacher.proj.weight"):
            with self.subTest(intruder=intruder):
                state = direct_backbone_state()
                state[intruder] = torch.zeros(2)
                with self.assertRaisesRegex(ValueError, "outside the backbone allowlist"):
                    panderm.detect_checkpoint_layout(state)

    def test_direct_non_tensor_value_is_rejected(self):
        state = direct_backbone_state()
        state["blocks.0.norm1.weight"] = {"not": "a tensor"}
        with self.assertRaisesRegex(ValueError, "non-tensor values"):
            panderm.remap_pretrained_state_dict(state)

    def test_direct_mixed_with_encoder_layout_is_rejected(self):
        state = direct_backbone_state()
        state["encoder.blocks.0.attn.weight"] = torch.zeros(MOCK_DIM, MOCK_DIM)
        with self.assertRaisesRegex(ValueError, "mixes the wrapped 'encoder.' layout"):
            panderm.detect_checkpoint_layout(state)

    def test_unknown_wrapper_and_unknown_payload_are_rejected(self):
        for payload in (
            {"model_ema": torch.zeros(2)},
            {"epoch": torch.zeros(2), "args": torch.zeros(2)},
        ):
            with self.subTest(payload=sorted(payload)):
                with self.assertRaisesRegex(
                    ValueError, "unrecognised PanDerm checkpoint layout"
                ):
                    panderm.detect_checkpoint_layout(payload)
        for payload in ({}, [], None):
            with self.subTest(payload=repr(payload)):
                with self.assertRaisesRegex(ValueError, "non-empty mapping"):
                    panderm.detect_checkpoint_layout(payload)

    def test_wrapped_encoder_layout_still_detected_and_accepted(self):
        state = mock_pretrained_state()
        self.assertEqual(
            panderm.detect_checkpoint_layout(state), panderm.LAYOUT_ENCODER_WRAPPED
        )
        model = panderm.build_panderm_classifier(
            7, model_factory=mock_factory, state_dict=state
        )
        self.assertEqual(
            model.pretrained_state_layout, panderm.LAYOUT_ENCODER_WRAPPED
        )

    def test_wrapped_missing_and_unexpected_contract_is_preserved(self):
        missing = mock_pretrained_state()
        missing.pop("encoder.blocks.0.attn.weight")
        with self.assertRaisesRegex(ValueError, "missing backbone parameters"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=missing
            )
        unexpected = mock_pretrained_state()
        unexpected["encoder.mystery.weight"] = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "cannot accept"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=unexpected
            )

    def test_layout_identities_are_the_two_named_values(self):
        self.assertEqual(panderm.LAYOUT_DIRECT_BACKBONE, "direct_backbone_v1")
        self.assertEqual(panderm.LAYOUT_ENCODER_WRAPPED, "encoder_wrapped_v1")

    # --- wrapped layout: an encoder.* is not a licence to accept anything -----
    def test_wrapped_unknown_top_level_component_is_rejected(self):
        """One ``encoder.`` key must not make the rest of the payload acceptable.

        ``load_state_dict`` would never see these keys, so without this check an
        optimizer blob or an unrecognised tower rides along silently.
        """
        for intruder in ("optimizer.state", "mystery.foo", "epoch", "scaler"):
            with self.subTest(intruder=intruder):
                state = mock_pretrained_state()
                state[intruder] = torch.zeros(2)
                with self.assertRaisesRegex(
                    ValueError, "outside the registered pretraining components"
                ):
                    panderm.detect_checkpoint_layout(state)

    def test_wrapped_plus_direct_backbone_key_is_a_mixed_layout(self):
        state = mock_pretrained_state()
        state["cls_token"] = torch.zeros(1, 1, MOCK_DIM)
        with self.assertRaisesRegex(ValueError, "mixes the wrapped 'encoder.' layout"):
            panderm.detect_checkpoint_layout(state)

    def test_wrapped_encoder_decoder_teacher_combination_still_accepted(self):
        state = mock_pretrained_state(include_extras=True)
        self.assertTrue(any(key.startswith("decoder.") for key in state))
        self.assertTrue(any(key.startswith("teacher.") for key in state))
        self.assertEqual(
            panderm.detect_checkpoint_layout(state), panderm.LAYOUT_ENCODER_WRAPPED
        )
        remapped = panderm.remap_pretrained_state_dict(state)
        self.assertFalse(
            any(key.startswith(("decoder.", "teacher.")) for key in remapped)
        )

    def test_model_or_state_dict_wrapper_unwrapping_is_unchanged(self):
        """``load_pretrained_state`` still unwraps, and the payload is then judged."""
        inner = direct_backbone_state()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save({"model": inner}, path)
            unwrapped = panderm.load_pretrained_state(path)
            self.assertEqual(
                panderm.detect_checkpoint_layout(unwrapped),
                panderm.LAYOUT_DIRECT_BACKBONE,
            )

    # --- dtype must match exactly; no silent load_state_dict cast ------------
    def test_direct_dtype_mismatch_is_rejected(self):
        state = direct_backbone_state()
        state["cls_token"] = state["cls_token"].to(torch.float64)
        with self.assertRaisesRegex(ValueError, "tensor dtype mismatch"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_wrapped_dtype_mismatch_is_rejected(self):
        state = mock_pretrained_state()
        state["encoder.blocks.0.attn.weight"] = state[
            "encoder.blocks.0.attn.weight"
        ].to(torch.float64)
        with self.assertRaisesRegex(ValueError, "tensor dtype mismatch"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )

    def test_matching_dtypes_are_accepted_and_stay_float32(self):
        for label, state in (
            ("direct", direct_backbone_state()),
            ("wrapped", mock_pretrained_state()),
        ):
            with self.subTest(layout=label):
                model = panderm.build_panderm_classifier(
                    7, model_factory=mock_factory, state_dict=state
                )
                dtypes = {value.dtype for value in model.state_dict().values()}
                self.assertEqual(dtypes, {torch.float32})

    def test_shape_mismatch_is_rejected(self):
        state = direct_backbone_state()
        state["cls_token"] = torch.zeros(1, 1, MOCK_DIM + 1)
        with self.assertRaisesRegex(ValueError, "tensor shape mismatch"):
            panderm.build_panderm_classifier(
                7, model_factory=mock_factory, state_dict=state
            )


class RetainingFactory:
    """Builds a mock model and keeps it, so post-rejection state is inspectable."""

    def __init__(self):
        self.model = None
        self.snapshot = None

    def __call__(self, **kwargs):
        self.model = MockPanDerm(num_classes=kwargs["num_classes"])
        # Parameters and buffers exactly as the factory produced them.
        self.snapshot = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        return self.model


class RejectionBeforeMutationTests(unittest.TestCase):
    """A rejected checkpoint must not leave a partially loaded model behind.

    ``load_state_dict(..., strict=False)`` copies every acceptable tensor in
    *before* reporting what was missing or unexpected. Validating afterwards
    therefore rejects a checkpoint that has already overwritten most of the
    backbone, which on Colab would be a silently half-initialised model. Each
    case below asserts the raise *and* that every parameter and buffer is
    bit-identical to what the factory built.
    """

    def failing_payloads(self):
        direct_incomplete = direct_backbone_state()
        direct_incomplete.pop("blocks.0.attn.weight")

        direct_unexpected = direct_backbone_state()
        direct_unexpected["blocks.0.mystery.weight"] = torch.zeros(2)

        direct_shape = direct_backbone_state()
        direct_shape["cls_token"] = torch.zeros(1, 1, MOCK_DIM + 1)

        direct_dtype = direct_backbone_state()
        direct_dtype["cls_token"] = direct_dtype["cls_token"].to(torch.float64)

        wrapped_missing = mock_pretrained_state()
        wrapped_missing.pop("encoder.blocks.0.attn.weight")

        wrapped_unexpected = mock_pretrained_state()
        wrapped_unexpected["encoder.mystery.weight"] = torch.zeros(2)

        wrapped_dtype = mock_pretrained_state()
        wrapped_dtype["encoder.blocks.0.mlp.weight"] = wrapped_dtype[
            "encoder.blocks.0.mlp.weight"
        ].to(torch.float64)

        return {
            "direct incomplete": direct_incomplete,
            "direct unexpected": direct_unexpected,
            "direct shape mismatch": direct_shape,
            "direct dtype mismatch": direct_dtype,
            "wrapped missing": wrapped_missing,
            "wrapped unexpected encoder key": wrapped_unexpected,
            "wrapped dtype mismatch": wrapped_dtype,
        }

    def test_every_rejection_leaves_the_model_bit_identical(self):
        for label, state in self.failing_payloads().items():
            with self.subTest(case=label):
                factory = RetainingFactory()
                with self.assertRaises(ValueError):
                    panderm.build_panderm_classifier(
                        7, model_factory=factory, state_dict=state
                    )
                rejected_before_model_mutation = True
                self.assertTrue(rejected_before_model_mutation)

                self.assertIsNotNone(factory.model, "factory never built a model")
                after = factory.model.state_dict()
                self.assertEqual(sorted(after), sorted(factory.snapshot))
                for name, before in factory.snapshot.items():
                    value = after[name]
                    self.assertEqual(value.shape, before.shape, name)
                    self.assertEqual(value.dtype, before.dtype, name)
                    self.assertTrue(torch.equal(value, before), f"{label}: {name} moved")

    def test_a_valid_payload_does_change_the_model(self):
        """Control: the comparison above is only meaningful if loading can move it."""
        factory = RetainingFactory()
        model = panderm.build_panderm_classifier(
            7, model_factory=factory, state_dict=direct_backbone_state()
        )
        after = model.state_dict()
        moved = [
            name
            for name, before in factory.snapshot.items()
            if not name.startswith("head.")
            and not torch.equal(after[name], before)
        ]
        self.assertGreater(len(moved), 0)

    def test_head_is_never_touched_by_a_successful_load(self):
        factory = RetainingFactory()
        model = panderm.build_panderm_classifier(
            7, model_factory=factory, state_dict=direct_backbone_state()
        )
        after = model.state_dict()
        for name, before in factory.snapshot.items():
            if name.startswith("head."):
                self.assertTrue(torch.equal(after[name], before), name)

    def test_validation_pass_alone_never_mutates(self):
        """``plan_pretrained_load`` is read-only even on a payload that would load."""
        factory = RetainingFactory()
        model = factory(num_classes=7)
        remapped = panderm.remap_pretrained_state_dict(direct_backbone_state())
        plan = panderm.plan_pretrained_load(model, remapped)
        self.assertEqual(plan["head_keys"], ["head.bias", "head.weight"])
        self.assertEqual(plan["validated_tensors"], len(remapped))
        after = model.state_dict()
        for name, before in factory.snapshot.items():
            self.assertTrue(torch.equal(after[name], before), name)


# --- real upstream dynamic-import contract -----------------------------------
# Everything above injects ``mock_factory``, so none of it exercises the actual
# ``importlib`` path used on Colab. The fixture below is the smallest module that
# reproduces the *real* upstream contract: upstream ``modeling_finetune`` wraps
# every factory in timm's ``register_model``, which resolves the defining module
# out of ``sys.modules`` at decoration time, i.e. while the module is still being
# executed by ``exec_module``.
TIMM_LIKE_REGISTRY = '''
    import pathlib
    import sys
    import warnings

    # The registry lives in a file, not a module global, because timm's registry
    # lives in the *timm* package and therefore survives re-executing this module.
    # That is exactly what makes a second exec_module overwrite its entries.
    REGISTRY_LOG = pathlib.Path(__REGISTRY_LOG__)
    IMPORT_TIME_OBSERVATIONS = []


    def register_model(fn):
        """Same import-time lookup and overwrite warning as timm 0.9.16.

        The real decorator's first statement is ``mod = sys.modules[fn.__module__]``,
        so a loader that never publishes the module under ``spec.name`` raises
        ``KeyError`` here, long before the factory is ever called. Registering the
        same name twice is what produces timm's "Overwriting ... in registry"
        warning, so re-executing this module is observable.
        """
        module = sys.modules[fn.__module__]
        already = (
            REGISTRY_LOG.read_text(encoding="utf-8").split()
            if REGISTRY_LOG.exists()
            else []
        )
        if fn.__name__ in already:
            warnings.warn(
                "Overwriting %s in registry" % fn.__name__, UserWarning
            )
        with REGISTRY_LOG.open("a", encoding="utf-8") as handle:
            handle.write(fn.__name__ + "\\n")
        IMPORT_TIME_OBSERVATIONS.append(
            {
                "factory": fn.__name__,
                "module_name": fn.__module__,
                "module_file": module.__file__,
            }
        )
        return fn
    '''

FACTORY_NAME = panderm_run.UPSTREAM_MODEL_FACTORY

# Spelled out rather than read from ``panderm`` so these tests keep failing with
# the *observed production* error -- ``KeyError: 'panderm_upstream_modeling_finetune'``
# -- instead of quietly reshaping themselves around whatever the loader happens
# to call the module. The two are pinned to each other below.
UPSTREAM_MODULE_NAME = "panderm_upstream_modeling_finetune"

# Imports cleanly and exposes the pinned factory.
WORKING_UPSTREAM_BODY = f'''
    @register_model
    def {FACTORY_NAME}(pretrained=False, **kwargs):
        return "built", pretrained, tuple(sorted(kwargs))
    '''

# Registers one factory (so the module really was published under spec.name),
# binds a global, then dies part-way through: a half-initialised module.
EXPLODING_UPSTREAM_BODY = f'''
    @register_model
    def {FACTORY_NAME}(pretrained=False, **kwargs):
        return "built", pretrained, tuple(sorted(kwargs))

    HALF_INITIALISED = True
    raise RuntimeError("upstream import exploded")
    '''

# Executes fine but defines a different factory, as an upstream revision that
# dropped or renamed the pinned entry point would.
WRONG_FACTORY_UPSTREAM_BODY = '''
    @register_model
    def panderm_large_patch16_224(pretrained=False, **kwargs):
        return "wrong-factory"
    '''


def registry_log_path(root):
    """Where the fixture records every ``register_model`` call."""
    return Path(root) / "timm_registry_log.txt"


def write_upstream_checkout(root, body):
    """Lay out a temporary ``classification/models/modeling_finetune.py``."""
    module_path = panderm.upstream_module_path(root)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    source = textwrap.dedent(TIMM_LIKE_REGISTRY).replace(
        "__REGISTRY_LOG__", repr(str(registry_log_path(root)))
    ) + textwrap.dedent(body)
    module_path.write_text(source, encoding="utf-8")
    return module_path


def registrations(root):
    """Factory names registered so far; one entry per module execution."""
    log = registry_log_path(root)
    return log.read_text(encoding="utf-8").split() if log.exists() else []


class UpstreamDynamicImportTests(unittest.TestCase):
    """Pin the timm ``register_model`` import-time ``sys.modules`` contract.

    Why this matters rather than "a callable comes back": on Colab the loader
    imports the pinned upstream checkout by path, and upstream decorates its
    factories with timm's ``register_model``. Because that decorator reads
    ``sys.modules[fn.__module__]`` *during* module execution, a loader that calls
    ``exec_module`` on a module it never registered under ``spec.name`` fails with
    ``KeyError: 'panderm_upstream_modeling_finetune'`` before any model is built.
    No injected-mock test can see that, so these tests own the contract.

    Registering the module early is only safe if it is transactional, so the
    rollback behaviour is pinned here too: a failed import must never leave a
    half-initialised module for a later import to pick up.
    """

    def setUp(self):
        self.module_name = UPSTREAM_MODULE_NAME
        # Snapshot/restore so a successful load (which legitimately keeps the
        # module in sys.modules, as timm's registry expects) cannot leak into
        # any other test in the suite.
        self.had_module = self.module_name in sys.modules
        self.previous_module = sys.modules.get(self.module_name)
        self.addCleanup(self._restore_sys_modules)
        sys.modules.pop(self.module_name, None)

    def _restore_sys_modules(self):
        if self.had_module:
            sys.modules[self.module_name] = self.previous_module
        else:
            sys.modules.pop(self.module_name, None)

    def test_loader_publishes_the_name_from_the_observed_production_failure(self):
        self.assertEqual(panderm.UPSTREAM_MODULE_NAME, UPSTREAM_MODULE_NAME)

    def test_fixture_fails_the_way_an_unregistered_dynamic_import_does(self):
        """The fixture is not vacuous: the pre-fix loader body still breaks on it.

        This is the exact ``module_from_spec`` + ``exec_module`` pair the loader
        used before the fix, run against the same file the loader is given below.
        If this ever stops raising, the fixture has drifted away from the real
        timm contract and the tests after it prove nothing.
        """
        with tempfile.TemporaryDirectory() as tmp:
            module_path = write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            spec = importlib.util.spec_from_file_location(
                self.module_name, module_path
            )
            module = importlib.util.module_from_spec(spec)
            with self.assertRaises(KeyError) as caught:
                spec.loader.exec_module(module)
            self.assertEqual(caught.exception.args[0], self.module_name)
            self.assertNotIn(self.module_name, sys.modules)

    def test_factory_is_imported_from_the_module_registered_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_path = write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            factory = panderm.load_upstream_model_factory(tmp)

            self.assertTrue(callable(factory))
            self.assertEqual(factory.__name__, FACTORY_NAME)
            self.assertEqual(factory.__module__, self.module_name)
            self.assertEqual(
                factory(pretrained=False, num_classes=7),
                ("built", False, ("num_classes",)),
            )

            loaded = sys.modules[self.module_name]
            self.assertIs(getattr(loaded, FACTORY_NAME), factory)
            self.assertEqual(Path(loaded.__file__), module_path)

            # Recorded by the timm-like decorator *while* the module executed, so
            # it is evidence about import time, not about the state afterwards.
            self.assertEqual(len(loaded.IMPORT_TIME_OBSERVATIONS), 1)
            observed = loaded.IMPORT_TIME_OBSERVATIONS[0]
            self.assertEqual(observed["factory"], FACTORY_NAME)
            self.assertEqual(observed["module_name"], self.module_name)
            self.assertEqual(Path(observed["module_file"]), module_path)

    def test_first_load_executes_the_module_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            self.assertEqual(registrations(tmp), [])
            panderm.load_upstream_model_factory(tmp)
            self.assertEqual(registrations(tmp), [FACTORY_NAME])

    def test_second_identical_load_reuses_the_module_without_re_executing(self):
        """A second Phase 3 run in the same runtime must not re-register.

        Re-executing would run every ``@register_model`` again over names timm has
        already registered, which is what produced the four overwrite warnings.
        """
        with tempfile.TemporaryDirectory() as tmp:
            write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            first = panderm.load_upstream_model_factory(tmp)
            first_module = sys.modules[self.module_name]
            second = panderm.load_upstream_model_factory(tmp)

            self.assertIs(second, first)
            self.assertIs(sys.modules[self.module_name], first_module)
            # One execution, therefore exactly one registration.
            self.assertEqual(registrations(tmp), [FACTORY_NAME])
            self.assertEqual(len(first_module.IMPORT_TIME_OBSERVATIONS), 1)

    def test_repeated_load_emits_no_registry_overwrite_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_path = write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            panderm.load_upstream_model_factory(tmp)
            with warnings.catch_warnings(record=True) as reused:
                warnings.simplefilter("always")
                panderm.load_upstream_model_factory(tmp)
            self.assertEqual([str(item.message) for item in reused], [])

            # Not vacuous: forcing a genuine re-exec of the same file does warn,
            # which is the behaviour observed in the Colab runtime.
            spec = importlib.util.spec_from_file_location(
                self.module_name, module_path
            )
            reloaded = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = reloaded
            with warnings.catch_warnings(record=True) as re_executed:
                warnings.simplefilter("always")
                spec.loader.exec_module(reloaded)
            self.assertEqual(
                [str(item.message) for item in re_executed],
                [f"Overwriting {FACTORY_NAME} in registry"],
            )
            self.assertEqual(registrations(tmp), [FACTORY_NAME, FACTORY_NAME])

    def test_cached_module_from_a_different_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            write_upstream_checkout(Path(tmp), WORKING_UPSTREAM_BODY)
            impostor_path = write_upstream_checkout(Path(other), WORKING_UPSTREAM_BODY)
            impostor = types.ModuleType(self.module_name)
            impostor.__file__ = str(impostor_path)
            setattr(impostor, FACTORY_NAME, lambda **kwargs: "impostor")
            sys.modules[self.module_name] = impostor

            with self.assertRaisesRegex(ImportError, "not from the pinned upstream"):
                panderm.load_upstream_model_factory(tmp)
            # Refused, never silently reused, and never silently replaced.
            self.assertIs(sys.modules[self.module_name], impostor)
            self.assertEqual(registrations(tmp), [])

    def test_cached_module_with_an_unusable_factory_is_refused(self):
        cases = {
            "factory missing": None,
            "factory not callable": "not-callable",
            "factory from another module": types.SimpleNamespace,
        }
        for label, replacement in cases.items():
            with self.subTest(cached=label), tempfile.TemporaryDirectory() as tmp:
                module_path = write_upstream_checkout(
                    Path(tmp), WORKING_UPSTREAM_BODY
                )
                cached = types.ModuleType(self.module_name)
                cached.__file__ = str(module_path)
                if replacement is not None:
                    setattr(cached, FACTORY_NAME, replacement)
                sys.modules[self.module_name] = cached

                with self.assertRaises(ImportError):
                    panderm.load_upstream_model_factory(tmp)
                self.assertIs(sys.modules[self.module_name], cached)
                sys.modules.pop(self.module_name, None)

    def test_failed_import_rolls_back_when_the_module_was_absent(self):
        cases = {
            "exec_module raises": (EXPLODING_UPSTREAM_BODY, RuntimeError),
            "pinned factory missing": (WRONG_FACTORY_UPSTREAM_BODY, ImportError),
        }
        for label, (body, error) in cases.items():
            with self.subTest(failure=label), tempfile.TemporaryDirectory() as tmp:
                write_upstream_checkout(Path(tmp), body)
                self.assertNotIn(self.module_name, sys.modules)
                with self.assertRaises(error):
                    panderm.load_upstream_model_factory(tmp)
                self.assertNotIn(self.module_name, sys.modules)

    def test_failure_leaves_a_pre_existing_object_exactly_as_it_was(self):
        for label, body in {
            "exec_module raises": EXPLODING_UPSTREAM_BODY,
            "pinned factory missing": WRONG_FACTORY_UPSTREAM_BODY,
        }.items():
            with self.subTest(failure=label), tempfile.TemporaryDirectory() as tmp:
                write_upstream_checkout(Path(tmp), body)
                occupant = types.ModuleType(self.module_name)
                occupant.pre_existing = True
                sys.modules[self.module_name] = occupant
                with self.assertRaises(ImportError):
                    panderm.load_upstream_model_factory(tmp)
                self.assertIs(sys.modules[self.module_name], occupant)
                self.assertTrue(sys.modules[self.module_name].pre_existing)
                sys.modules.pop(self.module_name, None)

    def test_original_exception_is_not_swallowed_or_downgraded(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_upstream_checkout(Path(tmp), EXPLODING_UPSTREAM_BODY)
            with self.assertRaisesRegex(RuntimeError, "upstream import exploded"):
                panderm.load_upstream_model_factory(tmp)


class FullTrainabilityTests(unittest.TestCase):
    def test_every_learnable_parameter_is_trainable_and_pos_embed_is_fixed(self):
        model = build_mock_model()
        total, trainable = panderm.parameter_counts(model)
        self.assertGreater(total, trainable)
        self.assertEqual(
            total - trainable,
            model.pos_embed.numel(),
        )
        self.assertFalse(model.pos_embed.requires_grad)
        self.assertGreater(panderm.assert_full_trainability(model), 0)

    def test_a_frozen_backbone_is_refused(self):
        model = build_mock_model()
        model.blocks[0].attn.weight.requires_grad = False
        with self.assertRaisesRegex(ValueError, "unexpectedly_frozen"):
            panderm.assert_full_trainability(model)

    def test_accidentally_trainable_fixed_pos_embed_is_refused(self):
        model = build_mock_model()
        model.pos_embed.requires_grad = True
        with self.assertRaisesRegex(ValueError, "unexpectedly_trainable"):
            panderm.assert_full_trainability(model)

    def test_backbone_receives_gradients_and_updates(self):
        model = build_mock_model()
        optimizer = panderm.build_optimizer(model, num_layers=MOCK_DEPTH)
        before = panderm.snapshot_parameters(model)
        images = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 3])
        nn.CrossEntropyLoss()(model(images), targets).backward()
        report = panderm.backbone_gradient_report(model)
        self.assertTrue(
            report["all_trainable_backbone_parameters_have_gradient"]
        )
        self.assertTrue(report["backbone_gradients_finite"])
        self.assertTrue(report["fixed_backbone_parameters_without_gradient"])
        self.assertEqual(report["fixed_backbone_parameter_names"], ["pos_embed"])
        self.assertEqual(report["blocks_with_gradient"], list(range(MOCK_DEPTH)))
        self.assertTrue(report["all_backbone_blocks_have_gradient"])
        self.assertTrue(report["head_parameters_have_gradient"])
        self.assertTrue(report["head_gradients_finite"])
        optimizer.step()
        self.assertGreater(panderm.changed_parameter_count(before, model), 0)

    def test_missing_gradient_from_a_learnable_backbone_parameter_is_reported(self):
        model = build_mock_model()
        images = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 3])
        nn.CrossEntropyLoss()(model(images), targets).backward()
        model.blocks[0].attn.weight.grad = None
        report = panderm.backbone_gradient_report(model)
        self.assertFalse(
            report["all_trainable_backbone_parameters_have_gradient"]
        )
        self.assertEqual(
            report["missing_trainable_backbone_gradient_names"],
            ["blocks.0.attn.weight"],
        )

    def test_model_identity_reports_full_finetune(self):
        model = build_mock_model()
        identity = panderm.model_identity(
            model, train_transform="T", eval_transform="E", checkpoint_sha256="a" * 64
        )
        self.assertEqual(identity["arch"], "panderm_base_vit_b16")
        self.assertEqual(identity["freeze_mode"], "full_finetune")
        self.assertFalse(identity["all_parameters_trainable"])
        self.assertTrue(identity["all_learnable_parameters_trainable"])
        self.assertEqual(identity["fixed_parameter_names"], ["pos_embed"])
        self.assertEqual(
            identity["fixed_parameter_identity"],
            {"pos_embed": "upstream_fixed_2d_sincos_detached"},
        )
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
        self.assertNotIn(id(model.pos_embed), seen)

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
        self.assertIsNone(full.pos_embed.grad)
        full_grads = {
            name: parameter.grad.clone()
            for name, parameter in full.named_parameters()
            if parameter.requires_grad
        }

        accumulated = build_mock_model()
        accumulated.load_state_dict(full.state_dict())
        for index in range(4):
            chunk = slice(index * 2, index * 2 + 2)
            loss = criterion(accumulated(images[chunk]), targets[chunk])
            (loss * panderm.accumulation_loss_scale(index, 4, 4)).backward()
        self.assertIsNone(accumulated.pos_embed.grad)
        for name, parameter in accumulated.named_parameters():
            if not parameter.requires_grad:
                continue
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
                [{
                    "epoch": 1,
                    "optimizer_steps": schedule.step_count,
                    "train_loss": 1.0,
                    "val_df_f1": 0.5,
                    "val_macro_f1": 0.3,
                }],
                args, identity,
                write_guard=AllowDurableWriteGuard(),
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
            "scaler_state_dict": {}, "epoch": 1, "global_step": 0,
            "best_val_df_f1": 0.1,
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
                path, model, optimizer, schedule, scaler, 1, 0.5,
                [{"epoch": 1, "optimizer_steps": schedule.step_count}],
                args, identity,
                write_guard=AllowDurableWriteGuard(),
            )
            saved = train_panderm.load_checkpoint_safe(path, map_location="cpu")

        fresh_model, fresh_opt, fresh_sched, fresh_scaler = self._components()
        self.assertNotEqual(fresh_sched.step_count, schedule.step_count)
        start, best, history = train_panderm.restore_checkpoint_state(
            saved,
            fresh_model,
            fresh_opt,
            fresh_sched,
            fresh_scaler,
            write_guard=AllowDurableWriteGuard(),
        )
        self.assertEqual(
            (start, best, history),
            (2, 0.5, [{"epoch": 1, "optimizer_steps": 3}]),
        )
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
                path, model, optimizer, schedule, scaler, 1, 0.5,
                [{"epoch": 1, "optimizer_steps": schedule.step_count}],
                args, identity,
                write_guard=AllowDurableWriteGuard(),
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
    def test_reviewed_checkpoint_hash_is_pinned(self):
        self.assertEqual(
            panderm_run.EXPECTED_CHECKPOINT_SHA256,
            REVIEWED_CHECKPOINT_SHA256,
        )
        self.assertNotEqual(
            panderm_run.EXPECTED_CHECKPOINT_SHA256,
            panderm_run.CHECKPOINT_SHA256_PLACEHOLDER,
        )

    def test_placeholder_hash_blocks_validation(self):
        with self.assertRaisesRegex(ValueError, "not pinned"):
            panderm_run.require_provenance_clearance(
                upstream_commit=panderm_run.UPSTREAM_COMMIT,
                checkpoint_sha256="a" * 64,
                expected_checkpoint_sha256=panderm_run.CHECKPOINT_SHA256_PLACEHOLDER,
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
                panderm_run.require_checkpoint_sha256(
                    path, panderm_run.CHECKPOINT_SHA256_PLACEHOLDER
                )
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


class AggregateResultsTests(unittest.TestCase):
    def _synthetic_run(self, seed, **overrides):
        base_id = {
            "schema_version": 1,
            "git_commit": "c" * 40,
            "run_version": panderm_run.RUN_VERSION,
            "upstream_repo": panderm_run.UPSTREAM_REPO,
            "upstream_commit": panderm_run.UPSTREAM_COMMIT,
            "checkpoint_filename": panderm_run.CHECKPOINT_FILENAME,
            "checkpoint_source_url": panderm_run.CHECKPOINT_SOURCE_URL,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_sha256_provenance": panderm_run.CHECKPOINT_SHA256_PROVENANCE,
            "checkpoint_format": panderm_run.CHECKPOINT_FORMAT,
            "model_identity": {"arch": "panderm_base_vit_b16"},
            "variant": panderm_run.VARIANT,
            "seed": seed,
            "fixed_split_identity": "fixed-split-sha",
            "manifest_sha256": {"train": "t", "val": "v"},
            "c1_construction": {"df_target_count": 585},
            "objective": {"num_classes": 7},
            "optimization": {"learning_rate": 5e-4},
            "dependency_versions": {"torch": "2.2.0"},
            "shared_root_uuid": "shared-uuid",
            "formal_output_identity": "formal-identity",
            "evaluation_scope": panderm_run.VALIDATION_ONLY,
            "license_review": {"license": "CC-BY-NC-ND-4.0"},
            "contamination_review": {"patient_level_overlap": "not_excludable"},
            "claim_boundary": panderm_run.CLAIM_BOUNDARY,
        }
        val_metrics = {
            "target_f1": 0.5 + seed * 0.1,
            "macro_f1": 0.6 + seed * 0.05,
            "target_recall": 0.4 + seed * 0.1,
            "per_class_recall": {
                "akiec": 0.5, "bcc": 0.6, "bkl": 0.7, "df": 0.4 + seed * 0.1,
                "mel": 0.8, "nv": 0.9, "vasc": 0.3,
            },
            "confusion_matrix": [[10] * 7] * 7,
        }
        run = {
            "seed": seed,
            "variant": panderm_run.VARIANT,
            "evaluation_scope": panderm_run.VALIDATION_ONLY,
            "test_metrics": None,
            "claim_boundary": panderm_run.CLAIM_BOUNDARY,
            "validation_metrics": val_metrics,
            "run_identity": base_id,
        }
        for key, val in overrides.items():
            if key in run:
                run[key] = val
            if key in base_id:
                base_id[key] = val
        return run

    def test_aggregate_results_happy_path(self):
        runs = [self._synthetic_run(seed) for seed in (0, 1, 2)]
        agg = panderm_run.aggregate_results(runs)
        self.assertEqual(agg["variant"], panderm_run.VARIANT)
        self.assertEqual(agg["seeds"], [0, 1, 2])
        self.assertEqual(agg["evaluation_scope"], panderm_run.VALIDATION_ONLY)
        self.assertIsNone(agg["test_metrics"])
        self.assertTrue(agg["formal_training_allowed"])
        self.assertFalse(agg["test_access_allowed"])
        self.assertEqual(agg["claim_boundary"], panderm_run.CLAIM_BOUNDARY)

        # Values: [0.5, 0.6, 0.7] -> mean = 0.6
        df_f1 = agg["validation_metrics"]["df_f1"]
        self.assertEqual(df_f1["values"], [0.5, 0.6, 0.7])
        self.assertAlmostEqual(df_f1["mean"], 0.6)
        self.assertAlmostEqual(df_f1["population_std"], 0.08164965809277261)

    def test_aggregate_results_missing_duplicate_or_extra_seeds_rejected(self):
        # Missing seed 2
        with self.assertRaisesRegex(ValueError, "expected exactly the"):
            panderm_run.aggregate_results([self._synthetic_run(0), self._synthetic_run(1)])

        # Duplicate seed 0
        with self.assertRaisesRegex(ValueError, "expected exactly the"):
            panderm_run.aggregate_results([self._synthetic_run(0), self._synthetic_run(0), self._synthetic_run(1)])

        # Extra seed 3
        with self.assertRaisesRegex(ValueError, "expected exactly the"):
            panderm_run.aggregate_results([self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2), self._synthetic_run(3)])

    def test_aggregate_results_wrong_variant_rejected(self):
        runs = [self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2, variant="C2")]
        with self.assertRaisesRegex(ValueError, "refusing to aggregate non-C1 variants"):
            panderm_run.aggregate_results(runs)

    def test_aggregate_results_full_evaluation_scope_rejected(self):
        runs = [self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2, evaluation_scope="full")]
        with self.assertRaisesRegex(ValueError, "PanDerm v1 formal aggregation must stay validation_only"):
            panderm_run.aggregate_results(runs)

    def test_aggregate_results_non_none_test_metrics_rejected(self):
        runs = [self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2, test_metrics={"accuracy": 0.9})]
        with self.assertRaisesRegex(ValueError, "test access is prohibited"):
            panderm_run.aggregate_results(runs)

    def test_aggregate_results_claim_boundary_drift_rejected(self):
        runs = [self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2, claim_boundary="unrestricted")]
        with self.assertRaisesRegex(ValueError, "refusing to aggregate mixed claim boundaries"):
            panderm_run.aggregate_results(runs)

    def test_aggregate_results_identity_drift_matrix_rejects_24_of_24(self):
        non_seed_keys = [key for key in panderm_run.IMMUTABLE_IDENTITY_KEYS if key != "seed"]
        self.assertEqual(len(non_seed_keys), 24)
        rejection_count = 0
        for key in non_seed_keys:
            runs = [self._synthetic_run(0), self._synthetic_run(1), self._synthetic_run(2)]
            # Drift run 2's key in its run_identity
            runs[2]["run_identity"][key] = "DRIFTED_VALUE"
            try:
                panderm_run.aggregate_results(runs)
            except ValueError as error:
                if "drifted identity" in str(error):
                    rejection_count += 1
        self.assertEqual(rejection_count, 24)


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
                path, model, optimizer, schedule, scaler, 1, 0.4,
                [{"epoch": 1, "optimizer_steps": schedule.step_count}],
                args, identity,
                write_guard=AllowDurableWriteGuard(),
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
