"""PanDerm-Base ViT-B/16 backbone, preprocessing and optimisation. Needs torch.

The upstream model definition is *imported from a pinned detached checkout*,
never vendored: only ``classification/models/modeling_finetune.py`` is loaded, by
path, so no third-party code is copied into this repository and no upstream
package ``__init__`` side effects run.

Local tests inject ``model_factory`` / ``state_dict`` and never download the real
400 MB checkpoint; the real weight path is exercised only by the Colab
validation notebook.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
from torchvision import transforms

from . import panderm_run

ARCH = panderm_run.ARCH
NUM_CLASSES = len(panderm_run.EXPECTED_C1_CLASS_COUNTS)

# ViT-B/16, matching upstream ``panderm_base_patch16_224_finetune``.
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
PATCH_SIZE = 16
INPUT_RESOLUTION = panderm_run.INPUT_RESOLUTION

# Upstream fine-tuning defaults kept as-is (see plan section 2.6).
INIT_SCALE = 0.001
USE_MEAN_POOLING = True
USE_REL_POS_BIAS = False
LAYER_SCALE_INIT_VALUE = 0.1
ATTN_DROP_RATE = 0.0
DROP_RATE = 0.0

# Upstream constructs ``pos_embed`` as a fixed 2-D sin/cos table and detaches it
# in ``forward_features``.  It is architectural state, not a learnable weight.
EXPECTED_FIXED_PARAMETER_NAMES = frozenset({"pos_embed"})

# ``--imagenet_default_mean_and_std``; upstream crop_pct is 224/256.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SIZE = 256
TRAIN_AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
TRAIN_COLOR_JITTER = 0.4
TRAIN_INTERPOLATION = "bicubic"
TRAIN_RE_PROB = 0.25
TRAIN_RE_MODE = "pixel"
TRAIN_RE_COUNT = 1

PRETRAINED_STATE_PREFIX = "encoder."
DROPPED_STATE_PREFIXES = ("decoder.", "teacher.")
# The only top-level components a wrapped pretraining checkpoint may contain:
# the encoder we load, plus the two companions we knowingly ignore. Anything
# else (optimizer state, metadata, an unrecognised tower) is refused rather than
# quietly skipped, so "it had an encoder." is never sufficient on its own.
WRAPPED_ALLOWED_PREFIXES = (PRETRAINED_STATE_PREFIX,) + DROPPED_STATE_PREFIXES

# Supported pretrained checkpoint layouts. They are named (rather than implied)
# so "which layout did we actually load" is printable and testable.
LAYOUT_ENCODER_WRAPPED = "encoder_wrapped_v1"
LAYOUT_DIRECT_BACKBONE = "direct_backbone_v1"

# The published PanDerm-Base weights are a plain backbone state dict: no wrapper,
# no ``encoder.`` prefix and no classifier head. Only these keys may appear; a
# ``head.`` entry is tolerated so it can be dropped, never loaded.
DIRECT_ALLOWED_EXACT = frozenset({"cls_token", "pos_embed"})
DIRECT_ALLOWED_PREFIXES = ("patch_embed.", "blocks.", "norm.", "head.")
# Sentinels: enough of the ViT to prove this really is a PanDerm backbone, not
# some other OrderedDict that merely happens to hold tensors.
DIRECT_SENTINEL_KEYS = (
    "cls_token",
    "pos_embed",
    "patch_embed.proj.weight",
    "patch_embed.proj.bias",
    "norm.weight",
    "norm.bias",
)
DIRECT_SENTINEL_PREFIX = "blocks.0."


# --- upstream model loading --------------------------------------------------
UPSTREAM_MODULE_NAME = "panderm_upstream_modeling_finetune"


def upstream_module_path(upstream_dir: str | Path) -> Path:
    return Path(upstream_dir) / panderm_run.UPSTREAM_MODEL_MODULE


def _upstream_factory_from_module(
    module: Any, module_path: Path
) -> Callable[..., nn.Module]:
    """Return the pinned factory from ``module``, or refuse the module.

    Used both for a freshly executed module and for one already in
    ``sys.modules``, so a reused module is held to exactly the same standard as a
    newly imported one.
    """
    origin = getattr(module, "__file__", None)
    if origin is None or Path(origin).resolve() != Path(module_path).resolve():
        raise ImportError(
            f"module '{UPSTREAM_MODULE_NAME}' is already loaded from {origin!r}, "
            f"not from the pinned upstream checkout {module_path}; refusing to "
            "reuse it"
        )
    factory = getattr(module, panderm_run.UPSTREAM_MODEL_FACTORY, None)
    if factory is None:
        raise ImportError(
            f"upstream module has no {panderm_run.UPSTREAM_MODEL_FACTORY}: "
            f"{module_path}"
        )
    if (
        not callable(factory)
        or getattr(factory, "__module__", None) != UPSTREAM_MODULE_NAME
    ):
        raise ImportError(
            f"{panderm_run.UPSTREAM_MODEL_FACTORY} is not defined by the pinned "
            f"upstream module {module_path}: {factory!r}"
        )
    return factory


def load_upstream_model_factory(upstream_dir: str | Path) -> Callable[..., nn.Module]:
    """Import ``panderm_base_patch16_224_finetune`` from the pinned checkout.

    ``modeling_finetune`` decorates its factories with ``timm``'s
    ``register_model``, which resolves ``sys.modules[fn.__module__]`` *while the
    module is still executing*, so the module must be published under
    ``spec.name`` before ``exec_module`` rather than after it. Publishing it
    early is only safe if it is also transactional: any failure restores
    ``sys.modules`` exactly as it was, so a half-initialised module can never be
    picked up by a later import.

    The import is also idempotent. Re-executing the module would re-run every
    ``@register_model`` over names timm has already registered, which overwrites
    its registry and warns once per factory, so an already-loaded module is
    reused instead -- but only after proving it is this exact pinned file with a
    usable factory. A same-named module from anywhere else is refused loudly
    rather than silently reused.
    """
    module_path = upstream_module_path(upstream_dir)
    if not module_path.is_file():
        raise FileNotFoundError(
            "pinned PanDerm upstream checkout is missing "
            f"{panderm_run.UPSTREAM_MODEL_MODULE}: {module_path}"
        )
    if UPSTREAM_MODULE_NAME in sys.modules:
        return _upstream_factory_from_module(
            sys.modules[UPSTREAM_MODULE_NAME], module_path
        )
    spec = importlib.util.spec_from_file_location(UPSTREAM_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        factory = _upstream_factory_from_module(module, module_path)
    except BaseException:
        # The name was absent above, so removing it restores the exact prior state.
        sys.modules.pop(spec.name, None)
        raise
    return factory


def _is_direct_backbone_key(key: Any) -> bool:
    return isinstance(key, str) and (
        key in DIRECT_ALLOWED_EXACT or key.startswith(DIRECT_ALLOWED_PREFIXES)
    )


def detect_checkpoint_layout(raw_state: Mapping[str, Any]) -> str:
    """Name the checkpoint layout, or refuse it.

    Being a mapping of tensors is *not* enough to be accepted as PanDerm weights:
    a layout this function cannot name is always an error, never a best-effort
    partial load.
    """
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(
            "PanDerm checkpoint payload must be a non-empty mapping, got "
            f"{type(raw_state)!r}"
        )
    encoder_keys = [
        key
        for key in raw_state
        if isinstance(key, str) and key.startswith(PRETRAINED_STATE_PREFIX)
    ]
    direct_keys = [key for key in raw_state if _is_direct_backbone_key(key)]
    if encoder_keys and direct_keys:
        raise ValueError(
            "PanDerm checkpoint mixes the wrapped 'encoder.' layout with a direct "
            "backbone layout; refusing an ambiguous checkpoint: "
            f"encoder={sorted(encoder_keys)[:3]} direct={sorted(direct_keys)[:3]}"
        )
    if encoder_keys:
        unknown = sorted(
            str(key)
            for key in raw_state
            if not (isinstance(key, str) and key.startswith(WRAPPED_ALLOWED_PREFIXES))
        )
        if unknown:
            raise ValueError(
                "wrapped PanDerm checkpoint has top-level keys outside the "
                "registered pretraining components "
                f"{list(WRAPPED_ALLOWED_PREFIXES)}: {unknown[:10]}"
            )
        return LAYOUT_ENCODER_WRAPPED
    if direct_keys:
        unknown = sorted(
            str(key) for key in raw_state if not _is_direct_backbone_key(key)
        )
        if unknown:
            raise ValueError(
                "direct PanDerm backbone checkpoint has keys outside the backbone "
                f"allowlist: {unknown[:10]}"
            )
        missing = [key for key in DIRECT_SENTINEL_KEYS if key not in raw_state]
        if not any(
            key.startswith(DIRECT_SENTINEL_PREFIX) for key in raw_state
        ):
            missing.append(DIRECT_SENTINEL_PREFIX + "*")
        if missing:
            raise ValueError(
                "direct PanDerm backbone checkpoint is missing required backbone "
                f"sentinels: {missing}"
            )
        return LAYOUT_DIRECT_BACKBONE
    raise ValueError(
        "unrecognised PanDerm checkpoint layout: no 'encoder.' parameters and no "
        f"direct backbone keys; first keys: {sorted(str(key) for key in raw_state)[:10]}"
    )


def _require_tensor_values(state: Mapping[str, Any], layout: str) -> None:
    non_tensor = sorted(
        str(key)
        for key, value in state.items()
        if not isinstance(value, torch.Tensor)
    )
    if non_tensor:
        raise ValueError(
            f"{layout} PanDerm checkpoint has non-tensor values: {non_tensor[:10]}"
        )


def remap_pretrained_state_dict(
    raw_state: Mapping[str, Any], *, layout: str | None = None
) -> dict[str, Any]:
    """Map a published checkpoint onto the fine-tuning model's parameter names.

    Two layouts are supported and they converge on the same contract: ``norm.``
    becomes ``fc_norm.`` and any pretrained classifier head is dropped so the
    7-class head is always trained from scratch.

    ``encoder_wrapped_v1`` keeps ``encoder.*`` and strips the prefix, ignoring the
    ``decoder.*``/``teacher.*`` companions. ``direct_backbone_v1`` is the layout
    the official ``panderm_bb_data6_checkpoint-499.pth`` actually ships: a plain
    backbone state dict with no wrapper and no head.
    """
    layout = detect_checkpoint_layout(raw_state) if layout is None else layout
    if layout == LAYOUT_ENCODER_WRAPPED:
        selected = {
            key[len(PRETRAINED_STATE_PREFIX):]: value
            for key, value in raw_state.items()
            if isinstance(key, str) and key.startswith(PRETRAINED_STATE_PREFIX)
        }
        for key in list(selected):
            if key.startswith(DROPPED_STATE_PREFIXES):
                selected.pop(key)
    elif layout == LAYOUT_DIRECT_BACKBONE:
        selected = dict(raw_state)
    else:
        raise ValueError(f"unsupported PanDerm checkpoint layout: {layout!r}")

    _require_tensor_values(selected, layout)
    for key in list(selected):
        if key.startswith("norm."):
            selected["fc_norm." + key[len("norm."):]] = selected.pop(key)
    for key in ("head.weight", "head.bias"):
        selected.pop(key, None)
    if not selected:
        raise ValueError(
            f"{layout} PanDerm checkpoint produced no backbone tensors"
        )
    return selected


def plan_pretrained_load(
    model: nn.Module, remapped: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a remapped payload against a *fresh* model, mutating nothing.

    This is deliberately a read-only pass: it never calls ``load_state_dict`` and
    never writes a parameter or buffer, so a checkpoint that fails any check
    leaves the model exactly as the factory built it. Doing the checks *after* a
    ``strict=False`` load would already have copied the acceptable tensors in
    before raising, leaving a half-loaded model behind.

    Shapes and dtypes are compared exactly. ``load_state_dict`` silently casts a
    mismatched dtype via ``copy_``, which would change the numerics of the very
    checkpoint we are supposed to be reproducing, so a float64 tensor is refused
    rather than quietly narrowed to the model's float32.
    """
    reference = model.state_dict()
    expected = set(reference)
    head_keys = {name for name in expected if name.startswith("head.")}
    if not head_keys:
        raise ValueError("PanDerm model has no classifier head to replace")

    leaked = sorted(key for key in head_keys if key in remapped)
    if leaked:
        raise ValueError(
            f"pretrained head leaked into the fresh classifier head: {leaked}"
        )

    unexpected = sorted(set(remapped) - expected)
    if unexpected:
        raise ValueError(
            f"PanDerm checkpoint has keys the model cannot accept: {unexpected}"
        )

    missing = sorted(expected - head_keys - set(remapped))
    if missing:
        raise ValueError(
            f"PanDerm checkpoint is missing backbone parameters: {missing}"
        )

    shape_mismatch: list[str] = []
    dtype_mismatch: list[str] = []
    for key in sorted(remapped):
        target = reference[key]
        value = remapped[key]
        if tuple(value.shape) != tuple(target.shape):
            shape_mismatch.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(target.shape)}"
            )
        elif value.dtype != target.dtype:
            dtype_mismatch.append(
                f"{key}: checkpoint {value.dtype} != model {target.dtype}"
            )
    if shape_mismatch:
        raise ValueError(
            f"PanDerm checkpoint tensor shape mismatch: {shape_mismatch[:10]}"
        )
    if dtype_mismatch:
        raise ValueError(
            f"PanDerm checkpoint tensor dtype mismatch: {dtype_mismatch[:10]}"
        )
    return {
        "head_keys": sorted(head_keys),
        "expected_non_head_keys": len(expected - head_keys),
        "validated_tensors": len(remapped),
    }


def load_pretrained_state(path: str | Path) -> dict[str, Any]:
    """Read the official checkpoint. Only tensors are unpickled."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model", "module", "state_dict"):
        if isinstance(raw, dict) and key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    if not isinstance(raw, dict):
        raise ValueError(f"unexpected PanDerm checkpoint payload type: {type(raw)!r}")
    return raw


def build_panderm_classifier(
    num_classes: int = NUM_CLASSES,
    *,
    checkpoint_path: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    drop_path: float = panderm_run.DROP_PATH,
    model_factory: Callable[..., nn.Module] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build PanDerm-Base with a fresh head and all learnable weights trainable.

    ``model_factory``/``state_dict`` exist so local tests can inject a mock and
    never touch the network or the real weights.
    """
    if float(drop_path) != panderm_run.DROP_PATH:
        raise ValueError(
            f"PanDerm drop_path must be exactly {panderm_run.DROP_PATH}, "
            f"got {drop_path}"
        )
    if model_factory is None:
        if upstream_dir is None:
            raise ValueError(
                "build_panderm_classifier needs upstream_dir (pinned checkout) or "
                "an injected model_factory"
            )
        model_factory = load_upstream_model_factory(upstream_dir)
    model = model_factory(
        pretrained=False,
        num_classes=num_classes,
        drop_rate=DROP_RATE,
        drop_path_rate=drop_path,
        attn_drop_rate=ATTN_DROP_RATE,
        drop_block_rate=None,
        use_mean_pooling=USE_MEAN_POOLING,
        init_scale=INIT_SCALE,
        use_rel_pos_bias=USE_REL_POS_BIAS,
        init_values=LAYER_SCALE_INIT_VALUE,
        lin_probe=False,
    )
    if state_dict is None:
        if checkpoint_path is None:
            raise ValueError(
                "build_panderm_classifier needs checkpoint_path or an injected "
                "state_dict; refusing to fine-tune randomly initialised weights"
            )
        state_dict = load_pretrained_state(checkpoint_path)
    state_layout = detect_checkpoint_layout(state_dict)
    remapped = remap_pretrained_state_dict(state_dict, layout=state_layout)

    # Stage 1: read-only validation. Nothing below this point has touched the
    # model, so any rejection leaves it exactly as the factory built it.
    load_plan = plan_pretrained_load(model, remapped)
    head_keys = set(load_plan["head_keys"])

    # Stage 2: the payload is fully validated, so this is the single mutation.
    incompatible = model.load_state_dict(remapped, strict=False)
    residual_unexpected = sorted(incompatible.unexpected_keys)
    residual_missing = sorted(set(incompatible.missing_keys) - head_keys)
    if residual_unexpected or residual_missing:
        raise ValueError(
            "PanDerm load contract diverged from validation: "
            f"unexpected={residual_unexpected} missing={residual_missing}"
        )

    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear) or head.out_features != num_classes:
        raise ValueError(
            f"PanDerm head must be nn.Linear(..., {num_classes}), got {head!r}"
        )
    model_parameter_names = {name for name, _ in model.named_parameters()}
    missing_fixed = sorted(EXPECTED_FIXED_PARAMETER_NAMES - model_parameter_names)
    if missing_fixed:
        raise ValueError(
            "PanDerm is missing its fixed sin/cos positional parameter: "
            f"{missing_fixed}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name not in EXPECTED_FIXED_PARAMETER_NAMES

    model.arch = ARCH
    model.pretrained_checkpoint = (
        None if checkpoint_path is None else str(checkpoint_path)
    )
    model.freeze_mode = "full_finetune"
    model.input_resolution = (INPUT_RESOLUTION, INPUT_RESOLUTION)
    model.panderm_runtime_drop_path = float(drop_path)
    model.pretrained_state_layout = state_layout
    model.loaded_backbone_keys = sorted(remapped)
    return model


def assert_full_trainability(model: nn.Module) -> int:
    """Require full fine-tuning except upstream's fixed sin/cos position table."""
    named = dict(model.named_parameters())
    missing_fixed = sorted(EXPECTED_FIXED_PARAMETER_NAMES - set(named))
    if missing_fixed:
        raise ValueError(
            "PanDerm is missing its fixed sin/cos positional parameter: "
            f"{missing_fixed}"
        )
    fixed = {name for name, parameter in named.items() if not parameter.requires_grad}
    if fixed != EXPECTED_FIXED_PARAMETER_NAMES:
        unexpectedly_frozen = sorted(fixed - EXPECTED_FIXED_PARAMETER_NAMES)
        unexpectedly_trainable = sorted(EXPECTED_FIXED_PARAMETER_NAMES - fixed)
        raise ValueError(
            "PanDerm full fine-tuning requires every learnable parameter to have "
            "requires_grad=True and only the fixed sin/cos position table frozen: "
            f"unexpectedly_frozen={unexpectedly_frozen[:10]} "
            f"unexpectedly_trainable={unexpectedly_trainable}"
        )
    backbone = [
        name
        for name in named
        if not name.startswith("head.")
    ]
    if not backbone:
        raise ValueError("PanDerm model exposes no backbone parameters")
    return len(backbone)


# --- preprocessing -----------------------------------------------------------
def build_eval_transform():
    """Upstream eval pipeline: Resize(256, bicubic) -> CenterCrop(224) -> norm."""
    return transforms.Compose(
        [
            transforms.Resize(
                RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(INPUT_RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(create_transform=None):
    """Upstream training pipeline via ``timm.data.create_transform``.

    timm is a hard upstream dependency anyway (``modeling_finetune`` imports it),
    so this uses upstream's own transform rather than an approximation.
    """
    if create_transform is None:
        try:
            from timm.data import create_transform
        except ImportError as exc:  # pragma: no cover - exercised on Colab
            raise ImportError(
                f"{ARCH} requires timm==0.9.16 for upstream preprocessing"
            ) from exc
    return create_transform(
        input_size=INPUT_RESOLUTION,
        is_training=True,
        color_jitter=TRAIN_COLOR_JITTER,
        auto_augment=TRAIN_AUTO_AUGMENT,
        interpolation=TRAIN_INTERPOLATION,
        re_prob=TRAIN_RE_PROB,
        re_mode=TRAIN_RE_MODE,
        re_count=TRAIN_RE_COUNT,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def preprocessing_identity(train_transform, eval_transform) -> dict[str, str]:
    return {"train": repr(train_transform), "eval": repr(eval_transform)}


# --- layer-wise learning-rate decay -----------------------------------------
def layer_id_for_parameter(name: str, num_max_layer: int) -> int:
    """Upstream BEiT rule, reimplemented (see plan section 2.4)."""
    if name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    if name.startswith("patch_embed"):
        return 0
    if name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    if name.startswith("blocks"):
        return int(name.split(".")[1]) + 1
    return num_max_layer - 1


def layer_decay_scales(
    num_layers: int = DEPTH, layer_decay: float = panderm_run.LAYER_DECAY
) -> list[float]:
    """``scale[i] = layer_decay ** (num_layers + 1 - i)`` for ``num_layers + 2`` ids."""
    return [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]


def _skip_weight_decay(model: nn.Module) -> set[str]:
    skip = getattr(model, "no_weight_decay", None)
    return set(skip()) if callable(skip) else set()


def build_param_groups(
    model: nn.Module,
    *,
    weight_decay: float = panderm_run.WEIGHT_DECAY,
    num_layers: int = DEPTH,
    layer_decay: float = panderm_run.LAYER_DECAY,
) -> list[dict[str, Any]]:
    """Group trainable parameters by (layer id, decay/no-decay) with lr scales.

    Every trainable parameter lands in exactly one group; 1-D parameters, biases
    and ``no_weight_decay()`` names get weight decay 0.
    """
    scales = layer_decay_scales(num_layers, layer_decay)
    skip = _skip_weight_decay(model)
    groups: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias") or name in skip:
            decay_name, decay_value = "no_decay", 0.0
        else:
            decay_name, decay_value = "decay", float(weight_decay)
        layer_id = layer_id_for_parameter(name, len(scales))
        key = f"layer_{layer_id}_{decay_name}"
        if key not in groups:
            groups[key] = {
                "group_name": key,
                "layer_id": layer_id,
                "params": [],
                "param_names": [],
                "weight_decay": decay_value,
                "lr_scale": scales[layer_id],
            }
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    if not groups:
        raise ValueError("PanDerm model has no trainable parameters")
    return [groups[key] for key in sorted(groups, key=lambda item: (groups[item]["layer_id"], item))]


def verify_optimizer_covers_parameters_once(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> int:
    """Require every trainable parameter to appear in the optimizer exactly once."""
    seen: list[int] = []
    for group in optimizer.param_groups:
        seen.extend(id(parameter) for parameter in group["params"])
        if "lr_scale" not in group:
            raise ValueError(f"optimizer group without lr_scale: {group.get('group_name')}")
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    duplicates = sorted({key for key in seen if seen.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} trainable parameter(s) appear in more than one "
            "optimizer group"
        )
    missing = expected - set(seen)
    extra = set(seen) - expected
    if missing or extra:
        raise ValueError(
            f"optimizer parameter coverage mismatch: {len(missing)} missing, "
            f"{len(extra)} unexpected"
        )
    return len(expected)


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = panderm_run.LEARNING_RATE,
    weight_decay: float = panderm_run.WEIGHT_DECAY,
    num_layers: int = DEPTH,
    layer_decay: float = panderm_run.LAYER_DECAY,
) -> torch.optim.AdamW:
    groups = build_param_groups(
        model,
        weight_decay=weight_decay,
        num_layers=num_layers,
        layer_decay=layer_decay,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": group["params"],
                "weight_decay": group["weight_decay"],
                "lr": learning_rate * group["lr_scale"],
                "lr_scale": group["lr_scale"],
                "group_name": group["group_name"],
                "layer_id": group["layer_id"],
            }
            for group in groups
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    verify_optimizer_covers_parameters_once(optimizer, model)
    return optimizer


# --- warm-up + cosine schedule, stepped per optimizer step -------------------
def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Optimizer steps in one epoch. Trailing partial accumulation is dropped."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    steps = num_batches // accumulation_steps
    if steps < 1:
        raise ValueError(
            f"{num_batches} batches at accumulation {accumulation_steps} yields no "
            "optimizer step"
        )
    return steps


class WarmupCosineSchedule:
    """Linear warm-up then cosine decay, advanced once per *optimizer* step.

    Stepping per micro-batch would silently compress the schedule by the
    accumulation factor, so ``step()`` is only ever called after an
    ``optimizer.step()``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float = panderm_run.LEARNING_RATE,
        min_lr: float = panderm_run.MIN_LR,
        warmup_epochs: int = panderm_run.WARMUP_EPOCHS,
        epochs: int = panderm_run.FORMAL_EPOCHS,
        steps_per_epoch: int,
    ):
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        if not 0 <= warmup_epochs <= epochs:
            raise ValueError(
                f"warmup_epochs {warmup_epochs} must be within 0..{epochs}"
            )
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.warmup_steps = self.warmup_epochs * self.steps_per_epoch
        self.total_steps = self.epochs * self.steps_per_epoch
        self.step_count = 0
        self.apply()

    def lr_at(self, step_index: int) -> float:
        if step_index < self.warmup_steps:
            return self.base_lr * (step_index + 1) / max(self.warmup_steps, 1)
        progress = (step_index - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def apply(self) -> float:
        lr = self.lr_at(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        return lr

    def step(self) -> float:
        self.step_count += 1
        return self.apply()

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "warmup_epochs": self.warmup_epochs,
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in ("base_lr", "min_lr", "warmup_epochs", "epochs", "steps_per_epoch"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(
                    f"scheduler {key} mismatch on resume: saved={state[key]!r} "
                    f"current={getattr(self, key)!r}"
                )
        self.step_count = int(state["step_count"])
        self.apply()


def scheduler_identity(schedule: WarmupCosineSchedule) -> dict[str, Any]:
    return {
        "name": "warmup_cosine",
        "step_unit": "optimizer_step",
        "base_lr": schedule.base_lr,
        "min_lr": schedule.min_lr,
        "warmup_epochs": schedule.warmup_epochs,
        "epochs": schedule.epochs,
        "steps_per_epoch": schedule.steps_per_epoch,
        "warmup_steps": schedule.warmup_steps,
        "total_steps": schedule.total_steps,
    }


# --- identity ----------------------------------------------------------------
def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def dependency_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"torch": str(torch.__version__)}
    for package in ("timm", "torchvision"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def model_identity(
    model: nn.Module,
    *,
    train_transform=None,
    eval_transform=None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    assert_full_trainability(model)
    total, trainable = parameter_counts(model)
    fixed_parameter_names = sorted(
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    return {
        "arch": ARCH,
        "model_name": panderm_run.UPSTREAM_MODEL_FACTORY,
        "upstream_selector": panderm_run.UPSTREAM_MODEL_SELECTOR,
        "pretrained_tag": panderm_run.CHECKPOINT_FILENAME,
        "checkpoint_sha256": checkpoint_sha256,
        "freeze_mode": "full_finetune",
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "use_mean_pooling": USE_MEAN_POOLING,
        "init_scale": INIT_SCALE,
        "layer_scale_init_value": LAYER_SCALE_INIT_VALUE,
        "use_rel_pos_bias": USE_REL_POS_BIAS,
        "drop_path": float(model.panderm_runtime_drop_path),
        "input_resolution": [INPUT_RESOLUTION, INPUT_RESOLUTION],
        "preprocessing_identity": preprocessing_identity(
            train_transform, eval_transform
        ),
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "total_parameter_count": total,
        "trainable_parameter_count": trainable,
        "all_parameters_trainable": total == trainable,
        "all_learnable_parameters_trainable": True,
        "fixed_parameter_names": fixed_parameter_names,
        "fixed_parameter_identity": {
            "pos_embed": "upstream_fixed_2d_sincos_detached",
        },
    }


def backbone_gradient_report(model: nn.Module) -> dict[str, Any]:
    """Evidence that the backbone (not just the head) received gradients."""
    assert_full_trainability(model)
    backbone = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    ]
    trainable = [
        (name, parameter) for name, parameter in backbone
        if parameter.requires_grad
    ]
    fixed = [
        (name, parameter) for name, parameter in backbone
        if not parameter.requires_grad
    ]
    with_grad = [
        name for name, parameter in trainable if parameter.grad is not None
    ]
    missing_grad = [
        name for name, parameter in trainable if parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    blocks_with_gradient = sorted({
        int(name.split(".")[1])
        for name in with_grad
        if name.startswith("blocks.")
        and len(name.split(".")) > 1
        and name.split(".")[1].isdigit()
    })
    expected_blocks = list(range(len(getattr(model, "blocks", ()))))
    head = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("head.") and parameter.requires_grad
    ]
    return {
        "backbone_parameter_count": len(backbone),
        "trainable_backbone_parameter_count": len(trainable),
        "fixed_backbone_parameter_names": [name for name, _ in fixed],
        "backbone_parameters_with_gradient": len(with_grad),
        "missing_trainable_backbone_gradient_names": missing_grad,
        "nonfinite_backbone_gradient_names": nonfinite,
        "backbone_gradients_finite": not nonfinite,
        "all_trainable_backbone_parameters_have_gradient": not missing_grad,
        "fixed_backbone_parameters_without_gradient": all(
            parameter.grad is None for _, parameter in fixed
        ),
        "blocks_with_gradient": blocks_with_gradient,
        "all_backbone_blocks_have_gradient": blocks_with_gradient == expected_blocks,
        "head_parameters_have_gradient": bool(head) and all(
            parameter.grad is not None for _, parameter in head
        ),
        "head_gradients_finite": bool(head) and all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for _, parameter in head
        ),
    }


def changed_parameter_count(
    before: Mapping[str, torch.Tensor], model: nn.Module
) -> int:
    """How many backbone tensors actually moved after an optimizer step."""
    changed = 0
    for name, parameter in model.named_parameters():
        if name.startswith("head.") or name not in before:
            continue
        if not torch.equal(before[name], parameter.detach().cpu()):
            changed += 1
    return changed


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def accumulation_loss_scale(
    micro_batch_index: int, accumulation_steps: int, batches_in_epoch: int
) -> float:
    """Scale factor applied to each micro-batch loss before ``backward()``.

    Always ``1 / accumulation_steps`` for micro-batches inside a complete
    accumulation window; trailing micro-batches that cannot complete a window
    are dropped (scale 0) so a short tail never applies an oversized update.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    complete = (batches_in_epoch // accumulation_steps) * accumulation_steps
    if micro_batch_index >= complete:
        return 0.0
    return 1.0 / accumulation_steps


def require_finite(value: float, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite {label}: {numeric}")
    return numeric


def finite_gradients(parameters: Iterable[nn.Parameter]) -> bool:
    """True when every populated gradient is finite.

    Callers that are inside an AMP ``GradScaler`` loop want this predicate, not
    the raising form: an fp16 overflow there is an expected, self-correcting
    event the scaler handles by skipping the step and backing the scale off.
    Callers outside one -- a smoke test, or any unscaled step -- want the
    raising form, because for them a non-finite gradient is a real defect.
    """
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def require_finite_gradients(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("non-finite gradient encountered; stopping the run")
