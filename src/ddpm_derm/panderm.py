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

from . import config, panderm_run

ARCH = panderm_run.ARCH

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


# --- upstream model loading --------------------------------------------------
UPSTREAM_MODULE_NAME = "panderm_upstream_modeling_finetune"


def upstream_module_path(upstream_dir: str | Path) -> Path:
    return Path(upstream_dir) / panderm_run.UPSTREAM_MODEL_MODULE


def load_upstream_model_factory(upstream_dir: str | Path) -> Callable[..., nn.Module]:
    """Import ``panderm_base_patch16_224_finetune`` from the pinned checkout.

    ``modeling_finetune`` decorates its factories with ``timm``'s
    ``register_model``, which resolves ``sys.modules[fn.__module__]`` *while the
    module is still executing*, so the module must be published under
    ``spec.name`` before ``exec_module`` rather than after it. Publishing it
    early is only safe if it is also transactional: any failure restores
    ``sys.modules`` exactly as it was, so a half-initialised module can never be
    picked up by a later import.
    """
    module_path = upstream_module_path(upstream_dir)
    if not module_path.is_file():
        raise FileNotFoundError(
            "pinned PanDerm upstream checkout is missing "
            f"{panderm_run.UPSTREAM_MODEL_MODULE}: {module_path}"
        )
    spec = importlib.util.spec_from_file_location(UPSTREAM_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    had_previous = spec.name in sys.modules
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        factory = getattr(module, panderm_run.UPSTREAM_MODEL_FACTORY, None)
        if factory is None:
            raise ImportError(
                f"upstream module has no {panderm_run.UPSTREAM_MODEL_FACTORY}: "
                f"{module_path}"
            )
        if not callable(factory) or getattr(factory, "__module__", None) != spec.name:
            raise ImportError(
                f"{panderm_run.UPSTREAM_MODEL_FACTORY} is not defined by the pinned "
                f"upstream module {module_path}: {factory!r}"
            )
    except BaseException:
        if had_previous:
            sys.modules[spec.name] = previous
        else:
            sys.modules.pop(spec.name, None)
        raise
    return factory


def remap_pretrained_state_dict(raw_state: Mapping[str, Any]) -> dict[str, Any]:
    """Apply upstream's documented pretrain->finetune key mapping.

    Keep ``encoder.*`` and strip the prefix, drop ``decoder.*``/``teacher.*``,
    rename ``norm.`` to ``fc_norm.``, and drop any pretrained classifier head so
    a fresh 7-class head is always trained from scratch.
    """
    encoder_keys = [key for key in raw_state if key.startswith(PRETRAINED_STATE_PREFIX)]
    if not encoder_keys:
        raise ValueError(
            "PanDerm checkpoint has no 'encoder.' parameters; refusing an "
            "unrecognised checkpoint layout"
        )
    remapped: dict[str, Any] = {}
    for key in encoder_keys:
        remapped[key[len(PRETRAINED_STATE_PREFIX):]] = raw_state[key]
    for key in list(remapped):
        if key.startswith(DROPPED_STATE_PREFIXES):
            remapped.pop(key)
    for key in list(remapped):
        if key.startswith("norm."):
            remapped["fc_norm." + key[len("norm."):]] = remapped.pop(key)
    for key in ("head.weight", "head.bias"):
        remapped.pop(key, None)
    return remapped


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
    num_classes: int = config.NUM_CLASSES,
    *,
    checkpoint_path: str | Path | None = None,
    upstream_dir: str | Path | None = None,
    drop_path: float = panderm_run.DROP_PATH,
    model_factory: Callable[..., nn.Module] | None = None,
    state_dict: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build PanDerm-Base with a fresh head and every parameter trainable.

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
    remapped = remap_pretrained_state_dict(state_dict)

    head_keys = {name for name in model.state_dict() if name.startswith("head.")}
    if not head_keys:
        raise ValueError("PanDerm model has no classifier head to replace")
    if any(key in remapped for key in head_keys):
        raise ValueError("pretrained head leaked into the fresh 7-class head")
    incompatible = model.load_state_dict(remapped, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(
            f"PanDerm checkpoint has keys the model cannot accept: {unexpected}"
        )
    missing = sorted(set(incompatible.missing_keys) - head_keys)
    if missing:
        raise ValueError(
            f"PanDerm checkpoint is missing backbone parameters: {missing}"
        )

    head = getattr(model, "head", None)
    if not isinstance(head, nn.Linear) or head.out_features != num_classes:
        raise ValueError(
            f"PanDerm head must be nn.Linear(..., {num_classes}), got {head!r}"
        )
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.arch = ARCH
    model.pretrained_checkpoint = (
        None if checkpoint_path is None else str(checkpoint_path)
    )
    model.freeze_mode = "full_finetune"
    model.input_resolution = (INPUT_RESOLUTION, INPUT_RESOLUTION)
    model.panderm_runtime_drop_path = float(drop_path)
    model.loaded_backbone_keys = sorted(remapped)
    return model


def assert_full_trainability(model: nn.Module) -> int:
    """Refuse to run unless every parameter is trainable (not a linear probe)."""
    frozen = [
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    if frozen:
        raise ValueError(
            "PanDerm full fine-tuning requires requires_grad=True everywhere; "
            f"frozen parameters: {frozen[:10]}"
        )
    backbone = [
        name
        for name, _ in model.named_parameters()
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
    versions: dict[str, Any] = {"torch": torch.__version__}
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
    total, trainable = parameter_counts(model)
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
    }


def backbone_gradient_report(model: nn.Module) -> dict[str, Any]:
    """Evidence that the backbone (not just the head) received gradients."""
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("head.")
    ]
    with_grad = [name for name, parameter in named if parameter.grad is not None]
    finite = [
        name
        for name, parameter in named
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    return {
        "backbone_parameter_count": len(named),
        "backbone_parameters_with_gradient": len(with_grad),
        "backbone_gradients_finite": len(finite) == len(with_grad),
        "all_backbone_parameters_have_gradient": len(with_grad) == len(named),
        "head_parameters_have_gradient": all(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("head.")
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


def require_finite_gradients(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("non-finite gradient encountered; stopping the run")
