"""Optimizer, scheduler and AMP scaler construction."""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(cfg, model: torch.nn.Module) -> Optimizer:
    optim_cfg = cfg.optim
    name = str(optim_cfg.get("name", "adamw")).lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    # BatchNorm and bias parameters are excluded from weight decay; decaying
    # them measurably hurts small-object heatmap models.
    decay, no_decay = [], []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 or param_name.endswith(".bias") else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(optim_cfg.get("momentum", 0.9)), nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    raise KeyError(f"unknown optimizer {name!r}; expected adam, adamw or sgd")


def build_scheduler(cfg, optimizer: Optimizer, steps_per_epoch: int = 1) -> Optional[Any]:
    """Build an epoch-stepped scheduler. Returns None when disabled.

    Warmup is folded into the cosine/step schedules via a LambdaLR so there is
    only ever one scheduler object to checkpoint.
    """
    sched_cfg = cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "none")).lower()
    if name in ("none", "off", ""):
        return None

    epochs = int(cfg.train.epochs)
    warmup = max(0, int(sched_cfg.get("warmup_epochs", 0)))
    base_lr = float(cfg.optim.get("lr", 1e-3))
    min_lr = float(sched_cfg.get("min_lr", 0.0))
    min_factor = (min_lr / base_lr) if base_lr > 0 else 0.0

    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg.train.get("monitor_mode", "max")),
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 3)),
            min_lr=min_lr,
        )

    if name == "cosine":

        def cosine(epoch: int) -> float:
            if warmup and epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = (epoch - warmup) / max(1, epochs - warmup)
            progress = min(1.0, max(0.0, progress))
            return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, cosine)

    if name == "step":
        step_size = max(1, int(sched_cfg.get("step_size", 10)))
        gamma = float(sched_cfg.get("gamma", 0.1))

        def stepwise(epoch: int) -> float:
            if warmup and epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            return max(min_factor, gamma ** ((epoch - warmup) // step_size))

        return LambdaLR(optimizer, stepwise)

    raise KeyError(f"unknown scheduler {name!r}; expected none, cosine, step or plateau")


def is_plateau(scheduler: Any) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def build_scaler(device_info) -> Any:
    """GradScaler, enabled only for CUDA float16 (bfloat16 does not need it)."""
    enabled = bool(device_info.amp_enabled and device_info.amp_dtype == torch.float16)
    try:  # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def current_lr(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
