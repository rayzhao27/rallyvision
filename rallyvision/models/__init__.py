"""Model definitions and the config-driven factory."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch.nn as nn

from .tracknet import STRIDE, TrackNet, check_input_size

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "tracknet": TrackNet,
}


def build_model(cfg, override_width: Optional[float] = None) -> nn.Module:
    """Instantiate the model described by ``cfg.model`` and ``cfg.data``."""
    model_cfg = cfg.model
    name = str(model_cfg.get("name", "tracknet")).lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")

    check_input_size(int(cfg.data.input_height), int(cfg.data.input_width))
    return MODEL_REGISTRY[name](
        num_frames=int(cfg.data.num_frames),
        in_channels_per_frame=3,
        out_channels=int(model_cfg.get("out_channels", 1)),
        width=float(override_width if override_width is not None else model_cfg.get("width", 1.0)),
        sigmoid_output=bool(model_cfg.get("sigmoid_output", True)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


__all__ = ["MODEL_REGISTRY", "STRIDE", "TrackNet", "build_model", "check_input_size"]
