"""Heatmap losses.

The target heatmap is ~0.01% positive pixels, so a plain BCE collapses to
"predict zero everywhere". The default is the penalty-reduced focal loss from
CornerNet/CenterNet, which down-weights easy negatives and treats pixels near
the gaussian peak as partial positives.

All losses take **logits** so they can use numerically stable formulations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalHeatmapLoss(nn.Module):
    """Penalty-reduced pixel-wise focal loss (CornerNet, eq. 1).

    Args:
        alpha: focusing exponent on the predicted probability.
        beta: exponent that discounts negatives close to the gaussian peak.
        pos_threshold: target values at or above this count as true positives.
            The gaussian peak is slightly below 1.0 when the ball centre falls
            between pixels, so this is not exactly 1.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, pos_threshold: float = 0.9):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pos_threshold = float(pos_threshold)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(logits.dtype)
        pos_mask = target.ge(self.pos_threshold).to(logits.dtype)
        neg_mask = 1.0 - pos_mask

        log_p = F.logsigmoid(logits)
        log_one_minus_p = F.logsigmoid(-logits)
        p = torch.sigmoid(logits)

        pos_loss = -((1.0 - p).clamp_min(1e-6) ** self.alpha) * log_p * pos_mask
        neg_weight = (1.0 - target).clamp_min(0.0) ** self.beta
        neg_loss = -neg_weight * (p.clamp_min(1e-6) ** self.alpha) * log_one_minus_p * neg_mask

        num_pos = pos_mask.sum()
        total = pos_loss.sum() + neg_loss.sum()
        # Frames with no visible ball contribute only negatives; normalizing by
        # the batch size keeps their gradient comparable to a positive frame's.
        denom = num_pos.clamp_min(1.0) if num_pos > 0 else torch.tensor(
            float(logits.shape[0]), device=logits.device, dtype=logits.dtype
        )
        return total / denom


class WeightedBCELoss(nn.Module):
    """BCE-with-logits with a constant positive-class weight."""

    def __init__(self, pos_weight: float = 100.0):
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), pos_weight=weight)


class BCEHeatmapLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))


class MSEHeatmapLoss(nn.Module):
    """MSE on probabilities; simple and stable but slow to localize."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(torch.sigmoid(logits), target.to(logits.dtype))


LOSS_REGISTRY = {
    "focal": FocalHeatmapLoss,
    "wbce": WeightedBCELoss,
    "bce": BCEHeatmapLoss,
    "mse": MSEHeatmapLoss,
}


def build_loss(cfg) -> nn.Module:
    """Build the loss described by ``cfg.loss``."""
    loss_cfg = cfg.loss
    name = str(loss_cfg.get("name", "focal")).lower()
    if name not in LOSS_REGISTRY:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(LOSS_REGISTRY)}")

    if name == "focal":
        focal = loss_cfg.get("focal", {})
        return FocalHeatmapLoss(
            alpha=float(focal.get("alpha", 2.0)),
            beta=float(focal.get("beta", 4.0)),
            pos_threshold=float(focal.get("pos_threshold", 0.9)),
        )
    if name == "wbce":
        return WeightedBCELoss(pos_weight=float(loss_cfg.get("pos_weight", 100.0)))
    return LOSS_REGISTRY[name]()
