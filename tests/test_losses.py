from __future__ import annotations

import pytest
import torch

from rallyvision.data.heatmap import render_gaussian
from rallyvision.losses import FocalHeatmapLoss, WeightedBCELoss, build_loss


def _target(height=32, width=48, x=20.0, y=10.0):
    return torch.from_numpy(render_gaussian(height, width, x=x, y=y, sigma=2.0)).unsqueeze(0).unsqueeze(0)


def test_focal_loss_is_lower_for_a_better_prediction():
    target = _target()
    good = torch.logit(target.clamp(1e-4, 1 - 1e-4))
    bad = torch.full_like(target, -4.0)
    loss = FocalHeatmapLoss()
    assert float(loss(good, target)) < float(loss(bad, target))


def test_focal_loss_is_finite_on_an_empty_target():
    target = torch.zeros(2, 1, 32, 48)
    logits = torch.randn(2, 1, 32, 48)
    value = FocalHeatmapLoss()(logits, target)
    assert torch.isfinite(value)
    assert float(value) > 0


def test_focal_loss_penalises_a_false_positive_on_an_empty_target():
    target = torch.zeros(1, 1, 32, 48)
    quiet = torch.full_like(target, -6.0)
    shouting = torch.full_like(target, 3.0)
    loss = FocalHeatmapLoss()
    assert float(loss(quiet, target)) < float(loss(shouting, target))


def test_focal_loss_is_differentiable():
    target = _target()
    logits = torch.randn_like(target, requires_grad=True)
    FocalHeatmapLoss()(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_focal_loss_handles_extreme_logits_without_nan():
    target = _target()
    for value in (-50.0, 50.0):
        out = FocalHeatmapLoss()(torch.full_like(target, value), target)
        assert torch.isfinite(out)


def test_weighted_bce_weights_positives_more():
    target = _target()
    logits = torch.full_like(target, -4.0)
    assert float(WeightedBCELoss(pos_weight=100.0)(logits, target)) > float(
        WeightedBCELoss(pos_weight=1.0)(logits, target)
    )


@pytest.mark.parametrize("name", ["focal", "bce", "wbce", "mse"])
def test_every_registered_loss_builds_and_runs(base_config, name):
    base_config["loss.name"] = name
    loss_fn = build_loss(base_config)
    target = _target()
    value = loss_fn(torch.randn_like(target), target)
    assert torch.isfinite(value)


def test_unknown_loss_name_is_rejected(base_config):
    base_config["loss.name"] = "nope"
    with pytest.raises(KeyError):
        build_loss(base_config)
