from __future__ import annotations

import pytest
import torch

from rallyvision.ball_tracking.model import build_model, check_input_size
from rallyvision.common.tracknet import STRIDE, TrackNet


def test_output_matches_input_resolution():
    model = TrackNet(num_frames=3, width=0.125).eval()
    x = torch.rand(2, 9, 72, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 72, 128)


def test_forward_returns_probabilities_when_sigmoid_enabled():
    model = TrackNet(num_frames=3, width=0.125, sigmoid_output=True).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 9, 72, 128))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_logits_and_forward_are_consistent():
    model = TrackNet(num_frames=3, width=0.125).eval()
    x = torch.rand(1, 9, 72, 128)
    with torch.no_grad():
        assert torch.allclose(model(x), torch.sigmoid(model.forward_logits(x)), atol=1e-6)


def test_channel_count_follows_num_frames():
    for num_frames in (1, 2, 3, 5):
        model = TrackNet(num_frames=num_frames, width=0.125).eval()
        assert model.in_channels == num_frames * 3
        with torch.no_grad():
            out = model(torch.rand(1, num_frames * 3, 72, 128))
        assert out.shape == (1, 1, 72, 128)


def test_head_bias_starts_negative_so_the_heatmap_starts_empty():
    torch.manual_seed(0)
    model = TrackNet(num_frames=3, width=0.125).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 9, 72, 128))
    assert float(out.mean()) < 0.2


def test_gradients_reach_the_first_layer():
    model = TrackNet(num_frames=3, width=0.125)
    out = model.forward_logits(torch.rand(1, 9, 72, 128))
    out.mean().backward()
    first = model.enc1[0].conv.weight
    assert first.grad is not None
    assert torch.isfinite(first.grad).all()
    assert float(first.grad.abs().sum()) > 0


def test_build_model_from_config(base_config):
    model = build_model(base_config)
    assert model.in_channels == int(base_config.data.num_frames) * 3
    with torch.no_grad():
        out = model(torch.rand(1, model.in_channels, base_config.data.input_height, base_config.data.input_width))
    assert out.shape[-2:] == (base_config.data.input_height, base_config.data.input_width)


def test_unknown_model_name_is_rejected(base_config):
    base_config["model.name"] = "nope"
    with pytest.raises(KeyError):
        build_model(base_config)


def test_input_size_must_be_divisible_by_stride():
    check_input_size(360, 640)
    with pytest.raises(ValueError, match=f"divisible by {STRIDE}"):
        check_input_size(361, 640)


def test_width_scales_parameter_count():
    small = sum(p.numel() for p in TrackNet(width=0.25).parameters())
    large = sum(p.numel() for p in TrackNet(width=0.5).parameters())
    assert large > small


def test_eval_mode_is_deterministic():
    model = TrackNet(num_frames=3, width=0.125).eval()
    x = torch.rand(1, 9, 72, 128)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))
