"""Export-path tests.

Tracing is checked everywhere, because a model that cannot be traced cannot be
converted. The actual Core ML conversion is marked and skipped unless
coremltools is installed on macOS.
"""

from __future__ import annotations

import platform

import pytest
import torch

from rallyvision.ball_tracking.export.coreml_export import CoreMLUnavailable, check_macos, trace_model
from rallyvision.ball_tracking.model import build_model
from rallyvision.common.tracknet import TrackNet


def _has_coremltools() -> bool:
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return platform.system() == "Darwin"


requires_coreml = pytest.mark.skipif(
    not _has_coremltools(), reason="needs coremltools on macOS"
)


def test_model_traces_cleanly():
    model = TrackNet(num_frames=3, width=0.125).eval()
    traced = trace_model(model, (1, 9, 72, 128))
    assert isinstance(traced, torch.jit.ScriptModule)


def test_traced_output_matches_the_eager_model():
    model = TrackNet(num_frames=3, width=0.125).eval()
    traced = trace_model(model, (1, 9, 72, 128))
    x = torch.rand(1, 9, 72, 128)
    with torch.no_grad():
        expected = torch.sigmoid(model.forward_logits(x))
        actual = traced(x)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_traced_model_always_outputs_probabilities():
    model = TrackNet(num_frames=3, width=0.125, sigmoid_output=False).eval()
    traced = trace_model(model, (1, 9, 72, 128))
    with torch.no_grad():
        out = traced(torch.rand(1, 9, 72, 128))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_trace_uses_the_configured_input_shape(base_config):
    model = build_model(base_config).eval()
    shape = (1, int(base_config.data.num_frames) * 3, int(base_config.data.input_height), int(base_config.data.input_width))
    traced = trace_model(model, shape)
    with torch.no_grad():
        assert traced(torch.rand(*shape)).shape == (1, 1, shape[2], shape[3])


@pytest.mark.skipif(platform.system() == "Darwin", reason="this check only fires off macOS")
def test_check_macos_blocks_non_mac_platforms():
    with pytest.raises(CoreMLUnavailable):
        check_macos()


@requires_coreml
@pytest.mark.coreml
@pytest.mark.slow
def test_coreml_export_and_validation_round_trip(tmp_path):
    from rallyvision.ball_tracking.export.coreml_export import export_coreml
    from rallyvision.ball_tracking.export.coreml_validate import validate_coreml

    model = TrackNet(num_frames=3, width=0.125).eval()
    shape = (1, 9, 72, 128)
    path = export_coreml(model, tmp_path / "m.mlpackage", input_shape=shape, precision="fp32")
    assert path.exists()

    report = validate_coreml(
        model, path, input_shape=shape, num_samples=2, max_abs_diff=0.05, max_peak_distance_px=3.0
    )
    assert report["num_samples"] == 2
