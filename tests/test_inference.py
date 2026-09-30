"""Video inference on a few seconds of synthetic footage. CPU, untrained model."""

from __future__ import annotations

from pathlib import Path

import pytest

from rallyvision.data.synthetic import generate_video
from rallyvision.device import resolve_device
from rallyvision.inference.video import probe_video, run_video_inference
from rallyvision.models import build_model


@pytest.fixture(scope="module")
def short_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "rally.mp4"
    return generate_video(path, num_frames=12, width=128, height=72, fps=15, seed=3)


def test_probe_reads_video_properties(short_video):
    info = probe_video(short_video)
    assert info.width == 128
    assert info.height == 72
    assert info.frame_count == 12
    assert "128x72" in info.summary()


def test_probe_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")


def test_inference_produces_one_track_entry_per_frame(base_config, short_video, tmp_path):
    model = build_model(base_config)
    result = run_video_inference(
        model,
        short_video,
        output_path=None,
        device_info=resolve_device("cpu"),
        cfg=base_config,
        progress=False,
    )
    assert result["video"]["frames_processed"] == 12
    assert len(result["track"]) == 12
    assert result["stats"]["frames"] == 12


def test_max_frames_stops_early(base_config, short_video):
    model = build_model(base_config)
    result = run_video_inference(
        model,
        short_video,
        output_path=None,
        device_info=resolve_device("cpu"),
        cfg=base_config,
        max_frames=5,
        progress=False,
    )
    assert result["video"]["frames_processed"] == 5


def test_annotated_video_is_written_and_readable(base_config, short_video, tmp_path):
    model = build_model(base_config)
    out = tmp_path / "annotated.mp4"
    result = run_video_inference(
        model,
        short_video,
        output_path=out,
        device_info=resolve_device("cpu"),
        cfg=base_config,
        max_frames=6,
        progress=False,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    assert result["output_video"] == str(out)
    written = probe_video(out)
    assert written.width == 128 and written.height == 72


def test_track_coordinates_stay_inside_the_frame(base_config, short_video):
    base_config["inference.detection_threshold"] = 0.0001  # force detections
    model = build_model(base_config)
    result = run_video_inference(
        model,
        short_video,
        output_path=None,
        device_info=resolve_device("cpu"),
        cfg=base_config,
        max_frames=6,
        progress=False,
    )
    for entry in result["track"]:
        if entry["x"] is not None:
            assert 0 <= entry["x"] <= 128
            assert 0 <= entry["y"] <= 72
