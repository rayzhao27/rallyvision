"""GPU-batched peak / keypoint decoders agree with the CPU versions (tiny synthetic heatmaps, CPU tensors)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rallyvision.bounce_detection.experiments.previous_transformers.preprocess_e2e import gpu_keypoints, gpu_top_peaks, merged_segments  # noqa: E402
from rallyvision.ball_tracking.inference.ball_select import top_peaks  # noqa: E402
from rallyvision.court_detection import decode_keypoints  # noqa: E402
from rallyvision.common.heatmap import render_gaussian  # noqa: E402


def test_gpu_top_peaks_matches_cpu():
    hm = np.maximum(render_gaussian(72, 128, 40.3, 20.6), 0.6 * render_gaussian(72, 128, 100, 50)).astype(np.float32)
    got = gpu_top_peaks(torch.from_numpy(hm[None]), size=9)[0]
    want = top_peaks(hm, size=9)
    np.testing.assert_allclose(got[:2], want, atol=1e-3)
    assert np.isnan(got[2]).all()


def test_gpu_keypoints_matches_decode_keypoints():
    hm = np.zeros((14, 36, 64), np.float32)
    for k in range(13):
        hm[k] = render_gaussian(36, 64, 3 + 4.3 * k, 5 + 2.1 * k)
    hm[13] = 0.1 * render_gaussian(36, 64, 30, 20)              # below threshold: missing
    got = gpu_keypoints(torch.from_numpy(hm[None]), 0.2)[0]
    want = decode_keypoints(hm, 0.2)[0]
    np.testing.assert_allclose(got, want, atol=1e-3)
    assert np.isnan(got[13]).all()


def test_merged_segments():
    clips = [{"start": 10, "end": 50}, {"start": 52, "end": 80}, {"start": 200, "end": 260}]
    segs = merged_segments(clips, back=4)
    assert [(lo, hi, len(c)) for lo, hi, c in segs] == [(6, 80, 2), (196, 260, 1)]
