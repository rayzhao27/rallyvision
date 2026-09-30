from __future__ import annotations

import numpy as np
import pytest

from rallyvision.data.heatmap import decode_batch, decode_peak, render_gaussian


def test_gaussian_peaks_at_the_requested_point():
    hm = render_gaussian(40, 60, x=30.0, y=20.0, sigma=2.0)
    assert hm.shape == (40, 60)
    assert hm[20, 30] == pytest.approx(1.0, abs=1e-6)
    assert hm.max() == pytest.approx(1.0, abs=1e-6)


def test_gaussian_decays_with_distance():
    hm = render_gaussian(40, 60, x=30.0, y=20.0, sigma=2.0)
    assert hm[20, 31] < hm[20, 30]
    assert hm[20, 34] < hm[20, 31]


def test_gaussian_outside_canvas_is_empty():
    assert render_gaussian(20, 20, x=-5, y=5).max() == 0.0
    assert render_gaussian(20, 20, x=5, y=25).max() == 0.0
    assert render_gaussian(20, 20, x=float("nan"), y=5).max() == 0.0


def test_render_then_decode_recovers_the_point():
    for x, y in [(10.0, 12.0), (33.0, 5.0), (47.5, 29.25)]:
        hm = render_gaussian(40, 60, x=x, y=y, sigma=2.5)
        point, score = decode_peak(hm, threshold=0.5)
        assert point is not None
        assert score > 0.5
        assert point[0] == pytest.approx(x, abs=0.6)
        assert point[1] == pytest.approx(y, abs=0.6)


def test_decode_returns_none_below_threshold():
    hm = render_gaussian(30, 30, x=10, y=10, sigma=2.0) * 0.2
    point, score = decode_peak(hm, threshold=0.5)
    assert point is None
    assert score == pytest.approx(0.2, abs=1e-6)


def test_decode_rejects_tiny_blobs():
    hm = np.zeros((30, 30), dtype=np.float32)
    hm[10, 10] = 1.0  # a single hot pixel
    assert decode_peak(hm, threshold=0.5, min_blob_area=1)[0] is not None
    assert decode_peak(hm, threshold=0.5, min_blob_area=5)[0] is None


def test_decode_handles_empty_heatmap():
    point, score = decode_peak(np.zeros((16, 16), dtype=np.float32), threshold=0.5)
    assert point is None
    assert score == 0.0


def test_decode_batch_shapes():
    batch = np.stack([render_gaussian(24, 32, x=i * 4 + 2, y=10, sigma=2.0) for i in range(4)])
    results = decode_batch(batch[:, None], threshold=0.5)
    assert len(results) == 4
    for i, (point, _score) in enumerate(results):
        assert point is not None
        assert point[0] == pytest.approx(i * 4 + 2, abs=0.6)


def test_subpixel_is_more_accurate_than_argmax():
    hm = render_gaussian(40, 40, x=20.5, y=15.5, sigma=2.5)
    coarse, _ = decode_peak(hm, threshold=0.5, subpixel=False)
    fine, _ = decode_peak(hm, threshold=0.5, subpixel=True)
    coarse_err = abs(coarse[0] - 20.5) + abs(coarse[1] - 15.5)
    fine_err = abs(fine[0] - 20.5) + abs(fine[1] - 15.5)
    assert fine_err <= coarse_err
