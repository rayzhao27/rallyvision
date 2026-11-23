from __future__ import annotations

import numpy as np
import pytest
import torch

from rallyvision.common.heatmap import render_gaussian
from rallyvision.ball_tracking.evaluation.metrics import BallMetrics, build_metrics, format_metrics

H, W = 32, 48


def _batch(points, visible, scale=(1.0, 1.0)):
    """Build a dataset-shaped batch from a list of (x, y) or None."""
    n = len(points)
    target = torch.zeros(n, 1, H, W)
    xy = torch.zeros(n, 2)
    for i, point in enumerate(points):
        if point is not None:
            target[i, 0] = torch.from_numpy(render_gaussian(H, W, point[0], point[1], sigma=2.0))
            xy[i] = torch.tensor(point)
        else:
            xy[i] = torch.tensor([-1.0, -1.0])
    return {
        "heatmap": target,
        "visible": torch.tensor([1.0 if v else 0.0 for v in visible]),
        "target_xy": xy,
        "to_original": torch.tensor([list(scale)] * n, dtype=torch.float32),
        "index": torch.arange(n),
        "clip_id": ["clip"] * n,
        "frame_file": [f"{i:04d}.jpg" for i in range(n)],
    }


def _prediction(points):
    out = torch.zeros(len(points), 1, H, W)
    for i, point in enumerate(points):
        if point is not None:
            out[i, 0] = torch.from_numpy(render_gaussian(H, W, point[0], point[1], sigma=2.0))
    return out


def test_perfect_prediction_scores_perfectly():
    points = [(10.0, 12.0), (30.0, 20.0)]
    metrics = BallMetrics(tolerances_px=[5, 10, 20], pr_tolerance_px=10)
    metrics.update(_prediction(points), _batch(points, [True, True]))
    result = metrics.compute()
    assert result["accuracy_at_5px"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["mean_error_px"] < 1.0


def test_missed_detection_counts_as_a_false_negative():
    metrics = BallMetrics(pr_tolerance_px=10)
    metrics.update(_prediction([None]), _batch([(10.0, 12.0)], [True]))
    result = metrics.compute()
    assert result["false_negatives"] == 1
    assert result["recall"] == 0.0
    assert result["accuracy_at_5px"] == 0.0


def test_detection_on_an_empty_frame_counts_as_a_false_positive():
    metrics = BallMetrics(pr_tolerance_px=10)
    metrics.update(_prediction([(10.0, 12.0)]), _batch([None], [False]))
    result = metrics.compute()
    assert result["false_positives"] == 1
    assert result["precision"] == 0.0


def test_correctly_predicting_nothing_is_a_true_negative():
    metrics = BallMetrics()
    metrics.update(_prediction([None]), _batch([None], [False]))
    result = metrics.compute()
    assert result["true_negatives"] == 1
    assert result["accuracy"] == 1.0


def test_offset_beyond_tolerance_is_a_false_positive():
    metrics = BallMetrics(tolerances_px=[5, 10, 20], pr_tolerance_px=5)
    metrics.update(_prediction([(25.0, 12.0)]), _batch([(10.0, 12.0)], [True]))
    result = metrics.compute()
    assert result["false_positives"] == 1
    assert result["accuracy_at_5px"] == 0.0
    assert result["accuracy_at_20px"] == 1.0
    assert result["mean_error_px"] == pytest.approx(15.0, abs=1.0)


def test_errors_are_reported_in_original_pixels():
    points = [(10.0, 12.0)]
    metrics = BallMetrics(tolerances_px=[5], pr_tolerance_px=10)
    # A 4x upscale means a 2px input-space error becomes 8px in the source video.
    metrics.update(_prediction([(12.0, 12.0)]), _batch(points, [True], scale=(4.0, 4.0)))
    result = metrics.compute()
    assert result["mean_error_px"] == pytest.approx(8.0, abs=1.0)


def test_accuracy_tolerances_are_monotonic():
    points = [(10.0, 12.0), (30.0, 20.0), (20.0, 8.0)]
    preds = [(10.0, 12.0), (37.0, 20.0), (20.0, 20.0)]
    metrics = BallMetrics(tolerances_px=[5, 10, 20], pr_tolerance_px=10)
    metrics.update(_prediction(preds), _batch(points, [True] * 3))
    result = metrics.compute()
    assert result["accuracy_at_5px"] <= result["accuracy_at_10px"] <= result["accuracy_at_20px"]


def test_empty_evaluation_does_not_crash():
    result = BallMetrics().compute()
    assert result["num_samples"] == 0
    assert np.isnan(result["mean_error_px"])
    assert result["f1"] == 0.0


def test_best_and_worst_samples_are_ordered():
    points = [(10.0, 12.0), (30.0, 20.0), (20.0, 8.0)]
    preds = [(10.0, 12.0), (37.0, 20.0), (22.0, 8.0)]
    metrics = BallMetrics(pr_tolerance_px=10)
    metrics.update(_prediction(preds), _batch(points, [True] * 3))
    assert metrics.best_samples(1)[0].error_px < metrics.worst_samples(1)[0].error_px


def test_build_metrics_reads_config(base_config):
    base_config["eval.tolerances_px"] = [3, 7]
    metrics = build_metrics(base_config)
    assert metrics.tolerances == [3.0, 7.0]


def test_format_metrics_includes_the_headline_numbers():
    points = [(10.0, 12.0)]
    metrics = BallMetrics()
    metrics.update(_prediction(points), _batch(points, [True]))
    text = format_metrics(metrics.compute())
    for expected in ("mean error", "accuracy@5px", "precision", "recall", "f1"):
        assert expected in text
