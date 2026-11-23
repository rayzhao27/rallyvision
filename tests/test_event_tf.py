"""Temporal event detector: features, targets, decoding, scoring, model shapes (tiny synthetic inputs only)."""
from __future__ import annotations

import numpy as np
import torch

from rallyvision.bounce_detection.experiments.previous_transformers.e2e_tennis import BOUNCE, HIT, OTHER, event_kind, frame_targets, parse_clip_name
from rallyvision.bounce_detection.experiments.previous_transformers.event_tf import (MOTION_DIM, EventTransformer, WindowDataset, decode_events, motion_features,
                                  predict_clip, score_bounces, soft_targets)


def test_labels():
    assert parse_clip_name("usopen_2019_mens_final_medvedev_nadal_100_400") == ("usopen_2019_mens_final_medvedev_nadal", 100, 400)
    assert event_kind("far_court_bounce") == (BOUNCE, "far") and event_kind("near_court_serve") == (HIT, "near")
    cls, side = frame_targets({"num_frames": 10, "events": [{"frame": 3, "label": "far_court_bounce"},
                                                             {"frame": 7, "label": "near_court_swing"}]})
    assert cls.tolist() == [0, 0, 0, 1, 0, 0, 0, 2, 0, 0] and side[3] == 2 and side[7] == 1


def _clip(n=40):
    t = np.arange(n, dtype=float)
    track = np.column_stack([100 + 3 * t, 200 + 0.5 * (t - 20) ** 2 / 4])
    track[10:13] = np.nan
    src = np.where(np.isfinite(track[:, 0]), 1, 0)
    peaks = np.full((n, 3, 3), np.nan)
    peaks[:, 0, 2] = 0.8
    H = np.tile(np.eye(3), (n, 1, 1))
    return track, src, peaks, H, np.ones(n, bool)


def test_motion_features_defined_everywhere():
    f = motion_features(*_clip(), fps=30.0)
    assert f.shape == (40, MOTION_DIM) and np.isfinite(f).all()
    assert f[11, 11] == 1 and f[0, 9] == 1          # missing / detected flags


def test_soft_targets_and_decoding():
    cls = np.zeros(20, int)
    cls[5], cls[12] = BOUNCE, HIT
    assert soft_targets(cls)[4:7].tolist() == [BOUNCE] * 3 and soft_targets(cls)[11:14].tolist() == [HIT] * 3
    prob = np.zeros((20, 3))
    prob[:, 0] = 1
    prob[[5, 6], 1] = [0.9, 0.8]
    prob[15, 1] = 0.3
    assert decode_events(prob, BOUNCE, 0.5, min_sep=3) == [5]


def test_score_bounces_by_side():
    n = 30
    track, src, peaks, H, hv = _clip(n)
    cls, side = np.zeros(n, int), np.zeros(n, int)
    cls[[5, 20]] = BOUNCE
    side[[5, 20]] = [1, 2]
    c = {"name": "c", "cls": cls, "side": side, "track": track, "H": H, "H_valid": hv}
    s = score_bounces([c], {"c": [6, 25]}, tol=2)
    assert s["all"][:3] == (1, 1, 1) and s["near"][0] == 1 and s["far"][2] == 1


def test_model_and_windows():
    track, src, peaks, H, hv = _clip(40)
    cls = np.zeros(40, int)
    cls[20] = BOUNCE
    c = {"name": "c", "cls": cls, "motion": motion_features(track, src, peaks, H, hv, 30.0)}
    ds = WindowDataset([c], window=16, stride=8, vision={"c": np.zeros((40, 4), np.float32)})
    x, y = ds[0]
    assert x.shape == (16, MOTION_DIM + 4) and y.shape == (16,)
    m = EventTransformer(MOTION_DIM + 4, d_model=16, layers=1, heads=2, ff=32, window=16)
    assert m(x[None]).shape == (1, 16, 3)
    p = predict_clip(m.eval(), np.concatenate([c["motion"], np.zeros((40, 4), np.float32)], 1), 16, 4, "cpu")
    assert p.shape == (40, 3) and np.allclose(p.sum(1), 1, atol=1e-5)
