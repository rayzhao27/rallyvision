"""Behaviour of the refactored modules equals the pre-refactor golden outputs (tests/golden, recorded 2026-11-22).

Uses the new module paths only. Inputs are saved intermediate results under local_archive/articrafts/ (local, not in git);
without them these tests are skipped. No video model runs except the small Transformer v3 head on a fixed input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "tests" / "golden"
A = REPO / "local_archive/articrafts"
VIDEO = REPO / "data/alcaraz_vs_sinner.mov"
if not (GOLD / "golden.json").exists():
    pytest.skip("local golden outputs are not available", allow_module_level=True)
G = json.loads((GOLD / "golden.json").read_text())
need = pytest.mark.skipif(not (A / "ball_select_v2/sinner/analysis.npz").exists(), reason="local local_archive/articrafts/ inputs missing")


def saved_peaks(name):
    a = np.load(A / f"ball_select_v2/{name}/analysis.npz")
    rows, n = a["peaks"], len(a["ball"])
    return [rows[rows[:, 0] == f, 1:] for f in range(n)], float(a["fps"])


def same(a, b):
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-6)


@need
@pytest.mark.parametrize("name", ["sinner", "carlos"])
def test_ball_tracking(name):
    from rallyvision.ball_tracking.inference.track_video import as_array, track
    peaks, fps = saved_peaks(name)
    assert same(as_array(track(peaks, fps, 0.16)), np.load(GOLD / f"ball_{name}.npy"))


@need
def test_court_homography():
    from rallyvision.court_detection import COURT_POINTS, fit_homography, image_to_court, project
    c = np.load(A / "motion_comp/sinner_court.npz")
    pts = np.array([[320.0, 180.0], [250.0, 120.0], [400.0, 260.0]])
    for f, ref in G["court_image_to_court"].items():
        assert image_to_court(c["H"][int(f)], pts).round(6).tolist() == ref
    kp = project(c["H"][500], COURT_POINTS) + np.random.default_rng(0).normal(0, 0.7, (14, 2))
    kp[[3, 9]] = np.nan
    fit = fit_homography(kp)
    assert {"valid": bool(fit["valid"]), "H": np.asarray(fit["H"]).round(8).tolist(), "inliers": fit["inliers"].tolist(),
            "err_px": round(float(fit["err_px"]), 6)} == G["court_fit"]


@need
def test_bounce_transformer_v3():
    from rallyvision.bounce_detection.inference.decode import decode_bounces
    from rallyvision.bounce_detection.inference.predict import predict
    from rallyvision.bounce_detection.model.event_transformer.model import load_event_transformer
    model, cfg, dec = load_event_transformer(A / "event_transformer/run_v3/event_transformer.pt", "cpu")
    assert dec == G["v3_decoding"] and dec["bounce"] == {"thr": 0.6, "sep": 15}
    x = np.random.default_rng(1).normal(size=(230, 768)).astype(np.float16)
    assert np.allclose(predict(model, x, cfg["window"], "cpu"), np.load(GOLD / "v3_predict_fixed_input.npy"), atol=1e-6)
    z = np.load(A / "event_transformer/audit/probs/sinner.npz")
    got = [{"t30": t, "video_frame": int(z["video_frame"][t]), "conf": round(float(s), 6), "side": side}
           for t, s, side in decode_bounces(z["p_v3"].astype(np.float32), dec["bounce"]["thr"], dec["bounce"]["sep"])]
    assert got == sorted(G["sinner_v3_events"], key=lambda e: e["t30"])


@need
def test_pipeline_fusion_reproduces_demo():
    from rallyvision.ball_tracking.inference.track_video import as_array, track
    from rallyvision.bounce_detection.inference.decode import decode_bounces
    from rallyvision.pipeline.fusion import bounce_locations
    peaks, fps = saved_peaks("sinner")            # saved peaks are already court-gated
    ball = as_array(track(peaks, fps, 0.16))
    z = np.load(A / "event_transformer/audit/probs/sinner.npz")
    events = [{"video_frame": int(z["video_frame"][t]), "confidence": round(float(s), 3), "side": side}
              for t, s, side in decode_bounces(z["p_v3"].astype(np.float32), 0.6, 15)]
    c = np.load(A / "motion_comp/sinner_court.npz")
    assert bounce_locations(events, ball, c["H"], c["valid"], fps, 2) == G["demo_bounces"]


@need
def test_gbdt_experiment_unchanged():
    from rallyvision.ball_tracking.inference.track_video import track
    from rallyvision.bounce_detection.experiments.gbdt.rally_gbdt import find_bounces
    from rallyvision.common.config import load_config
    from rallyvision.common.locations import artifact
    peaks, fps = saved_peaks("sinner")
    ev, params, _, _ = find_bounces(track(peaks, fps, 0.16), fps, load_config(),
                                    artifact("ball_tracking", "tracknet_tracks_640_t018"))
    assert json.loads(json.dumps(ev, default=float)) == G["gbdt_legacy_events"] and params == G["gbdt_legacy_params"]


@pytest.mark.skipif(not VIDEO.exists(), reason="demo video missing")
@need
def test_render_side_frames(monkeypatch):
    import cv2
    from rallyvision.pipeline import render
    captured = {}

    class Grab:
        def __init__(self, *a, **k):
            self.i = 0

        def write(self, img):
            if self.i in (0, 350, 700, 1000):
                captured[str(self.i)] = hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()
            self.i += 1

        def release(self):
            pass
    monkeypatch.setattr(cv2, "VideoWriter", Grab)
    monkeypatch.setattr(render.subprocess, "run", lambda *a, **k: None)
    out = Path("/tmp/_golden_test.mp4")
    out.with_suffix(".raw.mp4").touch()
    render.render_video(VIDEO, G["demo_bounces"], out, "side", 90, 1608)
    assert captured == G["render_side_frame_sha256"]


def test_golden_files_present():
    assert (GOLD / "golden.json").exists() and torch is not None
