"""Run the three subsystems on one video and fuse their outputs.

    ball_tracking     per-frame ball peaks (TrackNet 1280)          -> rallyvision.ball_tracking.inference.track_video
    court_detection   per-frame homography (court keypoints)        -> rallyvision.court_detection.inference.calibrate
    bounce_detection  bounce frames + confidence (Transformer v3)   -> rallyvision.bounce_detection.inference.predict
    fusion            court-gated ball track; bounces in court metres -> rallyvision.pipeline.fusion

Bounce detection runs first so that a problem with the external E2E-Spot shows before the long passes.
All per-frame outputs are indexed by source video frame (bounce frames are mapped back from its 30 fps sampling).
"""

from __future__ import annotations

import numpy as np

from rallyvision.ball_tracking.inference.track_video import as_array, detect_video, load_ball_model, track
from rallyvision.bounce_detection.inference.predict import detect_bounces
from rallyvision.common.locations import pipeline_config
from rallyvision.court_detection.inference.calibrate import calibrate_video
from rallyvision.court_detection.model import load_court_model
from rallyvision.common.locations import artifact
from rallyvision.pipeline.fusion import bounce_locations, gate_peaks


def analyze(video, device="cuda", cfg=None, spot_code=None, spot_model=None):
    """Everything the demo needs, as plain arrays and dicts (see the module docstring)."""
    cfg = cfg or pipeline_config()
    events, bounce_aux = detect_bounces(video, None, spot_code, spot_model, device)   # first: external E2E-Spot fails early
    ball_model, ball_cfg = load_ball_model(None, device)
    fps, peaks, _, geom = detect_video(video, ball_model, ball_cfg, device, cfg["ball"]["threshold"], cfg["ball"]["input_scale"])
    court_model, _ = load_court_model(artifact("court_detection", "model"), device)
    _, calib = calibrate_video(video, court_model, device, cfg["court"]["threshold"])
    if len(calib) != len(peaks):
        raise RuntimeError(f"frame count mismatch: ball {len(peaks)}, court {len(calib)}")
    gated = gate_peaks(peaks, calib)
    ball = as_array(track(gated, fps, cfg["ball"]["threshold"]))
    H = np.array([c["H"] if c["H"] is not None else np.full((3, 3), np.nan) for c in calib])
    H_valid = np.array([c["valid"] for c in calib])
    bounces = bounce_locations(events, ball, H, H_valid, fps, cfg["fusion"]["window"])
    return {"fps": fps, "geom": geom, "peaks": gated, "ball": ball, "H": H, "H_valid": H_valid,
            "events": events, "bounce_aux": bounce_aux, "bounces": bounces}
