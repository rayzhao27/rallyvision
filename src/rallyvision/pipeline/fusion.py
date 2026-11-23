"""Fusion of the three subsystems: court-gated ball track, and bounce events placed on the court.

The only place where ball tracking, court detection and bounce detection meet. No model runs here.
"""

from __future__ import annotations

import numpy as np

from rallyvision.ball_tracking.inference.ball_select import court_gate
from rallyvision.court_detection.homography import image_to_court


def gate_peaks(peaks, calib):
    """Per frame, drop ball peaks far outside the court (court_gate) when that frame's homography is valid."""
    out = []
    for p, c in zip(peaks, calib):
        out.append(court_gate(p, lambda xy, H=c["H"]: image_to_court(H, xy)) if c["valid"] else p)
    return out


def bounce_locations(events, ball: np.ndarray, H: np.ndarray, H_valid: np.ndarray, fps: float, window: int = 2):
    """Court-space position of each bounce.

    Within ±``window`` video frames of the predicted bounce frame, the frame where the tracked ball is
    lowest in the image (largest y: the contact) is taken; its image position goes through that frame's
    homography to court metres. A frame without a valid homography leaves the bounce unplaced.
    ``events``: dicts with video_frame, confidence, side (bounce detection output).
    """
    out = []
    for e in events:
        f0 = int(e["video_frame"])
        cand = [f for f in range(f0 - window, f0 + window + 1) if 0 <= f < len(ball) and np.isfinite(ball[f, 0])]
        if cand:
            f = max(cand, key=lambda f: ball[f, 1])
            xy = image_to_court(H[f], ball[f])[0] if H_valid[f] else None
        else:
            f, xy = None, None
        out.append({"video_frame": f0, "time_s": round(f0 / fps, 3), "conf": round(float(e["confidence"]), 3),
                    "side_pred": e["side"], "ball_frame": f,
                    "court_xy": None if xy is None or not np.isfinite(xy).all() else [round(float(v), 2) for v in xy],
                    "valid": bool(f is not None and H_valid[f])})
    return out
