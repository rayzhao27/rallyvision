"""Temporal post-processing of per-frame detections.

A per-frame heatmap peak is noisy: the ball disappears behind a player, and
bright court lines occasionally out-score the real ball. The tracker applies
two cheap, physically motivated rules:

    * reject a detection that implies an impossible jump from the last
      confident position;
    * interpolate short gaps, since a ball occluded for three frames was still
      travelling on roughly a straight line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Detection:
    """A single frame's result, before or after tracking."""

    frame: int
    x: Optional[float] = None
    y: Optional[float] = None
    score: float = 0.0
    source: str = "none"  # detected | interpolated | rejected | none

    @property
    def has_position(self) -> bool:
        return self.x is not None and self.y is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "x": None if self.x is None else round(float(self.x), 3),
            "y": None if self.y is None else round(float(self.y), 3),
            "score": round(float(self.score), 5),
            "source": self.source,
        }


class BallTracker:
    """Online tracker: feed it one detection per frame, in order."""

    def __init__(
        self,
        max_jump_px: float = 100.0,
        max_gap_frames: int = 4,
        smooth_window: int = 0,
    ):
        self.max_jump_px = float(max_jump_px)
        self.max_gap_frames = int(max_gap_frames)
        self.smooth_window = int(smooth_window)
        self.reset()

    def reset(self) -> None:
        self.track: List[Detection] = []
        self._last_point: Optional[Tuple[float, float]] = None
        self._last_frame: Optional[int] = None
        self._rejected = 0

    @property
    def num_rejected(self) -> int:
        return self._rejected

    def update(
        self,
        frame_index: int,
        point: Optional[Tuple[float, float]],
        score: float = 0.0,
    ) -> Detection:
        """Validate one detection and append it to the track."""
        if point is None:
            det = Detection(frame=frame_index, score=float(score), source="none")
            self.track.append(det)
            return det

        x, y = float(point[0]), float(point[1])
        if self._last_point is not None and self._last_frame is not None:
            gap = max(1, frame_index - self._last_frame)
            # Allow proportionally more travel after a longer gap.
            budget = self.max_jump_px * min(gap, self.max_gap_frames + 1)
            distance = float(np.hypot(x - self._last_point[0], y - self._last_point[1]))
            if distance > budget and gap <= self.max_gap_frames:
                self._rejected += 1
                det = Detection(frame=frame_index, score=float(score), source="rejected")
                self.track.append(det)
                return det

        det = Detection(frame=frame_index, x=x, y=y, score=float(score), source="detected")
        self.track.append(det)
        self._last_point = (x, y)
        self._last_frame = frame_index
        return det

    def finalize(self, interpolate: bool = True) -> List[Detection]:
        """Return the completed track, optionally with short gaps filled."""
        track = list(self.track)
        if interpolate:
            track = interpolate_gaps(track, self.max_gap_frames)
        if self.smooth_window and self.smooth_window > 1:
            track = smooth_track(track, self.smooth_window)
        return track


def interpolate_gaps(track: Sequence[Detection], max_gap_frames: int = 4) -> List[Detection]:
    """Linearly fill runs of at most ``max_gap_frames`` missing positions."""
    out = [Detection(d.frame, d.x, d.y, d.score, d.source) for d in track]
    known = [i for i, d in enumerate(out) if d.has_position]
    if len(known) < 2 or max_gap_frames <= 0:
        return out

    for a, b in zip(known, known[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        x0, y0 = out[a].x, out[a].y
        x1, y1 = out[b].x, out[b].y
        for step in range(1, gap + 1):
            t = step / (gap + 1)
            filled = out[a + step]
            filled.x = x0 + (x1 - x0) * t
            filled.y = y0 + (y1 - y0) * t
            filled.source = "interpolated"
    return out


def smooth_track(track: Sequence[Detection], window: int = 3) -> List[Detection]:
    """Centered moving average over contiguous runs of known positions."""
    out = [Detection(d.frame, d.x, d.y, d.score, d.source) for d in track]
    if window < 2:
        return out
    half = window // 2
    xs = [d.x for d in out]
    ys = [d.y for d in out]
    for i, det in enumerate(out):
        if not det.has_position:
            continue
        lo, hi = max(0, i - half), min(len(out), i + half + 1)
        window_x = [xs[j] for j in range(lo, hi) if xs[j] is not None]
        window_y = [ys[j] for j in range(lo, hi) if ys[j] is not None]
        if window_x:
            det.x = float(np.mean(window_x))
            det.y = float(np.mean(window_y))
    return out


def track_statistics(track: Sequence[Detection]) -> Dict[str, Any]:
    total = len(track)
    detected = sum(1 for d in track if d.source == "detected")
    interpolated = sum(1 for d in track if d.source == "interpolated")
    rejected = sum(1 for d in track if d.source == "rejected")
    positions = [(d.x, d.y) for d in track if d.has_position]
    speeds: List[float] = []
    for (x0, y0), (x1, y1) in zip(positions, positions[1:]):
        speeds.append(float(np.hypot(x1 - x0, y1 - y0)))
    return {
        "frames": total,
        "detected": detected,
        "interpolated": interpolated,
        "rejected": rejected,
        "missing": total - detected - interpolated,
        "coverage": (detected + interpolated) / total if total else 0.0,
        "mean_step_px": float(np.mean(speeds)) if speeds else 0.0,
        "max_step_px": float(np.max(speeds)) if speeds else 0.0,
    }


def build_tracker(cfg) -> BallTracker:
    tracker_cfg = cfg.get("tracker", {})
    return BallTracker(
        max_jump_px=float(tracker_cfg.get("max_jump_px", 100.0)),
        max_gap_frames=int(tracker_cfg.get("max_gap_frames", 4)),
        smooth_window=int(tracker_cfg.get("smooth_window", 0)),
    )
