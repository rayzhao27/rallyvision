"""Choosing the ball among heatmap peaks, for videos of any frame rate.

The detector was trained on 30 fps clips: its input is the current frame and
the two frames 1/30 s and 2/30 s earlier. At other frame rates the input frames
are picked by time (`input_offsets`), so the ball moves as far between them as
in training.

Taking only the highest peak fails when a head, racket or court mark out-scores
the ball. `select_ball` keeps the top peaks of every frame and follows the ball
in time: the next position is predicted from the last velocity, the peak nearest
to the prediction is accepted if it lies within a radius that grows with the time
since the last detection, and a peak far from the prediction is rejected rather
than allowed to pull the track away (a head next to a hidden ball). `drop_static` removes detections that do not
move for a long time (court marks, logos), `court_gate` removes peaks far
outside the court (graphics, crowd).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

TRAIN_FPS = 30.0

Point = Optional[Tuple[float, float]]


def input_offsets(fps: float, train_fps: float = TRAIN_FPS) -> Tuple[int, int]:
    """Frames back to the frames nearest t - 2/train_fps and t - 1/train_fps; (2, 1) at 30 fps."""
    if fps <= 0:
        raise ValueError(f"bad fps {fps}")
    near = max(1, int(round(fps / train_fps)))
    far = max(near + 1, int(round(2 * fps / train_fps)))
    return far, near


def top_peaks(heatmap: np.ndarray, k: int = 3, min_score: float = 0.05, size: int = 9) -> np.ndarray:
    """Up to k local maxima above min_score, best first, as rows (x, y, score).

    Positions are refined by the centroid of the 5x5 neighbourhood above half the peak value.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    ys, xs = np.nonzero((hm == maximum_filter(hm, size=size)) & (hm >= min_score))
    order = np.argsort(-hm[ys, xs])[:k]
    rows = []
    for y, x in zip(ys[order], xs[order]):
        y0, y1, x0, x1 = max(y - 2, 0), min(y + 3, hm.shape[0]), max(x - 2, 0), min(x + 3, hm.shape[1])
        w = np.clip(hm[y0:y1, x0:x1] - 0.5 * hm[y, x], 0, None)
        gy, gx = np.mgrid[y0:y1, x0:x1]
        rows.append((float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum()), float(hm[y, x])))
    return np.array(rows, dtype=float).reshape(-1, 3)


def select_ball(peaks: Sequence[np.ndarray], fps: float, acquire: float = 0.18, follow: float = 0.05,
                base_px: float = 10.0, speed_px_s: float = 900.0, max_gap_s: float = 0.25,
                confirm: int = 3, min_speed_px_s: float = 120.0, much_stronger: float = 2.0) -> List[Point]:
    """One ball position (or None) per frame from per-frame peak arrays (x, y, score).

    While the ball was seen within the last ``max_gap_s`` seconds, only peaks within
    ``base_px + speed_px_s * dt`` of the predicted position (and scoring at least ``follow``)
    can be the ball: the one nearest to the prediction, unless another one within the radius
    scores ``much_stronger`` times higher (the ball changed speed, e.g. a serve or a bounce).
    A peak outside the radius is rejected, however
    strong (a head or racket next to a weak or hidden ball); the frame is then missing.
    The track moves to a new place only when
      - it was lost for longer than ``max_gap_s`` (then the strongest peak >= ``acquire`` starts it), or
      - a strong peak (>= ``acquire``) outside the gate moved like a ball for ``confirm`` frames in a
        row (consistent positions, at least ``min_speed_px_s``) and scores ``much_stronger`` times
        higher than the tracked peak: the ball is elsewhere and the track was on a decoy.
    Pixels are those of the peaks (640x360 model input).
    """
    out: List[Point] = []
    last = vel = None
    last_t = 0.0
    pending: List[Tuple[float, np.ndarray]] = []    # (t, xy) of a possible new track outside the gate

    def radius(dt):
        return base_px + speed_px_s * dt

    for i, cand in enumerate(peaks):
        t = i / fps
        cand = np.asarray(cand, dtype=float).reshape(-1, 3)
        strongest = cand[0] if len(cand) and cand[0, 2] >= acquire else None   # rows are sorted by score
        tracking = last is not None and t - last_t <= max_gap_s
        in_pending = np.zeros(len(cand), dtype=bool)   # peaks that continue the possible new track
        pick = None
        if tracking:
            dt = t - last_t
            pred = last + (vel * dt if vel is not None else 0.0)
            dist = np.hypot(*(cand[:, :2] - pred).T)
            if pending:                                   # a peak belongs to the track that predicts it better
                (t0, p0), (t1, p1) = pending[0], pending[-1]
                v = (p1 - p0) / (t1 - t0) if len(pending) > 1 else np.zeros(2)
                d_pend = np.hypot(*(cand[:, :2] - (p1 + v * (t - t1))).T)
                in_pending = (d_pend <= radius(t - t1)) & (d_pend < dist)
            # an object already rejected stays rejected when the growing radius reaches it
            near = (cand[:, 2] >= follow) & (dist <= radius(dt)) & ~in_pending
            if near.any():
                pick = cand[near][np.argmin(dist[near])]
                best = cand[np.flatnonzero(near)[0]]       # rows are sorted by score
                if best[2] >= max(acquire, much_stronger * pick[2]):
                    pick = best

        # follow a strong peak that is not the ball as a possible new track
        if tracking and strongest is not None and (pick is None or not np.array_equal(pick, strongest)):
            if not (len(cand) and in_pending[0]):
                pending = []
            pending.append((t, strongest[:2]))
        elif not tracking:
            pending = []                                  # kept through frames where it is weak or not seen
        confirmed = False
        if len(pending) >= confirm:
            (t0, p0), (t1, p1) = pending[0], pending[-1]
            confirmed = np.hypot(*(p1 - p0)) / (t1 - t0) >= min_speed_px_s

        restart = strongest is not None and (not tracking or (confirmed and (pick is None or strongest[2] >= much_stronger * pick[2])))
        if restart:
            pick = strongest
            pending = []
        if pick is None:
            out.append(None)
            continue
        vel = None if restart or not tracking else (pick[:2] - last) / (t - last_t)
        last, last_t = pick[:2], t
        out.append((float(pick[0]), float(pick[1])))
    return out


def drop_static(points: Sequence[Point], fps: float, radius_px: float = 2.0, min_s: float = 1.0) -> List[Point]:
    """Remove runs of detections that stay within ``radius_px`` for at least ``min_s`` seconds."""
    pts = list(points)
    min_len = max(2, int(round(min_s * fps)))
    start = 0
    while start < len(pts):
        if pts[start] is None:
            start += 1
            continue
        end = start + 1                                  # grow the run while points stay put
        while end < len(pts) and pts[end] is not None and np.hypot(pts[end][0] - pts[start][0],
                                                                   pts[end][1] - pts[start][1]) <= radius_px:
            end += 1
        if end - start >= min_len:
            pts[start:end] = [None] * (end - start)
        start = end
    return pts


def court_gate(peaks: np.ndarray, image_to_court, margin_m: float = 5.0) -> np.ndarray:
    """Drop peaks whose ground-plane position lies beyond the sidelines or behind the near
    baseline by more than ``margin_m`` (score bugs, crowd). Beyond the far baseline is kept:
    a ball high in the air projects there.

    ``image_to_court`` maps [N, 2] peak pixels to court metres (x across, y towards the far end).
    """
    if len(peaks) == 0:
        return peaks
    xy = image_to_court(peaks[:, :2])
    keep = np.isfinite(xy).all(axis=1) & (np.abs(xy[:, 0]) <= 5.485 + margin_m) & (xy[:, 1] >= -11.885 - margin_m)
    return peaks[keep]
