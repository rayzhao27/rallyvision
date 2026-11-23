"""Bounce detection on a ball trajectory, first bounce per shot, sub-frame contact, and event scoring.

A bounce shows up as a sudden upward kink in the ball's image-y motion: the second
difference ``a[t] = y[t+1] - 2*y[t] + y[t-1]`` has a sharp negative peak (image y grows
downwards). Kink frames are the candidates; a rule or a classifier decides which are bounces.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def fill_short_gaps(y: Sequence[float], max_gap: int = 4) -> np.ndarray:
    """Linearly fill runs of at most ``max_gap`` NaNs between two known values."""
    y = np.asarray(y, dtype=float).copy()
    known = np.flatnonzero(np.isfinite(y))
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= max_gap + 1:
            y[a + 1 : b] = np.interp(np.arange(a + 1, b), [a, b], [y[a], y[b]])
    return y


def bridge_gaps(x, y, min_gap: int = 5, max_gap: int = 20, window: int = 7, meet_px: float = 8.0):
    """Fill gaps of ``min_gap``..``max_gap`` frames where the arcs on both sides meet.

    Between hits the ball flies on a parabola, so each side of a gap is fitted with a
    quadratic in x(t) and y(t) from its nearest ``window`` frames: the incoming arc is
    predicted forward and the outgoing arc backward (a Kalman-style forward/backward
    prediction with gravity). If the two come within ``meet_px`` inside the gap, the frames
    up to the closest point take the incoming arc and the rest the outgoing one, which
    restores the kink of a bounce hidden in the gap (the classifier still decides whether
    it is one). If they never meet, the gap stays empty: nothing is invented.
    Returns new arrays; frames that were known are unchanged.
    """
    x0, y0 = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xs, ys = x0.copy(), y0.copy()
    known = np.flatnonzero(np.isfinite(x0) & np.isfinite(y0))
    for a, b in zip(known, known[1:]):
        if not min_gap <= b - a - 1 <= max_gap:
            continue
        before = known[(known <= a) & (known > a - window)]
        after = known[(known >= b) & (known < b + window)]
        if len(before) < 4 or len(after) < 4:
            continue
        t = np.arange(a + 1, b)
        arc_in = [np.polyval(np.polyfit(before, v[before], 2), t) for v in (x0, y0)]
        arc_out = [np.polyval(np.polyfit(after, v[after], 2), t) for v in (x0, y0)]
        dist = np.hypot(arc_in[0] - arc_out[0], arc_in[1] - arc_out[1])
        k = int(np.argmin(dist))
        if dist[k] > meet_px:
            continue
        xs[a + 1:b] = np.r_[arc_in[0][:k + 1], arc_out[0][k + 1:]]
        ys[a + 1:b] = np.r_[arc_in[1][:k + 1], arc_out[1][k + 1:]]
    return xs, ys


def bounce_candidates(y: Sequence[float], max_gap: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Frames where the second difference has a negative local minimum (an upward kink), and its size.

    Strict local minima never sit on adjacent frames. Undefined values (gaps, clip ends)
    are never candidates and never block a neighbour.
    """
    y = fill_short_gaps(y, max_gap)
    a = np.full(len(y), np.inf)
    a[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
    a[np.isnan(a)] = np.inf
    frames = np.array([t for t in range(1, len(y) - 1) if a[t] < 0 and a[t] <= a[t - 1] and a[t] < a[t + 1]],
                      dtype=int)
    return frames, -a[frames]


def detect_bounces(y: Sequence[float], threshold: float, max_gap: int = 4) -> np.ndarray:
    """Frames whose upward kink is larger than ``threshold`` (px / frame^2)."""
    frames, strength = bounce_candidates(y, max_gap)
    return frames[strength > threshold]


def subframe_bounce(x, y, t0: int, window: int = 7, skip: int = 0, degree: int = 2):
    """Sub-frame contact estimate from the trajectory segments around a detected bounce frame ``t0``.

    Fits ``degree``-order polynomials to x(t) and y(t) on the ``window`` frames before
    ``t0 - skip`` and after ``t0 + skip`` (``t0`` itself is always left out), then returns
    ``(t, x, y)`` where the incoming and outgoing curves come closest, searched within
    ±(skip + 1) frames of ``t0``. Returns None if either side has too few points.
    Defaults were chosen in scripts/simulate_bounce_localization.py; it only beats the
    plain bounce-frame position when per-frame detection noise is low (≲1.5 px at 720p).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    fits = []
    for lo, hi in ((t0 - skip - window, t0 - skip - 1), (t0 + skip + 1, t0 + skip + window)):
        t = np.arange(max(lo, 0), min(hi, len(y) - 1) + 1)
        t = t[np.isfinite(x[t]) & np.isfinite(y[t])]
        if len(t) < degree + 2:
            return None
        fits.append((np.polyfit(t, x[t], degree), np.polyfit(t, y[t], degree)))
    ts = np.arange(t0 - skip - 1, t0 + skip + 1.001, 0.01)
    (bx, by), (ax, ay) = fits
    gap = np.hypot(np.polyval(bx, ts) - np.polyval(ax, ts), np.polyval(by, ts) - np.polyval(ay, ts))
    t = ts[int(np.argmin(gap))]
    return float(t), float((np.polyval(bx, t) + np.polyval(ax, t)) / 2), float((np.polyval(by, t) + np.polyval(ay, t)) / 2)


FEATURES = ("kink", "x", "y", "vx_before", "vy_before", "vx_after", "vy_after", "vx_product", "missing_frac")


def candidate_features(x, y, missing, idx, kink, window: int = 5) -> np.ndarray:
    """One row per candidate frame, columns as in ``FEATURES``.

    Velocities are mean px/frame over ``window`` frames before and after the
    candidate (NaN near clip ends or across long gaps). ``vx_product`` < 0 means
    the horizontal direction reversed, which is typical of a hit, not a bounce.
    ``missing_frac`` is the share of frames in the window without a real detection.
    ``kink`` is the signed second difference at the candidate (negative = upward kink).
    """
    x, y = fill_short_gaps(x), fill_short_gaps(y)
    missing = np.asarray(missing, dtype=bool)
    n = len(y)
    rows = []
    for t, s in zip(idx, kink):
        b, a = t - window, t + window
        vxb, vyb = ((x[t] - x[b]) / window, (y[t] - y[b]) / window) if b >= 0 else (np.nan, np.nan)
        vxa, vya = ((x[a] - x[t]) / window, (y[a] - y[t]) / window) if a < n else (np.nan, np.nan)
        rows.append([s, x[t], y[t], vxb, vyb, vxa, vya, vxb * vxa, missing[max(b, 0) : a + 1].mean()])
    return np.array(rows, dtype=float).reshape(-1, len(FEATURES))


def first_bounces(bounces: Sequence[int], hits: Sequence[int]) -> List[int]:
    """The first bounce after each hit (or after the clip start); later bounces of the same shot are dropped."""
    out, open_shot = [], True
    for frame, is_hit in sorted([(int(b), False) for b in bounces] + [(int(h), True) for h in hits]):
        if is_hit:
            open_shot = True
        elif open_shot:
            out.append(frame)
            open_shot = False
    return out


def first_bounces_by_gap(bounces: Sequence[int], max_same_shot_gap: int) -> List[int]:
    """First bounce of each shot without hit detection.

    A bounce at most ``max_same_shot_gap`` frames after the previous bounce is a later
    bounce of the same shot (e.g. a drop shot): a new shot needs time to reach the
    opponent and come back. ``max_same_shot_gap=0`` keeps every bounce.
    """
    out, prev = [], None
    for b in sorted(int(b) for b in bounces):
        if prev is None or b - prev > max_same_shot_gap:
            out.append(b)
        prev = b
    return out


def match_pairs(pred: Sequence[int], gt: Sequence[int], tol: int) -> List[Tuple[int, int]]:
    """One-to-one (pred index, gt index) matches within ``tol`` frames, closest pairs first."""
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol)
    used_p, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def match_events(pred: Sequence[int], gt: Sequence[int], tol: int) -> Tuple[int, int, int]:
    """Returns (tp, fp, fn) for a one-to-one match within ``tol`` frames."""
    tp = len(match_pairs(pred, gt, tol))
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp: int, fp: int, fn: int) -> List[float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return [p, r, f1]
