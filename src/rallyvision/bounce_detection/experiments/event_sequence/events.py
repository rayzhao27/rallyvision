"""Hit candidates and event-sequence decoding (bounce / hit / nothing) on a ball trajectory.

The bounce classifier (rallyvision.bounce) scores upward-kink candidates one by one. Here the
candidates also include 2D velocity-change peaks (hits change the ball's direction without an
upward kink), a separate classifier scores hits, and `decode_events` picks the best-scoring
sequence of events over the whole clip:

    score = sum over chosen events of  logit(p) - logit(threshold)      (classifier evidence)
          + lam * sum over consecutive events of  transition(prev, next, gap)

`transition` is learned from labelled event sequences (`TransitionModel`): how often a bounce or
a hit follows a bounce or a hit after a given gap, relative to chance. Nothing is forbidden;
rare orders and timings only cost score, so bounce -> bounce (drop shot) stays possible.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from rallyvision.bounce_detection.experiments.gbdt.bounce import bounce_candidates, fill_short_gaps

BOUNCE, HIT = "bounce", "hit"
TYPES = (BOUNCE, HIT)
GAP_EDGES = (7, 10, 15, 25, 40, 70)  # frames at 30 fps; buckets <7, 7-9, 10-14, 15-24, 25-39, 40-69, >=70


def velocity_change(x, y, w: int = 3) -> np.ndarray:
    """|v_after - v_before| in px/frame, mean velocities over ``w`` frames each side; NaN where undefined."""
    x, y = fill_short_gaps(x), fill_short_gaps(y)
    d = np.full(len(y), np.nan)
    for t in range(w, len(y) - w):
        dvx = (x[t + w] - x[t]) / w - (x[t] - x[t - w]) / w
        dvy = (y[t + w] - y[t]) / w - (y[t] - y[t - w]) / w
        d[t] = np.hypot(dvx, dvy)
    return d


def velocity_peaks(d: np.ndarray, threshold: float = 2.0, sep: int = 3) -> np.ndarray:
    """Frames where ``d`` >= threshold, strongest first, at least ``sep`` frames apart; sorted."""
    keep = []
    for t in np.argsort(-np.nan_to_num(d, nan=-1.0)):
        if not d[t] >= threshold:
            break
        if all(abs(t - k) >= sep for k in keep):
            keep.append(int(t))
    return np.array(sorted(keep), dtype=int)


def event_candidates(x, y, w: int = 3, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union of upward-kink and velocity-change candidates.

    Returns (frames, signed second difference at each frame, is_kink). The kink candidates are
    exactly those of `bounce_candidates`, so the bounce classifier sees what it was trained on.
    """
    kink_idx, _ = bounce_candidates(y)
    idx = np.union1d(kink_idx, velocity_peaks(velocity_change(x, y, w), threshold, sep=w)).astype(int)
    yf = fill_short_gaps(y)
    a = np.full(len(yf), np.nan)
    a[1:-1] = yf[2:] - 2 * yf[1:-1] + yf[:-2]
    return idx, a[idx], np.isin(idx, kink_idx)


def gap_bucket(gap: float) -> int:
    return int(np.searchsorted(GAP_EDGES, gap, side="right"))


class TransitionModel:
    """log( P(next type, gap bucket | previous type) / chance ) from labelled event sequences.

    Counts are smoothed with ``alpha`` per cell; "chance" is the uniform 1 / number of cells, so a
    common transition scores > 0 and a rare one < 0.
    """

    def __init__(self, sequences: Sequence[Sequence[Tuple[int, str]]], alpha: float = 1.0):
        n_cells = len(TYPES) * (len(GAP_EDGES) + 1)
        counts = {p: np.full((len(TYPES), len(GAP_EDGES) + 1), alpha) for p in TYPES}
        for seq in sequences:
            seq = sorted(seq)
            for (f0, t0), (f1, t1) in zip(seq, seq[1:]):
                counts[t0][TYPES.index(t1), gap_bucket(f1 - f0)] += 1
        self.table = {p: np.log(c / c.sum() * n_cells) for p, c in counts.items()}

    def __call__(self, prev: str, nxt: str, gap: float) -> float:
        return float(self.table[prev][TYPES.index(nxt), gap_bucket(gap)])


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def decode_events(frames, p_bounce, p_hit, thresholds: Dict[str, float], transition: TransitionModel,
                  lam: float, max_lookback: int = 150) -> List[Tuple[int, str]]:
    """Best sequence of (frame, type) events over the candidates (dynamic programming, O(n * lookback)).

    Each candidate is a bounce, a hit or nothing. A candidate alone is worth
    logit(p) - logit(threshold), so with ``lam = 0`` this is plain thresholding of each classifier
    (a candidate above both thresholds takes the stronger one). ``lam`` weights the transition
    score between consecutive chosen events. The previous event is searched among the candidates
    in the last ``max_lookback`` frames; earlier history adds a constant and is ignored.
    """
    frames = np.asarray(frames, dtype=int)
    unary = {BOUNCE: _logit(p_bounce) - _logit(thresholds[BOUNCE]),
             HIT: _logit(p_hit) - _logit(thresholds[HIT])}
    n = len(frames)
    best = np.full((n, 2), -np.inf)       # best score of a sequence ending with candidate i as type s
    back = np.full((n, 2, 2), -1, dtype=int)
    for i in range(n):
        for s, typ in enumerate(TYPES):
            score, arg = 0.0, (-1, -1)    # i is the first event
            for j in range(i - 1, -1, -1):
                if frames[i] - frames[j] > max_lookback:
                    break
                for r, prev in enumerate(TYPES):
                    v = best[j, r] + lam * transition(prev, typ, frames[i] - frames[j])
                    if v > score:
                        score, arg = v, (j, r)
            best[i, s] = unary[typ][i] + score
            back[i, s] = arg
    if n == 0 or best.max() <= 0:
        return []
    i, s = np.unravel_index(int(np.argmax(best)), best.shape)
    out = []
    while i >= 0:
        out.append((int(frames[i]), TYPES[s]))
        i, s = back[i, s]
    return out[::-1]


def first_bounces_of_shots(events: Sequence[Tuple[int, str]], max_same_shot_gap: int) -> List[int]:
    """First bounce after each hit; without a hit in between, a bounce more than
    ``max_same_shot_gap`` frames after the previous bounce also starts a new shot (a missed hit)."""
    out, open_shot, last_bounce = [], True, None
    for f, typ in sorted(events):
        if typ == HIT:
            open_shot = True
            continue
        if open_shot or (last_bounce is not None and f - last_bounce > max_same_shot_gap):
            out.append(f)
        open_shot, last_bounce = False, f
    return out
