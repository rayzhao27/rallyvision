"""Event decoding: per-frame probabilities -> event frames (greedy peak picking with a temporal NMS window)."""

from __future__ import annotations

import numpy as np

def peaks(score, min_sep=5, floor=0.05):
    """Greedy peak picking (highest first, min_sep frames apart) down to floor: [(frame, score)].
    Picking at a higher threshold is exactly the subset with score >= threshold."""
    keep = []
    for t in np.argsort(-score):
        if score[t] < floor:
            break
        if all(abs(t - k) >= min_sep for k, _ in keep):
            keep.append((int(t), float(score[t])))
    return keep


def decode_bounces(prob: np.ndarray, threshold: float, sep: int):
    """Bounce events from per-frame probabilities [n, 7]: (frame, confidence, side) with side far/near.

    The bounce score is far + near bounce probability; peaks at least ``sep`` frames apart with score
    >= ``threshold`` are kept; the side is the larger of the two probabilities at the peak.
    """
    score = prob[:, 1] + prob[:, 4]
    kept = [(t, s) for t, s in peaks(score, sep) if s >= threshold]
    return [(t, s, "far" if prob[t, 1] >= prob[t, 4] else "near") for t, s in sorted(kept)]
