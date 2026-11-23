"""Training data for the event Transformer: cached backbone features per clip, labels, GPU windows."""

from __future__ import annotations

import numpy as np
import torch

def load_split(cache, split):
    out = []
    for f in sorted((cache / split).glob("*.npz")):
        z = np.load(f)
        y = np.zeros(len(z["feat"]), np.int64)
        y[z["ev_frame"]] = z["ev_cls"]
        out.append({"name": f.stem, "feat": z["feat"], "spot": z["spot"].astype(np.float32), "fps": float(z["fps"]),
                    "y": y, "ev_frame": z["ev_frame"].tolist(), "ev_cls": z["ev_cls"].tolist()})
    return out


def dilate(y, r):
    """Each event also labels its ±r background neighbours (training targets only)."""
    out = y.copy()
    for t in np.flatnonzero(y):
        lo, hi = max(t - r, 0), min(t + r + 1, len(y))
        out[lo:hi][out[lo:hi] == 0] = y[t]
    return out


def gpu_windows(clips, key, window, stride, device):
    """All features of a split on the GPU (float16), labels, and the start of every window inside a clip."""
    feat = torch.from_numpy(np.concatenate([c["feat"] for c in clips])).to(device)
    y = torch.from_numpy(np.concatenate([c[key] for c in clips])).to(device)
    starts, off = [], 0
    for c in clips:
        n = len(c["y"])
        starts += list(range(off, off + max(n - window, 0) + 1, stride))
        off += n
    return feat, y, torch.tensor(starts, device=device)


def take(feat, y, starts, window):
    idx = starts[:, None] + torch.arange(window, device=feat.device)
    return feat[idx].float(), y[idx]


GROUP_OF = {"far_court_bounce": "far_bounce", "far_court_swing": "far_swing", "far_court_serve": "far_serve",
            "near_court_bounce": "near_bounce", "near_court_swing": "near_swing", "near_court_serve": "near_serve"}
