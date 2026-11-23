"""Temporal event detector: per-frame bounce / hit / other from ball motion, court geometry and image features.

Input per frame (one token each, ~1 s windows):
  motion   ball position and its velocity / acceleration over the window, detection state, peak scores,
           and the ball's ground-plane position through the court homography (meaningful at a bounce)
  vision   (optional) frozen-encoder features of a 64x64 crop around the ball and of the whole frame
A small Transformer encoder sees the whole window and labels every frame, so a bounce that leaves no
kink in the 2D track can still be found from context and appearance. No candidate generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from rallyvision.court_detection import image_to_court
from rallyvision.bounce_detection.experiments.previous_transformers.e2e_tennis import BOUNCE, HIT, OTHER, frame_targets

IMG_W, IMG_H = 640.0, 360.0
COURT_HALF = (5.5, 12.0)          # metres, to normalise ground-plane coordinates
MOTION_DIM = 15


# --------------------------------------------------------------------------- features
def motion_features(track, track_src, peaks, H, H_valid, fps) -> np.ndarray:
    """[n, MOTION_DIM] float32 per-frame motion features (see module doc)."""
    n = len(track)
    xy = np.asarray(track, dtype=float).copy()
    have = np.isfinite(xy).all(1)
    # positions are carried forward/backward over gaps so differences stay defined; flags say which are real
    idx = np.where(have, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(have) if have.any() else 0
    filled = xy[np.where(idx >= 0, idx, first)] if have.any() else np.zeros((n, 2))
    pos = filled / [IMG_W, IMG_H]
    vel = np.gradient(pos, axis=0) * fps / 30.0 * 10           # per 1/30 s, scaled to ~unit range
    acc = np.gradient(vel, axis=0)
    top = np.nan_to_num(np.asarray(peaks)[:, :, 2], nan=0.0)  # [n, 3] peak scores
    ground = np.zeros((n, 2))
    gvalid = np.zeros(n)
    for i in np.flatnonzero(have & np.asarray(H_valid, bool)):
        g = image_to_court(H[i], xy[i])[0]
        if np.isfinite(g).all():
            ground[i] = np.clip(g / COURT_HALF, -3, 3)
            gvalid[i] = 1
    src = np.asarray(track_src)
    f = np.column_stack([pos, vel, acc, top, src == 1, src == 2, src == 0, ground, gvalid])
    assert f.shape[1] == MOTION_DIM, f.shape
    return f.astype(np.float32)


def load_clip(path: Path) -> Dict:
    """One preprocessed clip (scripts/preprocess_e2e.py) -> motion features, targets, image arrays."""
    z = np.load(path)
    clip = {"num_frames": len(z["track"]), "events": json.loads(str(z["events"]))}
    cls, side = frame_targets(clip)
    return {
        "name": Path(path).stem, "split": str(z["split"]), "fps": float(z["fps"]),
        "motion": motion_features(z["track"], z["track_src"], z["peaks"], z["H"], z["H_valid"], float(z["fps"])),
        "cls": cls, "side": side, "events": clip["events"],
        "track": z["track"], "H": z["H"], "H_valid": z["H_valid"],
    }


def soft_targets(cls: np.ndarray, radius: int = 1) -> np.ndarray:
    """Frames within ``radius`` of an event get its class (the label frame itself is ±1 frame uncertain)."""
    out = cls.copy()
    for t in np.flatnonzero(cls != OTHER):
        lo, hi = max(t - radius, 0), min(t + radius + 1, len(cls))
        out[lo:hi][out[lo:hi] == OTHER] = cls[t]
    return out


class WindowDataset(torch.utils.data.Dataset):
    """Fixed-length windows over clips; features = motion (+ vision) per frame."""

    def __init__(self, clips: Sequence[Dict], window: int, stride: int, vision: Dict[str, np.ndarray] = None,
                 augment: bool = False):
        self.clips, self.window, self.vision, self.augment = clips, window, vision, augment
        self.items = [(ci, s) for ci, c in enumerate(clips)
                      for s in range(0, max(len(c["cls"]) - window, 0) + 1, stride)]

    def __len__(self):
        return len(self.items)

    def features(self, c, s, e):
        m = c["motion"][s:e].copy()
        if self.vision is None:
            return m
        return np.concatenate([m, self.vision[c["name"]][s:e]], axis=1)

    def __getitem__(self, i):
        ci, s = self.items[i]
        c = self.clips[ci]
        e = s + self.window
        x = self.features(c, s, e)
        y = soft_targets(c["cls"])[s:e]
        if self.augment and self.vision is None and np.random.rand() < 0.5:
            x = x.copy()                              # horizontal flip (motion only): x, vx, ax, ground x
            x[:, 0] = 1 - x[:, 0]
            x[:, [2, 4, 12]] *= -1
        return torch.from_numpy(x), torch.from_numpy(y)


# --------------------------------------------------------------------------- model
class EventTransformer(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, layers: int = 3, heads: int = 4, ff: int = 256,
                 window: int = 32, dropout: float = 0.1, motion_dim: int = MOTION_DIM):
        super().__init__()
        self.motion_dim = motion_dim
        self.motion = nn.Sequential(nn.Linear(motion_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.vision = (nn.Sequential(nn.LayerNorm(in_dim - motion_dim), nn.Linear(in_dim - motion_dim, d_model))
                       if in_dim > motion_dim else None)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(self, x):                            # x [B, T, in_dim] -> logits [B, T, 3]
        h = self.motion(x[..., :self.motion_dim])
        if self.vision is not None:
            h = h + self.vision(x[..., self.motion_dim:])
        h = h + self.pos[:, :x.shape[1]]
        return self.head(self.encoder(h))


@torch.no_grad()
def predict_clip(model, feats: np.ndarray, window: int, stride: int, device) -> np.ndarray:
    """Per-frame class probabilities [n, 3] for a whole clip, averaging overlapping windows."""
    n = len(feats)
    if n < window:                                   # pad short clips at the end
        feats = np.concatenate([feats, np.repeat(feats[-1:], window - n, 0)])
    starts = list(range(0, len(feats) - window + 1, stride))
    if starts[-1] != len(feats) - window:
        starts.append(len(feats) - window)
    x = torch.from_numpy(np.stack([feats[s:s + window] for s in starts])).to(device)
    p = torch.softmax(model(x).float(), -1).cpu().numpy()
    acc, cnt = np.zeros((len(feats), 3)), np.zeros(len(feats))
    for s, ps in zip(starts, p):
        acc[s:s + window] += ps
        cnt[s:s + window] += 1
    return (acc / cnt[:, None])[:n]


# --------------------------------------------------------------------------- decoding and scoring
def decode_events(prob: np.ndarray, cls: int, threshold: float, min_sep: int) -> List[int]:
    """Frames where prob[:, cls] is a local maximum above threshold, at least min_sep frames apart."""
    p = prob[:, cls]
    keep = []
    for t in np.argsort(-p):
        if p[t] < threshold:
            break
        if all(abs(t - k) >= min_sep for k in keep):
            keep.append(int(t))
    return sorted(keep)


def match(pred: Sequence[int], gt: Sequence[int], tol: int) -> List[Tuple[int, int]]:
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol)
    used_p, used_g, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ground_side(c: Dict, frame: int) -> int:
    """1 near / 2 far / 0 unknown for a predicted bounce, from the ball's ground-plane y."""
    xy = c["track"][frame]
    if not (np.isfinite(xy).all() and c["H_valid"][frame]):
        return 0
    g = image_to_court(c["H"][frame], xy)[0]
    return 0 if not np.isfinite(g).all() else (2 if g[1] > 0 else 1)


def score_bounces(clips, preds: Dict[str, List[int]], tol: int) -> Dict[str, Tuple]:
    """Bounce (tp, fp, fn) overall and per court half. GT side from the label; FP side from the ground plane."""
    tot = {"all": [0, 0, 0], "near": [0, 0, 0], "far": [0, 0, 0]}
    for c in clips:
        gt = np.flatnonzero(c["cls"] == BOUNCE)
        gside = c["side"][gt]
        pred = preds[c["name"]]
        pairs = match(pred, gt.tolist(), tol)
        mp, mg = {i for i, _ in pairs}, {j for _, j in pairs}
        for k, name in ((1, "near"), (2, "far")):
            tot[name][0] += sum(gside[j] == k for _, j in pairs)
            tot[name][2] += sum(gside[j] == k for j in range(len(gt)) if j not in mg)
            tot[name][1] += sum(ground_side(c, pred[i]) == k for i in range(len(pred)) if i not in mp)
        tot["all"][0] += len(pairs)
        tot["all"][1] += len(pred) - len(pairs)
        tot["all"][2] += len(gt) - len(pairs)
    return {k: (*v, *prf(*v)) for k, v in tot.items()}


def score_class(clips, preds: Dict[str, List[int]], cls: int, tol: int) -> Tuple:
    tp = fp = fn = 0
    for c in clips:
        gt = np.flatnonzero(c["cls"] == cls).tolist()
        m = len(match(preds[c["name"]], gt, tol))
        tp, fp, fn = tp + m, fp + len(preds[c["name"]]) - m, fn + len(gt) - m
    return (tp, fp, fn, *prf(tp, fp, fn))
