"""Frozen visual backbone: the pretrained E2E-Spot tennis model (RegNetY-008 + GSM), used as-is.

E2E-Spot (Hong et al., ECCV 2022, github.com/jhong93/spot, BSD-3) is an external dependency: its code
checkout and its pretrained checkpoint (e2e-spot-models/tennis_rny008gsm_gru_rgb) are loaded from the
paths in configs/paths.yaml. Frames go in exactly as in its own inference (398x224, centre crop 224,
ImageNet normalisation, 100-frame clips with 50-frame overlap). Per frame we keep the backbone features
[768] (input of our Transformer) and the original GRU's class probabilities [7] (the baseline).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from rallyvision.bounce_detection.model.event_transformer.model import CLASSES

W_IN, H_IN, CROP = 398, 224, 224


def load_spot(spot_dir: Path, model_dir: Path, device):
    sys.path.insert(0, str(spot_dir))
    from train_e2e import E2EModel
    from util.dataset import load_classes
    cfg = json.loads((model_dir / "config.json").read_text())
    classes = list(load_classes(str(spot_dir / "data/tennis/class.txt")))
    assert classes == CLASSES, classes
    m = E2EModel(len(classes) + 1, cfg["feature_arch"], cfg["temporal_arch"], clip_len=cfg["clip_len"],
                 modality=cfg["modality"], device=device)
    ckpt = sorted(model_dir.glob("checkpoint_*.pt"))[-1]
    m.load(torch.load(ckpt, map_location=device, weights_only=False))
    print("E2E-Spot checkpoint:", ckpt.name, "clip_len", cfg["clip_len"])
    return m._model.eval(), cfg["clip_len"]


def pad169(f):
    h, w = f.shape[:2]
    px, py = max(0, (round(h * 16 / 9) - w) // 2), max(0, (round(w * 9 / 16) - h) // 2)
    return cv2.copyMakeBorder(f, py, py, px, px, cv2.BORDER_CONSTANT)


def decode(path, keep=None):
    """RGB uint8 [n, 224, 398, 3] as E2E-Spot's frames (cv2 resize of the whole frame). keep: frame indices."""
    cap, out, i = cv2.VideoCapture(str(path)), [], 0
    want = None if keep is None else set(int(k) for k in keep)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if want is None or i in want:
            out.append(cv2.cvtColor(cv2.resize(pad169(f), (W_IN, H_IN)), cv2.COLOR_BGR2RGB))
        i += 1
    return np.stack(out)


@torch.no_grad()
def backbone(impl, frames, clip_len, device, clips_per_batch=8):
    """Per-frame backbone features [n, 768] and E2E-Spot GRU probabilities [n, 7], overlapping clips averaged."""
    n = len(frames)
    half = clip_len // 2
    starts = list(range(0, max(n - half, 1), half))
    if n > clip_len and starts[-1] + clip_len < n:
        starts.append(n - clip_len)
    left = (W_IN - CROP) // 2
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    feat = np.zeros((n, impl._feat_dim), np.float32)
    prob = np.zeros((n, len(CLASSES) + 1), np.float32)
    cnt = np.zeros(n, np.float32)
    for b in range(0, len(starts), clips_per_batch):
        ss = starts[b:b + clips_per_batch]
        xs = []
        for s in ss:
            x = torch.from_numpy(frames[s:s + clip_len, :, left:left + CROP]).to(device)
            x = (x.permute(0, 3, 1, 2).float() / 255 - mean) / std
            xs.append(F.pad(x, (0, 0, 0, 0, 0, 0, 0, clip_len - len(x))))     # pad time like E2E-Spot
        x = torch.stack(xs)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            f = impl._features(x.reshape(-1, 3, CROP, CROP)).reshape(len(ss), clip_len, -1)
            p = impl._pred_fine(f)
        p = torch.softmax((p[0] if isinstance(p, tuple) else p).float(), -1)
        f, p = f.float().cpu().numpy(), p.cpu().numpy()
        for k, s in enumerate(ss):
            e = min(s + clip_len, n)
            feat[s:e] += f[k, :e - s]
            prob[s:e] += p[k, :e - s]
            cnt[s:e] += 1
    return feat / cnt[:, None], prob / cnt[:, None]
