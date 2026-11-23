"""Court keypoint model: load the single-frame TrackNet (14 heatmaps) and decode its keypoints."""

from __future__ import annotations

import numpy as np
import torch

from rallyvision.common.heatmap import decode_peak
from rallyvision.court_detection.geometry import NUM_KEYPOINTS


def decode_keypoints(heatmaps: np.ndarray, threshold: float = 0.5):
    """[14, H, W] probabilities -> (points [14, 2] with NaN where missing, scores [14])."""
    pts = np.full((NUM_KEYPOINTS, 2), np.nan)
    scores = np.zeros(NUM_KEYPOINTS)
    for k in range(NUM_KEYPOINTS):
        p, scores[k] = decode_peak(heatmaps[k], threshold=threshold)
        if p is not None:
            pts[k] = p
    return pts, scores


def load_court_model(path, device="cpu"):
    """Rebuild the TrackNet saved by scripts/train_court.py."""
    from rallyvision.common.tracknet import TrackNet
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TrackNet(num_frames=1, out_channels=NUM_KEYPOINTS, width=ckpt["width"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
