"""Video helpers shared by the three subsystems."""

from __future__ import annotations

import cv2
import numpy as np


def pad_16x9(frame: np.ndarray):
    """Pillar-box (or letter-box) to 16:9. Returns the padded frame and (pad_x, pad_y)."""
    h, w = frame.shape[:2]
    pad_x = max(0, (int(round(h * 16 / 9)) - w) // 2)
    pad_y = max(0, (int(round(w * 9 / 16)) - h) // 2)
    return cv2.copyMakeBorder(frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT), (pad_x, pad_y)


def model_to_orig(scale: float, pad) -> np.ndarray:
    """3x3 map from 640x360 model pixels to original video pixels."""
    return np.array([[1 / scale, 0, -pad[0]], [0, 1 / scale, -pad[1]], [0, 0, 1]], dtype=np.float64)
