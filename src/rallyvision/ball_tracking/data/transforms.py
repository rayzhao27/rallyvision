"""Augmentation applied consistently across a temporal frame stack.

Every transform is sampled once per *stack*, not per frame: the model's whole
signal is inter-frame motion, so augmenting frames independently would destroy
the thing it is meant to learn. Ball coordinates are transformed alongside the
pixels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    enabled: bool = True
    horizontal_flip: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    translate: float = 0.05
    scale: float = 0.05

    @classmethod
    def from_config(cls, cfg) -> "AugmentConfig":
        if cfg is None:
            return cls(enabled=False)
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            horizontal_flip=float(cfg.get("horizontal_flip", 0.0)),
            brightness=float(cfg.get("brightness", 0.0)),
            contrast=float(cfg.get("contrast", 0.0)),
            translate=float(cfg.get("translate", 0.0)),
            scale=float(cfg.get("scale", 0.0)),
        )


class StackAugmentor:
    """Apply one randomly sampled geometric + photometric transform to a stack."""

    def __init__(self, config: AugmentConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()

    def __call__(
        self,
        frames: Sequence[np.ndarray],
        point: Optional[Tuple[float, float]],
    ) -> Tuple[List[np.ndarray], Optional[Tuple[float, float]]]:
        cfg = self.config
        if not cfg.enabled or not frames:
            return list(frames), point

        height, width = frames[0].shape[:2]
        out = [f for f in frames]
        xy = point

        if cfg.horizontal_flip > 0 and self.rng.random() < cfg.horizontal_flip:
            out = [np.ascontiguousarray(f[:, ::-1]) for f in out]
            if xy is not None:
                xy = (float(width - 1 - xy[0]), xy[1])

        if cfg.translate > 0 or cfg.scale > 0:
            scale = 1.0 + self.rng.uniform(-cfg.scale, cfg.scale) if cfg.scale > 0 else 1.0
            tx = self.rng.uniform(-cfg.translate, cfg.translate) * width if cfg.translate > 0 else 0.0
            ty = self.rng.uniform(-cfg.translate, cfg.translate) * height if cfg.translate > 0 else 0.0
            if abs(scale - 1.0) > 1e-6 or abs(tx) > 1e-6 or abs(ty) > 1e-6:
                cx, cy = width / 2.0, height / 2.0
                matrix = np.array(
                    [
                        [scale, 0.0, cx - scale * cx + tx],
                        [0.0, scale, cy - scale * cy + ty],
                    ],
                    dtype=np.float32,
                )
                out = [
                    cv2.warpAffine(
                        f,
                        matrix,
                        (width, height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                    for f in out
                ]
                if xy is not None:
                    nx = matrix[0, 0] * xy[0] + matrix[0, 2]
                    ny = matrix[1, 1] * xy[1] + matrix[1, 2]
                    xy = (float(nx), float(ny)) if 0 <= nx < width and 0 <= ny < height else None

        if cfg.brightness > 0 or cfg.contrast > 0:
            alpha = 1.0 + self.rng.uniform(-cfg.contrast, cfg.contrast) if cfg.contrast > 0 else 1.0
            beta = self.rng.uniform(-cfg.brightness, cfg.brightness) * 255.0 if cfg.brightness > 0 else 0.0
            if abs(alpha - 1.0) > 1e-6 or abs(beta) > 1e-6:
                # Clip to [0, 255]; cv2.convertScaleAbs takes |x|, which brightens
                # pixels pushed below zero instead of saturating them to black.
                out = [
                    np.clip(np.rint(f.astype(np.float32) * alpha + beta), 0, 255).astype(np.uint8)
                    for f in out
                ]

        return out, xy


def frames_to_tensor_array(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Stack RGB uint8 frames ``[H, W, 3]`` into a float32 ``[N*3, H, W]`` array in [0, 1]."""
    arrays = [np.asarray(f, dtype=np.float32) / 255.0 for f in frames]
    stacked = np.concatenate(arrays, axis=2)  # [H, W, N*3]
    return np.ascontiguousarray(stacked.transpose(2, 0, 1))


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    interp = cv2.INTER_AREA if (frame.shape[0] > height or frame.shape[1] > width) else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)
