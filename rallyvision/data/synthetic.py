"""Procedurally generated clips in the real dataset's on-disk layout.

This exists so the whole pipeline (prepare -> dataset -> train -> evaluate ->
export) can be exercised locally in seconds without downloading anything. It is
a test fixture, not a training set: never report metrics measured on it.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _draw_court(width: int, height: int, rng: random.Random) -> np.ndarray:
    frame = np.full((height, width, 3), (40, 90, 60), dtype=np.uint8)
    noise = np.random.default_rng(rng.randrange(1 << 30)).integers(
        0, 18, size=(height, width, 3), dtype=np.int16
    )
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    margin_x, margin_y = int(width * 0.1), int(height * 0.12)
    cv2.rectangle(frame, (margin_x, margin_y), (width - margin_x, height - margin_y), (235, 235, 235), 2)
    cv2.line(frame, (width // 2, margin_y), (width // 2, height - margin_y), (235, 235, 235), 2)
    cv2.line(frame, (margin_x, height // 2), (width - margin_x, height // 2), (225, 225, 225), 2)
    return frame


def _ball_trajectory(
    num_frames: int, width: int, height: int, rng: random.Random
) -> List[Tuple[bool, float, float]]:
    """A bouncing parabolic path with a short occlusion gap."""
    x0 = rng.uniform(0.15, 0.3) * width
    x1 = rng.uniform(0.7, 0.9) * width
    peak = rng.uniform(0.15, 0.35) * height
    base = rng.uniform(0.6, 0.85) * height
    hidden_start = rng.randrange(0, max(1, num_frames - 4))
    hidden_len = rng.choice([0, 0, 2, 3])

    path: List[Tuple[bool, float, float]] = []
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        x = x0 + (x1 - x0) * t
        y = base - (base - peak) * math.sin(math.pi * t) + rng.uniform(-1.5, 1.5)
        visible = not (hidden_start <= i < hidden_start + hidden_len)
        visible = visible and 0 <= x < width and 0 <= y < height
        path.append((visible, x, y))
    return path


def generate_clip(
    clip_dir: Path,
    num_frames: int = 24,
    width: int = 640,
    height: int = 360,
    seed: int = 0,
    label_filename: str = "Label.csv",
) -> Path:
    """Write one synthetic clip (frames + Label.csv) to ``clip_dir``."""
    rng = random.Random(seed)
    clip_dir.mkdir(parents=True, exist_ok=True)
    background = _draw_court(width, height, rng)
    path = _ball_trajectory(num_frames, width, height, rng)

    rows = ["file name,visibility,x-coordinate,y-coordinate,status"]
    for i, (visible, x, y) in enumerate(path):
        frame = background.copy()
        # A moving player-like blob, so the model cannot cheat by finding the
        # only non-static thing in the scene.
        px = int(width * (0.3 + 0.4 * abs(math.sin(i / 7.0))))
        cv2.circle(frame, (px, int(height * 0.7)), 14, (60, 60, 160), -1)
        if visible:
            cv2.circle(frame, (int(round(x)), int(round(y))), 4, (30, 240, 240), -1)
        name = f"{i:04d}.jpg"
        if not cv2.imwrite(str(clip_dir / name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise RuntimeError(f"failed to write {clip_dir / name}")
        if visible:
            rows.append(f"{name},1,{x:.2f},{y:.2f},0")
        else:
            rows.append(f"{name},0,,,0")

    (clip_dir / label_filename).write_text("\n".join(rows) + "\n")
    return clip_dir


def generate_dataset(
    root: Path | str,
    num_games: int = 2,
    clips_per_game: int = 2,
    frames_per_clip: int = 24,
    width: int = 640,
    height: int = 360,
    seed: int = 1337,
    label_filename: str = "Label.csv",
) -> Path:
    """Write a `gameN/ClipM` tree that `prepare_dataset` can consume directly."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    counter = 0
    for g in range(1, num_games + 1):
        for c in range(1, clips_per_game + 1):
            generate_clip(
                root / f"game{g}" / f"Clip{c}",
                num_frames=frames_per_clip,
                width=width,
                height=height,
                seed=seed + counter,
                label_filename=label_filename,
            )
            counter += 1
    return root


def generate_video(
    path: Path | str,
    num_frames: int = 60,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    seed: int = 7,
) -> Path:
    """Write a short synthetic rally video, for testing `infer_video.py`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    background = _draw_court(width, height, rng)
    trajectory = _ball_trajectory(num_frames, width, height, rng)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for i, (visible, x, y) in enumerate(trajectory):
            frame = background.copy()
            px = int(width * (0.3 + 0.4 * abs(math.sin(i / 7.0))))
            cv2.circle(frame, (px, int(height * 0.7)), 14, (60, 60, 160), -1)
            if visible:
                cv2.circle(frame, (int(round(x)), int(round(y))), 4, (30, 240, 240), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path
