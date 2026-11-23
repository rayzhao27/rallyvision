"""Ball detection and tracking on a video: TrackNet heatmap peaks per frame -> one ball position per frame.

    python -m rallyvision.ball_tracking.inference.track_video --video match.mp4 --out ball.npz

Detection (``detect_video``): every frame is pillar-boxed to 16:9 and resized to ``scale`` x 640x360
(scale 2 = 1280x720 for the formal model); the 3-frame TrackNet sees the current frame and the frames
nearest 1/30 s and 2/30 s earlier (its training spacing, at any frame rate). The top 3 heatmap peaks of
every frame are kept, in 640x360 pixels. Tracking (``track``): ``select_ball`` follows the ball among the
peaks with motion gating, ``drop_static`` removes long-still detections.

In the full pipeline the peaks are first restricted to the court area with the court homography
(``rallyvision.pipeline.fusion.gate_peaks``); run alone, this module tracks without that gate.
These are the same steps the pre-refactor run_rally.py used for the ball.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from rallyvision.ball_tracking.inference.ball_select import drop_static, input_offsets, select_ball, top_peaks
from rallyvision.ball_tracking.model import build_model
from rallyvision.common.checkpoint import load_model_weights
from rallyvision.common.config import load_config
from rallyvision.common.heatmap import decode_peak
from rallyvision.common.device import best_device
from rallyvision.common.locations import artifact
from rallyvision.common.video import pad_16x9

INPUT_W, INPUT_H = 640, 360


def load_ball_model(checkpoint=None, device="cpu", cfg=None):
    cfg = cfg or load_config()
    model = build_model(cfg)
    load_model_weights(checkpoint or artifact("ball_tracking", "model"), model)
    return model.to(device).eval(), cfg


def detect_video(video, model, cfg, device, threshold: float, scale: int = 2, legacy: bool = False):
    """fps, per-frame top peaks [k, 3] (x, y, score in 640x360 px), per-frame highest peak, frame geometry."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_blob = int(cfg.eval.get("min_blob_area", 0)) if scale == 1 else 0   # 1280 threshold chosen without it
    far, near = (2, 1) if legacy else input_offsets(fps)
    history = deque(maxlen=far + 1)
    peaks, argmax, geom = [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        padded, pad = pad_16x9(frame)
        if geom is None:
            geom = {"size": frame.shape[1::-1], "pad": pad, "scale": INPUT_W / padded.shape[1]}
        to_tensor = lambda w, h: torch.from_numpy(cv2.cvtColor(cv2.resize(padded, (w, h), interpolation=cv2.INTER_AREA),
                                                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        history.append(to_tensor(INPUT_W * scale, INPUT_H * scale))
        while len(history) < history.maxlen:    # repeat the first frame until the stack is full
            history.appendleft(history[0])
        stack = torch.cat([history[-1 - far], history[-1 - near], history[-1]])
        with torch.no_grad():
            hm = torch.sigmoid(model.forward_logits(stack[None].to(device)))[0, 0].cpu().numpy()
        p = top_peaks(hm, size=9 * scale)
        p[:, :2] /= scale
        peaks.append(p)
        a = decode_peak(hm, threshold=threshold, min_blob_area=min_blob)[0]
        argmax.append(None if a is None else (a[0] / scale, a[1] / scale))
    cap.release()
    if not peaks:
        raise ValueError(f"no frames read from {video}")
    return fps, peaks, argmax, geom


def track(peaks, fps: float, threshold: float):
    """One ball position (x, y) or None per frame (select_ball with motion gating, then drop_static)."""
    return drop_static(select_ball(peaks, fps, acquire=threshold), fps)


def as_array(balls) -> np.ndarray:
    return np.array([p if p is not None else (np.nan, np.nan) for p in balls], dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".npz with fps, peaks (frame, x, y, score), ball [n, 2]")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: configs/paths.yaml ball_tracking.model")
    ap.add_argument("--scale", type=int, default=2, help="model input = scale x 640x360")
    ap.add_argument("--threshold", type=float, default=0.16)
    ap.add_argument("--device", default=best_device())
    args = ap.parse_args()
    model, cfg = load_ball_model(args.checkpoint, args.device)
    fps, peaks, _, _ = detect_video(args.video, model, cfg, args.device, args.threshold, args.scale)
    ball = as_array(track(peaks, fps, args.threshold))
    np.savez(args.out, fps=fps, ball=ball,
             peaks=np.array([(f, *r) for f, p in enumerate(peaks) for r in p], dtype=float).reshape(-1, 4))
    print(f"{len(ball)} frames at {fps:.2f} fps; ball in {np.isfinite(ball[:, 0]).mean():.1%} -> {args.out}")


if __name__ == "__main__":
    main()
