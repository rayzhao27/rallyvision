"""Per-frame court calibration of a video: keypoints -> homography (court metres -> 640x360 image pixels).

    python -m rallyvision.court_detection.inference.calibrate --video match.mp4 --out court.npz

Each frame is pillar-boxed to 16:9, resized to 640x360 and passed through the keypoint model; keypoints
above ``threshold`` (0.2, chosen on val) give the homography via ``fit_homography`` with its sanity checks.
These are the same steps the pre-refactor run_rally.py used for the court.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from rallyvision.common.device import best_device
from rallyvision.common.locations import artifact
from rallyvision.common.video import pad_16x9
from rallyvision.court_detection.geometry import INPUT_H, INPUT_W
from rallyvision.court_detection.homography import fit_homography
from rallyvision.court_detection.model import decode_keypoints, load_court_model


def frame_tensor(frame: np.ndarray) -> torch.Tensor:
    """BGR frame -> RGB float tensor [3, 360, 640] (16:9 pillar-box, area resize)."""
    padded, _ = pad_16x9(frame)
    rgb = cv2.cvtColor(cv2.resize(padded, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


@torch.no_grad()
def calibrate_frame(model, x: torch.Tensor, device, threshold: float = 0.2) -> dict:
    """One frame tensor -> fit_homography result (valid, reason, H, inliers, err_px, points)."""
    hm = model.predict_heatmap(x[None].to(device))[0].cpu().numpy()
    return fit_homography(decode_keypoints(hm, threshold)[0])


def calibrate_video(video, model, device, threshold: float = 0.2):
    """fps and one calibration dict per frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps, calib = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        calib.append(calibrate_frame(model, frame_tensor(frame), device, threshold))
    cap.release()
    return fps, calib


def save(path: Path, fps: float, calib) -> None:
    np.savez(path, fps=fps, valid=np.array([c["valid"] for c in calib]),
             H=np.array([c["H"] if c["H"] is not None else np.full((3, 3), np.nan) for c in calib]),
             reason=np.array([c["reason"] for c in calib]), err_px=np.array([c["err_px"] for c in calib]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".npz with H, valid, reason, err_px per frame")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: configs/paths.yaml court_detection.model")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--device", default=best_device())
    args = ap.parse_args()
    model, _ = load_court_model(args.checkpoint or artifact("court_detection", "model"), args.device)
    fps, calib = calibrate_video(args.video, model, args.device, args.threshold)
    save(args.out, fps, calib)
    print(f"{len(calib)} frames, court valid in {np.mean([c['valid'] for c in calib]):.1%} -> {args.out}")


if __name__ == "__main__":
    main()
