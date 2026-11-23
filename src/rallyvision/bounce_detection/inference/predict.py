"""Bounce detection on a video with the formal model: Transformer v3 on the frozen E2E-Spot backbone.

    python -m rallyvision.bounce_detection.inference.predict --video match.mp4 --out bounces.json

Steps (the same as the reference demo): frames (resampled to 30 fps when the video is faster than
40 fps, as the training clips are 25/30 fps) -> backbone features -> Transformer v3 per-frame
probabilities (100-frame windows, 50% overlap, averaged) -> bounces with the decoding stored in the
checkpoint (threshold 0.6, NMS window 15 frames, both chosen on val). Needs no ball track or court.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from rallyvision.bounce_detection.inference.decode import decode_bounces
from rallyvision.bounce_detection.model.event_transformer.backbone import backbone, decode, load_spot
from rallyvision.bounce_detection.model.event_transformer.model import load_event_transformer
from rallyvision.common.device import best_device
from rallyvision.common.locations import artifact

@torch.no_grad()
def predict(model, feat, window, device):
    """Per-frame probabilities [n, 7]; windows with 50% overlap, averaged."""
    n = len(feat)
    x = feat if n >= window else np.concatenate([feat, np.zeros((window - n, feat.shape[1]), feat.dtype)])
    starts = list(range(0, len(x) - window + 1, window // 2))
    if starts[-1] != len(x) - window:
        starts.append(len(x) - window)
    xb = torch.from_numpy(np.stack([x[s:s + window] for s in starts]).astype(np.float32)).to(device)
    p = torch.softmax(model(xb).float(), -1).cpu().numpy()
    acc, cnt = np.zeros((len(x), p.shape[-1])), np.zeros(len(x))
    for s, q in zip(starts, p):
        acc[s:s + window] += q
        cnt[s:s + window] += 1
    return (acc / cnt[:, None])[:n]


def sample_frames(video) -> tuple:
    """(source fps, source frame index of every model frame): every frame, or a 30 fps subsampling above 40 fps."""
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps > 40:
        return fps, np.unique(np.round(np.arange(0, n, fps / 30.0)).astype(int).clip(0, n - 1))
    return fps, None


def video_probabilities(video, spot, model, window: int, device):
    """Per-model-frame probabilities of Transformer v3 and of the original E2E-Spot GRU, and the frame map."""
    impl, clip_len = spot
    fps, keep = sample_frames(video)
    frames = decode(video, keep)
    keep = np.arange(len(frames)) if keep is None else keep[:len(frames)]
    feat, gru = backbone(impl, frames, clip_len, device)
    return predict(model, feat.astype(np.float16), window, device), gru, keep, fps


def detect_bounces(video, checkpoint=None, spot_code=None, spot_model=None, device="cuda"):
    """Bounce events of a video: [{"video_frame", "time_s", "confidence", "side"}] and the probabilities."""
    model, cfg, dec = load_event_transformer(checkpoint or artifact("bounce_detection", "transformer_v3"), device)
    spot = load_spot(Path(spot_code or artifact("bounce_detection", "e2e_spot_code")),
                     Path(spot_model or artifact("bounce_detection", "e2e_spot_model")), device)
    prob, gru, keep, fps = video_probabilities(video, spot, model, cfg["window"], device)
    events = [{"video_frame": int(keep[t]), "time_s": round(float(keep[t]) / fps, 3), "confidence": round(float(s), 3),
               "side": side} for t, s, side in decode_bounces(prob, dec["bounce"]["thr"], dec["bounce"]["sep"])]
    return events, {"prob": prob, "gru": gru, "video_frame": keep, "fps": fps, "decoding": dec["bounce"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="bounces .json (per-frame probabilities go next to it as .npz)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: configs/paths.yaml bounce_detection.transformer_v3")
    ap.add_argument("--spot-code", type=Path, default=None)
    ap.add_argument("--spot-model", type=Path, default=None)
    ap.add_argument("--device", default=best_device())
    args = ap.parse_args()
    events, aux = detect_bounces(args.video, args.checkpoint, args.spot_code, args.spot_model, args.device)
    args.out.write_text(json.dumps({"video": str(args.video), "decoding": aux["decoding"], "bounces": events}, indent=1))
    np.savez(args.out.with_suffix(".npz"), prob=aux["prob"].astype(np.float16), gru=aux["gru"].astype(np.float16),
             video_frame=aux["video_frame"], fps=aux["fps"])
    print(f"{len(events)} bounces -> {args.out}")


if __name__ == "__main__":
    main()
