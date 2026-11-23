#!/usr/bin/env python
"""Video -> bounces on a top-down court, in one command.

    python scripts/run_rally.py --video ~/Downloads/carlos.MOV --out local_archive/articrafts/rally/carlos

Steps (each one is the evaluated component, unchanged):
  1. every frame is pillar-boxed to 16:9 and resized to 640x360 (the training geometry)
  2. ball detector (3-frame TrackNet) on the current frame and the frames nearest 1/30 s and
     2/30 s earlier (its training spacing, at any frame rate); the ball is followed among the
     top heatmap peaks (rallyvision.ball_select), then BallTracker with gaps measured in time
     (--legacy-ball: consecutive frames and the single highest peak, as evaluated before)
  3. court model (single-frame TrackNet, 14 keypoints) + homography with sanity checks
  4. track resampled to 30 fps; bounce classifier trained on the 7 train games
     (threshold and same-shot gap chosen on train, as in scripts/evaluate_bounce.py)
  5. first bounce of each shot, sub-frame contact point, mapped to court metres with
     the H of its own frame; an invalid H makes the bounce "uncertain", not guessed

Outputs in --out: rally.mp4 (the untouched video beside a panel showing the
top-down court, a dot added at each bounce), bounce_map.png, bounces.json.
Two passes over the video, so long or high-resolution clips never sit in memory.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from rallyvision.common.config import REPO_ROOT  # noqa: E402
REPO = REPO_ROOT

from rallyvision.common.cli import load_config  # noqa: E402
from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import VAL_GAMES, fit_gbdt, load_clips, track_data  # noqa: E402
from rallyvision.pipeline.court_map import CANVAS_H, CANVAS_W, draw_bounce_dots, draw_topdown, to_canvas  # noqa: E402
from rallyvision.ball_tracking.inference.ball_select import court_gate, drop_static, input_offsets, select_ball, top_peaks  # noqa: E402
from rallyvision.bounce_detection.experiments.gbdt.bounce import bridge_gaps, first_bounces_by_gap, subframe_bounce  # noqa: E402
from rallyvision.common.checkpoint import load_model_weights  # noqa: E402
from rallyvision.court_detection import (INPUT_H, INPUT_W, decode_keypoints, fit_homography,  # noqa: E402
                               image_to_court, load_court_model)
from rallyvision.common.heatmap import decode_peak  # noqa: E402
from rallyvision.ball_tracking.model import build_model  # noqa: E402
from rallyvision.ball_tracking.inference.tracker import BallTracker  # noqa: E402

TRAIN_FPS = 30.0
OUT_H = 720                                  # rendered video height


def pad_16x9(frame: np.ndarray):
    """Pillar-box (or letter-box) to 16:9. Returns the padded frame and (pad_x, pad_y)."""
    h, w = frame.shape[:2]
    pad_x = max(0, (int(round(h * 16 / 9)) - w) // 2)
    pad_y = max(0, (int(round(w * 9 / 16)) - h) // 2)
    return cv2.copyMakeBorder(frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT), (pad_x, pad_y)


def model_to_orig(scale: float, pad) -> np.ndarray:
    """3x3 map from 640x360 model pixels to original video pixels."""
    return np.array([[1 / scale, 0, -pad[0]], [0, 1 / scale, -pad[1]], [0, 0, 1]], dtype=np.float64)


def analyse(video: Path, ball_model, court_model, cfg, device, court_thr, ball_thr, ball_scale=1, legacy=False):
    """Pass 1: ball detection and court calibration on every frame (640x360 coordinates).

    The ball model sees the frame at ``ball_scale`` x 640x360; its peaks are scaled back to 640x360.

    Returns per frame the top ball peaks (court-gated) and the single highest peak.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    min_blob = int(cfg.eval.get("min_blob_area", 0)) if ball_scale == 1 else 0   # 1280 threshold chosen without it
    far, near = (2, 1) if legacy else input_offsets(fps)
    history = deque(maxlen=far + 1)
    peaks, argmax, calib, geom = [], [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        padded, pad = pad_16x9(frame)
        if geom is None:
            geom = {"size": frame.shape[1::-1], "pad": pad, "scale": INPUT_W / padded.shape[1]}
        to_tensor = lambda w, h: torch.from_numpy(cv2.cvtColor(cv2.resize(padded, (w, h), interpolation=cv2.INTER_AREA),
                                                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)
        x = to_tensor(INPUT_W, INPUT_H)
        history.append(x if ball_scale == 1 else to_tensor(INPUT_W * ball_scale, INPUT_H * ball_scale))
        while len(history) < history.maxlen:    # repeat the first frame until the stack is full
            history.appendleft(history[0])
        stack = torch.cat([history[-1 - far], history[-1 - near], history[-1]])
        with torch.no_grad():
            ball_hm = torch.sigmoid(ball_model.forward_logits(stack[None].to(device)))[0, 0].cpu().numpy()
            court_hm = court_model.predict_heatmap(x[None].to(device))[0].cpu().numpy()
        c = fit_homography(decode_keypoints(court_hm, court_thr)[0])
        p = top_peaks(ball_hm, size=9 * ball_scale)
        p[:, :2] /= ball_scale
        if c["valid"]:
            p = court_gate(p, lambda xy: image_to_court(c["H"], xy))
        peaks.append(p)
        a = decode_peak(ball_hm, threshold=ball_thr, min_blob_area=min_blob)[0]
        argmax.append(None if a is None else (a[0] / ball_scale, a[1] / ball_scale))
        calib.append(c)
        if len(calib) % 200 == 0:
            print(f"  {len(calib)} frames")
    cap.release()
    if not calib:
        raise ValueError(f"no frames read from {video}")
    return fps, peaks, argmax, calib, geom


def find_bounces(balls, fps, cfg, bounce_csv, bridge=False):
    """Track, resample to 30 fps, classify bounces, keep the first of each shot with a sub-frame fit."""
    gap = int(round(cfg.tracker.max_gap_frames * fps / TRAIN_FPS))     # same gap in seconds as at 30 fps
    tracker = BallTracker(max_jump_px=float(cfg.tracker.max_jump_px), max_gap_frames=gap)
    for f, p in enumerate(balls):
        tracker.update(f, p)
    trk = tracker.finalize()
    n = len(trk)
    x = np.array([d.x if d.has_position else np.nan for d in trk])
    y = np.array([d.y if d.has_position else np.nan for d in trk])
    missing = np.array([d.source != "detected" for d in trk])

    t_src = np.arange(n) / fps
    t30 = np.arange(0, t_src[-1], 1 / TRAIN_FPS)
    lo = np.clip(np.searchsorted(t_src, t30, side="right") - 1, 0, n - 2)
    w = (t30 - t_src[lo]) * fps
    x30, y30 = (1 - w) * x[lo] + w * x[lo + 1], (1 - w) * y[lo] + w * y[lo + 1]
    m30 = missing[lo] | missing[lo + 1]
    if bridge:                                  # filled frames stay flagged as missing
        x30, y30 = bridge_gaps(x30, y30)

    clips = load_clips(bounce_csv, REPO / "local_archive/articrafts/bounce_inputs/labels", cfg, bridge=bridge)
    detect = fit_gbdt([c for c in clips if c["game"] not in VAL_GAMES], "pred")   # train games only
    frames30 = np.arange(len(t30))
    bounces30 = detect({"frames": frames30, "pred": track_data(x30, y30, m30, frames30, [])})
    first = first_bounces_by_gap(bounces30, detect.params["same_shot_gap"])

    events = []
    for b in first:
        s = subframe_bounce(x30, y30, b)
        t = (s[0] if s else b) / TRAIN_FPS
        events.append({"time_s": round(float(t), 3), "video_frame": min(int(round(t * fps)), n - 1),
                       "xy_model": [float(s[1]), float(s[2])] if s else [float(x30[b]), float(y30[b])],
                       "subframe": s is not None})
    return events, detect.params, (x, y), float(np.mean(np.isfinite(x)))


def render(video: Path, out_path: Path, fps, calib, geom, bounces):
    """Pass 2: video (scaled to OUT_H) | side panel with the top-down court."""
    w0, h0 = geom["size"]
    vid_w = int(round(w0 * OUT_H / h0)) // 2 * 2
    panel_w = int(round(CANVAS_W * OUT_H / CANVAS_H)) // 2 * 2
    base = draw_topdown()

    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w + panel_w, OUT_H))
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or f >= len(calib):
            break
        img = cv2.resize(frame, (vid_w, OUT_H), interpolation=cv2.INTER_AREA)
        panel = draw_bounce_dots(base.copy(), bounces, upto_frame=f)
        shown = [b for b in bounces if b["video_frame"] <= f and b["court_xy"] is not None]
        if shown and f - shown[-1]["video_frame"] < int(fps):   # ring around the newest dot
            cv2.circle(panel, to_canvas(shown[-1]["court_xy"]), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"bounces: {len(shown)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if not calib[f]["valid"]:                               # the video itself stays untouched
            cv2.putText(panel, "court not visible", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 1, cv2.LINE_AA)
        writer.write(np.hstack([img, cv2.resize(panel, (panel_w, OUT_H), interpolation=cv2.INTER_AREA)]))
        f += 1
    cap.release()
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ball-checkpoint", type=Path, default=REPO / "local_archive/articrafts/best_original.pt")
    ap.add_argument("--ball-scale", type=int, default=1, help="ball model input = scale x 640x360 (2 for hr1280_v1)")
    ap.add_argument("--ball-threshold", type=float, help="default: config inference.detection_threshold")
    ap.add_argument("--bounce-csv", type=Path, nargs="+", default=[
        REPO / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv",
        REPO / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv"],
        help="per-frame tracks of this ball model on the TrackNet games (640x360), to train the bounce classifier")
    ap.add_argument("--court-checkpoint", type=Path, default=REPO / "local_archive/articrafts/court/court_v2/best.pt")
    ap.add_argument("--court-threshold", type=float, default=0.2, help="chosen on val")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--no-video", action="store_true", help="skip rendering rally.mp4")
    ap.add_argument("--legacy-ball", action="store_true", help="consecutive input frames + highest peak only")
    ap.add_argument("--bridge-gaps", action="store_true", help="fill longer track gaps where the arcs meet")
    args = ap.parse_args()
    for p in (args.video, args.ball_checkpoint, args.court_checkpoint, *args.bounce_csv):
        if not p.is_file():
            raise FileNotFoundError(p)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    ball_thr = args.ball_threshold if args.ball_threshold is not None else float(cfg.inference.detection_threshold)
    ball_model = build_model(cfg)
    load_model_weights(args.ball_checkpoint, ball_model)
    ball_model.to(args.device).eval()
    court_model, _ = load_court_model(args.court_checkpoint, args.device)

    print("pass 1: ball + court on every frame")
    fps, peaks, argmax, calib, geom = analyse(args.video, ball_model, court_model, cfg, args.device,
                                              args.court_threshold, ball_thr, args.ball_scale, args.legacy_ball)
    balls = argmax if args.legacy_ball else drop_static(select_ball(peaks, fps, acquire=ball_thr), fps)
    n_valid = sum(c["valid"] for c in calib)
    np.savez(args.out / "analysis.npz", fps=fps,                 # per-frame results, for diagnosis
             ball=np.array([p if p is not None else (np.nan, np.nan) for p in balls], dtype=float),
             court_valid=np.array([c["valid"] for c in calib]),
             peaks=np.array([(f, *r) for f, p in enumerate(peaks) for r in p], dtype=float).reshape(-1, 4))  # frame, x, y, score
    print(f"{len(calib)} frames at {fps:.2f} fps; ball found in {sum(p is not None for p in balls)}, "
          f"court valid in {n_valid} ({n_valid / len(calib):.1%})")

    print("bounces")
    events, params, _, coverage = find_bounces(balls, fps, cfg, args.bounce_csv, args.bridge_gaps)
    to_orig = model_to_orig(geom["scale"], geom["pad"])
    for e in events:
        c = calib[e["video_frame"]]
        e.update(valid=bool(c["valid"]), reason=c["reason"], court_xy=None,
                 xy_video=[round(float(v), 1) for v in (to_orig @ [*e["xy_model"], 1])[:2]])
        if c["H"] is not None:
            xy = image_to_court(c["H"], e["xy_model"])[0]
            if np.isfinite(xy).all():
                e["court_xy"] = [round(float(v), 3) for v in xy]
    bounces = events

    (args.out / "bounces.json").write_text(json.dumps({
        "video": args.video.name, "fps": fps, "frames": len(calib), "ball_coverage": coverage,
        "court_valid_frames": n_valid, "bounce_params": params, "court_threshold": args.court_threshold,
        "court_axes": "metres; origin net centre, x right, y towards the far (top-of-image) baseline",
        "bounces": bounces}, indent=2))
    cv2.imwrite(str(args.out / "bounce_map.png"), draw_bounce_dots(draw_topdown(), bounces))

    if not args.no_video:
        print("pass 2: rendering")
        render(args.video, args.out / "rally.mp4", fps, calib, geom, bounces)
    for i, b in enumerate(bounces):
        xy = "-" if b["court_xy"] is None else f"x={b['court_xy'][0]:+.2f} y={b['court_xy'][1]:+.2f} m"
        print(f"#{i + 1:2d} {b['time_s']:6.2f}s  {xy}  {'ok' if b['valid'] else 'UNCERTAIN: ' + b['reason']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
