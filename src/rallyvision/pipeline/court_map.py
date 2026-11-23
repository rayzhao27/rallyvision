#!/usr/bin/env python
"""Put detected bounces on a top-down court: per-frame court calibration + homography.

    python scripts/map_bounces.py --video ~/Downloads/carlos.MOV \
        --bounces local_archive/articrafts/bounce/demo_carlos/bounces.json \
        --checkpoint local_archive/articrafts/court/court_v2/best.pt --out local_archive/articrafts/court/demo_carlos

--bounces is the output of the ball + bounce stages (bounce pixels in the original
video, sub-frame time, video frame). The court model runs on every frame: frames
are pillar-boxed to 16:9 exactly as for the ball detector, keypoints are decoded,
and a homography court metres -> image is fitted and checked (rallyvision.court).
A bounce is mapped with the H of its own frame; when that H is invalid (cut-away,
close-up, too few keypoints) the bounce is reported as uncertain, not guessed.

The bounce pixel is the ball centre at contact, so the mapped point sits about one
ball radius (3.3 cm) above the ground; the resulting shift is a few cm and is not
corrected here.

Writes: bounces_court.json, bounce_map.png (top-down court), rectified.jpg (each
bounce frame warped to the top-down view, a visual check of H) and overlay.mp4
(court lines + bounces + a live mini-map).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


from rallyvision.court_detection import (COURT_LINES, COURT_POINTS, INPUT_W, decode_keypoints, draw_court,  # noqa: E402
                               fit_homography, image_to_court, load_court_model, preprocess, project)

# Top-down canvas: metres -> pixels, far baseline at the top.
PX_PER_M, MARGIN_M = 30, 3.0
HALF_W, HALF_L, NET_HALF = 5.485, 11.885, 5.485 + 0.914
CANVAS_W = int(round((2 * HALF_W + 2 * MARGIN_M) * PX_PER_M))
CANVAS_H = int(round((2 * HALF_L + 2 * MARGIN_M) * PX_PER_M))
COURT_TO_CANVAS = np.array([[PX_PER_M, 0, (HALF_W + MARGIN_M) * PX_PER_M],
                            [0, -PX_PER_M, (HALF_L + MARGIN_M) * PX_PER_M],
                            [0, 0, 1]], dtype=np.float64)


def to_canvas(xy) -> tuple:
    u, v = project(COURT_TO_CANVAS, np.asarray(xy, dtype=np.float64).reshape(1, 2))[0]
    return int(round(u)), int(round(v))


def draw_topdown(img=None) -> np.ndarray:
    """Standard court seen from above (white lines on green), with the net."""
    if img is None:
        img = np.full((CANVAS_H, CANVAS_W, 3), (70, 120, 60), np.uint8)
        cv2.rectangle(img, to_canvas((-HALF_W, HALF_L)), to_canvas((HALF_W, -HALF_L)), (150, 95, 45), -1)
    for a, b in COURT_LINES:
        cv2.line(img, to_canvas(COURT_POINTS[a]), to_canvas(COURT_POINTS[b]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, to_canvas((-NET_HALF, 0)), to_canvas((NET_HALF, 0)), (220, 220, 220), 1, cv2.LINE_AA)
    return img


def draw_bounce_dots(img, bounces, upto_frame=None):
    for i, b in enumerate(bounces):
        if upto_frame is not None and b["video_frame"] > upto_frame:
            continue
        if b["court_xy"] is None:
            continue
        p = to_canvas(b["court_xy"])
        if b["valid"]:
            cv2.circle(img, p, 7, (0, 230, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, p, 7, (160, 160, 160), 2, cv2.LINE_AA)
        cv2.putText(img, str(i + 1), (p[0] + 9, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--bounces", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.2, help="keypoint threshold (chosen on val)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    model, _ = load_court_model(args.checkpoint, args.device)
    events = json.loads(args.bounces.read_text())["bounces"]
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- pass 1: court model on every frame
    frames, calib = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        pad_x = max(0, (int(round(h * 16 / 9)) - w) // 2)      # same pillar-box as the ball detector
        padded = cv2.copyMakeBorder(frame, 0, 0, pad_x, pad_x, cv2.BORDER_CONSTANT)
        with torch.no_grad():
            hm = model.predict_heatmap(preprocess(padded)[None].to(args.device))[0].cpu().numpy()
        pts, _ = decode_keypoints(hm, args.threshold)
        r = fit_homography(pts)
        if r["H"] is not None:
            s = INPUT_W / padded.shape[1]
            # court -> original pixels: model px -> padded px (/s) -> original px (-pad_x)
            r["H_orig"] = np.array([[1, 0, -pad_x], [0, 1, 0], [0, 0, 1]]) @ np.diag([1 / s, 1 / s, 1]) @ r["H"]
        r["n_kps"] = int((~np.isnan(pts[:, 0])).sum())
        frames.append(frame)
        calib.append(r)
    cap.release()
    n = len(frames)
    if n == 0:
        raise ValueError("no frames read")
    n_valid = sum(c["valid"] for c in calib)
    print(f"{n} frames, court valid in {n_valid} ({n_valid / n:.1%})")

    # ---- map bounces with the H of their own frame
    bounces = []
    for e in events:
        f = min(max(int(e["video_frame"]), 0), n - 1)
        c = calib[f]
        b = {**e, "valid": bool(c["valid"]), "reason": c["reason"], "court_xy": None, "cm_per_px": None}
        if c.get("H_orig") is not None:
            xy = image_to_court(c["H_orig"], [e["x"], e["y"]])[0]
            if np.isfinite(xy).all():
                b["court_xy"] = [round(float(v), 3) for v in xy]
                # sensitivity: how far on court a 1-pixel error moves the point (larger = less precise)
                d = image_to_court(c["H_orig"], [[e["x"] + 1, e["y"]], [e["x"], e["y"] + 1]]) - xy
                b["cm_per_px"] = [round(float(np.linalg.norm(d[0])) * 100, 1), round(float(np.linalg.norm(d[1])) * 100, 1)]
        bounces.append(b)
    out = {"video": str(args.video.name), "fps": fps, "frames": n, "court_valid_frames": n_valid,
           "threshold": args.threshold,
           "court_axes": "metres; origin net centre, x right, y towards the far (top-of-image) baseline",
           "bounces": bounces}
    (args.out / "bounces_court.json").write_text(json.dumps(out, indent=2))

    # ---- top-down bounce map
    img = draw_bounce_dots(draw_topdown(), bounces)
    cv2.putText(img, "far", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "near (camera)", (8, CANVAS_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "bounce_map.png"), img)

    # ---- rectified bounce frames: the camera image warped to the top-down view
    tiles = []
    for i, b in enumerate(bounces):
        c = calib[min(max(int(b["video_frame"]), 0), n - 1)]
        if c.get("H_orig") is None:
            continue
        warp = COURT_TO_CANVAS @ np.linalg.inv(c["H_orig"])
        tile = cv2.warpPerspective(frames[min(max(int(b["video_frame"]), 0), n - 1)], warp, (CANVAS_W, CANVAS_H))
        draw_topdown(tile)
        if b["court_xy"] is not None:
            cv2.circle(tile, to_canvas(b["court_xy"]), 7, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"#{i + 1} {b['time_s']:.2f}s" + ("" if b["valid"] else " UNCERTAIN"), (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        cv2.imwrite(str(args.out / "rectified.jpg"), np.hstack(tiles))

    # ---- overlay video: court lines, bounce markers, live mini-map
    h, w = frames[0].shape[:2]
    mini_h = h // 2
    mini_w = int(round(CANVAS_W * mini_h / CANVAS_H))
    writer = cv2.VideoWriter(str(args.out / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    base_map = draw_topdown()
    for f, (frame, c) in enumerate(zip(frames, calib)):
        img = frame.copy()
        if c["valid"]:
            draw_court(img, c["H_orig"], (255, 160, 0), 2)
        else:
            cv2.putText(img, f"court: INVALID ({c['reason']})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        for i, b in enumerate(bounces):
            if 0 <= f - b["video_frame"] < int(fps):        # show each bounce for one second
                cv2.circle(img, (int(b["x"]), int(b["y"])), 12, (0, 230, 255) if b["valid"] else (160, 160, 160), 2, cv2.LINE_AA)
                cv2.putText(img, str(i + 1), (int(b["x"]) + 14, int(b["y"]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
        mini = cv2.resize(draw_bounce_dots(base_map.copy(), bounces, upto_frame=f), (mini_w, mini_h))
        img[h - mini_h - 10:h - 10, w - mini_w - 10:w - 10] = mini
        writer.write(img)
    writer.release()

    for i, b in enumerate(bounces):
        xy = "-" if b["court_xy"] is None else f"x={b['court_xy'][0]:+.2f} m, y={b['court_xy'][1]:+.2f} m"
        print(f"#{i + 1} {b['time_s']:5.2f}s frame {b['video_frame']:3d}  {xy}  "
              f"{'ok' if b['valid'] else 'UNCERTAIN: ' + b['reason']}  1px≈{b['cm_per_px']} cm")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
