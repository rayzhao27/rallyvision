#!/usr/bin/env python
"""Evaluate the court keypoint model on the video-level val or test split.

    python scripts/evaluate_court.py --data /content/court_all --checkpoint runs/court_v1/best.pt \
        --split val --out runs/court_v1/eval_val

Errors are in pixels of the 640x360 model input. Reports:
  keypoints   - precision / recall / F1 at --tol px, and error of detected points
  homography  - share of images with a valid H, and the error of all 14 points
                projected by H (this includes points the model missed or that
                were hidden), which is what the bounce map depends on
Writes report.md, metrics.json and a few overlay images (green = label,
red = detection, blue = court projected by H).

Use --split test once, after everything has been chosen on val.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch


from rallyvision.court_detection import (INPUT_H, INPUT_W, CourtDataset, decode_keypoints, draw_court,  # noqa: E402
                               fit_homography, load_coco_split, load_court_model, split_by_video)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=5.0, help="hit radius in px (640x360)")
    ap.add_argument("--overlays", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = []
    for name in ("train", "valid", "test"):
        if (args.data / name / "_annotations.coco.json").is_file():
            samples += load_coco_split(args.data / name)
    samples = split_by_video(samples)[args.split]
    if not samples:
        raise ValueError(f"empty {args.split} split")
    ds = CourtDataset(samples, augment=False)
    model, _ = load_court_model(args.checkpoint, args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    tp = fp = fn = 0
    det_err, h_err, h_err_missed = [], [], []
    n_valid, reasons = 0, Counter()
    overlay_every = max(1, len(ds) // max(args.overlays, 1))

    for i in range(len(ds)):
        x, gt, labeled = (t.numpy() for t in ds[i])
        with torch.no_grad():
            hm = model.predict_heatmap(torch.from_numpy(x)[None].to(args.device))[0].cpu().numpy()
        pts, _ = decode_keypoints(hm, args.threshold)

        inside = labeled & (gt[:, 0] >= 0) & (gt[:, 0] < INPUT_W) & (gt[:, 1] >= 0) & (gt[:, 1] < INPUT_H)
        found = ~np.isnan(pts[:, 0])
        err = np.linalg.norm(pts - gt, axis=1)
        hit = found & inside & (err <= args.tol)
        tp += int(hit.sum())
        fp += int((found & ~hit).sum())
        fn += int((inside & ~hit).sum())
        det_err += err[found & inside].tolist()

        h = fit_homography(pts)
        if h["valid"]:
            n_valid += 1
            e = np.linalg.norm(h["points"] - gt, axis=1)
            h_err += e[inside].tolist()
            h_err_missed += e[inside & ~found].tolist()
        else:
            reasons[re.sub(r"[\d.]+", "N", h["reason"])] += 1     # "only 3 keypoints" -> "only N keypoints"

        if args.overlays and i % overlay_every == 0 and i // overlay_every < args.overlays:
            img = cv2.cvtColor((x.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if h["H"] is not None:
                draw_court(img, h["H"], (255, 128, 0) if h["valid"] else (128, 128, 128))
            for p in gt[inside]:
                cv2.circle(img, tuple(np.round(p).astype(int)), 4, (0, 200, 0), 1, cv2.LINE_AA)
            for p in pts[found]:
                cv2.circle(img, tuple(np.round(p).astype(int)), 2, (0, 0, 255), -1, cv2.LINE_AA)
            label = "valid" if h["valid"] else f"INVALID: {h['reason']}"
            cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"overlay_{i:04d}.jpg"), img)

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    m = {
        "split": args.split, "images": len(ds), "threshold": args.threshold, "tol_px": args.tol,
        "keypoints": {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": fn,
                      "err_median_px": pct(det_err, 50), "err_p90_px": pct(det_err, 90)},
        "homography": {"valid_rate": n_valid / len(ds), "valid": n_valid, "invalid_reasons": dict(reasons),
                       "err_median_px": pct(h_err, 50), "err_p90_px": pct(h_err, 90),
                       "missed_points_err_median_px": pct(h_err_missed, 50), "missed_points": len(h_err_missed)},
    }
    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))

    k, hh = m["keypoints"], m["homography"]
    report = f"""# Court keypoints — {args.split} ({len(ds)} images, threshold {args.threshold})

Pixels are in the 640x360 model input. A hit is a detection within {args.tol:g} px of its label;
recall counts every labeled point inside the image.

| keypoints | value |
|---|---|
| precision | {k['precision']:.4f} |
| recall | {k['recall']:.4f} |
| F1 | {k['f1']:.4f} |
| error of detected points, median / p90 | {k['err_median_px']:.2f} / {k['err_p90_px']:.2f} px |

| homography | value |
|---|---|
| valid images | {n_valid} / {len(ds)} ({hh['valid_rate']:.1%}) |
| all 14 points projected by H, median / p90 error | {hh['err_median_px']:.2f} / {hh['err_p90_px']:.2f} px |
| points the model missed, recovered by H, median error | {hh['missed_points_err_median_px']:.2f} px (n={hh['missed_points']}) |
| invalid reasons | {dict(reasons) or '-'} |
"""
    (args.out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
