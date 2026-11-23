#!/usr/bin/env python
"""Choose the court keypoint threshold on val: the one with the highest keypoint F1.

    python scripts/sweep_court_threshold.py --data /content/court_all \
        --checkpoint runs/court_v2/best.pt --out runs/court_v2/sweep_val

The model runs once per image; every threshold is decoded from the same heatmaps.
Per threshold: keypoint precision / recall / F1 (hit = within --tol px of the
label, recall over every labeled point inside the image), homography valid rate,
and the error of all 14 points projected by H. Writes sweep.csv, sweep.png and
chosen.json. Val only: test is scored once afterwards with the chosen value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


from rallyvision.court_detection import (INPUT_H, INPUT_W, CourtDataset, decode_keypoints, fit_homography,  # noqa: E402
                               load_coco_split, load_court_model, split_by_video)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=5.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    samples = []
    for name in ("train", "valid", "test"):
        if (args.data / name / "_annotations.coco.json").is_file():
            samples += load_coco_split(args.data / name)
    ds = CourtDataset(split_by_video(samples)["val"], augment=False)
    if not len(ds):
        raise ValueError("empty val split")
    model, _ = load_court_model(args.checkpoint, args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    stats = {t: {"tp": 0, "fp": 0, "fn": 0, "valid": 0, "h_err": []} for t in thresholds}
    for i in range(len(ds)):
        x, gt, labeled = (t.numpy() for t in ds[i])
        with torch.no_grad():
            hm = model.predict_heatmap(torch.from_numpy(x)[None].to(args.device))[0].cpu().numpy()
        inside = labeled & (gt[:, 0] >= 0) & (gt[:, 0] < INPUT_W) & (gt[:, 1] >= 0) & (gt[:, 1] < INPUT_H)
        for t in thresholds:
            pts, _ = decode_keypoints(hm, t)
            found = ~np.isnan(pts[:, 0])
            hit = found & inside & (np.linalg.norm(pts - gt, axis=1) <= args.tol)
            s = stats[t]
            s["tp"] += int(hit.sum())
            s["fp"] += int((found & ~hit).sum())
            s["fn"] += int((inside & ~hit).sum())
            h = fit_homography(pts)
            if h["valid"]:
                s["valid"] += 1
                s["h_err"] += np.linalg.norm(h["points"] - gt, axis=1)[inside].tolist()

    rows = []
    for t in thresholds:
        s = stats[t]
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        e = np.asarray(s["h_err"])
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
                     "h_valid_rate": s["valid"] / len(ds),
                     "h_err_median_px": float(np.median(e)) if len(e) else float("nan"),
                     "h_err_p90_px": float(np.percentile(e, 90)) if len(e) else float("nan")})

    with open(args.out / "sweep.csv", "w") as f:
        f.write(",".join(rows[0]) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" for v in r.values()) + "\n")
    best = max(rows, key=lambda r: r["f1"])          # first maximum = lowest threshold on ties
    (args.out / "chosen.json").write_text(json.dumps({"rule": "max keypoint F1 on val", "images": len(ds), **best}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in ("precision", "recall", "f1", "h_valid_rate"):
        ax[0].plot(t, [r[k] for r in rows], marker=".", label=k)
    ax[0].axvline(best["threshold"], color="gray", ls="--")
    ax[0].set_xlabel("threshold"); ax[0].legend(); ax[0].set_title(f"val ({len(ds)} images)")
    ax[1].plot(t, [r["h_err_median_px"] for r in rows], marker=".", label="median")
    ax[1].plot(t, [r["h_err_p90_px"] for r in rows], marker=".", label="p90")
    ax[1].axvline(best["threshold"], color="gray", ls="--")
    ax[1].set_xlabel("threshold"); ax[1].set_ylabel("px (640x360)"); ax[1].legend(); ax[1].set_title("error of 14 points projected by H")
    fig.tight_layout(); fig.savefig(args.out / "sweep.png", dpi=120)

    print(f"{'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'H valid':>8} {'H med':>6} {'H p90':>6}")
    for r in rows:
        print(f"{r['threshold']:5.2f} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f} "
              f"{r['h_valid_rate']:8.3f} {r['h_err_median_px']:6.2f} {r['h_err_p90_px']:6.2f}"
              + ("  <- chosen" if r is best else ""))


if __name__ == "__main__":
    main()
