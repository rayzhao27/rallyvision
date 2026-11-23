#!/usr/bin/env python
"""Highest peak vs tracking among peaks (rallyvision.ball_select), on the TrackNet games.

    python scripts/evaluate_ball_select.py --train local_archive/articrafts/ball_select/train_peaks.csv \
        --val local_archive/articrafts/ball_select/val_peaks.csv --out local_archive/articrafts/ball_select

Inputs come from scripts/dump_ball_peaks.py. Ball metrics as in evaluation: a hit is
a detection within 10 px of a visible ball; a detection elsewhere is a false positive,
a visible ball without a hit a false negative. The tracking parameters are chosen on
the 7 train games only (ball F1); val is scored once. Both methods are then written
as *_per_sample.csv and passed through scripts/evaluate_bounce.py unchanged, so the
bounce numbers compare like for like. The clips are 30 fps, so the time-based input
frames make no difference here; the court gate needs a homography and is not used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rallyvision.common.config import REPO_ROOT  # noqa: E402
REPO = REPO_ROOT
from rallyvision.ball_tracking.inference.ball_select import drop_static, select_ball  # noqa: E402

FPS, TOL, ACQUIRE = 30.0, 10.0, 0.18
GRID = {"follow": [0.05, 0.1], "base_px": [5.0, 10.0], "speed_px_s": [600.0, 900.0, 1500.0]}


def load_peaks(path: Path):
    """{clip_id: (frame_files, peaks per frame, visible, gt [N, 2])}, frames in order."""
    df = pd.read_csv(path, keep_default_na=True)
    if df["x"].max() > 640 or df["y"].max() > 360:
        raise ValueError(f"{path}: coordinates exceed 640x360; ball_select pixel parameters assume model input pixels")
    df["frame"] = df["frame_file"].str.extract(r"(\d+)")[0].astype(int)
    clips = {}
    for clip_id, g in df.sort_values(["frame", "rank"]).groupby("clip_id"):
        frames = []
        for _, f in g.groupby("frame", sort=True):
            p = f[["x", "y", "score"]].dropna().to_numpy(float)
            frames.append((f["frame_file"].iloc[0], p, str(f["visible"].iloc[0]).lower() == "true",
                           (f["gt_x"].iloc[0], f["gt_y"].iloc[0])))
        files, peaks, vis, gt = zip(*frames)
        clips[clip_id] = (list(files), list(peaks), np.array(vis), np.array(gt, dtype=float))
    return clips


def highest(peaks):
    return [(p[0, 0], p[0, 1]) if len(p) and p[0, 2] >= ACQUIRE else None for p in peaks]


def tracked(params):
    return lambda peaks: drop_static(select_ball(peaks, FPS, acquire=ACQUIRE, **params), FPS)


def score(clips, method):
    tp = fp = fn = 0
    for _, peaks, vis, gt in clips.values():
        for p, v, g in zip(method(peaks), vis, gt):
            hit = p is not None and v and np.hypot(p[0] - g[0], p[1] - g[1]) <= TOL
            tp += hit
            fp += p is not None and not hit
            fn += v and not hit
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def write_per_sample(clips, method, path: Path):
    """The columns scripts/evaluate_bounce.py reads."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "frame_file", "visible", "gt_x", "gt_y", "pred_x", "pred_y"])
        for clip_id, (files, peaks, vis, gt) in clips.items():
            for name, p, v, g in zip(files, method(peaks), vis, gt):
                w.writerow([clip_id, name, bool(v), *(g if v else ("", "")), *(p if p is not None else ("", ""))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train, val = load_peaks(args.train), load_peaks(args.val)

    trials = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    best = max(trials, key=lambda prm: score(train, tracked(prm))["f1"])     # chosen on train only
    methods = {"highest": highest, "tracked": tracked(best)}

    lines = [f"# Ball selection on the TrackNet games\n", f"Tracking parameters chosen on train (ball F1): `{best}`\n",
             "| method | split | precision | recall | F1 |", "|---|---|---|---|---|"]
    for name, m in methods.items():
        for split, clips in (("train", train), ("val", val)):
            r = score(clips, m)
            lines.append(f"| {name} | {split} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
        csvs = []
        for split, clips in (("train", train), ("val", val)):
            csvs.append(args.out / f"{name}_{split}_per_sample.csv")
            write_per_sample(clips, m, csvs[-1])
        subprocess.run([sys.executable, "-m", "rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce", "--csv", *map(str, csvs),
                        "--out", str(args.out / f"bounce_{name}")], check=True, cwd=REPO)
    lines.append(f"\nBounce results per method: `bounce_highest/` and `bounce_tracked/` (scripts/evaluate_bounce.py).")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
