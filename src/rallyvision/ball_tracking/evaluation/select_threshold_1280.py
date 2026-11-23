# Copied from local_archive/articrafts/ball_1280/threshold_and_rescale.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Pick the 1280x720 model's detection threshold on the train games, then write 640x360 CSVs for evaluate_bounce.py.

A frame counts as detected when its peak score >= t. Hit = within 20 px at 1280 (= 10 px at 640, the v3
tolerance). Precision = hits / detections; recall = hits / (hits + visible frames without a detection),
the same definitions as the training evaluator. Val is scored once, at the train-chosen threshold.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

run = Path(sys.argv[1])
TOL = 20.0

def outcome(df, t):
    det = df.score >= t
    hit = det & df.visible & (df.error_px <= TOL)
    fn = df.visible & ~det
    return det, hit, fn

def prf(df, t):
    det, hit, fn = outcome(df, t)
    p = hit.sum() / max(det.sum(), 1); r = hit.sum() / (hit.sum() + fn.sum())
    return p, r, 2 * p * r / (p + r)

train = pd.read_csv(run / "train_t0_per_sample.csv"); val = pd.read_csv(run / "val_t0_per_sample.csv")
grid = np.round(np.arange(0.02, 0.91, 0.02), 2)
f1 = [prf(train, t)[2] for t in grid]
t = float(grid[int(np.argmax(f1))])
print(f"threshold chosen on train: {t}")
for name, df in (("train", train), ("val", val)):
    print(name, "P %.4f R %.4f F1 %.4f" % prf(df, t))

for name, df in (("train", train), ("val", val)):
    det, hit, fn = outcome(df, t)
    out = df.copy()
    for c in ("gt_x", "gt_y", "pred_x", "pred_y", "error_px"):
        out[c] = out[c] / 2
    out["detected"] = det
    out.loc[~det, ["pred_x", "pred_y", "error_px"]] = np.nan
    out["outcome"] = np.select([hit, det, fn], ["tp", "fp", "fn"], "tn")
    out.to_csv(run / "eval640" / f"{name}_per_sample.csv", index=False)
