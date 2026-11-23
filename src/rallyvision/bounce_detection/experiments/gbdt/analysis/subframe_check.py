# Copied from local_archive/articrafts/bounce/analysis/subframe_check.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""A4: on real TrackNet tracks (train games only, val untouched)
1) estimate per-frame position noise σ from trajectory smoothness (predicted vs labelled tracks),
2) compare bounce timing of the detected kink frame vs subframe_bounce() against the labels.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from rallyvision.common.config import REPO_ROOT  # noqa: E402
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/bounce/analysis"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.bounce import bounce_candidates, fill_short_gaps, subframe_bounce  # noqa: E402
from rallyvision.ball_tracking.data.labels import parse_label_csv  # noqa: E402

df = pd.read_csv(REPO / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv")
df["frame"] = df["frame_file"].str.extract(r"(\d+)")[0].astype(int)
labels_dir = REPO / "local_archive/articrafts/bounce_inputs/labels"
SCALE = 2.0  # 640x360 -> 1280x720

resid = {"pred": [], "gt": []}
off_frame, off_sub = [], []
for cid, g in df.sort_values("frame").groupby("clip_id"):
    frames = g["frame"].to_numpy()
    status = {int(Path(r["file"]).stem): r["status"] for r in parse_label_csv(labels_dir / cid / "Label.csv")}
    events = [i for i, f in enumerate(frames) if status.get(f) in (1, 2)]
    bounces = [i for i, f in enumerate(frames) if status.get(f) == 2 and status.get(f - 1) != 2]
    tracks = {"pred": (g["pred_x"].to_numpy(float), g["pred_y"].to_numpy(float)),
              "gt": (g["gt_x"].to_numpy(float), g["gt_y"].to_numpy(float))}
    # 1) residual of a local quadratic on 7-frame in-flight windows (no event within 4 frames, no gaps)
    for name, (x, y) in tracks.items():
        for s in range(0, len(frames) - 7):
            w = np.arange(s, s + 7)
            if any(abs(e - c) <= 4 for e in events for c in (w[0], w[-1])) or any(w[0] - 4 <= e <= w[-1] + 4 for e in events):
                continue
            if not (np.all(np.isfinite(x[w])) and np.all(np.isfinite(y[w]))):
                continue
            for v in (x[w], y[w]):
                resid[name].append(np.std(v - np.polyval(np.polyfit(w, v, 2), w), ddof=3))
    # 2) timing on predicted tracks
    px, py = fill_short_gaps(tracks["pred"][0]), fill_short_gaps(tracks["pred"][1])
    idx, strength = bounce_candidates(py)
    for b in bounces:
        near = np.abs(idx - b) <= 2
        if not near.any():
            continue
        t0 = int(idx[near][np.argmax(strength[near])])
        s = subframe_bounce(px, py, t0)
        if s is None:
            continue
        off_frame.append(t0 - b)
        off_sub.append(s[0] - b)

sigma = {k: float(np.median(v)) * SCALE for k, v in resid.items()}  # median per-axis residual σ, 720p px
of, osb = np.array(off_frame), np.array(off_sub)
res = {
    "per_axis_sigma_720p_px": sigma,
    "n_windows": {k: len(v) for k, v in resid.items()},
    "n_bounces": len(of),
    "frame_offset": {"mean": float(of.mean()), "counts": {int(k): int(v) for k, v in zip(*np.unique(of, return_counts=True))}},
    "subframe_offset": {"mean": float(osb.mean()), "median": float(np.median(osb)),
                        "share_in_[-1,0]": float(np.mean((osb >= -1) & (osb <= 0))),
                        "share_in_[-0.5,0.5]": float(np.mean(np.abs(osb) <= 0.5)),
                        "hist_0.25": {f"{a:.2f}": int(n) for a, n in zip(np.arange(-2.5, 2.5, 0.25),
                                      np.histogram(osb, bins=np.arange(-2.5, 2.75, 0.25))[0])}},
}
DATA.joinpath("subframe_check.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
