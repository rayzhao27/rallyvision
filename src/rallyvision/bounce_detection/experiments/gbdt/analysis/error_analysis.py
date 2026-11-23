# Copied from local_archive/articrafts/bounce/analysis/error_analysis.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Error analysis + timing accuracy for the bounce classifier on predicted tracks.

Uses nested out-of-fold detections on the train games (each game scored by a model
fit on the other six), so nothing here touches val.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/bounce/analysis"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import (VAL_GAMES, TOL, by_game, detections, fit_gbdt, load_clips,  # noqa: E402
                             match_pairs, prf, REPO_ROOT)
from rallyvision.common.cli import load_config  # noqa: E402

csv = REPO_ROOT / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv"
clips = load_clips([csv, REPO_ROOT / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv"],
                   REPO_ROOT / "local_archive/articrafts/bounce_inputs/labels", load_config())
train = [c for c in clips if c["game"] not in VAL_GAMES]

df = pd.read_csv(csv)
for col in ("visible", "detected"):
    df[col] = df[col].astype(str).str.lower().eq("true")
df["frame"] = df["frame_file"].str.extract(r"(\d+)")[0].astype(int)
df["good"] = df["detected"] & (df["error_px"] <= 10)
per_clip = {cid: g.set_index("frame") for cid, g in df.groupby("clip_id")}

fn_cat, fp_cat, offsets, speeds = Counter(), Counter(), [], []
game_rows = []
for game, rest, held in by_game(train):
    scorer, t = fit_gbdt(rest, "pred")
    g_tp = g_fp = g_fn = 0
    for c in held:
        s = scorer(c)
        dets = detections(c, "pred", s, t)
        pairs = match_pairs(dets, c["bounces"], TOL)
        mp, mg = {i for i, _ in pairs}, {j for _, j in pairs}
        g_tp, g_fp, g_fn = g_tp + len(pairs), g_fp + len(dets) - len(pairs), g_fn + len(c["bounces"]) - len(pairs)
        cand = c["frames"][c["pred"]["idx"]]
        first, last = c["frames"][0], c["frames"][-1]
        P = per_clip[c["clip"]]
        for i, j in pairs:
            offsets.append(dets[i] - c["bounces"][j])
        for j, b in enumerate(c["bounces"]):
            win = P.loc[max(first, b - 2): min(last, b + 2)]
            if 0 < b - first and last - b > 0 and b - 1 in P.index and b + 1 in P.index:
                p0, p1 = P.loc[b - 1, ["gt_x", "gt_y"]].to_numpy(float), P.loc[b + 1, ["gt_x", "gt_y"]].to_numpy(float)
                if np.all(np.isfinite([*p0, *p1])):
                    speeds.append(float(np.hypot(*(p1 - p0)) / 2 * 2))  # px/frame in 1280x720
            if j in mg:
                continue
            if b - first < 5 or last - b < 5:
                fn_cat["within 5 frames of clip edge"] += 1
            elif not np.any(np.abs(cand - b) <= TOL):
                fn_cat["no kink candidate (track too flat/noisy)"] += 1
            elif win["good"].mean() < 0.6:
                fn_cat["ball detector lost the ball around the bounce"] += 1
            else:
                fn_cat["candidate scored below threshold"] += 1
        for i, d in enumerate(dets):
            if i in mp:
                continue
            if any(abs(d - h) <= TOL for h in c["hits"]):
                fp_cat["at a hit"] += 1
            elif any(abs(d - b) <= 6 for b in c["bounces"]):
                fp_cat["near a real bounce (duplicate / off by 3-6 frames)"] += 1
            elif d - first < 5 or last - d < 5:
                fp_cat["within 5 frames of clip edge"] += 1
            else:
                fp_cat["in flight (track noise / gap)"] += 1
    gc = [c for c in held]
    vis = df[df["clip_id"].isin({c["clip"] for c in gc}) & df["visible"]]
    bounce_frames = [(c["clip"], b) for c in gc for b in c["bounces"]]
    bgood = np.mean([per_clip[cid].loc[b, "good"] for cid, b in bounce_frames if b in per_clip[cid].index])
    game_rows.append({"game": game, "clips": len(gc), "bounces": len(bounce_frames),
                      "f1": round(prf(g_tp, g_fp, g_fn)[2], 3), "threshold": t,
                      "ball_detect_rate_all": round(vis["good"].mean(), 3),
                      "ball_detect_rate_at_bounce": round(float(bgood), 3),
                      "candidates_per_bounce": round(sum(len(c["pred"]["idx"]) for c in gc) / max(1, len(bounce_frames)), 1)})

res = {"fn_categories": dict(fn_cat), "fp_categories": dict(fp_cat),
       "offset_counts": dict(sorted(Counter(offsets).items())),
       "exact_frame_share": float(np.mean(np.array(offsets) == 0)),
       "ball_speed_at_bounce_px_per_frame_720p": {q: float(np.percentile(speeds, p)) for q, p in
                                                  (("p25", 25), ("median", 50), ("p75", 75), ("p90", 90))},
       "per_game": game_rows}
DATA.joinpath("error_analysis.json").write_text(json.dumps(res, indent=2, default=int))
print(json.dumps(res, indent=2, default=int))
