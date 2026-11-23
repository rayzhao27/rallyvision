# Copied from local_archive/articrafts/ball_select_v2/eval_tracknet.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Ball selection on the TrackNet games by ball speed, image region and game: highest peak, old select_ball, new.

    python local_archive/articrafts/ball_select_v2/eval_tracknet.py

Peaks: v3 (640x360) dumps in local_archive/articrafts/ball_select/ (scripts/dump_ball_peaks.py), 30 fps clips.
Old and new use the tracking parameters chosen on train for the old version (ball_select/report.md);
the new version's extra parameters are its defaults. Hit = within 10 px of a visible ball.
Per stratum, every frame is put in the stratum of its true ball (speed from consecutive labels):
recall = hits / visible frames; wrong = detections more than 10 px from a visible ball / visible frames.
"""
import json
import sys
from pathlib import Path

import numpy as np

from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/ball_select_v2"   # inputs and results of this study (local archive)
HERE = DATA
REPO = REPO_ROOT
from rallyvision.ball_tracking.analysis.motion_gating.evaluate_ball_select import ACQUIRE, FPS, TOL, highest, load_peaks, write_per_sample  # noqa: E402
from rallyvision.ball_tracking.inference.ball_select import drop_static, select_ball                              # noqa: E402
from rallyvision.ball_tracking.experiments.legacy_ball_selection import ball_select as ball_select_old  # noqa: E402

PARAMS = {"follow": 0.1, "base_px": 5.0, "speed_px_s": 600.0}
NEW_PARAMS = {"follow": 0.1, "base_px": 10.0, "speed_px_s": 900.0}
METHODS = {
    "highest": highest,
    "old": lambda p: drop_static(ball_select_old.select_ball(p, FPS, acquire=ACQUIRE, **PARAMS), FPS),
    "new": lambda p: drop_static(select_ball(p, FPS, acquire=ACQUIRE, **PARAMS), FPS),
    "new_tuned": lambda p: drop_static(select_ball(p, FPS, acquire=ACQUIRE, **NEW_PARAMS), FPS),   # tune_new.py
}
SPEED_BINS = [0, 60, 200, 500, np.inf]          # px/s at 640x360 (60 px/s = 2 px per frame at 30 fps)
FAR_Y = 170                                     # image y above this: far half (approximate)


def rows(clips, method):
    """One row per frame: game, visible, speed bin, far, hit, wrong, detected."""
    out = []
    for clip_id, (_, peaks, vis, gt) in clips.items():
        game = clip_id.split("/")[-2]
        sp = np.full(len(gt), np.nan)
        sp[1:] = np.hypot(*np.diff(gt, axis=0).T) * FPS
        sp[0] = sp[1] if len(gt) > 1 else np.nan
        for p, v, g, s in zip(method(peaks), vis, gt, sp):
            hit = bool(p is not None and v and np.hypot(p[0] - g[0], p[1] - g[1]) <= TOL)
            out.append((game, bool(v), np.digitize(s, SPEED_BINS) - 1 if np.isfinite(s) else -1,
                        bool(v and g[1] < FAR_Y), hit, p is not None and not hit, p is not None))
    return out


def summary(r):
    vis = [x for x in r if x[1]]
    tp, det = sum(x[4] for x in r), sum(x[6] for x in r)
    p, rc = tp / max(det, 1), tp / max(len(vis), 1)
    return {"P": round(p, 4), "R": round(rc, 4), "F1": round(2 * p * rc / max(p + rc, 1e-9), 4)}


def stratum(r, keep):
    vis = [x for x in r if x[1] and keep(x)]
    return {"n": len(vis), "recall": round(sum(x[4] for x in vis) / max(len(vis), 1), 4),
            "wrong": round(sum(x[5] for x in vis) / max(len(vis), 1), 4)}


result = {}
for split in ("train", "val"):
    clips = load_peaks(REPO / f"local_archive/articrafts/ball_select/{split}_peaks.csv")
    result[split] = {}
    for name, m in METHODS.items():
        r = rows(clips, m)
        result[split][name] = {
            "overall": summary(r),
            "speed": {f"{SPEED_BINS[b]}-{SPEED_BINS[b + 1]}": stratum(r, lambda x, b=b: x[2] == b) for b in range(4)},
            "region": {"far": stratum(r, lambda x: x[3]), "near": stratum(r, lambda x: not x[3])},
            "game": {g: stratum(r, lambda x, g=g: x[0] == g) for g in sorted({x[0] for x in r})},
            "invisible_frames_with_detection": sum(x[6] for x in r if not x[1]),
        }
    if split == "val" or split == "train":
        write_per_sample(clips, METHODS["new_tuned"], HERE / f"new_{split}_per_sample.csv")

(HERE / "eval_tracknet.json").write_text(json.dumps(result, indent=1))
for split, res in result.items():
    print(f"\n===== {split}")
    for name, x in res.items():
        print(f"{name:8s} {x['overall']}  invisible-frame detections {x['invisible_frames_with_detection']}")
    for key in ("speed", "region", "game"):
        print(f"  -- by {key}: recall / wrong-object rate (n visible)")
        for k in res["new"][key]:
            print(f"  {k:10s} " + "   ".join(f"{name} {res[name][key][k]['recall']:.3f}/{res[name][key][k]['wrong']:.3f}"
                                            for name in res) + f"   (n {res['new'][key][k]['n']})")
