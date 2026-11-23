# Copied from local_archive/articrafts/ball_select_v2/compare.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Old vs new select_ball on the same saved heatmap peaks (hr1280_v1), same bounce stage as run_rally.

    python local_archive/articrafts/ball_select_v2/compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/ball_select_v2"   # inputs and results of this study (local archive)
HERE = DATA
REPO = REPO_ROOT
from rallyvision.common.cli import load_config                      # noqa: E402
from rallyvision.bounce_detection.experiments.gbdt.rally_gbdt import find_bounces                   # noqa: E402
from rallyvision.ball_tracking.inference.ball_select import drop_static, select_ball   # noqa: E402
from rallyvision.ball_tracking.experiments.legacy_ball_selection import ball_select as ball_select_old  # noqa: E402

THR = 0.16
CSV = [REPO / "local_archive/articrafts/ball_1280/hr1280_v1/eval640/train_per_sample.csv",
       REPO / "local_archive/articrafts/ball_1280/hr1280_v1/eval640/val_per_sample.csv"]
GT = {"sinner": [57, 163, 212, 308, 389, 456, 534, 605, 676, 764, 840, 950], "carlos": [11, 40, 70, 103, 133, 162]}
TOL_S = 2 / 30                                       # ±2 frames at 30 fps, in seconds


def score(frames, gt, tol):
    used, tp = set(), 0
    for g in gt:
        c = [i for i, f in enumerate(frames) if abs(f - g) <= tol and i not in used]
        if c:
            used.add(min(c, key=lambda i: abs(frames[i] - g)))
            tp += 1
    false = [frames[i] for i in range(len(frames)) if i not in used]
    missed = [g for g in gt if not any(abs(f - g) <= tol for f in frames)]
    return tp, false, missed


cfg = load_config()
report = {}
for clip in ("sinner", "carlos"):
    a = np.load(HERE / clip / "analysis.npz")
    fps, n = float(a["fps"]), len(a["ball"])
    rows = a["peaks"]
    peaks = [rows[rows[:, 0] == f, 1:] for f in range(n)]
    tol = int(round(TOL_S * fps))
    report[clip] = {}
    for name, fn in (("old", ball_select_old.select_ball), ("new", lambda p, fps, acquire: select_ball(p, fps, acquire=acquire, follow=0.1, base_px=10.0, speed_px_s=900.0))):
        balls = drop_static(fn(peaks, fps, acquire=THR), fps)
        events, params, _, cov = find_bounces(balls, fps, cfg, CSV)
        frames = [e["video_frame"] for e in events]
        tp, false, missed = score(frames, GT[clip], tol)
        report[clip][name] = {"balls": balls, "first_bounces": frames, "tp": tp, "false": false, "missed": missed,
                              "ball_frames": int(sum(b is not None for b in balls))}
    old, new = (report[clip][k]["balls"] for k in ("old", "new"))
    report[clip]["changed_frames"] = [f for f in range(n) if old[f] != new[f]]

for clip, r in report.items():
    print(f"\n== {clip}  (GT {len(GT[clip])} first bounces)")
    for k in ("old", "new"):
        x = r[k]
        print(f"  {k}: ball in {x['ball_frames']} frames; TP {x['tp']}  false {x['false']}  missed {x['missed']}")
        print(f"       first bounces {x['first_bounces']}")
    ch = r["changed_frames"]
    runs, s = [], None
    for f in ch + [None]:
        if s is not None and (f is None or f != prev + 1):
            runs.append((s, prev))
            s = None
        if f is not None and s is None:
            s = f
        prev = f
    print(f"  frames where the ball changed: {len(ch)} in runs {runs}")
    for k in ("old", "new"):
        r[k].pop("balls")
(HERE / "compare.json").write_text(json.dumps(report, indent=1))
