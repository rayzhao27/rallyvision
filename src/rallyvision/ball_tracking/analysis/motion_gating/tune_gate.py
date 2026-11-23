# Copied from local_archive/articrafts/ball_select_v2/tune_new.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Choose the new select_ball's gate parameters on the 7 train games (ball F1), same grid style as before."""
import itertools, json, sys
from pathlib import Path
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/ball_select_v2"   # inputs and results of this study (local archive)
HERE = DATA; REPO = REPO_ROOT
from rallyvision.ball_tracking.analysis.motion_gating.evaluate_ball_select import ACQUIRE, FPS, load_peaks, score
from rallyvision.ball_tracking.inference.ball_select import drop_static, select_ball
GRID = {"follow": [0.05, 0.1], "base_px": [5.0, 10.0], "speed_px_s": [600.0, 900.0, 1500.0, 2500.0]}
train = load_peaks(REPO / "local_archive/articrafts/ball_select/train_peaks.csv")
res = []
for vals in itertools.product(*GRID.values()):
    p = dict(zip(GRID, vals))
    s = score(train, lambda pk: drop_static(select_ball(pk, FPS, acquire=ACQUIRE, **p), FPS))
    res.append((s["f1"], p, s)); print(p, {k: round(v, 4) for k, v in s.items()})
best = max(res, key=lambda r: r[0])
print("chosen on train:", best[1])
(HERE / "tune_new.json").write_text(json.dumps({"chosen": best[1], "grid": [(r[1], r[2]) for r in res]}, indent=1))
