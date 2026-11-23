# Copied from local_archive/articrafts/rally/oracle/run_oracle.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Oracle: correct the ball centre near each true bounce (read off frame crops), rerun the deployed bounce stage."""
import sys, json, numpy as np
from rallyvision.bounce_detection.experiments.gbdt.rally_gbdt import find_bounces
from rallyvision.common.cli import load_config
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/rally/oracle"   # inputs and results of this study (local archive)
S = sys.argv[1]
# (window to clear, {frame: (x, y)} true positions in 640x360 model pixels); frames between labels are interpolated
FIX = {
  "sinner": [((218, 236), {219: (285, 91.5), 221: (284.5, 90.8), 223: (284, 90.8), 225: (283.5, 90.8), 227: (282.8, 91.3),
                           229: (282.5, 92.5), 231: (281.8, 93.3), 232: (282.3, 94.3), 233: (282, 95), 234: (281.5, 95.3),
                           237: (280.8, 99.3)}),
             ((840, 857), {841: (266, 98.3), 843: (267.8, 97.8), 845: (268.5, 96), 847: (269, 95.8), 851: (269.3, 95.5),
                           857: (271, 96.5)}),
             ((930, 950), {931: (372.8, 122.5), 933: (373.5, 122.5), 935: (374.8, 123), 937: (375.3, 123.8), 939: (376, 124.3),
                           941: (376.3, 125), 943: (376.8, 126), 945: (377.3, 127), 947: (377.8, 128.3), 949: (378, 130.8)})],
  "carlos": [((162, 173), {162: (241, 238.8), 163: (237.5, 240), 164: (228, 243.8), 165: (217.5, 248.8), 166: (203.8, 257.5),
                           167: (192.5, 265), 168: (183.8, 271.8)})],
}
GT = {"sinner": [57, 163, 212, 308, 389, 456, 534, 605, 676, 764, 840, 950], "carlos": [11, 40, 70, 103, 133, 162]}
TOL = {"sinner": 4, "carlos": 2}
cfg = load_config()

def score(clip, det):
    used, tp = set(), 0
    for g in GT[clip]:
        m = [d for d in det if abs(d - g) <= TOL[clip] and d not in used]
        if m: used.add(min(m, key=lambda d: abs(d - g))); tp += 1
    return tp, len(det) - tp, len(GT[clip]) - tp, [g for g in GT[clip] if not any(abs(d - g) <= TOL[clip] for d in det)]

tot = {}
for clip, base in (("sinner", f"{S}/sinner_legacy"), ("carlos", f"{S}/carlos_legacy")):
    d = np.load(base + "/analysis.npz"); ball, fps = d["ball"].copy(), float(d["fps"])
    oracle = ball.copy()
    for (a, b), labels in FIX[clip]:
        oracle[a:b + 1] = np.nan
        fr = sorted(labels); xy = np.array([labels[f] for f in fr])
        t = np.arange(fr[0], fr[-1] + 1)
        oracle[t] = np.c_[np.interp(t, fr, xy[:, 0]), np.interp(t, fr, xy[:, 1])]
    for name, tr in (("detected", ball), ("oracle", oracle)):
        pts = [None if np.isnan(p[0]) else (float(p[0]), float(p[1])) for p in tr]
        ev, _, _, _ = find_bounces(pts, fps, cfg)
        det = [e["video_frame"] for e in ev]
        tp, fp, fn, missed = score(clip, det)
        t = tot.setdefault(name, [0, 0, 0]); t[0] += tp; t[1] += fp; t[2] += fn
        print(f"{clip:7s} {name:9s} TP {tp:2d} FP {fp} FN {fn}  missed {missed}  detected {det}")
for name, (tp, fp, fn) in tot.items():
    print(f"TOTAL {name:9s} TP {tp} FP {fp} FN {fn}  P {tp/(tp+fp):.2f} R {tp/(tp+fn):.2f} F1 {2*tp/(2*tp+fp+fn):.2f}")
