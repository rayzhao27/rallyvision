# Copied from local_archive/articrafts/events/measure_candidates.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Stage 1: candidate recall on the 7 TrackNet train games (val untouched).

For bounces, first bounces and hits: share with a candidate within +-2 frames, and candidates per
100 frames. Generators: the current upward image-y kink, a 2D velocity-change peak, and their union.
"""
import itertools, sys
from pathlib import Path
import numpy as np
from rallyvision.common.config import REPO_ROOT  # noqa: E402
DATA = REPO_ROOT / "local_archive/articrafts/events"   # inputs and results of this study (local archive)
REPO = REPO_ROOT
import rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce as eb
from rallyvision.common.cli import load_config
from rallyvision.bounce_detection.experiments.gbdt.bounce import bounce_candidates, fill_short_gaps

_track_data = eb.track_data
def keep_xy(x, y, missing, frames, bounces):           # remember the raw track next to the features
    d = _track_data(x, y, missing, frames, bounces); d.update(x=np.asarray(x, float), y=np.asarray(y, float)); return d
eb.track_data = keep_xy

def velocity_change(x, y, w):
    """|v_after - v_before| (px/frame), mean velocities over w frames each side; NaN where undefined."""
    x, y = fill_short_gaps(x), fill_short_gaps(y)
    n = len(y); d = np.full(n, np.nan)
    for t in range(w, n - w):
        vb = np.array([x[t] - x[t - w], y[t] - y[t - w]]) / w
        va = np.array([x[t + w] - x[t], y[t + w] - y[t]]) / w
        d[t] = np.hypot(*(va - vb))
    return d

def peaks(d, thr, sep=3):
    """Local maxima of d above thr, at least sep frames apart (strongest kept)."""
    order = [t for t in np.argsort(-np.nan_to_num(d, nan=-1)) if np.isfinite(d[t]) and d[t] >= thr]
    keep = []
    for t in order:
        if all(abs(t - k) >= sep for k in keep): keep.append(t)
    return np.array(sorted(keep), dtype=int)

def recall(clips, gen, tol=2):
    hit = {k: [0, 0] for k in ("bounces", "first_bounces", "hits")}; n_c = n_f = 0
    for c in clips:
        cand = c["frames"][gen(c)] if len(gen(c)) else np.array([], int)
        n_c += len(cand); n_f += len(c["frames"])
        for k in hit:
            for e in c[k]:
                hit[k][0] += bool(len(cand)) and np.min(np.abs(cand - e)) <= tol; hit[k][1] += 1
    return {k: v[0] / max(v[1], 1) for k, v in hit.items()}, 100 * n_c / n_f

clips = eb.load_clips([REPO / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv",
                       REPO / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv"],
                      REPO / "local_archive/articrafts/bounce_inputs/labels", load_config())
train = [c for c in clips if c["game"] not in eb.VAL_GAMES]
print(f"train: {len(train)} clips, {sum(len(c['bounces']) for c in train)} bounces, {sum(len(c['hits']) for c in train)} hits\n")
print("| track | generator | bounce recall | first-bounce recall | hit recall | candidates / 100 frames |")
print("|---|---|---|---|---|---|")
for track in ("gt", "pred"):
    kink = lambda c: bounce_candidates(c[track]["y"])[0]
    rows = [("upward y-kink (current)", kink)]
    for w, thr in itertools.product((3, 5), (1.0, 2.0, 3.0)):
        vc = lambda c, w=w, thr=thr: peaks(velocity_change(c[track]["x"], c[track]["y"], w), thr)
        rows.append((f"velocity change w={w} >= {thr}", vc))
        rows.append((f"union kink + velocity w={w} >= {thr}",
                     lambda c, vc=vc: np.union1d(kink(c), vc(c)).astype(int)))
    for name, gen in rows:
        r, rate = recall(train, gen)
        print(f"| {track} | {name} | {r['bounces']:.3f} | {r['first_bounces']:.3f} | {r['hits']:.3f} | {rate:.1f} |")
