"""Event spotting metrics: one-to-one matching within a frame tolerance, P/R/F1, val-only decoding selection."""

from __future__ import annotations

import numpy as np

from rallyvision.bounce_detection.inference.decode import peaks

GROUPS = {"far_bounce": [1], "near_bounce": [4], "bounce": [1, 4], "far_swing": [2], "near_swing": [5],
          "swing": [2, 5], "far_serve": [3], "near_serve": [6], "serve": [3, 6]}
THR = np.round(np.arange(0.05, 0.96, 0.05), 2)


def match(pred, gt, tol):
    pairs = sorted((abs(a - b), i, j) for i, a in enumerate(pred) for j, b in enumerate(gt) if abs(a - b) <= tol)
    up, ug, out = set(), set(), []
    for _, i, j in pairs:
        if i not in up and j not in ug:
            up.add(i)
            ug.add(j)
            out.append((i, j))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


SEP_GRID = (3, 5, 7, 9, 11, 13, 15)        # frames; temporal NMS windows tried on val
BASE_SEP = 5                               # the fixed window of v1 / v2


def group_peaks(clips, probs, group, sep=BASE_SEP):
    return [peaks(p[:, GROUPS[group]].sum(1), sep) for p in probs]


def score_group(clips, pk, group, thr, tol):
    tp = fp = fn = 0
    errs = []
    for c, k in zip(clips, pk):
        pred = [t for t, s in k if s >= thr]
        gt = [f for f, cl in zip(c["ev_frame"], c["ev_cls"]) if cl in GROUPS[group]]
        m = match(pred, gt, tol)
        errs += [pred[i] - gt[j] for i, j in m]
        tp, fp, fn = tp + len(m), fp + len(pred) - len(m), fn + len(gt) - len(m)
    p, r, f1 = prf(tp, fp, fn)
    e = np.abs(errs) if errs else np.array([np.nan])
    return {"P": p, "R": r, "F1": f1, "tp": tp, "fp": fp, "fn": fn,
            "timing_err_mean": float(np.mean(e)), "timing_err_median": float(np.median(e)),
            "timing_bias_mean": float(np.mean(errs)) if errs else float("nan")}


def choose_decoding(clips, probs, tol, seps=SEP_GRID):
    """Per event group: NMS window and threshold with the best F1 on `clips` (ties: smaller window, lower threshold)."""
    out = {}
    for g in GROUPS:
        best = None
        for sep in seps:
            pk = group_peaks(clips, probs, g, sep)
            for t in THR:
                key = (score_group(clips, pk, g, t, tol)["F1"], -sep, -t)
                if best is None or key > best[0]:
                    best = (key, float(t), int(sep))
        out[g] = {"thr": best[1], "sep": best[2]}
    return out


def evaluate(clips, probs, dec, tols=(4, 2, 1)):
    res = {}
    for g in GROUPS:
        pk = group_peaks(clips, probs, g, dec[g]["sep"])
        res[g] = {f"tol{t}": score_group(clips, pk, g, dec[g]["thr"], t) for t in tols}
    return res


def close_pairs(clips, group, sep):
    """Labelled events of the group closer than `sep` frames to the previous one (an NMS window could merge them)."""
    n = tot = 0
    for c in clips:
        f = sorted(fr for fr, cl in zip(c["ev_frame"], c["ev_cls"]) if cl in GROUPS[group])
        tot += len(f)
        n += sum(1 for a, b in zip(f, f[1:]) if b - a < sep)
    return n, tot


def nms_diff(clips, probs, group, dec_a, dec_b, tol=4):
    """Labelled events found only with decoding a (lost by b) / only with b, and the change in false detections."""
    out = {"lost": 0, "gained": 0, "fp_removed": 0, "fp_added": 0}
    for c, p in zip(clips, probs):
        gt = [fr for fr, cl in zip(c["ev_frame"], c["ev_cls"]) if cl in GROUPS[group]]
        res = []
        for d in (dec_a, dec_b):
            pr = [t for t, sc in peaks(p[:, GROUPS[group]].sum(1), d["sep"]) if sc >= d["thr"]]
            m = match(pr, gt, tol)
            res.append(({j for _, j in m}, len(pr) - len(m)))
        (ga, fa), (gb, fb) = res
        out["lost"] += len(ga - gb)
        out["gained"] += len(gb - ga)
        out["fp_removed"] += max(fa - fb, 0)
        out["fp_added"] += max(fb - fa, 0)
    return out
