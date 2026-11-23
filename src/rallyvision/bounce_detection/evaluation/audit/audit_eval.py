# Copied from local_archive/articrafts/event_transformer/audit/audit_eval.py (module-first refactor, 2026-11-22); imports and data paths updated only.
"""Independent re-computation of the v3 test metrics from exported per-frame probabilities.

Written separately from scripts/train_event_transformer_e2e.py: own peak picking (suppression mask), own
one-to-one matching (optimal assignment, scipy.optimize.linear_sum_assignment), own val-only selection.

    python audit_eval.py probs_dir    # val.npz, test.npz, sinner.npz from the Colab export cell
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CLASSES = ["far_court_bounce", "far_court_swing", "far_court_serve", "near_court_bounce", "near_court_swing", "near_court_serve"]
GROUPS = {"bounce": [1, 4], "far_bounce": [1], "near_bounce": [4], "swing": [2, 5], "serve": [3, 6]}
THR = np.round(np.arange(0.05, 0.96, 0.05), 2)
SEPS = (3, 5, 7, 9, 11, 13, 15)
SINNER_GT = {"S1": (57, "far"), "S2": (163, "near"), "S3": (248, "far"), "S4": (308, "near"), "S5": (389, "far"),
             "S6": (456, "near"), "S7": (534, "far"), "S8": (605, "near"), "S9": (676, "far"), "S10": (764, "near"),
             "S11": (864, "far"), "S12": (950, "far")}


def nms_peaks(score, sep, floor=0.05):
    """Frames kept by greedy suppression: highest score first; a kept frame suppresses |dt| < sep."""
    order = np.argsort(-score, kind="stable")
    suppressed = np.zeros(len(score), bool)
    kept = []
    for t in order:
        if score[t] < floor:
            break
        if suppressed[t]:
            continue
        kept.append(t)
        suppressed[max(t - sep + 1, 0):t + sep] = True
    kept = np.array(sorted(kept), int)
    return kept, score[kept]


def assign(pred, gt, tol):
    """Maximum one-to-one matching within tol (then minimum total |dt|): list of (pred_i, gt_j, dt)."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    d = np.abs(np.asarray(pred)[:, None] - np.asarray(gt)[None, :]).astype(float)
    cost = np.where(d <= tol, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(i, j, int(pred[i] - gt[j])) for i, j in zip(r, c) if d[i, j] <= tol]


def load(path):
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    clips = []
    for k, name in enumerate(z["names"]):
        a, b = off[k], off[k + 1]
        ev = z["ev_offsets"]
        clips.append({"name": str(name), "fps": float(z["fps"][k]),
                      "ev_frame": z["ev_frame"][ev[k]:ev[k + 1]], "ev_cls": z["ev_cls"][ev[k]:ev[k + 1]],
                      "p": {m: z[f"p_{m}"][a:b].astype(np.float32) for m in ("v3", "v2", "gru")}})
    return clips


def counts(clips, model, group, thr, sep, tol, skip=()):
    tp = fp = fn = 0
    errs = []
    for c in clips:
        if c["name"] in skip:
            continue
        s = c["p"][model][:, GROUPS[group]].sum(1)
        fr, sc = nms_peaks(s, sep)
        pred = fr[sc >= thr]
        gt = c["ev_frame"][np.isin(c["ev_cls"], GROUPS[group])]
        m = assign(pred, gt, tol)
        tp, fp, fn = tp + len(m), fp + len(pred) - len(m), fn + len(gt) - len(m)
        errs += [abs(e) for _, _, e in m]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0,
            "timing_err_mean": float(np.mean(errs)) if errs else float("nan")}


def select(val, model, group, seps):
    best = None
    for sep in seps:
        for t in THR:
            key = (counts(val, model, group, t, sep, 2)["F1"], -sep, -t)
            if best is None or key > best[0]:
                best = (key, float(t), sep)
    return {"thr": best[1], "sep": best[2]}


def main(d):
    d = Path(d)
    val, test = load(d / "val.npz"), load(d / "test.npz")
    overlap_test = json.loads((d / "test_overlap_clips.json").read_text()) if (d / "test_overlap_clips.json").exists() else []
    out = {"selection": {}, "test": {}, "test_without_overlapping_clips": {}, "by_fps": {}}
    for m in ("v3", "v2", "gru"):
        out["selection"][m], out["test"][m], out["test_without_overlapping_clips"][m], out["by_fps"][m] = {}, {}, {}, {}
        for g in GROUPS:
            base = select(val, m, g, (5,))
            nms = select(val, m, g, SEPS)
            out["selection"][m][g] = {"base": base, "nms": nms}
            out["test"][m][g] = {f"{name}_tol{t}": counts(test, m, g, dec["thr"], dec["sep"], t)
                                 for name, dec in (("base", base), ("nms", nms)) for t in (1, 2, 4)}
            out["test_without_overlapping_clips"][m][g] = counts(test, m, g, nms["thr"], nms["sep"], 4, skip=set(overlap_test))
            for fps in (25.0, 29.97):
                sub = [c for c in test if abs(c["fps"] - fps) < 0.1]
                out["by_fps"][m][f"{g}_{fps}"] = {"tol4": counts(sub, m, g, nms["thr"], nms["sep"], 4)["F1"],
                                                  "tol133ms": counts(sub, m, g, nms["thr"], nms["sep"], round(0.133 * fps))["F1"]}
    # Sinner: per labelled bounce, nearest prediction of each model
    z = np.load(d / "sinner.npz")
    vf, src_fps = z["video_frame"], float(z["src_fps"])
    gt30 = {n: int(np.argmin(np.abs(vf - f))) for n, (f, _) in SINNER_GT.items()}
    out["sinner"] = {}
    for m in ("v3", "v2", "gru"):
        p = z[f"p_{m}"].astype(np.float32)
        dec = out["selection"][m]["bounce"]["nms"]
        s = p[:, GROUPS["bounce"]].sum(1)
        fr, sc = nms_peaks(s, dec["sep"])
        keep = sc >= dec["thr"]
        pred, conf = fr[keep], sc[keep]
        names = list(SINNER_GT)
        m_ = assign(pred, [gt30[n] for n in names], 2)
        matched = {j: (i, e) for i, j, e in m_}
        rows = {}
        for j, n in enumerate(names):
            if j in matched:
                i, e = matched[j]
                rows[n] = {"found": True, "pred_video_frame": int(vf[pred[i]]), "conf": round(float(conf[i]), 3),
                           "err_video_frames": int(vf[pred[i]] - SINNER_GT[n][0]),
                           "err_ms": round(1000 * (vf[pred[i]] - SINNER_GT[n][0]) / src_fps, 1)}
            else:
                near = np.abs(np.arange(len(s)) - gt30[n]) <= 2
                rows[n] = {"found": False, "max_prob_within_2": round(float(s[near].max()), 3)}
        fps_ = [int(vf[pred[i]]) for i in range(len(pred)) if i not in {i for i, _, _ in m_}]
        tp = len(m_)
        out["sinner"][m] = {"decoding": dec, "tp": tp, "fp": len(pred) - tp, "fn": 12 - tp, "false": fps_, "per_bounce": rows}
    Path(d / "audit_eval.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = main(sys.argv[1])
    for m in ("v3", "v2", "gru"):
        print(f"\n== {m}: decoding chosen on val (nms):", {g: r["selection"][m][g]["nms"] for g in ("bounce", "far_bounce", "near_bounce")})
        for g in GROUPS:
            row = r["test"][m][g]
            print(f"  {g:12s} " + "  ".join(f"{k}: {row[k]['P']:.3f}/{row[k]['R']:.3f}/{row[k]['F1']:.3f}"
                                         for k in ("base_tol4", "nms_tol4", "nms_tol2", "nms_tol1")))
    print("\nSinner:")
    for m, s in r["sinner"].items():
        print(f"  {m}: TP {s['tp']} FP {s['fp']} FN {s['fn']}  false {s['false']}  decoding {s['decoding']}")
