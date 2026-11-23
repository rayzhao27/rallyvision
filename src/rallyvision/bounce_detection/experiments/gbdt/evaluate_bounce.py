#!/usr/bin/env python3
"""Evaluate bounce detection and first-bounce-per-shot selection on ground-truth and predicted tracks.

Candidates are upward kinks in the ball's image-y motion. Two methods decide which are bounces:
  rule  - kink size above a threshold
  gbdt  - gradient-boosted classifier on trajectory features, probability above a threshold

Only the first bounce of each shot goes on the court map (a drop shot's second bounce is not a
new shot). It is picked without hit detection: a bounce within G frames of the previous bounce
belongs to the same shot. Reported for each method and track:
  bounce F1        - every labelled bounce
  first-bounce F1  - the first bounce after each labelled hit (the truth for the map)

Everything that is chosen (threshold, classifier, G) is chosen on the train games only.
Train CV is leave-one-game-out and nested: the held-out game is never used to fit the
classifier or pick a parameter. The val games are scored once.

    python scripts/evaluate_bounce.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from rallyvision.common.cli import REPO_ROOT, load_config
from rallyvision.bounce_detection.experiments.gbdt.bounce import (bounce_candidates, bridge_gaps, candidate_features, first_bounces,
                                first_bounces_by_gap, match_events, match_pairs, prf)
from rallyvision.ball_tracking.data.labels import parse_label_csv
from rallyvision.ball_tracking.inference.tracker import build_tracker

VAL_GAMES = ("game5", "game8", "game9")
TOL = 2  # frames; matching tolerance for training labels, parameter selection and the headline numbers
RULE_GRID = np.round(np.arange(0.25, 30.01, 0.25), 2)
PROB_GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)
GAP_GRID = range(0, 41, 2)  # frames; 0 = every bounce counts as a shot's first
HIT, BOUNCE = 1, 2  # Label.csv status codes


# --------------------------------------------------------------------------- data
def track_data(x, y, missing, frames, bounces):
    idx, strength = bounce_candidates(y)
    positive = np.zeros(len(idx), dtype=int)
    for i, _ in match_pairs(frames[idx].tolist(), bounces, TOL):
        positive[i] = 1
    X = candidate_features(x, y, missing, idx, -strength)  # signed kink: negative = upward
    return {"idx": idx, "strength": strength, "X": X, "label": positive}


def load_clips(csv_paths, labels_dir, cfg, bridge=False):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df["visible"] = df["visible"].astype(str).str.lower().eq("true")
    df["frame"] = df["frame_file"].str.extract(r"(\d+)")[0].astype(int)
    clips = []
    for clip_id, g in df.sort_values("frame").groupby("clip_id"):
        frames = g["frame"].to_numpy()
        if np.any(np.diff(frames) != 1):
            raise ValueError(f"{clip_id}: frames are not consecutive")
        status = {int(Path(r["file"]).stem): r["status"] for r in parse_label_csv(labels_dir / clip_id / "Label.csv")}
        # an event labelled on consecutive frames is one event: keep its first frame
        events = {s: [int(f) for f in frames if status.get(f) == s and status.get(f - 1) != s] for s in (HIT, BOUNCE)}

        tracker = build_tracker(cfg)
        for f, (x, y) in zip(frames, g[["pred_x", "pred_y"]].to_numpy()):
            tracker.update(int(f), (x, y) if np.isfinite(x) else None)
        track = tracker.finalize()
        px = np.array([d.x if d.has_position else np.nan for d in track])
        py = np.array([d.y if d.has_position else np.nan for d in track])
        p_missing = np.array([d.source != "detected" for d in track])
        if bridge:                      # filled frames stay flagged as missing
            px, py = bridge_gaps(px, py)

        b = events[BOUNCE]
        clips.append({
            "clip": clip_id,
            "game": clip_id.split("/")[-2],
            "frames": frames,
            "bounces": b,
            "hits": events[HIT],
            "first_bounces": first_bounces(b, events[HIT]),
            "gt": track_data(g["gt_x"].to_numpy(float), g["gt_y"].to_numpy(float), ~g["visible"].to_numpy(), frames, b),
            "pred": track_data(px, py, p_missing, frames, b),
        })
    return clips


def by_game(clips):
    games = sorted({c["game"] for c in clips})
    return [(g, [c for c in clips if c["game"] != g], [c for c in clips if c["game"] == g]) for g in games]


# --------------------------------------------------------------------------- detectors
# A detector maps a clip to its detected bounce frames; detect.params records what was chosen.
def make_detector(track, score_fn, threshold, gap=0):
    def detect(c):
        return c["frames"][c[track]["idx"][score_fn(c) > threshold]].tolist()
    detect.params = {"threshold": float(threshold), "same_shot_gap": int(gap)}
    return detect


def counts(clips, detect, kind, tol=TOL):
    """Summed (tp, fp, fn); kind "bounce" scores every bounce, "first" the first bounce of each shot."""
    tp = fp = fn = 0
    for c in clips:
        if kind == "bounce":
            pred, gt = detect(c), c["bounces"]
        else:
            pred, gt = first_bounces_by_gap(detect(c), detect.params["same_shot_gap"]), c["first_bounces"]
        a, b, d = match_events(pred, gt, tol)
        tp, fp, fn = tp + a, fp + b, fn + d
    return tp, fp, fn


def select(clips, track, score_fn, grid):
    """Threshold by bounce F1, then the same-shot gap by first-bounce F1, both on `clips`."""
    f1 = [prf(*counts(clips, make_detector(track, score_fn, t), "bounce"))[2] for t in grid]
    t = grid[int(np.argmax(f1))]
    f1 = [prf(*counts(clips, make_detector(track, score_fn, t, g), "first"))[2] for g in GAP_GRID]
    return t, GAP_GRID[int(np.argmax(f1))]


def fit_rule(train, track):
    score_fn = lambda c: c[track]["strength"]
    return make_detector(track, score_fn, *select(train, track, score_fn, RULE_GRID))


def _train_model(clips, track):
    X = np.vstack([c[track]["X"] for c in clips])
    y = np.concatenate([c[track]["label"] for c in clips])
    model = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    return lambda c: model.predict_proba(c[track]["X"])[:, 1] if len(c[track]["X"]) else np.zeros(0)


def fit_gbdt(train, track):
    # parameters from out-of-fold probabilities within `train`, then refit the model on all of `train`
    oof = {}
    for _, rest, held in by_game(train):
        model = _train_model(rest, track)
        oof.update({c["clip"]: model(c) for c in held})
    t, gap = select(train, track, lambda c: oof[c["clip"]], PROB_GRID)
    return make_detector(track, _train_model(train, track), t, gap)


# --------------------------------------------------------------------------- evaluation
def bootstrap_f1(clips, detect, kind, n=1000, seed=0):
    """95% CI of F1 by resampling clips."""
    per_clip = np.array([counts([c], detect, kind) for c in clips])
    rng = np.random.default_rng(seed)
    f1s = [prf(*per_clip[rng.integers(0, len(clips), len(clips))].sum(axis=0))[2] for _ in range(n)]
    return [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]


def fp_near_hits(clips, detect):
    """False bounce detections within TOL frames of a labelled hit."""
    n = 0
    for c in clips:
        dets = detect(c)
        matched = {i for i, _ in match_pairs(dets, c["bounces"], TOL)}
        n += sum(any(abs(d - h) <= TOL for h in c["hits"]) for i, d in enumerate(dets) if i not in matched)
    return n


def evaluate(train, val, track, fit):
    kinds = ("bounce", "first")
    cv = {k: [0, 0, 0] for k in kinds}
    for _, rest, held in by_game(train):
        detect = fit(rest, track)
        for k in kinds:
            cv[k] = [a + b for a, b in zip(cv[k], counts(held, detect, k))]
    detect = fit(train, track)
    tp, fp, fn = counts(val, detect, "bounce")
    return {
        **detect.params,
        "train_cv_f1": {k: prf(*cv[k])[2] for k in kinds},
        "val": {k: prf(*counts(val, detect, k)) for k in kinds},
        "val_f1_ci95": {k: bootstrap_f1(val, detect, k) for k in kinds},
        "val_bounce_f1_tol1_tol3": [prf(*counts(val, detect, "bounce", t))[2] for t in (1, 3)],
        "val_per_game_f1": {k: {g: prf(*counts([c for c in val if c["game"] == g], detect, k))[2] for g in VAL_GAMES}
                            for k in kinds},
        "val_bounce_counts": {"tp": tp, "fp": fp, "fn": fn, "fp_near_hit": fp_near_hits(val, detect)},
    }


def write_markdown(results, clips, path):
    train = [c for c in clips if c["game"] not in VAL_GAMES]
    val = [c for c in clips if c["game"] in VAL_GAMES]
    f = lambda v: f"{v:.3f}"
    n = lambda cs, key: sum(len(c[key]) for c in cs)
    names = {"bounce": "every bounce", "first": "first bounce of each shot"}
    lines = [
        "# Bounce detection\n",
        f"Events — train: {n(train, 'bounces')} bounces, {n(train, 'first_bounces')} first bounces; "
        f"val: {n(val, 'bounces')} bounces, {n(val, 'first_bounces')} first bounces. "
        f"Everything is chosen on train games only (F1 at ±{TOL} frames); val is scored once. "
        "Train CV is nested leave-one-game-out.\n",
        "| method | track | event | threshold / same-shot gap | train CV F1 | val P | val R | val F1 (95% CI) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (method, track), r in results.items():
        for k, (p, rc, f1) in r["val"].items():
            lo, hi = r["val_f1_ci95"][k]
            lines.append(f"| {method} | {track} | {names[k]} | {r['threshold']:g} / {r['same_shot_gap']} | "
                         f"{f(r['train_cv_f1'][k])} | {f(p)} | {f(rc)} | {f(f1)} ({f(lo)}–{f(hi)}) |")
    lines += ["", f"Per val game (F1 at ±{TOL}):", "", "| method | track | event | " + " | ".join(VAL_GAMES) + " |",
              "|---|---|---|---|---|---|"]
    for (method, track), r in results.items():
        for k, per_game in r["val_per_game_f1"].items():
            lines.append(f"| {method} | {track} | {names[k]} | " + " | ".join(f(per_game[g]) for g in VAL_GAMES) + " |")
    lines += ["", "Bounce details on val:", ""]
    for (method, track), r in results.items():
        c = r["val_bounce_counts"]
        t1, t3 = r["val_bounce_f1_tol1_tol3"]
        lines.append(f"- {method} / {track}: tp={c['tp']} fp={c['fp']} fn={c['fn']}, {c['fp_near_hit']} false "
                     f"bounces are at a hit; bounce F1 at ±1 / ±3 = {f(t1)} / {f(t3)}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, nargs="+", default=[
        REPO_ROOT / "local_archive/articrafts/bounce_inputs/train_t018_per_sample.csv",
        REPO_ROOT / "local_archive/articrafts/diag/eval_t018/val_t018_per_sample.csv",
    ])
    ap.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "local_archive/articrafts/bounce_inputs/labels")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "local_archive/articrafts/bounce")
    ap.add_argument("--bridge-gaps", action="store_true", help="fill longer gaps in predicted tracks (bridge_gaps)")
    args = ap.parse_args()

    clips = load_clips(args.csv, args.labels_dir, load_config(), bridge=args.bridge_gaps)
    train = [c for c in clips if c["game"] not in VAL_GAMES]
    val = [c for c in clips if c["game"] in VAL_GAMES]
    if not train or {c["game"] for c in val} != set(VAL_GAMES):
        raise SystemExit(f"expected train games and all of {VAL_GAMES} in the CSVs")

    results = {(m, t): evaluate(train, val, t, fit) for m, fit in (("rule", fit_rule), ("gbdt", fit_gbdt))
               for t in ("gt", "pred")}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bounce_results.json").write_text(json.dumps({f"{m}/{t}": r for (m, t), r in results.items()}, indent=2))
    write_markdown(results, clips, args.out / "bounce_results.md")
    print((args.out / "bounce_results.md").read_text())


if __name__ == "__main__":
    main()
