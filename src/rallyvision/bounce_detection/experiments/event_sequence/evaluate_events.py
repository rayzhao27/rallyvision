#!/usr/bin/env python3
"""Hit classifier + event-sequence decoding vs the bounce-only baseline, first bounce of each shot.

    python scripts/evaluate_events.py --csv TRAIN.csv VAL.csv --out DIR [--rally analysis.npz --gt gt.json]

Baseline: the bounce GBDT with its train-chosen threshold and same-shot gap (scripts/evaluate_bounce.py,
unchanged). New: the same bounce GBDT, a hit GBDT on kink + velocity-change candidates, and
rallyvision.events.decode_events with transitions learned from the training labels.

Everything is chosen on the train games. Thresholds, the same-shot gap and the transition weight are
chosen from out-of-fold probabilities (leave-one-game-out inside the training set), with transitions
counted on the other games only. Train CV is nested leave-one-game-out. Val is scored once; the rally
video (optional) once, with classifiers fitted on the train games.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from rallyvision.common.cli import REPO_ROOT, load_config
from rallyvision.bounce_detection.experiments.gbdt.evaluate_bounce import (PROB_GRID, TOL, VAL_GAMES, _train_model, by_game, counts, fit_gbdt, load_clips,
                             make_detector, select, track_data)
from rallyvision.bounce_detection.experiments.gbdt.bounce import candidate_features, first_bounces_by_gap, match_events, match_pairs, prf
from rallyvision.bounce_detection.experiments.event_sequence.events import BOUNCE, HIT, TransitionModel, decode_events, event_candidates, first_bounces_of_shots
from rallyvision.ball_tracking.inference.tracker import BallTracker, build_tracker

LAM_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TRAIN_FPS = 30.0


# --------------------------------------------------------------------------- data
def add_events(c, x, y, missing):
    """Union candidates, hit features and hit labels; c["pred"] (bounce candidates) is left as it is."""
    idx, a, is_kink = event_candidates(x, y)
    if not np.array_equal(idx[is_kink], c["pred"]["idx"]):
        raise AssertionError(f"{c['clip']}: kink candidates differ from the bounce pipeline")
    label = np.zeros(len(idx), dtype=int)
    for i, _ in match_pairs(c["frames"][idx].tolist(), c["hits"], TOL):
        label[i] = 1
    c["ev"] = {"idx": idx, "is_kink": is_kink, "X": candidate_features(x, y, missing, idx, a), "label": label}
    return c


def load(csv_paths, labels_dir, cfg):
    clips = {c["clip"]: c for c in load_clips(csv_paths, labels_dir, cfg)}
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df["frame"] = df["frame_file"].str.extract(r"(\d+)")[0].astype(int)
    for clip_id, g in df.sort_values("frame").groupby("clip_id"):   # same track as load_clips
        tracker = build_tracker(cfg)
        for f, (x, y) in zip(g["frame"], g[["pred_x", "pred_y"]].to_numpy()):
            tracker.update(int(f), (x, y) if np.isfinite(x) else None)
        t = tracker.finalize()
        add_events(clips[clip_id], np.array([d.x if d.has_position else np.nan for d in t]),
                   np.array([d.y if d.has_position else np.nan for d in t]),
                   np.array([d.source != "detected" for d in t]))
    return list(clips.values())


# --------------------------------------------------------------------------- models
def train_hit(clips):
    X = np.vstack([c["ev"]["X"] for c in clips])
    y = np.concatenate([c["ev"]["label"] for c in clips])
    model = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    return lambda c: model.predict_proba(c["ev"]["X"])[:, 1] if len(c["ev"]["X"]) else np.zeros(0)


def probabilities(c, bounce_model, hit_model):
    """(p_bounce, p_hit) on the union candidates; p_bounce is 0 where there is no upward kink."""
    pb = np.zeros(len(c["ev"]["idx"]))
    pb[c["ev"]["is_kink"]] = bounce_model(c)
    return pb, hit_model(c)


def sequences(clips):
    return [[(f, BOUNCE) for f in c["bounces"]] + [(f, HIT) for f in c["hits"]] for c in clips]


def out_of_fold(clips):
    """Per clip: probabilities from classifiers fitted on the other games, and transitions counted there."""
    out = {}
    for _, rest, held in by_game(clips):
        bm, hm, tm = _train_model(rest, "pred"), train_hit(rest), TransitionModel(sequences(rest))
        out.update({c["clip"]: (*probabilities(c, bm, hm), tm) for c in held})
    return out


# --------------------------------------------------------------------------- detectors
def decode_clip(c, pb, ph, tm, params):
    events = decode_events(c["frames"][c["ev"]["idx"]], pb, ph, {BOUNCE: params["t_bounce"], HIT: params["t_hit"]},
                           tm, params["lam"])
    return events


def scores(clips, events_of, gap):
    """Summed (tp, fp, fn) for first bounces, every bounce and hits."""
    tot = {k: np.zeros(3, int) for k in ("first", "bounce", "hit")}
    for c in clips:
        ev = events_of(c)
        b = [f for f, t in ev if t == BOUNCE]
        tot["first"] += match_events(first_bounces_of_shots(ev, gap), c["first_bounces"], TOL)
        tot["bounce"] += match_events(b, c["bounces"], TOL)
        tot["hit"] += match_events([f for f, t in ev if t == HIT], c["hits"], TOL)
    return tot


def fit_events(train):
    """Choose t_bounce + gap (as the baseline), t_hit, then lam, on out-of-fold probabilities; refit on all of train."""
    oof = out_of_fold(train)
    t_b, gap = select(train, "pred", lambda c: oof[c["clip"]][0][c["ev"]["is_kink"]], PROB_GRID)
    hit_f1 = [prf(*sum(np.array(match_events(c["frames"][c["ev"]["idx"][oof[c["clip"]][1] > t]].tolist(),
                                             c["hits"], TOL)) for c in train))[2] for t in PROB_GRID]
    t_h = float(PROB_GRID[int(np.argmax(hit_f1))])
    f1 = []
    for lam in LAM_GRID:
        p = {"t_bounce": float(t_b), "t_hit": t_h, "lam": lam}
        f1.append(prf(*scores(train, lambda c: decode_clip(c, *oof[c["clip"]], p), gap)["first"])[2])
    params = {"t_bounce": float(t_b), "same_shot_gap": int(gap), "t_hit": t_h, "lam": LAM_GRID[int(np.argmax(f1))],
              "train_oof_first_f1_by_lam": dict(zip(map(str, LAM_GRID), np.round(f1, 4).tolist()))}
    bm, hm, tm = _train_model(train, "pred"), train_hit(train), TransitionModel(sequences(train))

    def detect(c):
        return decode_clip(c, *probabilities(c, bm, hm), tm, params)

    def hits_only(c):              # the hit classifier alone, thresholded, no decoding
        return c["frames"][c["ev"]["idx"][hm(c) > t_h]].tolist()
    detect.params, detect.models, detect.hits_only = params, (bm, hm, tm), hits_only
    return detect


def hit_only_counts(clips, detect):
    return sum(np.array(match_events(detect.hits_only(c), c["hits"], TOL)) for c in clips)


def baseline_events(detect):
    """The baseline as an event list (bounces only), so both are scored by the same code."""
    return lambda c: [(f, BOUNCE) for f in detect(c)]


def baseline_scores(clips, detect):
    tot = scores(clips, baseline_events(detect), detect.params["same_shot_gap"])
    # first bounces exactly as the baseline picks them (by gap only), for the headline numbers
    tot["first"] = np.array(counts(clips, detect, "first"))
    return tot


# --------------------------------------------------------------------------- rally video
def track_30fps(balls, fps, cfg):
    """BallTracker with gaps in time, then linear resampling to 30 fps (as scripts/run_rally.py)."""
    tr = BallTracker(max_jump_px=float(cfg.tracker.max_jump_px),
                     max_gap_frames=int(round(cfg.tracker.max_gap_frames * fps / TRAIN_FPS)))
    for f, p in enumerate(balls):
        tr.update(f, tuple(p) if np.isfinite(p[0]) else None)
    t = tr.finalize()
    x = np.array([d.x if d.has_position else np.nan for d in t])
    y = np.array([d.y if d.has_position else np.nan for d in t])
    m = np.array([d.source != "detected" for d in t])
    ts = np.arange(len(t)) / fps
    t30 = np.arange(0, ts[-1], 1 / TRAIN_FPS)
    lo = np.clip(np.searchsorted(ts, t30, side="right") - 1, 0, len(t) - 2)
    w = (t30 - ts[lo]) * fps
    return (1 - w) * x[lo] + w * x[lo + 1], (1 - w) * y[lo] + w * y[lo + 1], m[lo] | m[lo + 1]


def rally(npz, gt_json, base, new, cfg):
    a = np.load(npz)
    fps = float(a["fps"])
    x, y, m = track_30fps(a["ball"], fps, cfg)
    frames = np.arange(len(x))
    c = add_events({"clip": "rally", "frames": frames, "hits": [], "pred": track_data(x, y, m, frames, [])}, x, y, m)
    bm, hm, tm = new.models
    pb, ph = probabilities(c, bm, hm)
    to_video = lambda f: int(round(f / TRAIN_FPS * fps))
    gt = [b["frame"] for b in json.loads(Path(gt_json).read_text())["bounces"]]
    tol = int(np.ceil(TOL * fps / TRAIN_FPS))
    p_at = {int(f): (round(float(b), 3), round(float(h), 3)) for f, b, h in zip(c["ev"]["idx"], pb, ph)}

    out = {}
    for name, events, gap in (("baseline", baseline_events(base)(c), base.params["same_shot_gap"]),
                              ("decoded", new(c), new.params["same_shot_gap"])):
        first = first_bounces_by_gap([f for f, _ in events], gap) if name == "baseline" else first_bounces_of_shots(events, gap)
        first_v = [to_video(f) for f in first]
        pairs = match_pairs(first_v, gt, tol)
        out[name] = {
            "events": [{"video_frame": to_video(f), "type": t, "p_bounce_p_hit": p_at[f]} for f, t in events],
            "first_bounces": first_v,
            "prf": prf(len(pairs), len(first_v) - len(pairs), len(gt) - len(pairs)),
            "false": [first_v[i] for i in range(len(first_v)) if i not in {i for i, _ in pairs}],
            "missed": [gt[j] for j in range(len(gt)) if j not in {j for _, j in pairs}],
        }
    return out


# --------------------------------------------------------------------------- main
def fmt(tot):
    return {k: {"tp_fp_fn": [int(v) for v in t], "p_r_f1": [round(v, 4) for v in prf(*t)]} for k, t in tot.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    ap.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "local_archive/articrafts/bounce_inputs/labels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rally", type=Path, help="analysis.npz from scripts/run_rally.py")
    ap.add_argument("--gt", type=Path, help="ground-truth first bounces of the rally video")
    args = ap.parse_args()
    cfg = load_config()

    clips = load(args.csv, args.labels_dir, cfg)
    train = [c for c in clips if c["game"] not in VAL_GAMES]
    val = [c for c in clips if c["game"] in VAL_GAMES]
    if not train or {c["game"] for c in val} != set(VAL_GAMES):
        raise SystemExit(f"expected train games and all of {VAL_GAMES} in the CSVs")

    cv = {"baseline": {k: np.zeros(3, int) for k in ("first", "bounce", "hit")},
          "decoded": {k: np.zeros(3, int) for k in ("first", "bounce", "hit", "hit_classifier_only")}}
    per_fold = []
    for g, rest, held in by_game(train):
        b, d = fit_gbdt(rest, "pred"), fit_events(rest)
        sb, sd = baseline_scores(held, b), scores(held, d, d.params["same_shot_gap"])
        sd["hit_classifier_only"] = hit_only_counts(held, d)
        for k in cv["baseline"]:
            cv["baseline"][k] += sb[k]
        for k in cv["decoded"]:
            cv["decoded"][k] += sd[k]
        per_fold.append({"held_out": g, "params": {k: v for k, v in d.params.items() if k != "train_oof_first_f1_by_lam"}})
        print(f"fold {g}: {per_fold[-1]['params']}")

    base, new = fit_gbdt(train, "pred"), fit_events(train)
    result = {
        "params": {"baseline": base.params, "decoded": new.params},
        "train_cv": {k: fmt(v) for k, v in cv.items()},
        "cv_folds": per_fold,
        "val": {"baseline": fmt(baseline_scores(val, base)),
                "decoded": fmt({**scores(val, new, new.params["same_shot_gap"]),
                                 "hit_classifier_only": hit_only_counts(val, new)})},
    }
    if args.rally:
        result["rally"] = rally(args.rally, args.gt, base, new, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "events_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("params", "train_cv", "val")}, indent=1))
    if args.rally:
        for name, r in result["rally"].items():
            print(name, "P/R/F1", np.round(r["prf"], 3), "false", r["false"], "missed", r["missed"])


if __name__ == "__main__":
    main()
