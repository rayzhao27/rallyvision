#!/usr/bin/env python3
"""Recall diagnosis for the ball tracker — zero training, pure post-hoc analysis.

It consumes a per-sample evaluation CSV and answers four questions the headline
precision/recall numbers do not:

  1. Operating point.      Where is the F1 (and F2) optimal detection threshold,
                           and how much recall is left on the table at the
                           default 0.5?  Produces the full P/R/F1-vs-threshold
                           sweep and the PR curve.
  2. Concentration.        Is recall loss spread evenly, or concentrated in a few
                           matches/clips?  Per-game and per-clip recall tables.
  3. Covariates.           Do misses correlate with ball speed (motion blur),
                           proximity to the frame edge, or (optional) frame
                           brightness/contrast?
  4. Recoverability.       Are misses isolated single frames (the tracker
                           interpolates them away) or long runs (unrecoverable)?
                           Reports raw detector recall vs post-tracker recall
                           using the project's own BallTracker.

IMPORTANT — the input CSV must be produced at detection_threshold = 0.0, e.g.

    python scripts/evaluate_ball_tracker.py --run <run> --split val \
        --threshold 0.0 --prefix val_t0

so that every frame carries its peak score AND the peak location.  A CSV written
at the default 0.5 discards the location of sub-threshold peaks, which makes a
true threshold sweep impossible; this script detects that case and warns.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rallyvision.common.config import REPO_ROOT

REPO = REPO_ROOT


# --------------------------------------------------------------------------- #
# Reuse the project's real tracker (pure numpy) without importing the package
# __init__ (which may pull in torch).
# --------------------------------------------------------------------------- #
def _load_tracker():
    try:
        spec = importlib.util.spec_from_file_location(
            "rv_tracker", REPO / "rallyvision" / "tracker.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclass needs the module registered
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.BallTracker, mod.interpolate_gaps
    except Exception as exc:  # pragma: no cover
        print(f"[warn] could not import project tracker ({exc}); using inline copy")
        return None, None


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def game_from_clip(clip_id: str) -> str:
    parts = str(clip_id).replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "game")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("visible", "detected"):
        if col in df.columns:
            df[col] = _to_bool(df[col])
    for col in ("score", "gt_x", "gt_y", "pred_x", "pred_y", "error_px"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "clip_id" not in df.columns:
        df["clip_id"] = ""
    if "frame_file" not in df.columns:
        df["frame_file"] = df.index.astype(str)
    df["game"] = df["clip_id"].map(game_from_clip)
    # Stable within-clip ordering for temporal analysis.
    df = df.sort_values(["clip_id", "frame_file", "index"], kind="stable").reset_index(drop=True)
    return df


def sanity_check(df: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    vis = df["visible"]
    vis_no_err = int((vis & df["error_px"].isna()).sum())
    vis_total = int(vis.sum())
    if vis_total and vis_no_err > 0.02 * vis_total:
        frac = vis_no_err / vis_total
        notes.append(
            f"{vis_no_err}/{vis_total} ({frac:.0%}) visible frames have no error_px. "
            "This CSV was almost certainly produced at a threshold > 0, so sub-threshold "
            "peak LOCATIONS are missing and the threshold sweep below is a lower bound only. "
            "Re-run eval with --threshold 0.0 for a clean sweep."
        )
    return notes


# --------------------------------------------------------------------------- #
# 1. Threshold sweep
# --------------------------------------------------------------------------- #
def confusion_at(df: pd.DataFrame, t: float, tol: float) -> Tuple[int, int, int, int]:
    vis = df["visible"].to_numpy()
    sc = df["score"].to_numpy()
    err = df["error_px"].to_numpy()
    det = sc >= t
    within = err <= tol  # NaN -> False
    tp = int(np.sum(vis & det & within))
    fp = int(np.sum((vis & det & ~within) | (~vis & det)))
    # A visible ball localized > tol away is BOTH a false alarm and a miss. Counting it
    # only as FP (as rallyvision/metrics.py does) lets recall climb to 1.0 as t -> 0
    # even though those frames are wrong. With this, recall = hits / visible frames,
    # consistent with group_recall() and tracker_recall().
    fn = int(np.sum(vis & ~(det & within)))
    tn = int(np.sum(~vis & ~det))
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom else 0.0


def threshold_sweep(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    vis_scores = np.unique(df.loc[df["visible"], "score"].dropna().to_numpy())
    grid = np.unique(np.concatenate([
        np.array([0.0, 0.5, 1.0]),
        vis_scores,
        np.linspace(0, 1, 201),
    ]))
    grid = grid[(grid >= 0) & (grid <= 1)]
    rows = []
    for t in grid:
        tp, fp, fn, tn = confusion_at(df, float(t), tol)
        p, r, f1 = prf(tp, fp, fn)
        rows.append({
            "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1,
            "f2": fbeta(p, r, 2.0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Per-clip / per-game recall
# --------------------------------------------------------------------------- #
def group_recall(df: pd.DataFrame, key: str, thresholds: Dict[str, float], tol: float) -> pd.DataFrame:
    out = []
    for name, g in df[df["visible"]].groupby(key):
        n = len(g)
        row = {key: name, "n_visible": int(n)}
        for label, t in thresholds.items():
            det = g["score"].to_numpy() >= t
            within = g["error_px"].to_numpy() <= tol
            tp = int(np.sum(det & within))
            row[f"recall@{label}"] = tp / n if n else 0.0
        miss = g[g["score"] < thresholds[list(thresholds)[0]]]
        row["mean_fn_score"] = float(miss["score"].mean()) if len(miss) else float("nan")
        row["mean_brightness"] = float(g["brightness"].mean()) if "brightness" in g else float("nan")
        out.append(row)
    res = pd.DataFrame(out)
    first = f"recall@{list(thresholds)[0]}"
    return res.sort_values(first, kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Covariates: speed, edge proximity
# --------------------------------------------------------------------------- #
def add_speed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["speed_px"] = np.nan
    for _, idx in df.groupby("clip_id").groups.items():
        g = df.loc[idx]
        gx = g["gt_x"].to_numpy(); gy = g["gt_y"].to_numpy()
        sp = np.full(len(g), np.nan)
        for i in range(1, len(g)):
            if np.isfinite(gx[i]) and np.isfinite(gx[i - 1]):
                sp[i] = math.hypot(gx[i] - gx[i - 1], gy[i] - gy[i - 1])
        df.loc[idx, "speed_px"] = sp
    return df


def covariate_recall(df: pd.DataFrame, col: str, bins, t: float, tol: float) -> pd.DataFrame:
    sub = df[df["visible"] & df[col].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["bin"] = pd.cut(sub[col], bins=bins, include_lowest=True)
    rows = []
    for b, g in sub.groupby("bin", observed=True):
        n = len(g)
        det = g["score"].to_numpy() >= t
        within = g["error_px"].to_numpy() <= tol
        tp = int(np.sum(det & within))
        rows.append({"bin": str(b), "n": int(n), "recall": tp / n if n else 0.0})
    return pd.DataFrame(rows)


def add_edge_distance(df: pd.DataFrame, width: int, height: int) -> pd.DataFrame:
    df = df.copy()
    x = df["gt_x"].to_numpy(); y = df["gt_y"].to_numpy()
    df["edge_dist_px"] = np.minimum.reduce([x, width - x, y, height - y])
    return df


# --------------------------------------------------------------------------- #
# 4. Run-length + post-tracker recall
# --------------------------------------------------------------------------- #
def miss_run_lengths(df: pd.DataFrame, t: float) -> List[int]:
    runs: List[int] = []
    for _, g in df.groupby("clip_id"):
        g = g[g["visible"]]
        det = (g["score"].to_numpy() >= t)
        run = 0
        for d in det:
            if not d:
                run += 1
            elif run:
                runs.append(run); run = 0
        if run:
            runs.append(run)
    return runs


def tracker_recall(df: pd.DataFrame, t: float, tol: float, cfg_tracker: dict) -> Dict[str, float]:
    BallTracker, _ = _load_tracker()
    raw_tp = raw_vis = trk_tp = 0
    for _, g in df.groupby("clip_id"):
        g = g.reset_index(drop=True)
        if BallTracker is not None:
            trk = BallTracker(
                max_jump_px=float(cfg_tracker.get("max_jump_px", 100.0)),
                max_gap_frames=int(cfg_tracker.get("max_gap_frames", 4)),
                smooth_window=int(cfg_tracker.get("smooth_window", 0)),
            )
            for i in range(len(g)):
                detected = g.loc[i, "score"] >= t
                pt = (g.loc[i, "pred_x"], g.loc[i, "pred_y"]) if (
                    detected and np.isfinite(g.loc[i, "pred_x"])
                ) else None
                trk.update(i, pt, float(g.loc[i, "score"]))
            track = trk.finalize(interpolate=bool(cfg_tracker.get("interpolate_gaps", True)))
            tx = {d.frame: (d.x, d.y) for d in track if d.has_position}
        else:
            tx = {}
        for i in range(len(g)):
            if not bool(g.loc[i, "visible"]):
                continue
            raw_vis += 1
            gx, gy = g.loc[i, "gt_x"], g.loc[i, "gt_y"]
            if (g.loc[i, "score"] >= t) and np.isfinite(g.loc[i, "error_px"]) and g.loc[i, "error_px"] <= tol:
                raw_tp += 1
            if i in tx and np.isfinite(gx):
                px, py = tx[i]
                if math.hypot(px - gx, py - gy) <= tol:
                    trk_tp += 1
    return {
        "raw_recall": raw_tp / raw_vis if raw_vis else 0.0,
        "tracker_recall": trk_tp / raw_vis if raw_vis else 0.0,
        "n_visible": raw_vis,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_sweep(sweep: pd.DataFrame, t_default: float, t_f1: float, t_f2: float, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["threshold"], sweep["precision"], label="precision", lw=2)
    ax.plot(sweep["threshold"], sweep["recall"], label="recall", lw=2)
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", lw=2, color="k")
    ax.plot(sweep["threshold"], sweep["f2"], label="F2 (recall-weighted)", lw=1.2, ls="--", color="gray")
    ax.axvline(t_default, color="red", ls=":", label=f"default t={t_default:g}")
    ax.axvline(t_f1, color="green", ls=":", label=f"F1-opt t={t_f1:.3f}")
    ax.set_xlabel("detection threshold"); ax.set_ylabel("score")
    ax.set_title("Precision / Recall / F1 vs detection threshold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3); ax.legend(loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_pr(sweep: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(sweep["recall"], sweep["precision"], c=sweep["threshold"], cmap="viridis", s=12)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("PR curve (color = threshold)")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01); ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label="threshold"); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_fn_hist(df: pd.DataFrame, t_default: float, t_f1: float, out: Path):
    vis = df[df["visible"]]
    fn = vis[vis["score"] < t_default]["score"]
    tp = vis[(vis["score"] >= t_default) & (vis["error_px"] <= 10)]["score"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(tp, bins=bins, alpha=0.6, label=f"detected (score≥{t_default:g})", color="tab:green")
    ax.hist(fn, bins=bins, alpha=0.7, label=f"missed (score<{t_default:g})", color="tab:red")
    ax.axvline(t_default, color="red", ls=":", label=f"default t={t_default:g}")
    ax.axvline(t_f1, color="green", ls=":", label=f"F1-opt t={t_f1:.3f}")
    ax.set_xlabel("peak score"); ax.set_ylabel("visible frames"); ax.set_yscale("log")
    ax.set_title("Peak-score distribution: detected vs missed"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_group_recall(tbl: pd.DataFrame, key: str, labels: List[str], out: Path):
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(tbl) + 3), 5))
    x = np.arange(len(tbl)); w = 0.8 / max(1, len(labels))
    for i, lab in enumerate(labels):
        ax.bar(x + i * w, tbl[f"recall@{lab}"], width=w, label=f"recall@{lab}")
    ax.set_xticks(x + w * (len(labels) - 1) / 2); ax.set_xticklabels(tbl[key], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("recall"); ax.set_ylim(0, 1.01); ax.set_title(f"Recall by {key}")
    ax.grid(alpha=0.3, axis="y"); ax.legend(); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_runlength(runs: List[int], max_gap: int, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if runs:
        m = max(runs)
        ax.hist(runs, bins=np.arange(0.5, m + 1.5, 1), color="tab:red", alpha=0.8)
    ax.axvline(max_gap + 0.5, color="green", ls=":", label=f"tracker gap budget = {max_gap}")
    ax.set_xlabel("consecutive-miss run length (visible frames)"); ax.set_ylabel("count")
    ax.set_title("Miss run lengths (runs ≤ budget are interpolable)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_covariate(tbl: pd.DataFrame, title: str, xlabel: str, out: Path):
    if tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tbl))
    ax.bar(x, tbl["recall"], color="tab:blue", alpha=0.8)
    for i, n in enumerate(tbl["n"]):
        ax.text(i, tbl["recall"].iloc[i] + 0.01, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(tbl["bin"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("recall"); ax.set_ylim(0, 1.05); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
# Optional: frame brightness/contrast
# --------------------------------------------------------------------------- #
def add_brightness(df: pd.DataFrame, frames_dir: Path, sample: int = 1500) -> pd.DataFrame:
    import cv2
    df = df.copy()
    df["brightness"] = np.nan
    df["contrast"] = np.nan
    vis_idx = df.index[df["visible"]].tolist()
    if len(vis_idx) > sample:
        rng = np.random.default_rng(0)
        vis_idx = sorted(rng.choice(vis_idx, size=sample, replace=False).tolist())
    read = 0
    for i in vis_idx:
        rel = str(df.loc[i, "clip_id"]) + "/" + str(df.loc[i, "frame_file"])
        p = frames_dir / rel
        if not p.is_file():
            p = frames_dir / str(df.loc[i, "frame_file"])
        if not p.is_file():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        df.loc[i, "brightness"] = float(img.mean())
        df.loc[i, "contrast"] = float(img.std())
        read += 1
    print(f"[info] brightness read for {read}/{len(vis_idx)} sampled visible frames")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path, help="per-sample CSV (produced at --threshold 0.0)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: <csv_dir>/recall_diagnosis)")
    ap.add_argument("--pr-tol", type=float, default=10.0, help="precision/recall pixel tolerance")
    ap.add_argument("--default-threshold", type=float, default=0.5)
    # gt_x/gt_y in the per-sample CSV are in model-input pixels (data.input_width/height),
    # not the original 1280x720 frame.
    ap.add_argument("--width", type=int, default=640, help="frame width of the CSV coordinates (for edge distance)")
    ap.add_argument("--height", type=int, default=360, help="frame height of the CSV coordinates")
    ap.add_argument("--frames-dir", type=Path, default=None, help="optional: root of val frames for brightness")
    ap.add_argument("--max-gap", type=int, default=4)
    ap.add_argument("--max-jump-px", type=float, default=100.0)
    args = ap.parse_args()

    out = args.out or (args.csv.parent / "recall_diagnosis")
    out.mkdir(parents=True, exist_ok=True)
    tol = args.pr_tol
    t0 = args.default_threshold

    df = load_csv(args.csv)
    notes = sanity_check(df)
    df = add_speed(df)
    df = add_edge_distance(df, args.width, args.height)
    if args.frames_dir:
        df = add_brightness(df, args.frames_dir)

    # 1. sweep
    sweep = threshold_sweep(df, tol)
    f1_row = sweep.loc[sweep["f1"].idxmax()]
    f2_row = sweep.loc[sweep["f2"].idxmax()]
    t_f1 = float(f1_row["threshold"]); t_f2 = float(f2_row["threshold"])
    def_tp, def_fp, def_fn, _ = confusion_at(df, t0, tol)
    def_p, def_r, def_f1 = prf(def_tp, def_fp, def_fn)

    thr = {f"{t0:g}": t0, "f1opt": t_f1}
    per_game = group_recall(df, "game", thr, tol)
    per_clip = group_recall(df, "clip_id", thr, tol)

    runs = miss_run_lengths(df, t0)
    cfg_tracker = {"max_jump_px": args.max_jump_px, "max_gap_frames": args.max_gap, "interpolate_gaps": True}
    trk = tracker_recall(df, t0, tol, cfg_tracker)
    trk_opt = tracker_recall(df, t_f1, tol, cfg_tracker)

    speed_tbl = covariate_recall(df, "speed_px", bins=[0, 10, 25, 50, 100, 1e9], t=t0, tol=tol)
    edge_tbl = covariate_recall(df, "edge_dist_px", bins=[0, 20, 50, 100, 200, 1e9], t=t0, tol=tol)

    # plots
    plot_sweep(sweep, t0, t_f1, t_f2, out / "threshold_sweep.png")
    plot_pr(sweep, out / "pr_curve.png")
    plot_fn_hist(df, t0, t_f1, out / "fn_score_hist.png")
    plot_group_recall(per_game, "game", [f"{t0:g}", "f1opt"], out / "recall_by_game.png")
    plot_runlength(runs, args.max_gap, out / "miss_runlength.png")
    plot_covariate(speed_tbl, "Recall vs ball speed (motion blur)", "GT speed (px/frame)", out / "recall_by_speed.png")
    plot_covariate(edge_tbl, "Recall vs distance to frame edge", "edge distance (px)", out / "recall_by_edge.png")

    per_clip.to_csv(out / "per_clip_recall.csv", index=False)
    per_game.to_csv(out / "per_game_recall.csv", index=False)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)

    runs_arr = np.array(runs) if runs else np.array([0])
    recoverable = int(np.sum(runs_arr <= args.max_gap))
    summary = {
        "n_samples": int(len(df)),
        "n_visible": int(df["visible"].sum()),
        "pr_tolerance_px": tol,
        "default_threshold": t0,
        "default": {"precision": def_p, "recall": def_r, "f1": def_f1,
                    "tp": def_tp, "fp": def_fp, "fn": def_fn},
        "f1_optimal": {"threshold": t_f1, "precision": float(f1_row["precision"]),
                       "recall": float(f1_row["recall"]), "f1": float(f1_row["f1"])},
        "f2_optimal": {"threshold": t_f2, "precision": float(f2_row["precision"]),
                       "recall": float(f2_row["recall"]), "f2": float(f2_row["f2"])},
        "tracker_recall_at_default": trk,
        "tracker_recall_at_f1opt": trk_opt,
        "miss_runs": {"total": len(runs), "recoverable_le_gap": recoverable,
                      "max_run": int(runs_arr.max())},
        "notes": notes,
    }
    (out / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2))

    # report
    def fmt(v):
        return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
    lines: List[str] = []
    lines.append("# Recall diagnosis\n")
    if notes:
        lines.append("> **Data caveat**\n>")
        for n in notes:
            lines.append(f"> - {n}")
        lines.append("")
    lines.append(f"Samples: {len(df):,}  ·  visible frames: {int(df['visible'].sum()):,}  ·  PR tolerance: {tol:g}px\n")
    lines.append(f"Recall here = visible frames localized within {tol:g}px / all visible frames "
                 "(a far-off detection counts as both FP and FN). This is stricter than the "
                 "training-log recall, which counts far-off detections only as FP.\n")
    lines.append("## 1. Operating point\n")
    lines.append("| threshold | precision | recall | F1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| default {t0:g} | {fmt(def_p)} | {fmt(def_r)} | {fmt(def_f1)} |")
    lines.append(f"| **F1-opt {t_f1:.3f}** | {fmt(float(f1_row['precision']))} | {fmt(float(f1_row['recall']))} | **{fmt(float(f1_row['f1']))}** |")
    lines.append(f"| F2-opt {t_f2:.3f} | {fmt(float(f2_row['precision']))} | {fmt(float(f2_row['recall']))} | F2={fmt(float(f2_row['f2']))} |")
    dr = float(f1_row["recall"]) - def_r
    df1 = float(f1_row["f1"]) - def_f1
    lines.append("")
    lines.append(f"Moving the threshold from {t0:g} to {t_f1:.3f} changes recall by **{dr:+.3f}** "
                 f"and F1 by **{df1:+.4f}** (precision {fmt(def_p)} → {fmt(float(f1_row['precision']))}). "
                 "This is a free, no-retraining gain — see `threshold_sweep.png` / `pr_curve.png`.\n")
    lines.append("## 2. Where recall is lost (concentration)\n")
    lines.append("Per game (sorted worst first):\n")
    lines.append("| game | n_visible | recall@%g | recall@f1opt | mean FN score |" % t0)
    lines.append("|---|---|---|---|---|")
    for _, r in per_game.iterrows():
        lines.append(f"| {r['game']} | {int(r['n_visible'])} | {fmt(r[f'recall@{t0:g}'])} | "
                     f"{fmt(r['recall@f1opt'])} | {fmt(r['mean_fn_score'])} |")
    lines.append("\nFull per-clip table in `per_clip_recall.csv`; bar chart in `recall_by_game.png`.\n")
    lines.append("## 3. Covariates\n")
    if not speed_tbl.empty:
        lines.append("Recall vs ball speed (motion-blur proxy):\n")
        lines.append("| speed bin (px/frame) | n | recall |")
        lines.append("|---|---|---|")
        for _, r in speed_tbl.iterrows():
            lines.append(f"| {r['bin']} | {int(r['n'])} | {fmt(r['recall'])} |")
        lines.append("")
    if not edge_tbl.empty:
        lines.append("Recall vs distance to frame edge:\n")
        lines.append("| edge dist (px) | n | recall |")
        lines.append("|---|---|---|")
        for _, r in edge_tbl.iterrows():
            lines.append(f"| {r['bin']} | {int(r['n'])} | {fmt(r['recall'])} |")
        lines.append("")
    if "brightness" in df and df["brightness"].notna().any():
        vis = df[df["visible"] & df["brightness"].notna()]
        fn_b = vis[vis["score"] < t0]["brightness"].mean()
        tp_b = vis[vis["score"] >= t0]["brightness"].mean()
        lines.append(f"Frame brightness — missed frames mean {fn_b:.1f} vs detected {tp_b:.1f} "
                     "(lower = darker; confirms/denies the dim-match hypothesis).\n")
    lines.append("## 4. Recoverability (raw detector vs post-tracker)\n")
    lines.append(f"At default {t0:g}: raw detector recall **{trk['raw_recall']:.4f}** → "
                 f"post-tracker recall **{trk['tracker_recall']:.4f}** "
                 f"(tracker interpolates gaps ≤ {args.max_gap}).")
    lines.append(f"At F1-opt {t_f1:.3f}: raw **{trk_opt['raw_recall']:.4f}** → "
                 f"post-tracker **{trk_opt['tracker_recall']:.4f}**.")
    lines.append(f"\nMiss runs: {len(runs)} total, {recoverable} are ≤ gap budget ({args.max_gap}) and thus "
                 f"interpolable; longest unrecoverable run = {int(runs_arr.max())} frames. See `miss_runlength.png`.\n")
    (out / "recall_diagnosis.md").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[done] wrote report + plots to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
