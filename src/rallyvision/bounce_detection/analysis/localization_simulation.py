#!/usr/bin/env python3
"""How accurately can a bounce be placed on the court from a single broadcast camera?

Simulates tennis shots with gravity, air drag and a bounce, films them with a fixed
broadcast-style camera at a random sub-frame phase, adds detector-like pixel noise and
dropped frames, then maps the estimated bounce point back to the court with the exact
ground-plane homography. Real video has no centimetre ground truth; here it is exact.

Compares
  frame         - ball position at the detected bounce frame
  frame_oracle  - ball position at the frame nearest the true contact (pure quantization)
  sub_dD_wW     - subframe_bounce(): degree-D fits on W frames either side

    python scripts/simulate_bounce_localization.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rallyvision.common.cli import REPO_ROOT
from rallyvision.bounce_detection.experiments.gbdt.bounce import bounce_candidates, fill_short_gaps, subframe_bounce

W, H = 1280, 720
G, DRAG, RADIUS = 9.81, 0.02, 0.033  # m/s^2, 1/m (0.5*rho*Cd*A/m for a tennis ball), m
RESTITUTION, FRICTION = 0.75, 0.75   # vertical speed kept / horizontal speed kept at the bounce
HALF_W, HALF_L = 5.485, 11.885       # doubles court, metres; net at y = 0
DT, T_END = 1e-3, 2.0
SUBFRAME = [(1, 5), (2, 7)]  # (degree, window) variants to compare; (2, 7) is the default


def broadcast_camera(position=(0.0, -26.0, 10.5), target=(0.0, 1.5, 0.0)):
    """Pinhole camera behind the near baseline; focal length chosen so the court fills ~70% of the height."""
    c, t = np.array(position), np.array(target)
    fwd = (t - c) / np.linalg.norm(t - c)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    corners = np.array([[sx * HALF_W, sy * HALF_L, 0] for sx in (-1, 1) for sy in (-1, 1)])
    cam = (corners - c) @ R.T
    f = 0.7 * H / np.ptp(cam[:, 1] / cam[:, 2])
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    Hg = K @ np.column_stack([R[:, 0], R[:, 1], -R @ c])  # ground plane (z=0) -> image
    return K, R, c, np.linalg.inv(Hg)


def project(P, K, R, c):
    q = (P - c) @ R.T @ K.T
    return q[..., :2] / q[..., 2:3]


def to_ground(uv, Hinv):
    q = np.append(uv, 1.0) @ Hinv.T
    return q[:2] / q[2]


def simulate_shots(n, rng):
    """Vectorised flight of n shots; returns positions on a 1 ms grid, contact time and contact point."""
    side = rng.choice([1.0, -1.0], n)  # +1: hit from the far side, bounce on the near half
    p = np.column_stack([rng.uniform(-4, 4, n), side * rng.uniform(10.5, 12.5, n), rng.uniform(0.6, 1.4, n)])
    target = np.column_stack([rng.uniform(-4, 4, n), -side * rng.uniform(2, 11.5, n)])
    d = (target - p[:, :2]) / np.linalg.norm(target - p[:, :2], axis=1, keepdims=True)
    speed, elev = rng.uniform(15, 35, n), np.radians(rng.uniform(-3, 12, n))
    v = np.column_stack([d * (np.cos(elev) * speed)[:, None], np.sin(elev) * speed])
    steps = int(T_END / DT)
    path = np.empty((steps, n, 3))
    bounced = np.zeros(n, bool)
    t_contact, p_contact = np.full(n, np.nan), np.full((n, 2), np.nan)
    for k in range(steps):
        v += (-DRAG * np.linalg.norm(v, axis=1, keepdims=True) * v - [0, 0, G]) * DT
        p += v * DT
        hit = ~bounced & (p[:, 2] <= RADIUS)
        t_contact[hit], p_contact[hit] = (k + 1) * DT, p[hit, :2]
        v[hit, 2] *= -RESTITUTION
        v[hit, :2] *= FRICTION
        p[hit, 2] = RADIUS
        bounced |= hit
        path[k] = p
    ok = (bounced & (np.abs(p_contact[:, 0]) < HALF_W) & (np.abs(p_contact[:, 1]) < HALF_L)
          & (p_contact[:, 1] * side < 0) & (t_contact > 0.3) & (t_contact < T_END - 0.5))
    return path[:, ok], t_contact[ok], p_contact[ok], side[ok]


def run(n_shots, fps_list, noise_list, drop, seed):
    rng = np.random.default_rng(seed)
    K, R, c, Hinv = broadcast_camera()
    path, t_contact, p_contact, side = simulate_shots(n_shots, rng)
    grid = np.arange(1, path.shape[0] + 1) * DT
    methods = ["frame", "frame_oracle"] + [f"sub_d{d}_w{w}" for d, w in SUBFRAME]
    results = {}
    for fps, noise in [(f, n) for f in fps_list for n in noise_list]:
        err = {m: [] for m in methods}
        terr = {m: [] for m in methods if m.startswith("sub")}
        where, missed = [], 0
        for i in range(len(t_contact)):
            phase = rng.uniform(0, 1 / fps)
            times = np.arange(t_contact[i] - 0.6 + phase, t_contact[i] + 0.4, 1 / fps)
            P = np.stack([np.interp(times, grid, path[:, i, a]) for a in range(3)], axis=1)
            uv = project(P, K, R, c) + rng.normal(0, noise, (len(times), 2))
            uv[rng.random(len(times)) < drop] = np.nan
            x, y = fill_short_gaps(uv[:, 0]), fill_short_gaps(uv[:, 1])
            true_idx = (t_contact[i] - times[0]) * fps
            idx, strength = bounce_candidates(y)
            near = np.abs(idx - true_idx) <= 2
            if not near.any():
                missed += 1
                continue
            t0 = int(idx[near][np.argmax(strength[near])])
            oracle = int(np.clip(round(true_idx), 0, len(times) - 1))
            truth = p_contact[i]
            est = {"frame": (x[t0], y[t0]), "frame_oracle": (x[oracle], y[oracle])}
            for d, w in SUBFRAME:
                s = subframe_bounce(x, y, t0, window=w, degree=d)
                est[f"sub_d{d}_w{w}"] = None if s is None else s[1:]
                if s is not None:
                    terr[f"sub_d{d}_w{w}"].append(abs(s[0] - true_idx))
            if any(v is None or not np.all(np.isfinite(v)) for v in est.values()):
                missed += 1
                continue
            for m, (u, vv) in est.items():
                err[m].append(np.linalg.norm(to_ground(np.array([u, vv]), Hinv) - truth))
            where.append("near" if side[i] > 0 else "far")
        where = np.array(where)
        summary = {"n": len(where), "missed_or_unfit": missed, "methods": {}}
        for m in methods:
            e = np.array(err[m]) * 100  # cm
            row = {"median_cm": float(np.median(e)), "p90_cm": float(np.percentile(e, 90))}
            for s in ("near", "far"):
                row[f"{s}_median_cm"] = float(np.median(e[where == s]))
            if m in terr:
                row["median_time_error_frames"] = float(np.median(terr[m]))
            summary["methods"][m] = row
        results[f"{fps}fps, σ={noise}px"] = summary
    return results


def write_markdown(results, args, path):
    lines = ["# Bounce localization: simulation\n",
             f"{args.shots} simulated shots per frame rate (15–35 m/s, broadcast camera behind the near baseline), "
             f"detector noise σ (per axis, 1280×720) as listed, {args.drop:.0%} dropped frames. "
             "Error = distance on court between estimated and true contact point.\n"]
    for fps, s in results.items():
        lines += [f"## {fps} (n={s['n']}, missed or unfit {s['missed_or_unfit']})\n",
                  "| method | median (cm) | p90 (cm) | near-court median | far-court median | time error (frames) |",
                  "|---|---|---|---|---|---|"]
        for m, r in s["methods"].items():
            te = f"{r['median_time_error_frames']:.2f}" if "median_time_error_frames" in r else "–"
            lines.append(f"| {m} | {r['median_cm']:.1f} | {r['p90_cm']:.1f} | {r['near_median_cm']:.1f} | "
                         f"{r['far_median_cm']:.1f} | {te} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shots", type=int, default=3000)
    ap.add_argument("--fps", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--noise-px", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.3],
                    help="per-axis σ at 1280×720; 2.3 ≈ the tracker's 1.37 px median error at 640×360 (incl. label noise)")
    ap.add_argument("--drop", type=float, default=0.10, help="fraction of frames with no detection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "local_archive/articrafts/bounce/simulation")
    args = ap.parse_args()

    results = run(args.shots, args.fps, args.noise_px, args.drop, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "simulation.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, args, args.out / "simulation.md")
    print((args.out / "simulation.md").read_text())


if __name__ == "__main__":
    main()
