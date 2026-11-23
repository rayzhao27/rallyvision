"""Regression check of a full `rallyvision` run on alcaraz_vs_sinner.mov against the reference demo.

    rallyvision --video alcaraz_vs_sinner.mov --out OUT --crop 90 1608      # GPU + E2E-Spot (Colab)
    python tests/regression/check_sinner_demo.py OUT

Reference = the saved stage outputs the demo was built from (local_archive/articrafts/, local):
    bounce probabilities  event_transformer/audit/probs/sinner.npz      (Transformer v3, Colab)
    ball track            ball_select_v2/sinner/analysis.npz            (TrackNet 1280 + select_ball)
    homographies          motion_comp/sinner_court.npz                  (court_v2)
    demo bounces          event_transformer/sinner_bounces_v3.json
Exits with 1 if any check fails. Small numeric differences (GPU vs earlier runs) are reported, not hidden.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[2] / "local_archive/articrafts"


def main(out: Path) -> int:
    r = np.load(out / "analysis.npz")
    got = json.loads((out / "bounces.json").read_text())
    z = np.load(A / "event_transformer/audit/probs/sinner.npz")
    ref_ball = np.load(A / "ball_select_v2/sinner/analysis.npz")["ball"]
    c = np.load(A / "motion_comp/sinner_court.npz")
    demo = json.loads((A / "event_transformer/sinner_bounces_v3.json").read_text())["bounces"]
    fps, n = float(r["fps"]), len(r["ball"])
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # bounce detection (Transformer v3 + E2E-Spot)
    same_frames = np.array_equal(r["bounce_frame"], z["video_frame"])
    check("bounce model frames", same_frames, f"{len(r['bounce_frame'])} vs {len(z['video_frame'])} sampled frames at 30 fps")
    if same_frames:
        d = np.abs(r["bounce_prob"].astype(np.float32) - z["p_v3"].astype(np.float32))
        check("v3 probabilities", d.max() < 0.02, f"max |diff| {d.max():.4f}, mean {d.mean():.5f}")
    key = lambda b: (b["video_frame"], b["side_pred"])
    check("bounce events (frame, side)", [key(b) for b in got["bounces"]] == [key(b) for b in demo],
          f"{len(got['bounces'])} vs demo {len(demo)}: {[b['video_frame'] for b in got['bounces']]}")
    check("decoding", got["decoding"] == {"thr": 0.6, "sep": 15}, got["decoding"])
    # ball tracking
    both = np.isfinite(r["ball"][:, 0]) & np.isfinite(ref_ball[:, 0])
    same_vis = (np.isfinite(r["ball"][:, 0]) == np.isfinite(ref_ball[:, 0])).mean()
    err = np.linalg.norm(r["ball"][both] - ref_ball[both], axis=1)
    check("ball track", n == len(ref_ball) and same_vis > 0.99 and np.mean(err < 1) > 0.99,
          f"{n} frames; same visibility {same_vis:.2%}; within 1 px {np.mean(err < 1):.2%} (640x360 px)")
    # court detection
    check("court valid frames", (r["H_valid"] == c["valid"]).mean() > 0.99,
          f"agree {(r['H_valid'] == c['valid']).mean():.2%}; valid {r['H_valid'].mean():.1%}")
    # fusion: court position and time of every bounce
    pairs = list(zip(got["bounces"], demo))
    dxy = [np.linalg.norm(np.subtract(g["court_xy"], d["court_xy"])) for g, d in pairs if g["court_xy"] and d["court_xy"]]
    check("bounce court positions", len(dxy) == len(demo) and max(dxy, default=0) < 0.1,
          f"max distance to demo {max(dxy, default=float('nan')):.3f} m over {len(dxy)} bounces")
    check("time sync", all(abs(b["time_s"] - b["video_frame"] / fps) < 1e-3 and 0 <= b["video_frame"] < n
                           for b in got["bounces"]), f"time_s = video_frame / {fps:.3f} fps, frames inside [0, {n})")
    # rendered video
    mp4 = out / "rally.mp4"
    if mp4.exists():
        p = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(mp4)], capture_output=True, text=True)
        s = json.loads(p.stdout)["streams"][0]
        num, den = map(int, s["avg_frame_rate"].split("/"))
        check("rally.mp4 decodes", int(s["nb_read_frames"]) == n and abs(num / den - fps) < 0.01,
              f"{s['nb_read_frames']} frames at {num / den:.3f} fps (expected {n} at {fps:.3f})")
    print(f"\n{sum(checks)}/{len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
