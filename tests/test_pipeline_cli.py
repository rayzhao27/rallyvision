"""The `rallyvision` CLI: orchestration, alignment, fusion and outputs, without running the video models.

The three model stages are replaced by the saved Sinner outputs of the reference demo (local_archive/articrafts/, local);
everything else (input checks, gating, tracking, fusion, bounces.json, analysis.npz, bounce_map.png) is the
real code. The full run on video (models included) is checked by tests/regression/check_sinner_demo.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
A = REPO / "local_archive/articrafts"
VIDEO = REPO / "data/alcaraz_vs_sinner.mov"
GOLDEN = REPO / "tests/golden/golden.json"
G = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else None
need = pytest.mark.skipif(not (A / "ball_select_v2/sinner/analysis.npz").exists() or not VIDEO.exists()
                          or G is None, reason="local golden outputs, inputs, or demo video missing")


def fake_spot(tmp_path):
    code, model = tmp_path / "spot", tmp_path / "model"
    (code / "data/tennis").mkdir(parents=True)
    model.mkdir()
    for f in (code / "train_e2e.py", code / "data/tennis/class.txt", model / "config.json", model / "checkpoint_041.pt"):
        f.touch()
    return code, model


def run_cli(monkeypatch, *args):
    from rallyvision.pipeline import __main__ as cli
    monkeypatch.setattr(sys, "argv", ["rallyvision", *map(str, args)])
    cli.main()


@need
def test_cli_reproduces_demo_from_saved_stage_outputs(tmp_path, monkeypatch):
    from rallyvision.bounce_detection.inference.decode import decode_bounces
    from rallyvision.pipeline import analyzer
    a = np.load(A / "ball_select_v2/sinner/analysis.npz")
    fps, n = float(a["fps"]), len(a["ball"])
    peaks = [a["peaks"][a["peaks"][:, 0] == f, 1:] for f in range(n)]
    c = np.load(A / "motion_comp/sinner_court.npz")
    calib = [{"H": c["H"][f] if c["valid"][f] else None, "valid": bool(c["valid"][f])} for f in range(n)]
    z = np.load(A / "event_transformer/audit/probs/sinner.npz")
    prob, keep = z["p_v3"].astype(np.float32), z["video_frame"]
    events = [{"video_frame": int(keep[t]), "time_s": round(float(keep[t]) / fps, 3), "confidence": round(float(s), 3),
               "side": side} for t, s, side in decode_bounces(prob, 0.6, 15)]
    aux = {"prob": prob, "video_frame": keep, "fps": fps, "decoding": {"thr": 0.6, "sep": 15}}
    monkeypatch.setattr(analyzer, "load_ball_model", lambda *a: (None, None))
    monkeypatch.setattr(analyzer, "detect_video", lambda *a: (fps, peaks, None, None))
    monkeypatch.setattr(analyzer, "load_court_model", lambda *a: (None, None))
    monkeypatch.setattr(analyzer, "calibrate_video", lambda *a: (fps, calib))
    monkeypatch.setattr(analyzer, "detect_bounces", lambda *a: (events, aux))
    code, model = fake_spot(tmp_path)
    out = tmp_path / "out"
    run_cli(monkeypatch, "--video", VIDEO, "--out", out, "--no-video", "--spot-code", code, "--spot-model", model)
    assert json.loads((out / "bounces.json").read_text())["bounces"] == G["demo_bounces"]
    r = np.load(out / "analysis.npz")
    assert np.array_equal(np.isnan(r["ball"]), np.isnan(np.load(REPO / "tests/golden/ball_sinner.npy")))
    assert len(r["H"]) == len(r["ball"]) == n and np.array_equal(r["bounce_frame"], keep)
    assert (out / "bounce_map.png").stat().st_size > 0


@need
def test_cli_stops_before_any_model_when_e2e_spot_missing(tmp_path, monkeypatch, capsys):
    from rallyvision.pipeline import analyzer
    monkeypatch.setattr(analyzer, "analyze", lambda *a: pytest.fail("models must not run"))
    with pytest.raises(SystemExit, match="E2E-Spot .* not set up"):
        run_cli(monkeypatch, "--video", VIDEO, "--out", tmp_path, "--spot-code", tmp_path / "nope", "--spot-model", tmp_path)


def test_cli_rejects_unreadable_video(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="cannot read video"):
        run_cli(monkeypatch, "--video", tmp_path / "missing.mp4", "--out", tmp_path)
