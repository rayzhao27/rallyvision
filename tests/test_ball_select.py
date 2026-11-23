"""Ball selection: time-based input frames, peak extraction, tracking among peaks, filters."""

from __future__ import annotations

import numpy as np

from rallyvision.ball_tracking.inference.ball_select import court_gate, drop_static, input_offsets, select_ball, top_peaks
from rallyvision.common.heatmap import render_gaussian


def test_input_offsets_follow_time_not_frames():
    assert input_offsets(30) == (2, 1)        # training geometry unchanged
    assert input_offsets(60) == (4, 2)
    assert input_offsets(54.6) == (4, 2)      # 1.82 and 3.64 frames -> nearest frames
    assert input_offsets(25) == (2, 1)
    assert input_offsets(10) == (2, 1)        # never two identical frames


def test_top_peaks_sorted_with_subpixel_position():
    hm = np.maximum(render_gaussian(90, 160, 40.3, 20.6), 0.5 * render_gaussian(90, 160, 120, 70))
    p = top_peaks(hm, k=3)
    assert p.shape == (2, 3) and p[0, 2] > p[1, 2]
    np.testing.assert_allclose(p[0, :2], [40.3, 20.6], atol=0.3)


def _ball_and_decoy(n=30, fps=30.0):
    """Ball moving right at 300 px/s; a stronger static decoy appears 60 px away (out of reach) at frame 10."""
    frames = []
    for i in range(n):
        ball = (100 + 300 * i / fps, 100.0, 0.4)
        rows = [ball] + ([(200.0, 160.0, 0.9)] if i >= 10 else [])
        frames.append(np.array(sorted(rows, key=lambda r: -r[2])))
    return frames


def test_select_ball_stays_on_ball_when_decoy_scores_higher():
    pts = select_ball(_ball_and_decoy(), fps=30.0, acquire=0.3)
    assert all(p is not None and abs(p[1] - 100) < 1e-6 for p in pts)


def test_select_ball_radius_scales_with_time():
    # same motion in pixels per second: at 60 fps the ball moves half as far per frame
    for fps in (30.0, 60.0):
        frames = [np.array([[100 + 600 * i / fps, 100.0, 0.5]]) for i in range(20)]
        assert all(p is not None for p in select_ball(frames, fps=fps, base_px=5, speed_px_s=700))


def test_select_ball_reacquires_after_long_gap():
    frames = [np.array([[100.0, 100.0, 0.5]])] + [np.zeros((0, 3))] * 20 + [np.array([[400.0, 50.0, 0.5]])]
    pts = select_ball(frames, fps=30.0)
    assert pts[0] is not None and pts[-1] == (400.0, 50.0)


def test_drop_static_removes_long_still_runs_only():
    still = [(50.0, 50.0)] * 40
    moving = [(100.0 + 5 * i, 80.0) for i in range(40)]
    out = drop_static(still + [None] + moving, fps=30.0, min_s=1.0)
    assert all(p is None for p in out[:40]) and out[41:] == moving
    assert drop_static(still[:20], fps=30.0, min_s=1.0) == still[:20]      # 0.67 s: kept


def test_court_gate_keeps_far_air_balls_drops_graphics():
    to_court = lambda xy: xy                       # identity: peaks already in metres for the test
    peaks = np.array([[0.0, 20.0, 0.9],            # beyond far baseline: a ball in the air
                      [-9.0, -20.0, 0.9],          # behind the near baseline, outside: graphics
                      [12.0, 0.0, 0.9],            # far beyond a sideline: crowd
                      [3.0, -5.0, 0.9]])
    kept = court_gate(peaks, to_court)
    assert kept[:, :2].tolist() == [[0.0, 20.0], [3.0, -5.0]]


def test_select_ball_rejects_a_strong_head_while_the_ball_is_hidden():
    # ball moves right; frames 10-20 it is not detected and a strong, slow "head" sits 45 px away;
    # the ball comes back on its arc at frame 21
    fps, frames = 54.6, []
    for i in range(30):
        ball = np.array([[100 + 2 * i, 100.0, 0.5]])
        head = np.array([[100 + 2 * 10 + 40, 80.0 - 0.1 * i, 0.9]])
        frames.append(head if 10 <= i <= 20 else ball)
    pts = select_ball(frames, fps=fps, acquire=0.18)
    assert all(pts[i] is None for i in range(10, 21))           # the head is never taken
    assert pts[21] == (142.0, 100.0)                             # the ball is picked up again


def test_select_ball_moves_to_a_fast_ball_after_a_few_frames():
    # track starts on a decoy (strong for 2 frames, then weak); the real ball (strong, 600 px/s)
    # appears 200 px away from frame 2
    fps, frames = 30.0, []
    for i in range(12):
        rows = [(300.0, 50.0, 0.6 if i < 2 else 0.1)] + ([(100 + 20 * i, 200.0, 0.8)] if i >= 2 else [])
        frames.append(np.array(sorted(rows, key=lambda r: -r[2])))
    pts = select_ball(frames, fps=fps, acquire=0.5)
    assert pts[0] == pts[1] == (300.0, 50.0)
    switch = next(i for i, p in enumerate(pts) if p is not None and p[1] == 200.0)
    assert switch <= 2 + 3 - 1 and all(p[1] == 200.0 for p in pts[switch:])
