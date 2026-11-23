from __future__ import annotations

import numpy as np

from rallyvision.bounce_detection.experiments.gbdt.bounce import (FEATURES, bounce_candidates, bridge_gaps, candidate_features, detect_bounces,
                                fill_short_gaps, first_bounces, match_events, subframe_bounce)


def _bouncing_ball():
    """Ball falls (image y grows), bounces at frame 20, rises again."""
    t = np.arange(40, dtype=float)
    return np.where(t <= 20, 100 + 0.5 * t**2, 300 - 10 * (t - 20) + 0.5 * (t - 20) ** 2)


def test_bounce_found_at_the_kink():
    assert detect_bounces(_bouncing_ball(), threshold=5).tolist() == [20]


def test_smooth_parabola_has_no_bounce():
    t = np.arange(40, dtype=float)
    assert detect_bounces(100 + 0.5 * (t - 20) ** 2, threshold=5).size == 0


def test_short_gap_filled_long_gap_kept():
    y = fill_short_gaps([0, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan, np.nan, 9], max_gap=4)
    assert y[:4].tolist() == [0, 1, 2, 3]
    assert np.isnan(y[4:9]).all()


def test_bounce_survives_a_missing_frame():
    y = _bouncing_ball()
    y[18] = np.nan
    assert detect_bounces(y, threshold=5).tolist() == [20]


def test_match_events():
    assert match_events([10], [10], tol=2) == (1, 0, 0)
    assert match_events([12], [10], tol=2) == (1, 0, 0)
    assert match_events([13], [10], tol=2) == (0, 1, 1)
    assert match_events([9, 11], [10], tol=2) == (1, 1, 0)  # one label, counted once
    assert match_events([], [10], tol=2) == (0, 0, 1)
    assert match_events([10], [], tol=2) == (0, 1, 0)


def test_candidate_features():
    y = _bouncing_ball()
    x_straight = np.arange(40, dtype=float) * 3          # keeps moving right: like a bounce
    x_reversed = np.where(np.arange(40) <= 20, 3.0, -3.0).cumsum()  # turns back: like a hit
    missing = np.zeros(40, dtype=bool)
    missing[22] = True
    f = dict(zip(FEATURES, candidate_features(x_straight, y, missing, [20], [29.0])[0]))
    assert f["kink"] == 29.0 and f["y"] == y[20]
    assert f["vx_product"] > 0 and f["vy_before"] > 0 > f["vy_after"]
    assert f["missing_frac"] == 1 / 11
    assert candidate_features(x_reversed, y, missing, [20], [29.0])[0][FEATURES.index("vx_product")] < 0



def test_candidate_features_near_clip_edge_and_empty():
    y = _bouncing_ball()
    row = candidate_features(y, y, np.zeros(40, bool), [2], [1.0])[0]
    assert np.isnan(row[FEATURES.index("vx_before")]) and np.isfinite(row[FEATURES.index("vx_after")])
    assert candidate_features(y, y, np.zeros(40, bool), [], []).shape == (0, len(FEATURES))


def test_first_bounces():
    # hit 0, bounce 10, second bounce 25 (drop shot), hit 40, bounce 50
    assert first_bounces([10, 25, 50], [0, 40]) == [10, 50]
    assert first_bounces([10, 25], []) == [10]          # clip start opens a shot
    assert first_bounces([], [5]) == []


def test_subframe_bounce_recovers_contact_between_frames():
    # ball falls at 20 px/frame, contact at t=20.4, rises at 15 px/frame; x moves steadily
    t = np.arange(40, dtype=float)
    y = np.where(t <= 20.4, 400 - 20 * (20.4 - t), 400 - 15 * (t - 20.4))
    x = 100 + 4 * t
    tc, xc, yc = subframe_bounce(x, y, t0=20, degree=1)
    assert abs(tc - 20.4) < 0.02 and abs(yc - 400) < 0.5 and abs(xc - (100 + 4 * 20.4)) < 0.5
    assert subframe_bounce(x, np.full(40, np.nan), t0=20) is None


def test_first_bounces_by_gap():
    from rallyvision.bounce_detection.experiments.gbdt.bounce import first_bounces_by_gap
    # bounce 10, drop-shot second bounce 25 (gap 15), next shot's bounce 60, third bounce 70
    assert first_bounces_by_gap([10, 25, 60, 70], max_same_shot_gap=20) == [10, 60]
    assert first_bounces_by_gap([10, 25], max_same_shot_gap=0) == [10, 25]
    assert first_bounces_by_gap([], max_same_shot_gap=20) == []


def test_bridge_gaps_restores_a_hidden_bounce():
    t = np.arange(60, dtype=float)
    x = 3 * t
    y = np.where(t <= 30, 100 + 4 * t + 0.05 * t ** 2, 265 - 2 * (t - 30) + 0.05 * (t - 30) ** 2)
    yg = y.copy()
    yg[24:38] = np.nan                                   # the bounce at t=30 falls in a 14-frame gap
    xb, yb = bridge_gaps(x, yg)
    assert np.isfinite(yb).all()
    np.testing.assert_allclose(yb, y, atol=1.5)
    frames, _ = bounce_candidates(yb)
    assert np.min(np.abs(frames - 30)) <= 1


def test_bridge_gaps_leaves_gap_when_arcs_do_not_meet():
    t = np.arange(40, dtype=float)
    y = np.where(t <= 20, 100 + 2 * t, 300 + 2 * t)      # a jump: the two sides never meet
    y[15:26] = np.nan
    _, yb = bridge_gaps(t, y)
    assert np.isnan(yb[15:26]).all()
