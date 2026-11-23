from __future__ import annotations

import numpy as np

from rallyvision.bounce_detection.experiments.gbdt.bounce import bounce_candidates
from rallyvision.bounce_detection.experiments.event_sequence.events import (BOUNCE, HIT, TransitionModel, decode_events, event_candidates,
                                first_bounces_of_shots, velocity_change, velocity_peaks)

# typical rally in labels: hit, bounce ~20 frames later, next hit ~15 frames after that
RALLY = [[(0, HIT), (20, BOUNCE), (35, HIT), (55, BOUNCE), (70, HIT)]] * 20
TH = {BOUNCE: 0.3, HIT: 0.3}


def test_velocity_peaks_find_a_direction_change():
    t = np.arange(40, dtype=float)
    x = np.where(t <= 20, 3 * t, 60 - 3 * (t - 20))          # turns back at 20: a hit, no upward kink
    y = 100 + 0.0 * t
    assert velocity_peaks(velocity_change(x, y)).tolist() == [20]


def test_event_candidates_contain_all_kink_candidates():
    t = np.arange(60, dtype=float)
    y = np.where(t <= 30, 100 + 0.5 * t ** 2, 550 - 10 * (t - 30) + 0.5 * (t - 30) ** 2)
    x = np.where(t <= 45, 2 * t, 90 - 2 * (t - 45))
    idx, a, is_kink = event_candidates(x, y)
    kink, _ = bounce_candidates(y)
    assert set(kink) <= set(idx) and set(idx[is_kink]) == set(kink)
    assert 45 in idx and not is_kink[list(idx).index(45)]


def test_transition_model_prefers_common_orders():
    tm = TransitionModel(RALLY)
    assert tm(HIT, BOUNCE, 20) > 0 > tm(HIT, BOUNCE, 3)        # common timing vs too soon
    assert tm(BOUNCE, HIT, 15) > tm(BOUNCE, BOUNCE, 15)


def test_decode_without_transitions_is_thresholding():
    ev = decode_events([10, 20, 30], [0.8, 0.2, 0.0], [0.1, 0.1, 0.9], TH, TransitionModel(RALLY), lam=0)
    assert ev == [(10, BOUNCE), (30, HIT)]


def test_decode_drops_a_weak_bounce_right_after_a_hit():
    # hit at 0, weak bounce 3 frames later (implausible), strong bounce at 20
    frames, pb, ph = [0, 3, 20], [0.0, 0.4, 0.9], [0.9, 0.0, 0.0]
    tm = TransitionModel(RALLY)
    assert decode_events(frames, pb, ph, TH, tm, lam=0) == [(0, HIT), (3, BOUNCE), (20, BOUNCE)]
    assert decode_events(frames, pb, ph, TH, tm, lam=1) == [(0, HIT), (20, BOUNCE)]


def test_decode_keeps_bounce_bounce_when_both_are_strong():
    tm = TransitionModel(RALLY + [[(0, HIT), (20, BOUNCE), (40, BOUNCE)]] * 3)
    ev = decode_events([0, 20, 40], [0.0, 0.95, 0.95], [0.95, 0.0, 0.0], TH, tm, lam=1)
    assert ev == [(0, HIT), (20, BOUNCE), (40, BOUNCE)]


def test_first_bounces_of_shots():
    ev = [(5, BOUNCE), (10, HIT), (30, BOUNCE), (45, BOUNCE), (60, HIT), (80, BOUNCE), (200, BOUNCE)]
    assert first_bounces_of_shots(ev, max_same_shot_gap=40) == [5, 30, 80, 200]
    assert decode_events([], [], [], TH, TransitionModel(RALLY), lam=1) == []
