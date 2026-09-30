from __future__ import annotations

from rallyvision.tracker import BallTracker, build_tracker, interpolate_gaps, track_statistics


def test_clean_track_passes_through():
    tracker = BallTracker(max_jump_px=50, max_gap_frames=3)
    for i in range(5):
        tracker.update(i, (10.0 * i, 20.0), 0.9)
    track = tracker.finalize()
    assert all(d.source == "detected" for d in track)
    assert [d.x for d in track] == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_implausible_jump_is_rejected():
    tracker = BallTracker(max_jump_px=20, max_gap_frames=3)
    tracker.update(0, (10.0, 10.0), 0.9)
    tracker.update(1, (500.0, 400.0), 0.9)  # teleport
    tracker.update(2, (30.0, 10.0), 0.9)
    assert tracker.num_rejected == 1
    assert tracker.track[1].source == "rejected"
    assert tracker.track[2].source == "detected"


def test_short_gaps_are_interpolated():
    tracker = BallTracker(max_jump_px=1000, max_gap_frames=3)
    tracker.update(0, (0.0, 0.0), 0.9)
    tracker.update(1, None, 0.1)
    tracker.update(2, None, 0.1)
    tracker.update(3, (30.0, 30.0), 0.9)
    track = tracker.finalize(interpolate=True)
    assert track[1].source == "interpolated"
    assert track[1].x == 10.0 and track[1].y == 10.0
    assert track[2].x == 20.0


def test_long_gaps_are_left_empty():
    tracker = BallTracker(max_jump_px=1000, max_gap_frames=2)
    tracker.update(0, (0.0, 0.0), 0.9)
    for i in range(1, 6):
        tracker.update(i, None, 0.0)
    tracker.update(6, (60.0, 60.0), 0.9)
    track = tracker.finalize(interpolate=True)
    assert all(not d.has_position for d in track[1:6])


def test_interpolate_gaps_is_a_pure_function():
    tracker = BallTracker()
    tracker.update(0, (0.0, 0.0))
    tracker.update(1, None)
    tracker.update(2, (10.0, 0.0))
    original = list(tracker.track)
    filled = interpolate_gaps(original, max_gap_frames=2)
    assert original[1].x is None
    assert filled[1].x == 5.0


def test_statistics_describe_the_track():
    tracker = BallTracker(max_jump_px=1000, max_gap_frames=2)
    tracker.update(0, (0.0, 0.0), 0.9)
    tracker.update(1, None, 0.1)
    tracker.update(2, (20.0, 0.0), 0.9)
    stats = track_statistics(tracker.finalize())
    assert stats["frames"] == 3
    assert stats["detected"] == 2
    assert stats["interpolated"] == 1
    assert stats["coverage"] == 1.0
    assert stats["mean_step_px"] == 10.0


def test_smoothing_reduces_jitter():
    tracker = BallTracker(max_jump_px=1000, smooth_window=3)
    for i, x in enumerate([0.0, 10.0, 0.0, 10.0, 0.0]):
        tracker.update(i, (x, 0.0), 0.9)
    smoothed = tracker.finalize(interpolate=False)
    middle = [d.x for d in smoothed[1:-1]]
    assert all(1.0 < x < 9.0 for x in middle)


def test_tracker_reset_clears_state():
    tracker = BallTracker()
    tracker.update(0, (1.0, 1.0))
    tracker.reset()
    assert tracker.track == []
    assert tracker.num_rejected == 0


def test_build_tracker_reads_config(base_config):
    base_config["tracker.max_jump_px"] = 42
    tracker = build_tracker(base_config)
    assert tracker.max_jump_px == 42.0
