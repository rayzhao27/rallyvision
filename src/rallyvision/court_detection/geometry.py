"""Court geometry: the 14 keypoints, their positions in metres, lines and the flip permutation.

A single-frame TrackNet predicts 14 heatmaps, one per court keypoint. Peaks
above a threshold become keypoints; four or more of them give a homography
from court metres to image pixels, which also places the keypoints that were
missed or hidden.

Keypoint names follow the Roboflow "tennis-court-all" export (same layout as
TennisCourtDetector). "Far" is the end at the top of the image.

    0  far baseline  x left doubles sideline     1  far baseline  x right doubles sideline
    2  near baseline x left doubles sideline     3  near baseline x right doubles sideline
    4  far baseline  x left singles sideline     5  near baseline x left singles sideline
    6  far baseline  x right singles sideline    7  near baseline x right singles sideline
    8  far service line  x left singles          9  far service line  x right singles
    10 near service line x left singles          11 near service line x right singles
    12 far T (service line x centre line)        13 near T
"""

from __future__ import annotations

import numpy as np

NUM_KEYPOINTS = 14
INPUT_W, INPUT_H = 640, 360

# Court coordinates in metres. Origin at the net centre, x to the right,
# y towards the far baseline.
_HW_D, _HW_S, _HL, _SL = 5.485, 4.115, 11.885, 6.40
COURT_POINTS = np.array([
    [-_HW_D, _HL], [_HW_D, _HL], [-_HW_D, -_HL], [_HW_D, -_HL],
    [-_HW_S, _HL], [-_HW_S, -_HL], [_HW_S, _HL], [_HW_S, -_HL],
    [-_HW_S, _SL], [_HW_S, _SL], [-_HW_S, -_SL], [_HW_S, -_SL],
    [0.0, _SL], [0.0, -_SL],
], dtype=np.float64)

# Court lines as pairs of keypoint indices (net not included).
COURT_LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]

# Keypoint index after a horizontal flip (left and right swap).
FLIP_INDEX = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]
