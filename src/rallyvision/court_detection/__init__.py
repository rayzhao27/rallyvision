"""Court detection and homography.

A single-frame TrackNet predicts 14 court keypoints; four or more give a homography from court
metres to image pixels (``fit_homography``), used to map image points onto the court (``image_to_court``).

    geometry     keypoint layout and court dimensions
    data         training data (Roboflow COCO export), targets and augmentation
    model        load the keypoint model, decode keypoints from its heatmaps
    homography   fit / project / image_to_court / draw_court
    inference    per-frame calibration of a video
    training, evaluation, analysis, experiments
"""

from rallyvision.court_detection.data import (CourtDataset, augment, augment_appearance, augment_geometry,  # noqa: F401
                                              load_coco_split, preprocess, render_targets, split_by_video, video_id)
from rallyvision.court_detection.geometry import (COURT_LINES, COURT_POINTS, FLIP_INDEX, INPUT_H, INPUT_W,  # noqa: F401
                                                  NUM_KEYPOINTS)
from rallyvision.court_detection.homography import draw_court, fit_homography, image_to_court, project  # noqa: F401
from rallyvision.court_detection.model import decode_keypoints, load_court_model  # noqa: F401
