"""Court keypoints: label loading, flip naming, targets, decoding, homography, training smoke test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from rallyvision.court_detection import (COURT_POINTS, FLIP_INDEX, INPUT_H, INPUT_W, CourtDataset, augment_geometry,
                               decode_keypoints, draw_court, fit_homography, image_to_court, load_coco_split,
                               project, render_targets, split_by_video, video_id)

REPO = Path(__file__).resolve().parent.parent
# Storage order of the Roboflow export (categories[].keypoints).
ROBOFLOW_ORDER = [0, 4, 6, 1, 2, 5, 7, 3, 8, 12, 9, 11, 13, 10]


def broadcast_h(w=640, h=360) -> np.ndarray:
    """A plausible court->image homography: court fills the middle of a 16:9 frame."""
    img_pts = np.array([[0.30 * w, 0.20 * h], [0.70 * w, 0.20 * h], [0.10 * w, 0.85 * h], [0.90 * w, 0.85 * h]])
    H, _ = cv2.findHomography(COURT_POINTS[:4], img_pts)
    return H


def write_fake_export(root: Path, videos=("IMG_9400_MOV", "NeHvxtGxj7s", "Bangkok-Open-_mov", "glasgow_mp4"),
                      frames=3) -> np.ndarray:
    """Roboflow-like COCO export: 1280x720 court renders stretched to 640x640, keypoints in storage order."""
    H = broadcast_h(1280, 720)
    kps = project(H, COURT_POINTS)
    canvas = np.full((720, 1280, 3), (60, 120, 40), np.uint8)
    draw_court(canvas, H, (255, 255, 255), 3)
    img = cv2.resize(canvas, (640, 640))
    stretched = kps * [640 / 1280, 640 / 720]

    for split in ("train", "valid", "test"):
        d = root / split
        d.mkdir(parents=True)
        images, anns = [], []
        for v in videos:
            for f in range(frames):
                name = f"{v}-{f:04d}_jpg.rf.{split}{f}abc.jpg"
                cv2.imwrite(str(d / name), img)
                flat = []
                for k in ROBOFLOW_ORDER:
                    flat += [float(stretched[k, 0]), float(stretched[k, 1]), 2]
                if v == "glasgow_mp4":           # one point unlabeled
                    flat[3 * 13:3 * 13 + 3] = [0, 0, 0]
                images.append({"id": len(images), "file_name": name, "width": 640, "height": 640})
                anns.append({"id": len(anns), "image_id": images[-1]["id"], "category_id": 1, "keypoints": flat})
        coco = {"images": images, "annotations": anns, "categories": [
            {"id": 0, "name": "tennis-court", "supercategory": "none"},
            {"id": 1, "name": "court", "keypoints": [str(k) for k in ROBOFLOW_ORDER]}]}
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
    return kps


def test_load_reorders_by_name(tmp_path):
    kps = write_fake_export(tmp_path)
    samples = load_coco_split(tmp_path / "train")
    assert len(samples) == 12
    s = next(x for x in samples if "IMG_9400" in x["file"])
    np.testing.assert_allclose(s["kps"] * [1280, 720], kps, atol=1e-6)
    assert s["labeled"].all()
    g = next(x for x in samples if "glasgow" in x["file"])
    assert not g["labeled"][ROBOFLOW_ORDER[13]] and g["labeled"].sum() == 13


def test_video_id():
    assert video_id("IMG_9400_MOV-0014_jpg.rf.ab12cd.jpg") == "IMG_9400_MOV"
    assert video_id("NeHvxtGxj7s_250_png.rf.ab12cd.jpg") == "NeHvxtGxj7s"
    # Real export: YouTube frames carry a double suffix; both frames are the same video.
    assert video_id("1-UMg_jEcdA_3350_png_jpg.rf.7a1f9cd3.jpg") == "1-UMg_jEcdA"
    assert video_id("1-UMg_jEcdA_2150_png_jpg.rf.a57546c1.jpg") == "1-UMg_jEcdA"
    assert video_id("full_match_video_MP4-0027_jpg.rf.1332f4d4.jpg") == "full_match_video_MP4"
    assert video_id("GianlucaBalotta_UCI_1_17_26_mp4-0003_jpg.rf.x9.jpg") == "GianlucaBalotta_UCI_1_17_26_mp4"


def test_split_keeps_each_video_in_one_split():
    samples = [{"file": f"vid{v}-{f:04d}_jpg.rf.x.jpg"} for v in range(200) for f in range(5)]
    splits = split_by_video(samples)
    owners = {}
    for name, items in splits.items():
        for s in items:
            assert owners.setdefault(video_id(s["file"]), name) == name
    assert all(len(v) for v in splits.values())


def test_flip_index_mirrors_the_court():
    np.testing.assert_allclose(COURT_POINTS[FLIP_INDEX], COURT_POINTS * [-1, 1])
    assert sorted(FLIP_INDEX) == list(range(14))


def test_geometry_augment_moves_keypoints_with_the_image():
    H = broadcast_h()
    kps = project(H, COURT_POINTS)
    labeled = np.ones(14, bool)
    rng = np.random.default_rng(0)
    for _ in range(30):
        img = np.zeros((INPUT_H, INPUT_W, 3), np.uint8)
        for k, (x, y) in enumerate(kps):          # brightness encodes the keypoint index
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (10 + 15 * k,) * 3, -1)
        out, k2, lab2 = augment_geometry(img, kps, labeled, rng)
        seen = {}
        for k in range(14):
            x, y = np.round(k2[k]).astype(int)
            if 0 <= x < INPUT_W and 0 <= y < INPUT_H:
                seen[k] = int(out[y, x, 0])
        # Either no flip (point k is the dot drawn for k) or a flip (it is the dot drawn for its mirror).
        same = all(abs(v - (10 + 15 * k)) <= 8 for k, v in seen.items())
        flipped = all(abs(v - (10 + 15 * FLIP_INDEX[k])) <= 8 for k, v in seen.items())
        assert seen and (same or flipped), seen


def test_render_targets_and_decode_roundtrip():
    kps = torch.tensor([[[100.3, 50.7]] * 14])
    kps[0, 1] = torch.tensor([-20.0, 50.0])        # outside the image
    labeled = torch.ones(1, 14, dtype=torch.bool)
    labeled[0, 2] = False
    hm = render_targets(kps, labeled).numpy()[0]
    assert hm.shape == (14, INPUT_H, INPUT_W)
    assert hm[1].max() == 0 and hm[2].max() == 0
    pts, scores = decode_keypoints(hm, threshold=0.5)
    assert np.isnan(pts[1]).all() and np.isnan(pts[2]).all()
    np.testing.assert_allclose(pts[0], [100.3, 50.7], atol=0.3)


def test_homography_recovers_missing_points_and_rejects_bad_frames():
    H = broadcast_h()
    truth = project(H, COURT_POINTS)

    pts = truth.copy()
    pts[[0, 3, 5, 12]] = np.nan                    # four points missed
    pts[7] += [30, -20]                            # one wrong detection
    r = fit_homography(pts)
    assert r["valid"] and not r["inliers"][7]
    assert np.abs(r["points"] - truth).max() < 0.5
    np.testing.assert_allclose(image_to_court(r["H"], truth[13]), COURT_POINTS[[13]], atol=1e-3)

    only_baseline = np.full_like(truth, np.nan)
    only_baseline[[2, 5, 7, 3]] = truth[[2, 5, 7, 3]]
    assert not fit_homography(only_baseline)["valid"]

    three = np.full_like(truth, np.nan)
    three[[0, 1, 2]] = truth[[0, 1, 2]]
    assert not fit_homography(three)["valid"]


def test_dataset_item_shapes(tmp_path):
    write_fake_export(tmp_path)
    ds = CourtDataset(load_coco_split(tmp_path / "train"), augment=True)
    x, kps, labeled = ds[0]
    assert x.shape == (3, INPUT_H, INPUT_W) and 0 <= x.min() and x.max() <= 1
    assert kps.shape == (14, 2) and labeled.dtype == torch.bool


def test_train_and_evaluate_smoke(tmp_path):
    write_fake_export(tmp_path / "data", videos=tuple(f"vid{i}_mp4" for i in range(40)), frames=1)
    run = tmp_path / "run"
    common = [sys.executable]
    subprocess.run(common + ["-m", "rallyvision.court_detection.training.train", "--data", str(tmp_path / "data"), "--out", str(run),
                             "--epochs", "2", "--batch", "2", "--width", "0.125", "--workers", "0", "--device", "cpu",
                             "--max-batches", "2"], check=True, cwd=REPO)
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    subprocess.run(common + ["-m", "rallyvision.court_detection.training.train", "--data", str(tmp_path / "data"), "--out", str(run),
                             "--epochs", "3", "--batch", "2", "--width", "0.125", "--workers", "0", "--device", "cpu",
                             "--max-batches", "1", "--resume"], check=True, cwd=REPO)
    assert len((run / "history.csv").read_text().strip().splitlines()) == 4   # header + 3 epochs
    subprocess.run(common + ["-m", "rallyvision.court_detection.evaluation.evaluate", "--data", str(tmp_path / "data"),
                             "--checkpoint", str(run / "best.pt"), "--split", "val", "--out", str(run / "eval"),
                             "--device", "cpu", "--overlays", "2"], check=True, cwd=REPO)
    assert (run / "eval" / "metrics.json").is_file()
    subprocess.run(common + ["-m", "rallyvision.court_detection.evaluation.sweep_threshold", "--data", str(tmp_path / "data"),
                             "--checkpoint", str(run / "best.pt"), "--out", str(run / "sweep"), "--device", "cpu"],
                   check=True, cwd=REPO)
    chosen = json.loads((run / "sweep" / "chosen.json").read_text())
    assert 0.05 <= chosen["threshold"] <= 0.95 and (run / "sweep" / "sweep.png").is_file()
