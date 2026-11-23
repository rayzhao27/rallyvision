"""Homography from detected keypoints (court metres -> image pixels), projection and drawing."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from rallyvision.court_detection.geometry import COURT_LINES, COURT_POINTS, NUM_KEYPOINTS


def fit_homography(pts: np.ndarray, ransac_px: float = 5.0, max_err_px: float = 5.0, min_area_m2: float = 20.0) -> Dict:
    """Court metres -> image pixels from the detected keypoints, with sanity checks.

    Returns {"valid", "reason", "H", "inliers" [14 bool], "err_px" (median inlier error),
    "points" [14, 2] all keypoints projected by H}. A frame is invalid when fewer than 4
    keypoints agree, when they cover too little of the court (near-collinear, e.g. one
    baseline only), when their error is large, or when the projected court is not a
    convex quadrilateral.
    """
    out = {"valid": False, "reason": "", "H": None, "inliers": np.zeros(NUM_KEYPOINTS, bool),
           "err_px": float("nan"), "points": None}
    found = ~np.isnan(pts[:, 0])
    if found.sum() < 4:
        out["reason"] = f"only {int(found.sum())} keypoints"
        return out

    H, mask = cv2.findHomography(COURT_POINTS[found], pts[found], cv2.RANSAC, ransac_px)
    if H is None:
        out["reason"] = "findHomography failed"
        return out
    inliers = np.zeros(NUM_KEYPOINTS, bool)
    inliers[np.flatnonzero(found)[mask.ravel() > 0]] = True
    out.update(H=H, inliers=inliers)
    if inliers.sum() < 4:
        out["reason"] = f"only {int(inliers.sum())} inliers"
        return out

    hull = cv2.convexHull(COURT_POINTS[inliers].astype(np.float32))
    if cv2.contourArea(hull) < min_area_m2:
        out["reason"] = "inliers nearly collinear"
        return out

    proj = project(H, COURT_POINTS)
    out["points"] = proj
    out["err_px"] = float(np.median(np.linalg.norm(proj[inliers] - pts[inliers], axis=1)))
    if out["err_px"] > max_err_px:
        out["reason"] = f"error {out['err_px']:.1f}px"
        return out

    corners = project(H, COURT_POINTS[[0, 1, 3, 2]])
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)):
        out["reason"] = "court not convex"
        return out
    out["valid"] = True
    return out


def project(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a homography to [N, 2] points. Points behind the camera come back as NaN."""
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p[:, :2] / p[:, 2:3]
    out[p[:, 2] <= 1e-9] = np.nan
    return out


def image_to_court(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Image pixels -> court metres, using the inverse of a court->image homography."""
    return project(np.linalg.inv(H), np.asarray(xy, dtype=np.float64).reshape(-1, 2))


def draw_court(img: np.ndarray, H: np.ndarray, color=(255, 128, 0), thickness: int = 1) -> np.ndarray:
    """Draw the court lines projected by H onto a BGR image (in place) and return it."""
    p = project(H, COURT_POINTS)
    for a, b in COURT_LINES:
        if np.isfinite(p[[a, b]]).all():
            cv2.line(img, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), color, thickness, cv2.LINE_AA)
    return img
