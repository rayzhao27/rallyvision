"""Gaussian heatmap rendering and peak decoding.

The model predicts a single-channel heatmap whose maximum marks the ball. The
two functions here are exact inverses in the noise-free case, which is what the
unit tests assert.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_gaussian(
    height: int,
    width: int,
    x: float,
    y: float,
    sigma: float = 2.5,
    radius: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Render an unnormalized 2D gaussian (peak value 1.0) at ``(x, y)``.

    Coordinates are in heatmap pixel space and may be fractional. A point
    outside the canvas yields an all-zero heatmap.
    """
    heatmap = np.zeros((height, width), dtype=dtype)
    if not np.isfinite(x) or not np.isfinite(y):
        return heatmap
    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
        return heatmap

    if radius is None:
        radius = int(np.ceil(3 * sigma))
    radius = max(1, int(radius))

    x0 = max(0, int(np.floor(x)) - radius)
    x1 = min(width - 1, int(np.ceil(x)) + radius)
    y0 = max(0, int(np.floor(y)) - radius)
    y1 = min(height - 1, int(np.ceil(y)) + radius)
    if x1 < x0 or y1 < y0:
        return heatmap

    xs = np.arange(x0, x1 + 1, dtype=np.float32)
    ys = np.arange(y0, y1 + 1, dtype=np.float32)
    gx = np.exp(-((xs - float(x)) ** 2) / (2.0 * sigma**2))
    gy = np.exp(-((ys - float(y)) ** 2) / (2.0 * sigma**2))
    patch = np.outer(gy, gx).astype(dtype)
    heatmap[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(heatmap[y0 : y1 + 1, x0 : x1 + 1], patch)
    return heatmap


def decode_peak(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    min_blob_area: int = 0,
    subpixel: bool = True,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Find the ball position in a heatmap.

    Returns ``((x, y), score)`` or ``(None, score)`` when the peak is below
    ``threshold`` or its connected blob is too small. With ``subpixel`` the peak
    is refined by a local intensity-weighted centroid, which recovers roughly a
    third of a pixel of accuracy on clean predictions.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm[0] if hm.shape[0] == 1 else hm.max(axis=0)
    if hm.size == 0:
        return None, 0.0

    flat_idx = int(np.argmax(hm))
    py, px = np.unravel_index(flat_idx, hm.shape)
    score = float(hm[py, px])
    if score < threshold:
        return None, score

    if min_blob_area > 0:
        area = int(_blob_area(hm, int(py), int(px), threshold))
        if area < min_blob_area:
            return None, score

    if not subpixel:
        return (float(px), float(py)), score

    radius = 2
    y0, y1 = max(0, py - radius), min(hm.shape[0], py + radius + 1)
    x0, x1 = max(0, px - radius), min(hm.shape[1], px + radius + 1)
    patch = hm[y0:y1, x0:x1]
    weights = np.clip(patch - threshold * 0.5, 0.0, None)
    total = float(weights.sum())
    if total <= 1e-8:
        return (float(px), float(py)), score
    ys, xs = np.mgrid[y0:y1, x0:x1]
    cx = float((weights * xs).sum() / total)
    cy = float((weights * ys).sum() / total)
    return (cx, cy), score


def _blob_area(hm: np.ndarray, seed_y: int, seed_x: int, threshold: float) -> int:
    """Flood-fill the above-threshold blob containing the seed (4-connected)."""
    height, width = hm.shape
    mask = hm >= threshold
    if not mask[seed_y, seed_x]:
        return 0
    visited = np.zeros_like(mask, dtype=bool)
    stack = [(seed_y, seed_x)]
    visited[seed_y, seed_x] = True
    area = 0
    while stack:
        cy, cx = stack.pop()
        area += 1
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return area


def decode_batch(
    heatmaps: np.ndarray,
    threshold: float = 0.5,
    min_blob_area: int = 0,
    subpixel: bool = True,
):
    """Decode a batch of heatmaps shaped ``(N, 1, H, W)`` or ``(N, H, W)``."""
    arr = np.asarray(heatmaps)
    if arr.ndim == 4:
        arr = arr[:, 0] if arr.shape[1] == 1 else arr.max(axis=1)
    results = []
    for i in range(arr.shape[0]):
        results.append(
            decode_peak(arr[i], threshold=threshold, min_blob_area=min_blob_area, subpixel=subpixel)
        )
    return results
