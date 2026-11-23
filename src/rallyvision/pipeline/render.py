"""Demo video: the untouched broadcast plus a top-down court with the bounces (side panel or top-right inset).

Earlier bounces are tennis-ball yellow, the newest one magenta with a white edge; the net is drawn at court
width; nothing is drawn on the video itself. This is the renderer of the reference Transformer-v3 demo
(previously local_archive/articrafts/event_transformer/render_side.py and render_pip.py), unchanged.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from rallyvision.court_detection.geometry import COURT_LINES, COURT_POINTS
from rallyvision.pipeline.court_map import CANVAS_H, CANVAS_W, HALF_W, to_canvas

OUT_H = 720
INSET_FRAC = 0.44            # inset height / video height (pip layout)
MARGIN = 14
OLD, NEW = (60, 220, 250), (255, 80, 230)    # BGR: yellow, magenta


def court_canvas():
    img = np.full((CANVAS_H, CANVAS_W, 3), (70, 120, 60), np.uint8)
    cv2.rectangle(img, to_canvas((-HALF_W, 11.885)), to_canvas((HALF_W, -11.885)), (150, 95, 45), -1)
    for a, b in COURT_LINES:
        cv2.line(img, to_canvas(COURT_POINTS[a]), to_canvas(COURT_POINTS[b]), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(img, to_canvas((-HALF_W, 0)), to_canvas((HALF_W, 0)), (220, 220, 220), 2, cv2.LINE_AA)   # net, court width
    return img


def draw_bounces(panel, bounces, frame_idx, scale, r_old, r_new):
    """Bounces up to ``frame_idx`` on the court canvas: earlier yellow, the newest magenta with a white edge."""
    shown = [b for b in bounces if b["video_frame"] <= frame_idx]
    for k, b in enumerate(shown):
        p = to_canvas(b["court_xy"])
        if k == len(shown) - 1:
            cv2.circle(panel, p, r_new + max(1, round(2 / scale)), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, p, r_new, NEW, -1, cv2.LINE_AA)
        else:
            cv2.circle(panel, p, r_old + max(1, round(1 / scale)), (40, 40, 40), -1, cv2.LINE_AA)
            cv2.circle(panel, p, r_old, OLD, -1, cv2.LINE_AA)
    return panel


def bounce_map(bounces) -> np.ndarray:
    """Still image: every placed bounce on the top-down court (the last frame of the video's court panel)."""
    placed = [b for b in bounces if b["court_xy"]]
    return draw_bounces(court_canvas(), placed, float("inf"), 1.0, 7, 10)


def render_video(video, bounces, out, layout: str = "side", crop_top=None, crop_bottom=None) -> str:
    """Write the demo MP4 (H.264). ``bounces``: dicts with video_frame and court_xy (None = not placed).

    side: video (rows [crop_top, crop_bottom) of the source) | court panel; pip: court inset top-right.
    """
    bounces = [b for b in bounces if b["court_xy"]]
    cap = cv2.VideoCapture(str(video))
    fps, w0, h0 = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
    top, bottom = int(crop_top or 0), int(crop_bottom or h0)
    h0 = bottom - top
    W = int(round(w0 * OUT_H / h0)) // 2 * 2
    base = court_canvas()
    if layout == "side":
        pw = int(round(CANVAS_W * OUT_H / CANVAS_H)) // 2 * 2
        scale = OUT_H / CANVAS_H
        r_old, r_new = max(1, round(7 / scale)), max(1, round(10 / scale))      # ~7 and ~10 px on the panel
        size = (W + pw, OUT_H)
    elif layout == "pip":
        ih = int(OUT_H * INSET_FRAC)
        iw = int(round(CANVAS_W * ih / CANVAS_H))
        scale = ih / CANVAS_H
        r_old, r_new = max(1, round(6 / scale)), max(1, round(8.5 / scale))   # ~6 and ~8.5 px after scaling
        x0, y0 = W - iw - MARGIN, MARGIN
        size = (W, OUT_H)
    else:
        raise ValueError(f"layout must be side or pip, not {layout!r}")
    raw = Path(out).with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        panel = draw_bounces(base.copy(), bounces, f, scale, r_old, r_new)
        if layout == "side":
            writer.write(np.hstack([cv2.resize(frame[top:bottom], (W, OUT_H), interpolation=cv2.INTER_AREA),
                                    cv2.resize(panel, (pw, OUT_H), interpolation=cv2.INTER_AREA)]))
        else:
            img = cv2.resize(frame[top:bottom], (W, OUT_H), interpolation=cv2.INTER_AREA)
            small = cv2.resize(panel, (iw, ih), interpolation=cv2.INTER_AREA)
            roi = img[y0:y0 + ih, x0:x0 + iw]
            img[y0:y0 + ih, x0:x0 + iw] = cv2.addWeighted(small, 0.92, roi, 0.08, 0)
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        f += 1
    writer.release()
    # H.264; the source audio is copied over when there is one ("1:a?" = optional)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(video), "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-shortest", str(out)], check=True)
    raw.unlink()
    return f"{f} frames -> {out} ({layout}, {size[0]}x{size[1]})"
