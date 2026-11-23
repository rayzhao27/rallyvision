"""Video inference: run the tracker over a clip and write an annotated MP4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from rallyvision.common.heatmap import decode_peak
from rallyvision.ball_tracking.data.transforms import bgr_to_rgb, resize_frame
from rallyvision.common.device import DeviceInfo, autocast_context
from rallyvision.common.logging_utils import get_logger
from rallyvision.ball_tracking.inference.tracker import BallTracker, Detection, track_statistics
from rallyvision.ball_tracking.viz import annotate_frame
from rallyvision.common.utils import format_duration, write_json

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    def summary(self) -> str:
        duration = self.frame_count / self.fps if self.fps else 0.0
        return (
            f"{self.path.name}: {self.width}x{self.height} @ {self.fps:.2f} fps, "
            f"{self.frame_count} frames ({format_duration(duration)})"
        )


def probe_video(path: Path | str) -> VideoInfo:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    try:
        return VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


@torch.no_grad()
def run_video_inference(
    model: torch.nn.Module,
    video_path: Path | str,
    output_path: Optional[Path | str],
    device_info: DeviceInfo,
    cfg,
    max_frames: Optional[int] = None,
    draw_heatmap: Optional[bool] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Track the ball through a video and optionally write an annotated MP4.

    Frames are pushed through a rolling buffer so each network input is the
    current frame plus its ``num_frames - 1`` predecessors, exactly matching
    how the dataset builds training samples.
    """
    model.eval().to(device_info.device)

    video = probe_video(video_path)
    logger.info("input video      : %s", video.summary())

    num_frames = int(cfg.data.num_frames)
    stride = int(cfg.data.frame_stride)
    in_h, in_w = int(cfg.data.input_height), int(cfg.data.input_width)
    downsample = int(cfg.data.heatmap.get("downsample", 1))
    threshold = float(cfg.inference.get("detection_threshold", cfg.eval.get("detection_threshold", 0.5)))
    min_blob_area = int(cfg.eval.get("min_blob_area", 0))
    trail_length = int(cfg.inference.get("trail_length", 8))
    if draw_heatmap is None:
        draw_heatmap = bool(cfg.inference.get("draw_heatmap", False))

    sx = video.width / float(in_w)
    sy = video.height / float(in_h)

    tracker = BallTracker(
        max_jump_px=float(cfg.tracker.get("max_jump_px", 100.0)),
        max_gap_frames=int(cfg.tracker.get("max_gap_frames", 4)),
        smooth_window=int(cfg.tracker.get("smooth_window", 0)),
    )

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video.path}")

    buffer: deque = deque(maxlen=(num_frames - 1) * stride + 1)
    original_frames: List[np.ndarray] = []
    heatmaps: List[Optional[np.ndarray]] = []
    frame_index = 0

    iterator = _progress(total=(min(video.frame_count, max_frames) if max_frames else video.frame_count),
                         enabled=progress)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            small = resize_frame(bgr_to_rgb(frame_bgr), in_w, in_h)
            buffer.append(small.astype(np.float32) / 255.0)
            # Repeat the first frame until the temporal buffer is primed.
            while len(buffer) < buffer.maxlen:
                buffer.appendleft(buffer[0])

            indices = [len(buffer) - 1 - offset * stride for offset in reversed(range(num_frames))]
            stack = np.concatenate([buffer[i] for i in indices], axis=2).transpose(2, 0, 1)
            tensor = torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0).to(device_info.device)

            with autocast_context(device_info):
                logits = model.forward_logits(tensor)
            probs = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

            point, score = decode_peak(probs, threshold=threshold, min_blob_area=min_blob_area)
            original_point: Optional[Tuple[float, float]] = None
            if point is not None:
                original_point = (point[0] * downsample * sx, point[1] * downsample * sy)
            tracker.update(frame_index, original_point, score)

            if output_path is not None:
                original_frames.append(bgr_to_rgb(frame_bgr))
                heatmaps.append(probs if draw_heatmap else None)

            frame_index += 1
            if iterator is not None:
                iterator.update(1)
    finally:
        cap.release()
        if iterator is not None:
            iterator.close()

    track = tracker.finalize(interpolate=bool(cfg.tracker.get("interpolate_gaps", True)))
    stats = track_statistics(track)
    logger.info(
        "tracked %d frames: %d detected, %d interpolated, %d rejected (coverage %.1f%%)",
        stats["frames"],
        stats["detected"],
        stats["interpolated"],
        stats["rejected"],
        100 * stats["coverage"],
    )

    written = None
    if output_path is not None:
        written = write_annotated_video(
            output_path,
            original_frames,
            track,
            heatmaps=heatmaps,
            fps=float(cfg.inference.get("output_fps") or video.fps),
            trail_length=trail_length,
        )
        logger.info("annotated video  : %s", written)

    return {
        "video": {
            "path": str(video.path),
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "frames_processed": frame_index,
        },
        "track": [d.to_dict() for d in track],
        "stats": stats,
        "output_video": str(written) if written else None,
        "settings": {
            "detection_threshold": threshold,
            "num_frames": num_frames,
            "input_height": in_h,
            "input_width": in_w,
        },
    }


def write_annotated_video(
    output_path: Path | str,
    frames_rgb: List[np.ndarray],
    track: List[Detection],
    heatmaps: Optional[List[Optional[np.ndarray]]] = None,
    fps: float = 30.0,
    trail_length: int = 8,
) -> Path:
    """Write the annotated MP4. Uses mp4v, which Colab and QuickTime both play."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        raise ValueError("no frames to write")

    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open a video writer for {output_path}")

    try:
        for i, frame in enumerate(frames_rgb):
            det = track[i] if i < len(track) else None
            point = (det.x, det.y) if det and det.has_position else None
            trail = [
                (t.x, t.y)
                for t in track[max(0, i - trail_length) : i + 1]
                if t.has_position
            ]
            lines = [f"frame {i}"]
            if det is not None:
                lines.append(
                    f"{det.source}"
                    + (f"  score {det.score:.2f}" if det.source == "detected" else "")
                )
            annotated = annotate_frame(
                frame,
                pred=point,
                trail=trail,
                heatmap=(heatmaps[i] if heatmaps and i < len(heatmaps) else None),
                text_lines=lines,
            )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path


def save_track(path: Path | str, result: Dict[str, Any]) -> Path:
    return write_json(path, result)


def _progress(total: Optional[int], enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="inference", leave=False)
