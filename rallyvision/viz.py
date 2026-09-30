"""Visualization helpers for the notebook's sanity checks and for annotated video."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .data.heatmap import decode_peak

GT_COLOR = (0, 255, 0)  # green, RGB
PRED_COLOR = (255, 64, 64)  # red, RGB
TRAIL_COLOR = (255, 200, 0)


def frames_from_stack(stack: torch.Tensor, num_frames: int) -> List[np.ndarray]:
    """Split a ``[N*3, H, W]`` tensor back into a list of uint8 RGB frames."""
    array = stack.detach().cpu().numpy()
    channels_per_frame = array.shape[0] // num_frames
    frames = []
    for i in range(num_frames):
        chunk = array[i * channels_per_frame : (i + 1) * channels_per_frame]
        frames.append((np.clip(chunk.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
    return frames


def heatmap_to_color(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm[0]
    hm = np.clip(hm, 0.0, 1.0)
    colored = cv2.applyColorMap((hm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim == 3:
        hm = hm[0]
    if hm.shape[:2] != frame_rgb.shape[:2]:
        hm = cv2.resize(hm, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = heatmap_to_color(hm)
    mask = np.clip(hm, 0.0, 1.0)[..., None]
    return np.clip(frame_rgb * (1 - alpha * mask) + colored * (alpha * mask), 0, 255).astype(np.uint8)


def draw_marker(
    frame: np.ndarray,
    point: Optional[Tuple[float, float]],
    color: Tuple[int, int, int],
    radius: int = 7,
    label: Optional[str] = None,
    cross: bool = False,
) -> np.ndarray:
    if point is None:
        return frame
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.circle(frame, (x, y), radius, color, 2)
    if cross:
        cv2.line(frame, (x - radius - 4, y), (x + radius + 4, y), color, 1)
        cv2.line(frame, (x, y - radius - 4), (x, y + radius + 4), color, 1)
    if label:
        cv2.putText(frame, label, (x + radius + 4, y - radius), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return frame


def annotate_frame(
    frame_rgb: np.ndarray,
    pred: Optional[Tuple[float, float]] = None,
    gt: Optional[Tuple[float, float]] = None,
    trail: Optional[Sequence[Tuple[float, float]]] = None,
    heatmap: Optional[np.ndarray] = None,
    text_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Draw predictions, ground truth, a motion trail and a text header."""
    out = frame_rgb.copy()
    if heatmap is not None:
        out = overlay_heatmap(out, heatmap)

    if trail:
        points = [p for p in trail if p is not None]
        for i in range(1, len(points)):
            fade = i / max(1, len(points))
            color = tuple(int(c * (0.35 + 0.65 * fade)) for c in TRAIL_COLOR)
            cv2.line(
                out,
                (int(points[i - 1][0]), int(points[i - 1][1])),
                (int(points[i][0]), int(points[i][1])),
                color,
                2,
                cv2.LINE_AA,
            )

    draw_marker(out, gt, GT_COLOR, radius=9, label="gt", cross=True)
    draw_marker(out, pred, PRED_COLOR, radius=6, label="pred")

    for i, line in enumerate(text_lines or []):
        cv2.putText(out, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return out


@torch.no_grad()
def predict_samples(
    model: torch.nn.Module,
    dataset,
    indices: Sequence[int],
    device: torch.device,
    detection_threshold: float = 0.5,
    min_blob_area: int = 0,
) -> List[Dict[str, Any]]:
    """Run the model on specific dataset indices and return everything needed to plot."""
    model.eval()
    results: List[Dict[str, Any]] = []
    for index in indices:
        sample = dataset[int(index)]
        frames = sample["frames"].unsqueeze(0).to(device)
        probs = torch.sigmoid(model.forward_logits(frames).float())[0, 0].cpu().numpy()

        point, score = decode_peak(probs, threshold=detection_threshold, min_blob_area=min_blob_area)
        downsample = getattr(dataset, "heatmap_downsample", 1)
        pred_xy = None if point is None else (point[0] * downsample, point[1] * downsample)

        visible = bool(sample["visible"].item() > 0.5)
        gt_xy = tuple(sample["target_xy"].tolist()) if visible else None
        scale = sample["to_original"].tolist()

        error = None
        if gt_xy is not None and pred_xy is not None:
            error = float(
                np.hypot(
                    (pred_xy[0] - gt_xy[0]) * scale[0],
                    (pred_xy[1] - gt_xy[1]) * scale[1],
                )
            )

        results.append(
            {
                "index": int(index),
                "frames": frames_from_stack(sample["frames"], dataset.num_frames),
                "heatmap": probs,
                "target_heatmap": sample["heatmap"][0].numpy(),
                "gt_xy": gt_xy,
                "pred_xy": pred_xy,
                "score": float(score),
                "visible": visible,
                "error_px": error,
                "clip_id": sample.get("clip_id", ""),
                "frame_file": sample.get("frame_file", ""),
            }
        )
    return results


def plot_predictions(
    results: Sequence[Dict[str, Any]],
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Grid of ``input frame | prediction overlay | predicted heatmap`` rows."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(results)
    if rows == 0:
        raise ValueError("nothing to plot")
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.1 * rows), squeeze=False)

    for r, result in enumerate(results):
        frame = result["frames"][-1]
        axes[r][0].imshow(frame)
        axes[r][0].set_title(f"{result['clip_id']} / {result['frame_file']}", fontsize=8)

        annotated = annotate_frame(frame, pred=result["pred_xy"], gt=result["gt_xy"])
        axes[r][1].imshow(annotated)
        error = result["error_px"]
        if error is not None:
            status = f"error {error:.1f}px"
        elif result["visible"] and result["pred_xy"] is None:
            status = "MISS (no detection)"
        elif not result["visible"] and result["pred_xy"] is not None:
            status = "FALSE POSITIVE"
        else:
            status = "no ball / no detection"
        axes[r][1].set_title(f"{status}  (score {result['score']:.2f})", fontsize=8)

        axes[r][2].imshow(result["heatmap"], cmap="inferno", vmin=0.0, vmax=1.0)
        axes[r][2].set_title("predicted heatmap", fontsize=8)

        for c in range(3):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if not show:
        import matplotlib.pyplot as plt  # noqa: F811

        plt.close(fig)
    return fig


def plot_history(history: Sequence[Dict[str, Any]], save_path: Optional[Path] = None, show: bool = False):
    """Loss and accuracy curves from a run's `history.json`."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h.get("epoch", i + 1) for i, h in enumerate(history)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for key, label in (("train/loss", "train"), ("val/loss", "val")):
        values = [h.get(key) for h in history]
        if any(v is not None for v in values):
            axes[0].plot(epochs, values, marker="o", ms=3, label=label)
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key in ("val/accuracy_at_5px", "val/accuracy_at_10px", "val/accuracy_at_20px"):
        values = [h.get(key) for h in history]
        if any(v is not None for v in values):
            axes[1].plot(epochs, values, marker="o", ms=3, label=key.split("/")[-1])
    axes[1].set_title("validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    for key, label in (("val/mean_error_px", "mean"), ("val/median_error_px", "median")):
        values = [h.get(key) for h in history]
        if any(v is not None for v in values):
            axes[2].plot(epochs, values, marker="o", ms=3, label=label)
    axes[2].set_title("validation localization error (px)")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig
