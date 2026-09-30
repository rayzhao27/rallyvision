"""Ball-localization metrics.

Every distance is reported in **original video pixels**, not network-input
pixels, so the numbers stay comparable if the input resolution changes.

Detection outcomes follow the TrackNet convention:

    TP  ball visible, predicted, and within the tolerance
    FP  predicted when the ball is not visible, or predicted too far away
    FN  ball visible but nothing predicted
    TN  ball not visible and nothing predicted

Accuracy@Npx is computed over frames where the ball is actually visible, so it
answers "when there is a ball, how often do we put it within N pixels?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .data.heatmap import decode_batch


@dataclass
class SampleResult:
    """One evaluated frame."""

    index: int
    clip_id: str = ""
    frame_file: str = ""
    visible: bool = False
    detected: bool = False
    score: float = 0.0
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    pred_x: Optional[float] = None
    pred_y: Optional[float] = None
    error_px: Optional[float] = None
    outcome: str = "tn"  # tp | fp | fn | tn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "clip_id": self.clip_id,
            "frame_file": self.frame_file,
            "visible": self.visible,
            "detected": self.detected,
            "score": round(self.score, 6),
            "gt_x": _round(self.gt_x),
            "gt_y": _round(self.gt_y),
            "pred_x": _round(self.pred_x),
            "pred_y": _round(self.pred_y),
            "error_px": _round(self.error_px),
            "outcome": self.outcome,
        }


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


class BallMetrics:
    """Streaming accumulator over batches; call :meth:`compute` at the end."""

    def __init__(
        self,
        tolerances_px: Sequence[float] = (5, 10, 20),
        pr_tolerance_px: float = 10.0,
        detection_threshold: float = 0.5,
        min_blob_area: int = 0,
        heatmap_downsample: int = 1,
        keep_samples: bool = True,
    ):
        self.tolerances = [float(t) for t in tolerances_px]
        self.pr_tolerance = float(pr_tolerance_px)
        self.detection_threshold = float(detection_threshold)
        self.min_blob_area = int(min_blob_area)
        self.heatmap_downsample = int(heatmap_downsample)
        self.keep_samples = bool(keep_samples)
        self.reset()

    def reset(self) -> None:
        self.samples: List[SampleResult] = []
        self._errors: List[float] = []
        self._visible_count = 0
        self._invisible_count = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0
        self._within: Dict[float, int] = {t: 0 for t in self.tolerances}

    # -- update -----------------------------------------------------------
    @torch.no_grad()
    def update(self, pred_heatmaps: torch.Tensor, batch: Dict[str, Any]) -> List[SampleResult]:
        """Accumulate one batch.

        Args:
            pred_heatmaps: probabilities shaped ``[N, 1, H, W]``.
            batch: a batch dict from :class:`~rallyvision.data.dataset.BallDataset`.
        """
        probs = pred_heatmaps.detach().float().cpu().numpy()
        decoded = decode_batch(
            probs,
            threshold=self.detection_threshold,
            min_blob_area=self.min_blob_area,
            subpixel=True,
        )

        visible = batch["visible"].detach().cpu().numpy().reshape(-1)
        target_xy = batch["target_xy"].detach().cpu().numpy().reshape(-1, 2)
        to_original = batch["to_original"].detach().cpu().numpy().reshape(-1, 2)
        indices = batch["index"].detach().cpu().numpy().reshape(-1)
        clip_ids = batch.get("clip_id") or [""] * len(indices)
        frame_files = batch.get("frame_file") or [""] * len(indices)

        results: List[SampleResult] = []
        scale_up = float(self.heatmap_downsample)
        for i, (point, score) in enumerate(decoded):
            is_visible = bool(visible[i] > 0.5)
            sx, sy = float(to_original[i][0]), float(to_original[i][1])

            gt_x = gt_y = None
            if is_visible:
                gt_x = float(target_xy[i][0]) * sx
                gt_y = float(target_xy[i][1]) * sy

            pred_x = pred_y = None
            if point is not None:
                pred_x = point[0] * scale_up * sx
                pred_y = point[1] * scale_up * sy

            error = None
            if gt_x is not None and pred_x is not None:
                error = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

            result = SampleResult(
                index=int(indices[i]),
                clip_id=str(clip_ids[i]) if i < len(clip_ids) else "",
                frame_file=str(frame_files[i]) if i < len(frame_files) else "",
                visible=is_visible,
                detected=point is not None,
                score=float(score),
                gt_x=gt_x,
                gt_y=gt_y,
                pred_x=pred_x,
                pred_y=pred_y,
                error_px=error,
                outcome=_classify(is_visible, point is not None, error, self.pr_tolerance),
            )
            self._ingest(result)
            results.append(result)

        if self.keep_samples:
            self.samples.extend(results)
        return results

    def _ingest(self, r: SampleResult) -> None:
        if r.visible:
            self._visible_count += 1
        else:
            self._invisible_count += 1

        if r.error_px is not None:
            self._errors.append(r.error_px)
            for tol in self.tolerances:
                if r.error_px <= tol:
                    self._within[tol] += 1

        self._tp += r.outcome == "tp"
        self._fp += r.outcome == "fp"
        self._fn += r.outcome == "fn"
        self._tn += r.outcome == "tn"

    # -- results ----------------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        total = self._visible_count + self._invisible_count
        errors = np.asarray(self._errors, dtype=np.float64)

        metrics: Dict[str, Any] = {
            "num_samples": total,
            "num_visible": self._visible_count,
            "num_invisible": self._invisible_count,
            "num_localized": int(errors.size),
            "true_positives": self._tp,
            "false_positives": self._fp,
            "false_negatives": self._fn,
            "true_negatives": self._tn,
        }

        if errors.size:
            metrics["mean_error_px"] = float(errors.mean())
            metrics["median_error_px"] = float(np.median(errors))
            metrics["p90_error_px"] = float(np.percentile(errors, 90))
            metrics["max_error_px"] = float(errors.max())
            metrics["rmse_px"] = float(np.sqrt((errors**2).mean()))
        else:
            for key in ("mean_error_px", "median_error_px", "p90_error_px", "max_error_px", "rmse_px"):
                metrics[key] = float("nan")

        # Denominator is visible frames: a missed detection counts against
        # accuracy rather than being silently excluded.
        for tol in self.tolerances:
            key = f"accuracy_at_{int(tol)}px" if float(tol).is_integer() else f"accuracy_at_{tol}px"
            metrics[key] = (self._within[tol] / self._visible_count) if self._visible_count else 0.0

        precision_denom = self._tp + self._fp
        recall_denom = self._tp + self._fn
        precision = self._tp / precision_denom if precision_denom else 0.0
        recall = self._tp / recall_denom if recall_denom else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["accuracy"] = ((self._tp + self._tn) / total) if total else 0.0
        metrics["detection_rate"] = (
            sum(1 for s in self.samples if s.detected) / total if (total and self.keep_samples) else None
        )
        metrics["pr_tolerance_px"] = self.pr_tolerance
        metrics["detection_threshold"] = self.detection_threshold
        return {k: v for k, v in metrics.items() if v is not None}

    def per_sample_records(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.samples]

    def worst_samples(self, n: int = 8) -> List[SampleResult]:
        """Largest localization errors first; misses (FN) count as worst."""
        localized = [s for s in self.samples if s.error_px is not None]
        misses = [s for s in self.samples if s.visible and not s.detected]
        localized.sort(key=lambda s: -(s.error_px or 0.0))
        return (misses + localized)[:n]

    def best_samples(self, n: int = 8) -> List[SampleResult]:
        localized = [s for s in self.samples if s.error_px is not None]
        localized.sort(key=lambda s: s.error_px or 0.0)
        return localized[:n]


def _classify(visible: bool, detected: bool, error: Optional[float], tolerance: float) -> str:
    if visible and detected:
        return "tp" if (error is not None and error <= tolerance) else "fp"
    if visible and not detected:
        return "fn"
    if not visible and detected:
        return "fp"
    return "tn"


def build_metrics(cfg, keep_samples: bool = True) -> BallMetrics:
    eval_cfg = cfg.eval
    return BallMetrics(
        tolerances_px=list(eval_cfg.get("tolerances_px", [5, 10, 20])),
        pr_tolerance_px=float(eval_cfg.get("pr_tolerance_px", 10)),
        detection_threshold=float(eval_cfg.get("detection_threshold", 0.5)),
        min_blob_area=int(eval_cfg.get("min_blob_area", 0)),
        heatmap_downsample=int(cfg.data.heatmap.get("downsample", 1)),
        keep_samples=keep_samples,
    )


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Human-readable block for logs and the notebook."""
    order = [
        ("num_samples", "samples", "{:,}"),
        ("num_visible", "visible frames", "{:,}"),
        ("num_localized", "localized", "{:,}"),
        ("mean_error_px", "mean error", "{:.2f} px"),
        ("median_error_px", "median error", "{:.2f} px"),
        ("p90_error_px", "p90 error", "{:.2f} px"),
        ("accuracy_at_5px", "accuracy@5px", "{:.4f}"),
        ("accuracy_at_10px", "accuracy@10px", "{:.4f}"),
        ("accuracy_at_20px", "accuracy@20px", "{:.4f}"),
        ("precision", "precision", "{:.4f}"),
        ("recall", "recall", "{:.4f}"),
        ("f1", "f1", "{:.4f}"),
        ("accuracy", "accuracy", "{:.4f}"),
    ]
    lines = []
    for key, label, fmt in order:
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            try:
                lines.append(f"{label:<16}: {fmt.format(value)}")
            except (ValueError, TypeError):
                lines.append(f"{label:<16}: {value}")
    counts = (
        f"tp={metrics.get('true_positives', 0)} fp={metrics.get('false_positives', 0)} "
        f"fn={metrics.get('false_negatives', 0)} tn={metrics.get('true_negatives', 0)}"
    )
    lines.append(f"{'counts':<16}: {counts}")
    return "\n".join(lines)
