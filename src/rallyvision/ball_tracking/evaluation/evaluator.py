"""Validation and full evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from rallyvision.common.device import DeviceInfo, autocast_context
from rallyvision.ball_tracking.evaluation.metrics import BallMetrics, build_metrics
from rallyvision.common.utils import write_csv, write_json


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device_info: DeviceInfo,
    cfg,
    loss_fn: Optional[torch.nn.Module] = None,
    max_batches: Optional[int] = None,
    keep_samples: bool = False,
    progress: bool = False,
    desc: str = "eval",
) -> Tuple[Dict[str, Any], BallMetrics]:
    """Run the model over ``loader`` and return ``(metrics, accumulator)``."""
    model.eval()
    metrics = build_metrics(cfg, keep_samples=keep_samples)

    total_loss = 0.0
    num_batches = 0
    iterator: Any = loader
    if progress:
        iterator = _maybe_tqdm(loader, desc=desc, total=_limited_len(loader, max_batches))

    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        batch = _move(batch, device_info.device)
        with autocast_context(device_info):
            logits = model.forward_logits(batch["frames"])
            if loss_fn is not None:
                loss = loss_fn(logits, batch["heatmap"])
                total_loss += float(loss.detach())
        probs = torch.sigmoid(logits.float())
        metrics.update(probs, batch)
        num_batches += 1

    result = metrics.compute()
    if loss_fn is not None and num_batches:
        result["loss"] = total_loss / num_batches
    result["num_batches"] = num_batches
    return result, metrics


def evaluate_and_save(
    model: torch.nn.Module,
    loader: DataLoader,
    device_info: DeviceInfo,
    cfg,
    output_dir,
    prefix: str = "eval",
    loss_fn: Optional[torch.nn.Module] = None,
    save_per_sample: bool = True,
    progress: bool = True,
) -> Tuple[Dict[str, Any], BallMetrics]:
    """Evaluate and write ``<prefix>_metrics.json`` (+ per-sample CSV)."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, accumulator = evaluate(
        model,
        loader,
        device_info,
        cfg,
        loss_fn=loss_fn,
        keep_samples=save_per_sample,
        progress=progress,
        desc=prefix,
    )

    payload = {
        "metrics": metrics,
        "config": {
            "detection_threshold": float(cfg.eval.get("detection_threshold", 0.5)),
            "min_blob_area": int(cfg.eval.get("min_blob_area", 0)),
            "tolerances_px": list(cfg.eval.get("tolerances_px", [5, 10, 20])),
            "pr_tolerance_px": float(cfg.eval.get("pr_tolerance_px", 10)),
            "input_height": int(cfg.data.input_height),
            "input_width": int(cfg.data.input_width),
            "num_frames": int(cfg.data.num_frames),
        },
        "device": {"kind": device_info.kind, "name": device_info.name},
    }
    write_json(output_dir / f"{prefix}_metrics.json", payload)

    if save_per_sample and accumulator.samples:
        write_csv(output_dir / f"{prefix}_per_sample.csv", accumulator.per_sample_records())
    return metrics, accumulator


def _limited_len(loader: DataLoader, max_batches: Optional[int]) -> Optional[int]:
    try:
        total = len(loader)
    except TypeError:
        return None
    return min(total, max_batches) if max_batches is not None else total


def _maybe_tqdm(iterable, desc: str, total: Optional[int]):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)
